# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math

import torch

from molt.utils.logprob_audit import LogprobAuditWriter, compute_logprob_audit_metrics, rank_action_token_budget


def test_logprob_audit_metrics_use_only_action_tokens_and_report_raw_ratio():
    learner = torch.tensor([[0.0, math.log(2.0), 99.0]])
    rollout = torch.zeros(1, 3)
    mask = torch.tensor([[True, True, False]])

    metrics = compute_logprob_audit_metrics(learner, rollout, mask)

    torch.testing.assert_close(metrics["audit/delta_mean"], torch.tensor(math.log(2.0) / 2.0))
    torch.testing.assert_close(metrics["audit/ratio_mean"], torch.tensor(1.5))
    torch.testing.assert_close(metrics["audit/ratio_second_moment"], torch.tensor(2.5))
    torch.testing.assert_close(metrics["audit/microbatch_ess"], torch.tensor(9.0 / 5.0))
    torch.testing.assert_close(metrics["audit/nonfinite_rate"], torch.tensor(0.0))


def test_logprob_audit_metrics_surface_nonfinite_and_support_failure():
    learner = torch.tensor([[0.0, 0.0]])
    rollout = torch.tensor([[-torch.inf, torch.nan]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    metrics = compute_logprob_audit_metrics(learner, rollout, mask)

    torch.testing.assert_close(metrics["audit/nonfinite_rate"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["audit/support_violation_rate"], torch.tensor(0.5))


def test_logprob_audit_writer_retains_prefix_alignment_and_versions(tmp_path):
    writer = LogprobAuditWriter(str(tmp_path), max_action_tokens=2, metadata={"run_id": "test"})
    written = writer.write_batch(
        torch.tensor([[10, 11, 12, 13]]),
        torch.tensor([[False, True, True]]),
        torch.tensor([[-9.0, -0.2, -0.3]]),
        torch.tensor([[-8.0, -0.4, -0.7]]),
        info={"rollout_actor_version": torch.tensor([4]), "learner_version": torch.tensor([5])},
        indices=[7],
        group_ids=["group"],
        rollout_ids=["rollout"],
    )

    assert written == 2
    rows = [json.loads(line) for line in (tmp_path / "audit_records.jsonl").read_text().splitlines()]
    assert rows[0]["record_type"] == "header"
    record = rows[1]
    assert record["sequence_token_ids"] == [10, 11, 12, 13]
    assert record["action_step_positions"] == [1, 2]
    assert record["action_token_ids"] == [12, 13]
    assert record["rollout_actor_version"] == 4
    assert record["learner_version"] == 5
    assert record["rollout_logprob_status"] == ["finite", "finite"]
    assert record["learner_logprob_status"] == ["finite", "finite"]
    assert record["support_violations"] == [False, False]


def test_logprob_audit_writer_preserves_nonfinite_classes_and_support(tmp_path):
    writer = LogprobAuditWriter(str(tmp_path), max_action_tokens=2, metadata={})
    writer.write_batch(
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[True, True]]),
        torch.tensor([[0.0, torch.nan]]),
        torch.tensor([[-torch.inf, torch.inf]]),
        info={},
        indices=None,
        group_ids=[],
        rollout_ids=[],
    )

    rows = [json.loads(line) for line in (tmp_path / "audit_records.jsonl").read_text().splitlines()]
    assert rows[0]["schema_version"] == 2
    record = rows[1]
    assert record["schema_version"] == 2
    assert record["rollout_log_probs"] == [None, None]
    assert record["learner_log_probs"] == [0.0, None]
    assert record["rollout_logprob_status"] == ["neg_inf", "pos_inf"]
    assert record["learner_logprob_status"] == ["finite", "nan"]
    assert record["support_violations"] == [True, False]


def test_logprob_audit_writer_does_not_split_trajectory_at_budget(tmp_path):
    writer = LogprobAuditWriter(str(tmp_path), max_action_tokens=1, metadata={})
    written = writer.write_batch(
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[True, True]]),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        info={},
        indices=None,
        group_ids=[],
        rollout_ids=[],
    )

    assert written == 0
    assert len((tmp_path / "audit_records.jsonl").read_text().splitlines()) == 1


def test_logprob_audit_writer_supports_rank_sharded_filename(tmp_path):
    writer = LogprobAuditWriter(
        str(tmp_path), max_action_tokens=1, metadata={"rank": 3}, filename="audit_records.rank00003.jsonl"
    )
    writer.write_batch(
        torch.tensor([[1, 2]]),
        torch.tensor([[True]]),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        info={},
        indices=None,
        group_ids=[],
        rollout_ids=[],
    )

    rows = [json.loads(line) for line in (tmp_path / "audit_records.rank00003.jsonl").read_text().splitlines()]
    assert rows[0]["rank"] == 3
    assert rows[1]["record_type"] == "trajectory"


def test_rank_action_token_budget_preserves_total_and_balances_remainder():
    budgets = [rank_action_token_budget(10, rank, 4) for rank in range(4)]
    assert budgets == [3, 3, 2, 2]
    assert sum(budgets) == 10
