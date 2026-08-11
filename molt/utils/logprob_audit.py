# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded rollout-versus-learner log-probability audit records."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


def compute_logprob_audit_metrics(
    learner_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return raw, action-token-only mismatch metrics without hiding bad values."""

    mask = action_mask.bool()
    learner = learner_log_probs.detach().float()[mask]
    rollout = rollout_log_probs.detach().float()[mask]
    if learner.numel() == 0:
        zero = learner_log_probs.new_zeros((), dtype=torch.float32)
        return {
            "audit/delta_mean": zero,
            "audit/abs_delta_mean": zero,
            "audit/abs_delta_p50": zero,
            "audit/abs_delta_p90": zero,
            "audit/abs_delta_p95": zero,
            "audit/abs_delta_p99": zero,
            "audit/abs_delta_p999": zero,
            "audit/abs_delta_max": zero,
            "audit/ratio_mean": zero,
            "audit/ratio_second_moment": zero,
            "audit/microbatch_ess": zero,
            "audit/nonfinite_rate": zero,
            "audit/support_violation_rate": zero,
        }

    support_violation = torch.isfinite(learner) & torch.isneginf(rollout)
    finite_inputs = torch.isfinite(learner) & torch.isfinite(rollout)
    delta = learner - rollout
    ratio = torch.exp(delta)
    finite_ratio = finite_inputs & torch.isfinite(delta) & torch.isfinite(ratio)
    nonfinite = ~finite_ratio

    finite_delta = delta[finite_ratio]
    finite_weights = ratio[finite_ratio]
    zero = learner.new_zeros(())
    if finite_delta.numel():
        absolute = finite_delta.abs()
        quantiles = torch.quantile(
            absolute,
            torch.tensor([0.50, 0.90, 0.95, 0.99, 0.999], device=absolute.device),
        )
        ratio_sum = finite_weights.sum()
        ratio_second_sum = finite_weights.square().sum()
        ess = ratio_sum.square() / ratio_second_sum if ratio_second_sum > 0 else zero
        delta_mean = finite_delta.mean()
        absolute_mean = absolute.mean()
        absolute_max = absolute.max()
        ratio_mean = finite_weights.mean()
        ratio_second_moment = finite_weights.square().mean()
    else:
        quantiles = learner.new_zeros(5)
        ess = zero
        delta_mean = zero
        absolute_mean = zero
        absolute_max = zero
        ratio_mean = zero
        ratio_second_moment = zero

    return {
        "audit/delta_mean": delta_mean,
        "audit/abs_delta_mean": absolute_mean,
        "audit/abs_delta_p50": quantiles[0],
        "audit/abs_delta_p90": quantiles[1],
        "audit/abs_delta_p95": quantiles[2],
        "audit/abs_delta_p99": quantiles[3],
        "audit/abs_delta_p999": quantiles[4],
        "audit/abs_delta_max": absolute_max,
        "audit/ratio_mean": ratio_mean,
        "audit/ratio_second_moment": ratio_second_moment,
        "audit/microbatch_ess": ess,
        "audit/nonfinite_rate": nonfinite.float().mean(),
        "audit/support_violation_rate": support_violation.float().mean(),
    }


def _json_float(value: torch.Tensor) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _float_status(value: torch.Tensor) -> str:
    number = float(value)
    if math.isnan(number):
        return "nan"
    if number == math.inf:
        return "pos_inf"
    if number == -math.inf:
        return "neg_inf"
    return "finite"


def _metadata_value(info: dict[str, Any], key: str, index: int) -> Any:
    value = info.get(key)
    if isinstance(value, list):
        value = value[index] if index < len(value) else None
    elif isinstance(value, torch.Tensor) and value.dim() > 0:
        value = value[index] if index < value.shape[0] else None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"audit metadata {key} must be scalar per sample")
        return value.item()
    return value


class LogprobAuditWriter:
    """Write whole-trajectory JSONL records up to an action-token budget."""

    def __init__(
        self,
        output_dir: str,
        max_action_tokens: int,
        metadata: dict[str, Any],
        filename: str = "audit_records.jsonl",
    ):
        if max_action_tokens <= 0:
            raise ValueError("logprob audit max_action_tokens must be positive")
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / filename
        self._stream = self.path.open("x", encoding="utf-8")
        self.max_action_tokens = max_action_tokens
        self.action_tokens = 0
        self.records = 0
        self._write({"record_type": "header", "schema_version": 2, **metadata})

    def _write(self, payload: dict[str, Any]) -> None:
        self._stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        self._stream.flush()

    def write_batch(
        self,
        sequences: torch.Tensor,
        action_mask: torch.Tensor,
        learner_log_probs: torch.Tensor,
        rollout_log_probs: torch.Tensor,
        *,
        info: dict[str, Any],
        indices: list[int] | None,
        group_ids: list[str],
        rollout_ids: list[str],
    ) -> int:
        sequences = sequences.detach().cpu()
        action_mask = action_mask.detach().bool().cpu()
        learner_log_probs = learner_log_probs.detach().float().cpu()
        rollout_log_probs = rollout_log_probs.detach().float().cpu()
        written = 0
        for batch_index in range(sequences.shape[0]):
            positions = torch.nonzero(action_mask[batch_index], as_tuple=False).flatten()
            action_count = int(positions.numel())
            if action_count == 0:
                continue
            if self.action_tokens + action_count > self.max_action_tokens:
                continue
            learner = learner_log_probs[batch_index, positions]
            rollout = rollout_log_probs[batch_index, positions]
            delta = learner - rollout
            ratios = torch.exp(delta)
            token_ids = sequences[batch_index, 1:][positions]
            payload = {
                "record_type": "trajectory",
                "schema_version": 2,
                "record_index": self.records,
                "sample_index": indices[batch_index] if indices and batch_index < len(indices) else None,
                "group_id": group_ids[batch_index] if batch_index < len(group_ids) else None,
                "rollout_id": rollout_ids[batch_index] if batch_index < len(rollout_ids) else None,
                "rollout_actor_version": _metadata_value(info, "rollout_actor_version", batch_index),
                "learner_version": _metadata_value(info, "learner_version", batch_index),
                "sequence_token_ids": sequences[batch_index].tolist(),
                "action_step_positions": positions.tolist(),
                "action_token_ids": token_ids.tolist(),
                "rollout_log_probs": [_json_float(value) for value in rollout],
                "learner_log_probs": [_json_float(value) for value in learner],
                "rollout_logprob_status": [_float_status(value) for value in rollout],
                "learner_logprob_status": [_float_status(value) for value in learner],
                "support_violations": [
                    bool(torch.isfinite(learner_value) and torch.isneginf(rollout_value))
                    for learner_value, rollout_value in zip(learner, rollout)
                ],
                "log_ratios": [_json_float(value) for value in delta],
                "importance_ratios": [_json_float(value) for value in ratios],
            }
            self._write(payload)
            self.records += 1
            self.action_tokens += action_count
            written += action_count
        return written
