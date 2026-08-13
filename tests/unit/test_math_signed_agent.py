# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
from pathlib import Path

import pytest

_AGENT_PATH = Path(__file__).resolve().parents[2] / "examples" / "python" / "agents" / "math_signed.py"
_SPEC = importlib.util.spec_from_file_location("math_signed_agent", _AGENT_PATH)
_AGENT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AGENT)


@pytest.mark.parametrize(
    ("action", "expected_reward", "expected_exact"),
    ((r"\boxed{42}", 1.0, 1.0), (r"\boxed{41}", -0.1, 0.0)),
)
def test_signed_math_reward_preserves_binary_accuracy(action, expected_reward, expected_exact):
    result = asyncio.run(
        _AGENT.SignedMathEnv().step(
            {
                "observation_text": "Solve the problem.",
                "action_text": action,
                "label": {"ground_truth": "42"},
            }
        )
    )

    assert result.reward.item() == pytest.approx(expected_reward)
    assert result.info["math_exact"].item() == expected_exact
