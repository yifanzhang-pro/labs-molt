# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-turn math RL with +1 correct and -0.1 incorrect rewards."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from molt.agents import StepEnvRunner

_MATH_SPEC = importlib.util.spec_from_file_location("math_agent", Path(__file__).with_name("math.py"))
_MATH = importlib.util.module_from_spec(_MATH_SPEC)
_MATH_SPEC.loader.exec_module(_MATH)


class SignedMathEnv(_MATH.MathEnv):
    async def step(self, state):
        result = await super().step(state)
        if result.reward.item() == 0.0:
            result.reward = torch.tensor(-0.1, dtype=torch.float32)
        return result


class AgentRunner(StepEnvRunner):
    def __init__(self):
        super().__init__(SignedMathEnv)
