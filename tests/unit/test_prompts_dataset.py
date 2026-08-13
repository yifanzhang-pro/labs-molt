# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified chat-format dataset contract: one dataset, two runners.

Locks the invariants of the f193c302 data path — the bug classes these guard
(silent system-turn drop, silent image drop, template double-render) are all
invisible to reward/vllm_kl at run time.
"""

from types import SimpleNamespace

import pytest
from PIL import Image

from molt.agents.chat_agent import _wire_messages
from molt.datasets.prompts_dataset import PromptDataset, preprocess_data
from molt.trainer.rl_trainer import compute_eval_metrics

ROW = {
    "prompt": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "<image>\nWhat is x?"},
    ],
    "answer": "42",
}
TOOLS = [{"type": "function", "function": {"name": "py", "parameters": {}}}]


def _template(chat, tokenize, add_generation_prompt, **kwargs):
    """Stub HF chat template: records tools, renders roles so drops are visible."""
    _template.tools = kwargs.get("tools")
    _template.enable_thinking = kwargs.get("enable_thinking")
    return "|".join(f"{m['role']}:{m['content']}" for m in chat) + "|gen"


# --------------------------- preprocess_data modes ---------------------------


def test_step_prerender_keeps_system_and_tools():
    prompt, label = preprocess_data(ROW, "prompt", "answer", _template, prerender=True, tools=TOOLS)
    assert prompt.startswith("system:Be brief.|user:")  # system turn rendered, not dropped
    assert label == "42"
    assert _template.tools is TOOLS  # tools reach the template (Hermes preamble)


def test_step_prerender_can_hard_disable_thinking():
    preprocess_data(ROW, "prompt", "answer", _template, prerender=True, disable_thinking=True)
    assert _template.enable_thinking is False


def test_chat_passthrough_hands_full_messages_unrendered():
    prompt, label = preprocess_data(ROW, "prompt", "answer", _template, prerender=False, tools=TOOLS)
    assert prompt == ROW["prompt"]  # verbatim: system preserved, no template applied
    assert label == "42"


def test_chat_passthrough_wraps_string_rows():
    prompt, _ = preprocess_data({"prompt": "hi", "answer": ""}, "prompt", "answer", _template, prerender=False)
    assert prompt == [{"role": "user", "content": "hi"}]


def test_raw_mode_feeds_strings_verbatim_and_rejects_chat_rows():
    prompt, _ = preprocess_data({"prompt": "pre-rendered", "answer": ""}, "prompt", "answer", None)
    assert prompt == "pre-rendered"
    with pytest.raises(ValueError, match="apply_chat_template"):
        preprocess_data(ROW, "prompt", "answer", None)


# ------------------------------ PromptDataset --------------------------------


class _Tok:
    apply_chat_template = staticmethod(_template)
    image_token = "<image>"  # literal branch: no structured split


def _dataset(rows, prerender, *, data_disable_thinking=False, disable_thinking=None):
    strategy = SimpleNamespace(
        args=SimpleNamespace(
            data=SimpleNamespace(
                input_key="prompt",
                label_key="answer",
                tools_key="tools",
                image_key="images",
                apply_chat_template=True,
                disable_thinking=data_disable_thinking,
            )
        )
    )
    return PromptDataset(rows, _Tok(), strategy, prerender=prerender, disable_thinking=disable_thinking)


def test_dataset_row_is_five_tuple_and_collate_matches():
    rows = [{**ROW, "tools": TOOLS, "images": ["a.png"]}]
    ds = _dataset(rows, prerender=False)
    item = ds[0]
    assert len(item) == 5  # (datasource, prompt, label, images, tools) — dispatch chain contract
    datasources, prompts, labels, images, tools = ds.collate_fn([item])
    assert prompts == [ROW["prompt"]] and images == [["a.png"]] and tools == [TOOLS]


def test_dataset_can_enable_thinking_when_training_disables_it():
    ds = _dataset([ROW], prerender=True, data_disable_thinking=True, disable_thinking=False)

    ds[0]

    assert _template.enable_thinking is None


# ------------------------------ _wire_messages -------------------------------


def _img():
    return Image.new("RGB", (4, 4), "red")


def test_wire_inlines_images_at_markers_preserving_order():
    out = _wire_messages(ROW["prompt"], [_img()])
    assert out[0] == ROW["prompt"][0]  # system untouched
    parts = out[1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "text"]
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[1]["text"] == "\nWhat is x?"


def test_wire_keeps_excess_markers_so_mismatch_fails_loudly():
    out = _wire_messages([{"role": "user", "content": "a<image>b<image>c"}], [_img()])
    parts = out[1 - 1]["content"]
    rebuilt = "".join("<image>" if p["type"] == "image_url" else p["text"] for p in parts)
    assert rebuilt == "a<image>b<image>c"  # nothing silently vanishes
    assert sum(p["type"] == "image_url" for p in parts) == 1


def test_wire_without_images_or_markers_is_identity():
    row = [{"role": "user", "content": "plain"}]
    assert _wire_messages(row, None) == row
    assert _wire_messages(row, [_img()]) == row  # no marker -> images stay on ctx.images
    assert _wire_messages("bare string", None) == [{"role": "user", "content": "bare string"}]


# ---------------------------- eval metrics mapping ----------------------------


def test_eval_metrics_maps_chat_list_prompts_to_datasource():
    # chat eval dataloader yields messages lists; samples carry the last user text
    eval_dataloader = [(["geo3k"], [ROW["prompt"]], ["42"], [None], [None])]
    sample = SimpleNamespace(
        prompts=["<image>\nWhat is x?"],
        group_ids=["g1"],
        rewards=[1.0],
        response_length=[7],
        truncated=[False],
    )
    metrics = compute_eval_metrics(eval_dataloader, [sample], n_samples_per_prompt=1)
    assert metrics["eval_geo3k_pass1"] == 1.0  # datasource resolved via the last user turn text


def test_eval_metrics_exposes_avg32_as_mean_sample_reward():
    eval_dataloader = [(["aime"], ["problem"], ["42"], [None], [None])]
    samples = [
        SimpleNamespace(
            prompts=["problem"],
            group_ids=["g1"],
            rewards=[float(index < 8)],
            response_length=[7],
            truncated=[False],
        )
        for index in range(32)
    ]

    metrics = compute_eval_metrics(eval_dataloader, samples, n_samples_per_prompt=32)

    assert metrics["eval_aime_avg32"] == 0.25
    assert metrics["eval_aime_pass1"] == 0.25
    assert metrics["eval_aime_pass32"] == 1.0
    assert metrics["eval_num_samples"] == 32.0
