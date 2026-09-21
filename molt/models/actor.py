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
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

from typing import Optional

import torch
from torch.distributed.tensor import DTensor

from molt.trainer.fsdp.packing import log_probs_from_vocab_parallel_logits, unshard_dtensor

from .base import BaseModel, _AttrDict
from .utils import compute_entropy, log_probs_from_logits


class Actor(BaseModel):
    """Policy model wrapper for RLHF.

    Reuses ``BaseModel`` for construction and the shared ``_forward_backbone``
    (packing / VLM / CP prep + model call); ``forward`` turns the model logits into
    per-token log-probs, optionally entropy, and the action-span log-probs.
    """

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cp_context_stack=None,
        return_entropy=False,
        routed_experts: Optional[torch.Tensor] = None,
        kl_token_ids: Optional[torch.Tensor] = None,
        return_full_log_probs: bool = False,
        **mm_inputs,
    ) -> _AttrDict:
        """Run the policy forward and return one named output dict.

        Always returns an ``_AttrDict`` (key- and attribute-accessible) with:

        - ``logits``:           model logits (TP-sharded DTensor unless gathered).
        - ``log_probs``:        ``[B, S-1]`` log-probs of the realized next tokens.
        - ``action_log_probs``: ``[B, num_actions]`` masked to the generated span —
                                only when ``action_mask`` is given (RL / reference).
        - ``entropy``:          ``[B, S-1]`` — only when ``return_entropy`` (RL).
        - ``aux_loss``:         MoE load-balancing loss — only for NeMo custom MoE.
        - ``kl_log_probs``:     [B, S-1, K/M/V] conditionals when KL records are requested.
        - ``kl_full_vocabulary``: whether the requested conditionals cover the vocabulary.

        ``kl_token_ids`` uses [B, K/M, S-1], the replay buffer's sequence-last format.
        """
        output, rolled_sequences, cp_forward, indices, batch, seqlen = self._forward_backbone(
            sequences, attention_mask, position_ids, cp_context_stack, mm_inputs, routed_experts=routed_experts
        )
        logits = output["logits"]
        full_logits = None

        if return_entropy:
            entropy_logits = unshard_dtensor(logits).to(torch.float32)
            if not cp_forward:
                full_logits = entropy_logits
                output["logits"] = full_logits
            # Compute entropy on the same temperature-scaled distribution as the
            # policy log-probs below (which divide by self.temperature). Without
            # this the entropy-regularization term operates on a different (T=1)
            # distribution than the policy whenever rollout.temperature != 1.0.
            # Standard practice scales logits before both log-probs *and*
            # entropy. Divide a copy so output["logits"]/full_logits stay raw
            # (log_probs_from_logits applies its own temperature division).
            entropy_src = entropy_logits / self.temperature if self.temperature != 1.0 else entropy_logits
            entropy = compute_entropy(entropy_src)
            # entropy is seq-local even under TP+CP (unshard_dtensor only
            # collapses the TP vocab dim), so restore the full sequence axis the
            # same way as log_probs below.
            entropy = self._restore_full_sequence(
                entropy, cp_forward=cp_forward, batch=batch, seqlen=seqlen, indices=indices
            )
            output["entropy"] = entropy[:, :-1]

        if isinstance(logits, DTensor):
            log_probs = log_probs_from_vocab_parallel_logits(
                logits,
                rolled_sequences,
                temperature=self.temperature,
            )
        else:
            # Pass bf16 directly: log_probs_from_logits chunks the fp32 upcast
            # internally to avoid the [B*S, V] memory spike that OOMs on
            # large-vocab models (Qwen3.6: 152K vocab × 65K tokens = 37 GiB).
            log_probs_input = logits if (cp_forward or full_logits is None) else full_logits
            log_probs = log_probs_from_logits(log_probs_input, rolled_sequences, temperature=self.temperature)

        log_probs = self._restore_full_sequence(
            log_probs, cp_forward=cp_forward, batch=batch, seqlen=seqlen, indices=indices
        )

        if kl_token_ids is not None or return_full_log_probs:
            # Gather the TP vocabulary; score only stored IDs before restoring the CP/packed axis.
            conditional_logits = unshard_dtensor(logits)
            if return_full_log_probs:
                dtype = torch.float64 if conditional_logits.dtype == torch.float64 else torch.float32
                conditional = torch.log_softmax(conditional_logits.to(dtype) / self.temperature, dim=-1)
            else:
                if kl_token_ids.shape[0] != batch or kl_token_ids.shape[2] != seqlen - 1:
                    raise ValueError("kl_token_ids must have shape [B, K/M, S-1]")
                ids = torch.nn.functional.pad(kl_token_ids.transpose(1, 2).long(), (0, 0, 0, 1))
                width = ids.shape[-1]
                if cp_forward:
                    layout = getattr(self._cp_sharder, "shard_layout", None)
                    positions = getattr(layout, "input_token_stream_positions", None)
                    if positions is not None:
                        # AutoModel's repositioned-row verb accepts scalar streams only.
                        positions = positions.to(ids.device)
                        valid = positions >= 0
                        padded = ids.new_zeros((batch, layout.padded_seq_len, width))
                        padded[valid.nonzero(as_tuple=True)[0], positions[valid]] = ids[valid]
                        ids = padded
                    ids = self._cp_sharder.shard_token_tensor(ids, seq_dim=1, fill=0)
                elif self.packing_samples:
                    ids = ids.reshape(-1, width).index_select(0, indices)
                    pad = conditional_logits.numel() // conditional_logits.shape[-1] - ids.shape[0]
                    ids = torch.nn.functional.pad(ids, (0, 0, 0, pad))
                ids = ids.reshape(*conditional_logits.shape[:-1], width)
                conditional = log_probs_from_logits(conditional_logits, ids, temperature=self.temperature)
            conditional = self._restore_full_sequence(
                conditional, cp_forward=cp_forward, batch=batch, seqlen=seqlen, indices=indices
            )
            output["kl_log_probs"] = conditional[:, :-1]
            output["kl_full_vocabulary"] = return_full_log_probs or conditional.shape[-1] == logits.shape[-1]

        # Drop the final column: logits[t] predicts token[t+1], so the last
        # position has no target. log_probs / action_log_probs / entropy are all
        # shifted [:, :-1] and stay mutually aligned.
        output["log_probs"] = log_probs[:, :-1]

        # RL / reference (action_mask given) additionally expose the action-span
        # log-probs, zeroed outside the generated tokens.
        if action_mask is not None:
            output["action_log_probs"] = output["log_probs"][:, -action_mask.shape[1] :] * action_mask.float()

        return output
