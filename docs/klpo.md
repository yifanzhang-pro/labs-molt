# Native KLPO interface

This fork's `feat/klpo-all-kl` branch supports KLPO **token regression** and
**sequence regression**, each with Monte Carlo KL (MC-KL), Top-K Aggregated KL
(TopK-KL), Binary KL, or Full KL. Install the [KLPO package](https://github.com/yifanzhang-pro/KLPO)
alongside Molt. The backend calls `klpo.molt.KLPOLoss`; the mathematical formulas
have one implementation in KLPO.

The default is **KLPO token regression + MC-KL**, with M=128 independent
auxiliary draws per prefix, matching the paper's default route and estimator.

```bash
python /path/to/KLPO/scripts/train_molt.py \
  --molt-path /path/to/labs-molt --model /path/to/model \
  --train-data /path/to/train --eval-data /path/to/eval \
  --route token --kl-estimator mc --mc-samples 128
# Other choices: --route sequence; --kl-estimator binary|full;
# --kl-estimator topk --top-k 128
```

Native CLI flags are `--actor.loss_mode klpo`, `--actor.klpo_route`,
`--actor.klpo_kl_estimator`, `--actor.klpo_beta`, `--actor.klpo_top_k`,
`--actor.klpo_mc_samples`, and `--actor.klpo_tail_floor`. The launcher supplies the
required synchronous single-update schedule and raw-reward configuration.
`molt.KLPO_API_VERSION = 2` identifies this interface and its `topk` estimator name.

| Estimator | Generation-time records | Trainer scoring |
| --- | --- | --- |
| MC-KL (default) | M independent q draws with replacement and log q | Same IDs, duplicates preserved |
| TopK-KL | Sampler's top K IDs and original log q | Same IDs, plus an aggregated tail |
| Binary KL | Realized action log q | Realized action log p |
| Full KL | Full conditional log q in vocabulary order | Full conditional log p |

M is the auxiliary sample count; K is the head size. Token MC allows M=1.
Sequence MC uses an independent cross estimator and requires M >= 2. Neither
requires extra response rollouts or a critic. MC KL and its sequence U-statistic
loss can be negative and are not clamped.

The `/molt/v1/generate` endpoint reads vLLM's processed generation logprobs and
returns exact token IDs with compact NumPy records. MC uses a separate RNG from
response generation; auxiliary draws never execute an action or tool. TopK-KL never
renormalizes the head or inserts an action outside it; K >= V takes the full-KL
limit. Full/MC capture currently requests all vocabulary logprobs inside the
engine, reducing MC to M records before HTTP transfer. Thus MC reduces network
and replay storage but still incurs full-distribution collection cost; it is not
a fused sparse GPU sampling kernel. Full KL costs O(TV) memory/transport.

`Trajectory.kl_log_probs/kl_token_ids` preserve context and tool gaps.
`Experience` stores `[B, K/M/V, T-1]` tensors so offload, padding, splitting, and
replay use the existing sequence-last machinery. Full KL omits redundant IDs.
`Actor.forward(kl_token_ids=..., return_full_log_probs=...)` returns
`kl_log_probs[B,T-1,K/M/V]` with autograd and `kl_full_vocabulary`.
TP vocabulary gathering, chunked selected-ID normalization, packing, and CP
sequence restoration precede the loss. The loss receives raw terminal rewards
and a global response count; gradient accumulation does not divide it twice.

KLPO drains a rollout batch before refitting, uses queue depth one, and updates
once. This avoids adaptive reuse of a fixed MC bank. Training uses matching
positive temperatures and full-support sampling with no penalties or minimum
EOS suppression. Evaluation skips auxiliary capture. Partial rollouts, MTP,
routing replay, and context-compacted response segments are rejected in KLPO
mode; existing non-KLPO modes retain their behavior. Multi-turn trajectories
must remain complete and keep tool tokens masked out.

CPU tests exercise sampler capture, transport/replay, duplicate MC records,
packed/CP scoring and gradients, and CLI validation. Real CUDA/vLLM training,
GPU collectives, and performance still require validation on a GPU host.
