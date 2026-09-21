"""Native sampler -> trajectory -> replay -> trainer KL tensor contract (CPU)."""

import asyncio
import base64
import io
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from molt.agents.base import Trajectory
from molt.models.actor import Actor
from molt.models.base import BaseModel, _AttrDict
from molt.models.utils import log_probs_from_logits
from molt.trainer.algorithm.experience import make_experience_batch, split_experience_batch
from molt.trainer.rollout.router import RouterGenerateClient
from test_samples_generator import SamplesGenerator
from test_vllm_engine import vllm_engine


@pytest.fixture
def engine(monkeypatch):
    module = ModuleType("vllm.sampling_params")
    module.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    module.RequestOutputKind = SimpleNamespace(DELTA="delta")
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", module)
    probs = np.array([[0.55, 0.30, 0.10, 0.05], [0.10, 0.15, 0.25, 0.50]])
    actions = [3, 0]  # Both are outside their sampler top-2 head.
    calls = []

    async def generate(prompt, sp, request_id):
        calls.append(sp)
        for row, action in zip(probs, actions):
            ids = list(range(4)) if sp.logprobs == -1 else list(np.argsort(-row)[: sp.logprobs])
            entries = {action: SimpleNamespace(logprob=float(np.log(row[action])))}
            entries.update({int(v): SimpleNamespace(logprob=float(np.log(row[v]))) for v in ids})
            yield SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[action], logprobs=[entries], finish_reason="stop")]
            )
        yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[], logprobs=None, finish_reason="stop")])

    async def abort(request_id):
        pass

    cls = vllm_engine.RolloutRayActor
    cls = cls.__ray_metadata__.modified_class if hasattr(cls, "__ray_metadata__") else cls
    stub = SimpleNamespace(
        kwargs={"logprobs_mode": "processed_logprobs"},
        llm=SimpleNamespace(model_config=SimpleNamespace(get_vocab_size=lambda: 4), generate=generate, abort=abort)
    )
    return cls.generate_kl, stub, probs, actions, calls


@pytest.mark.parametrize("estimator", ["binary", "topk", "mc", "full"])
def test_sampler_transport_and_replay_preserve_conditionals(engine, estimator):
    generate_kl, engine_stub, probs, actions, calls = engine
    config = {"estimator": estimator, "top_k": 2, "mc_samples": 17, "temperature": 0.7}
    client = RouterGenerateClient(None)

    async def post(path, body, sid):
        assert path == "/molt/v1/generate" and sid == "fixed-session"
        return await generate_kl(engine_stub, body)

    client._post = post
    sp = SimpleNamespace(max_tokens=2, temperature=0.7, top_p=1.0, extra_args={"molt_kl": config})
    result, _ = asyncio.run(client.generate([1, 2], sp, session_id="fixed-session"))
    gen = result.outputs[0]
    np.testing.assert_allclose(
        [gen.logprobs[v][a].logprob for v, a in enumerate(actions)], np.log(probs[np.arange(2), actions])
    )
    assert calls[0].output_kind == "delta"
    if estimator == "binary":
        assert not hasattr(gen, "kl_log_probs")
        return
    q = gen.kl_log_probs
    ids = getattr(gen, "kl_token_ids", None)
    if estimator == "topk":
        np.testing.assert_array_equal(ids, [[0, 1], [3, 2]])
    elif estimator == "mc":
        assert ids.shape == (2, 17) and np.unique(ids[0]).size < 17
    else:
        assert ids is None and q.shape == (2, 4)
    expected = np.log(probs) if ids is None else np.log(np.take_along_axis(probs, ids, axis=1))
    np.testing.assert_allclose(q, expected, rtol=1e-6)
    traj = Trajectory("p", "l", None, "p", [1, 2], rollout_log_probs=[0.0, 0.0])
    traj.append_action(
        actions, [gen.logprobs[i][a].logprob for i, a in enumerate(actions)], kl_token_ids=ids, kl_log_probs=q
    )
    traj.append_feedback("a", "tool", [2, 1])
    traj.append_action(actions, [-1.0, -2.0], kl_token_ids=ids, kl_log_probs=q)
    exp, reason = SamplesGenerator._process_response_into_experience(traj, set(), 20)
    assert reason is None
    assert exp.action_mask.tolist() == [[False, True, True, False, False, True, True]]
    torch.testing.assert_close(exp.kl_log_probs[0, :, 1:3].T, torch.tensor(q))
    single = split_experience_batch(exp)[0]
    short = split_experience_batch(exp)[0]
    for name in ("sequences", "attention_mask", "action_mask", "kl_token_ids", "kl_log_probs", "rollout_log_probs"):
        if getattr(short, name) is not None:
            setattr(short, name, getattr(short, name)[..., :-4])
    batch = make_experience_batch([single, short])
    assert batch.kl_log_probs.shape == (2, q.shape[-1], 7)
    assert torch.equal(batch.kl_log_probs[1, :, 3:], torch.zeros_like(batch.kl_log_probs[1, :, 3:]))


def test_mc_draws_use_separate_rng_and_preserve_duplicates(engine, monkeypatch):
    generate, stub, probs, actions, _ = engine
    received = []

    class AuxiliaryRNG:
        def choice(self, vocab, size, replace, p):
            assert replace and size == 3
            received.append(p)
            return np.array([1, 1, 3])

    monkeypatch.setattr(np.random, "default_rng", lambda: AuxiliaryRNG())
    body = {
        "token_ids": [1],
        "sampling_params": {"temperature": 1},
        "kl": {"estimator": "mc", "top_k": 2, "mc_samples": 3, "temperature": 1},
    }
    out = asyncio.run(generate(stub, body))["choices"][0]
    with np.load(io.BytesIO(base64.b64decode(out["kl_records"]))) as arrays:
        np.testing.assert_array_equal(arrays["kl_token_ids"], [[1, 1, 3], [1, 1, 3]])
    np.testing.assert_allclose(received, probs)


@pytest.mark.parametrize("bad", [{"top_p": 0.9}, {"min_tokens": 1}, {"temperature": 0.5}, {"repetition_penalty": 1.2}])
def test_capture_rejects_sampler_trainer_distribution_mismatch(engine, bad):
    generate, stub, *_ = engine
    body = {
        "token_ids": [1],
        "sampling_params": {"temperature": 1, **bad},
        "kl": {"estimator": "full", "top_k": 2, "mc_samples": 2, "temperature": 1},
    }
    with pytest.raises(ValueError, match="sampling must match"):
        asyncio.run(generate(stub, body))


@pytest.mark.parametrize("packing", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_actor_conditional_scores_and_gradients_match_dense_softmax(packing, full):
    torch.manual_seed(4)
    logits = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    sequences = torch.tensor([[1, 2, 3, 0], [4, 2, 1, 3]])
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    ids = torch.tensor([[[4, 1, 0], [4, 0, 0]], [[2, 4, 0], [2, 1, 3]]])
    indices = attention.flatten().nonzero().flatten() if packing else None
    if packing:
        local_logits = logits.flatten(0, 1)[indices].unsqueeze(0)
        rolled = sequences.roll(-1, 1).flatten()[indices].unsqueeze(0)
    else:
        local_logits = logits
        rolled = sequences.roll(-1, 1)
    stub = SimpleNamespace(temperature=0.7, packing_samples=packing)
    stub._forward_backbone = lambda *a, **kw: (_AttrDict(logits=local_logits), rolled, False, indices, 2, 4)
    stub._restore_full_sequence = lambda t, **kw: BaseModel._restore_full_sequence(stub, t, **kw)
    out = Actor.forward(
        stub, sequences, mask, attention_mask=attention, kl_token_ids=None if full else ids, return_full_log_probs=full
    )
    expected = torch.log_softmax(logits / 0.7, -1)[:, :-1]
    if not full:
        expected = expected.gather(-1, ids.transpose(1, 2))
    torch.testing.assert_close(out.kl_log_probs[mask], expected[mask])
    (grad,) = torch.autograd.grad(out.kl_log_probs[mask].sum(), logits, retain_graph=True)
    (want,) = torch.autograd.grad(expected[mask].sum(), logits)
    torch.testing.assert_close(grad, want)


def test_multi_id_scoring_keeps_duplicate_gradients_and_temperature():
    logits = torch.randn(1, 300, 8, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([2, 2, 7]).expand(1, 300, 3)
    got = log_probs_from_logits(logits, ids, temperature=0.6)
    expected = torch.log_softmax(logits / 0.6, -1).gather(-1, ids)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(
        torch.autograd.grad(got.sum(), logits, retain_graph=True)[0], torch.autograd.grad(expected.sum(), logits)[0]
    )


@pytest.mark.parametrize("route", ["sequence", "token"])
@pytest.mark.parametrize("estimator", ["binary", "topk", "mc", "full"])
def test_native_cli_accepts_all_routes(monkeypatch, route, estimator):
    import ast
    from pathlib import Path
    import molt.cli.train_rl_ray as cli

    source = Path(cli.__file__).read_text()
    main = next(node for node in ast.parse(source).body if isinstance(node, ast.If))
    namespace = dict(vars(cli), __name__="__main__")
    captured = []
    namespace["train"] = captured.append
    flags = [
        "train",
        "--actor.model_name_or_path",
        "model",
        "--vllm.num_engines",
        "1",
        "--train.agent_path",
        "agent.py",
        "--actor.loss_mode",
        "klpo",
        "--actor.klpo_mc_samples",
        "2",
        "--algo.advantage.estimator",
        "reinforce",
        "--algo.advantage.no_whiten",
        "--algo.kl.init_coef",
        "0",
        "--algo.advantage.is_correction_level",
        "off",
        "--train.force_on_policy",
        "--train.force_sync_mode",
        "--train.async_queue_size",
        "1",
    ]
    if (route, estimator) != ("token", "mc"):
        flags += ["--actor.klpo_route", route, "--actor.klpo_kl_estimator", estimator]
    monkeypatch.setattr(sys, "argv", flags)
    exec(compile(ast.Module(body=main.body, type_ignores=[]), cli.__file__, "exec"), namespace)
    assert captured[0].actor.klpo_kl_estimator == estimator
    assert captured[0].actor.klpo_route == route


@pytest.mark.parametrize("layout_kind", ["round_robin", "thd", "reposition"])
def test_cp_auxiliary_axis_follows_model_sharder(layout_kind):
    logits = torch.randn(2, 4, 5, requires_grad=True)
    ids = torch.tensor([[[1, 2, 3], [1, 0, 2]], [[2, 1, 3], [4, 2, 3]]])
    sequence = torch.tensor([[0, 2, 4, 1], [3, 0, 2, 4]])
    positions = torch.tensor([[1, 3, 0, 2], [2, 0, 3, 1]])
    layout = SimpleNamespace(
        input_token_stream_positions=positions if layout_kind == "reposition" else None, padded_seq_len=4
    )

    def shard(t, seq_dim=1, fill=0):
        if layout_kind == "round_robin":
            return t[:, [0, 2, 1, 3]]
        if layout_kind == "thd":
            return t.reshape(1, 8, *t.shape[2:])
        return t

    def gather(t, seq_dim=1, trim=False, fill=None):
        if layout_kind == "round_robin":
            return t[:, [0, 2, 1, 3]]
        if layout_kind == "thd":
            return t.reshape(2, 4, *t.shape[2:])
        return t.gather(1, positions) if t.ndim == 2 else t

    if layout_kind == "reposition":
        local = torch.zeros_like(logits).scatter(1, positions[..., None].expand_as(logits), logits)
        rolled = torch.zeros_like(sequence).scatter(1, positions, sequence.roll(-1, 1))
    else:
        local, rolled = shard(logits), shard(sequence.roll(-1, 1))
    stub = SimpleNamespace(
        temperature=1,
        packing_samples=False,
        _cp_sharder=SimpleNamespace(shard_layout=layout, shard_token_tensor=shard, gather_token_tensor=gather),
    )
    stub._forward_backbone = lambda *a, **kw: (_AttrDict(logits=local), rolled, True, None, 2, 4)
    stub._restore_full_sequence = lambda t, **kw: BaseModel._restore_full_sequence(stub, t, **kw)
    out = Actor.forward(stub, sequence, torch.ones(2, 3, dtype=torch.bool), kl_token_ids=ids)
    expected = logits.log_softmax(-1)[:, :-1].gather(-1, ids.transpose(1, 2))
    torch.testing.assert_close(out.kl_log_probs, expected)
    torch.testing.assert_close(
        torch.autograd.grad(out.kl_log_probs.sum(), logits, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), logits)[0],
    )
