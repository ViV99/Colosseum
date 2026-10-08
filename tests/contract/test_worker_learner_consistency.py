"""Contract: the learner reproduces the worker's log-probs and values at equal weights.

Chunks come from the real RolloutLoop (in-process) and are re-evaluated by the
real APPO (``model.unroll`` from ``cat_batch(chunk.initial_state)``).
"""

from __future__ import annotations

import pytest
import torch

from colosseum.core.types import WorkerCommand, state_dict_to_numpy
from colosseum.networks.state import tree_leaves
from harness import (
    learner_eval,
    make_loop,
    player_index,
    run_steps,
    run_until_chunks,
    simple_factory,
    weights_payload,
)
from helpers import CORE_KINDS

TOL = 1e-5


def _max_diff(a, b) -> float:
    assert a.shape == b.shape, (tuple(a.shape), tuple(b.shape))
    return float((a - b).abs().max())


def _is_zero_state(state) -> bool:
    leaves = tree_leaves(state)
    return bool(leaves) and all(int(torch.count_nonzero(leaf)) == 0 for leaf in leaves)


@pytest.mark.parametrize("core", CORE_KINDS)
def test_learner_reproduces_worker_logprobs_and_values(core):
    torch.manual_seed(0)
    learner_model = simple_factory(core)
    loop, rec = make_loop(
        model_factories={"agent_0": lambda: simple_factory(core)},
        num_envs=2, chunk_length=8,  # episodes last 5 steps -> chunks span boundaries
        initial_weights={"agent_0": weights_payload("agent_0", learner_model, 0)},
    )
    chunks = run_until_chunks(loop, rec, 8)  # two chunks per slot
    loop.close()

    assert any(bool(c.dones[:-1].any()) for c in chunks), "no chunk spans an episode boundary"
    if core != "none":
        assert any(
            float(sum(leaf.float().abs().sum() for leaf in c.initial_state.values())) > 0
            for c in chunks
        ), "no chunk starts mid-episode with a non-zero state"

    lp, v, worker_lp, worker_v = learner_eval(learner_model, chunks)
    assert _max_diff(lp, worker_lp) < TOL
    assert _max_diff(v, worker_v) < TOL


@pytest.mark.parametrize("core", CORE_KINDS)
def test_chunk_starting_right_after_a_done_reproduces_from_a_zero_state(core):
    """chunk_length == episode_length: every chunk is closed by a done, so each
    slot's next chunk starts at an episode start with the model's initial state."""
    torch.manual_seed(0)
    learner_model = simple_factory(core)
    loop, rec = make_loop(
        model_factories={"agent_0": lambda: simple_factory(core)},
        num_envs=2, chunk_length=5,  # == CountingEnv episode_length
        initial_weights={"agent_0": weights_payload("agent_0", learner_model, 0)},
    )
    chunks = run_until_chunks(loop, rec, 8)  # two chunks per slot
    loop.close()

    assert all(bool(c.dones[-1]) and not bool(c.dones[:-1].any()) for c in chunks)
    after_done = chunks[4:]  # each slot's second chunk follows the done that closed its first
    for c in after_done:
        if core == "none":
            assert c.initial_state is None
        else:
            assert _is_zero_state(c.initial_state), sorted(c.initial_state)

    lp, v, worker_lp, worker_v = learner_eval(learner_model, after_done)
    assert _max_diff(lp, worker_lp) < TOL
    assert _max_diff(v, worker_v) < TOL


@pytest.mark.parametrize("core", CORE_KINDS)
def test_resumed_parked_buffers_reproduce(core):
    """A seat flip parks partial buffers; another seat resumes them. The resumed
    chunk mixes two seats' episodes (closed by a done, state reset) and the
    learner still reproduces the worker's log-probs and values."""
    torch.manual_seed(0)
    models = {"a": simple_factory(core), "b": simple_factory(core)}
    loop, rec = make_loop(
        agent_ids=["a", "b"],
        model_factories={"a": lambda: simple_factory(core), "b": lambda: simple_factory(core)},
        num_envs=2, chunk_length=8,  # episodes last 5 steps
        slot_agent_map=[["a", "b"], ["a", "b"]],
        initial_weights={aid: weights_payload(aid, m, 0) for aid, m in models.items()},
    )
    # Every seat changes agent, so each agent's seat-0/seat-1 buffers are parked and
    # resumed by a seat with the other index (CountingEnv obs carry the seat, not the env).
    rec.commands.append(WorkerCommand(
        slot_agent_map=[["b", "a"], ["b", "a"]],
        slot_network_map=[["latest", "latest"], ["latest", "latest"]],
        collect_mask=[[True, True], [True, True]],
    ))
    run_steps(loop, 5)  # episode 0 ends: every seat parks its 5-step buffer
    assert loop.stats["parked_buffers"] == 1  # three were resumed at once, one remains parked
    run_steps(loop, 11)

    resumed = [c for c in rec.chunks if len(player_index(c)) == 2]
    assert resumed, "no chunk resumes a parked buffer in another seat"
    for c in resumed:
        assert bool(c.dones[4]) and c.observations[5, 0] == 0  # seat switch right after a done
    for agent_id, model in models.items():
        chunks = [c for c in rec.chunks if c.agent_id == agent_id]
        assert chunks and any(len(player_index(c)) == 2 for c in chunks), agent_id
        lp, v, worker_lp, worker_v = learner_eval(model, chunks)
        assert _max_diff(lp, worker_lp) < TOL, agent_id
        assert _max_diff(v, worker_v) < TOL, agent_id


def test_heterogeneous_agents_and_checkpoint_opponents():
    """Two agents with different cores share envs; a frozen stateful checkpoint plays too."""
    torch.manual_seed(0)
    model_a = simple_factory("lstm")
    model_b = simple_factory("window")
    ckpt_b = state_dict_to_numpy(simple_factory("window").state_dict())
    loop, rec = make_loop(
        agent_ids=["a", "b"],
        model_factories={"a": lambda: simple_factory("lstm"), "b": lambda: simple_factory("window")},
        num_envs=3, chunk_length=8,
        slot_agent_map=[["a", "b"], ["b", "a"], ["a", "b"]],
        slot_network_map=[["latest", "latest"], ["latest", "latest"], ["latest", "ckpt_v1"]],
        collect_mask=[[True, True], [True, True], [True, False]],
        checkpoint_state_dicts_by_agent={"a": {}, "b": {"ckpt_v1": ckpt_b}},
        initial_weights={
            "a": weights_payload("a", model_a, 0),
            "b": weights_payload("b", model_b, 0),
        },
    )
    run_until_chunks(loop, rec, 10)
    loop.close()

    # Preconditions: the frozen checkpoint (not b's latest model) is seated in env 2,
    # seat 1 with the checkpoint's weights, and the checkpoint differs from b's latest.
    assert loop._slot_network_map[2][1] == "ckpt_v1" and not loop._collect_mask[2][1]
    seated = loop._resolve_model("b", "ckpt_v1")
    assert seated is loop._models["b"]["ckpt_v1"]
    assert seated is not loop._models["b"]["latest"]
    for k, v in seated.state_dict().items():
        assert torch.equal(v, torch.from_numpy(ckpt_b[k])), k
    assert any(not torch.equal(v, model_b.state_dict()[k]) for k, v in seated.state_dict().items())
    # 5 collecting slots (a: 3, b: 2), chunk_length 8: two chunks each after 16 steps.
    assert len(rec.chunks) == 10
    assert sum(c.agent_id == "a" for c in rec.chunks) == 6
    assert sum(c.agent_id == "b" for c in rec.chunks) == 4

    for agent_id, model in (("a", model_a), ("b", model_b)):
        chunks = [c for c in rec.chunks if c.agent_id == agent_id]
        assert any(bool(c.dones[:-1].any()) for c in chunks), f"{agent_id}: no done inside a chunk"
        lp, v, worker_lp, worker_v = learner_eval(model, chunks)
        assert _max_diff(lp, worker_lp) < TOL, agent_id
        assert _max_diff(v, worker_v) < TOL, agent_id
