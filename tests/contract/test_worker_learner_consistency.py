"""Contract: the learner reproduces the worker's log-probs and values at equal weights.

Chunks come from the real RolloutLoop (in-process) and are re-evaluated by the
real APPO (``model.unroll`` from ``cat_batch(chunk.initial_state)``).
"""

from __future__ import annotations

import pytest
import torch

from colosseum.core.types import state_dict_to_numpy
from harness import learner_eval, make_loop, run_until_chunks, simple_factory, weights_payload
from helpers import CORE_KINDS

TOL = 1e-5


def _max_diff(a, b) -> float:
    return float((a - b).abs().max())


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

    for agent_id, model in (("a", model_a), ("b", model_b)):
        chunks = [c for c in rec.chunks if c.agent_id == agent_id]
        assert len(chunks) >= 2
        lp, v, worker_lp, worker_v = learner_eval(model, chunks)
        assert _max_diff(lp, worker_lp) < TOL, agent_id
        assert _max_diff(v, worker_v) < TOL, agent_id
