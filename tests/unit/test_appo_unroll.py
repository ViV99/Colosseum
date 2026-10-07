"""APPO trains every PolicyModel through model.unroll from the chunks' initial states."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.types import TrajectoryChunk
from colosseum.networks.state import slice_batch, tree_leaves
from colosseum.transport.serialization import deserialize_chunk, serialize_chunk
from helpers import CORE_KINDS, make_simple_model

OBS, A, T = 8, 4, 6


def _chunk(model, initial_state, dones=None, masks=None) -> TrajectoryChunk:
    d = torch.zeros(T) if dones is None else dones
    return TrajectoryChunk(
        agent_id="a",
        observations=torch.randn(T, OBS),
        actions=torch.randint(0, A, (T,)),
        action_log_probs=torch.full((T,), -1.3),
        rewards=torch.randn(T),
        dones=d,
        values=torch.randn(T),
        bootstrap_value=torch.tensor(0.0),
        behavior_policy_version=0,
        initial_state=initial_state,
        action_masks=masks,
    )


def _reference(model, chunks):
    """Per-chunk step loop with resets: expected [T*B] time-major log-probs/values."""
    lps = torch.zeros(T, len(chunks))
    vals = torch.zeros(T, len(chunks))
    with torch.no_grad():
        for b, c in enumerate(chunks):
            state = c.initial_state
            for t in range(T):
                mask = None if c.action_masks is None else c.action_masks[t:t + 1]
                out = model.step(c.observations[t:t + 1], state, mask)
                lps[t, b] = out.dist.log_prob(c.actions[t:t + 1])[0]
                vals[t, b] = out.value[0]
                state = out.state
                if bool(c.dones[t]):
                    state = model.initial_state(1)
    return lps.reshape(-1), vals.reshape(-1)


@pytest.mark.parametrize("core", CORE_KINDS)
def test_evaluate_chunks_matches_step_loop(core):
    torch.manual_seed(0)
    model = make_simple_model(obs_dim=OBS, num_actions=A, core=core)
    algo = APPO(model, AlgorithmConfig(), device="cpu")
    warm = model.initial_state(2)
    if warm is not None:
        with torch.no_grad():
            for _ in range(3):
                warm = model.step(torch.randn(2, OBS), warm).state
    dones = torch.zeros(T)
    dones[2] = 1.0
    masks = torch.ones(T, A, dtype=torch.bool)
    masks[:, 3] = False
    chunks = [
        _chunk(model, None if warm is None else slice_batch(warm, 0), dones=dones),
        _chunk(model, None if warm is None else slice_batch(warm, 1), masks=masks),
    ]
    chunks[0].action_masks = torch.ones(T, A, dtype=torch.bool)
    chunks[1].actions = torch.randint(0, 3, (T,))
    lp, v = algo.evaluate_chunks(chunks)
    ref_lp, ref_v = _reference(model, chunks)
    assert lp.shape == (T * 2,) and v.shape == (T * 2,)
    assert torch.allclose(lp, ref_lp, atol=1e-5)
    assert torch.allclose(v, ref_v, atol=1e-5)


@pytest.mark.parametrize("core", CORE_KINDS)
def test_train_step_on_stateful_chunks_updates_all_parameters(core):
    torch.manual_seed(0)
    model = make_simple_model(obs_dim=OBS, num_actions=A, core=core)
    algo = APPO(model, AlgorithmConfig(learning_rate=1e-2), device="cpu")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    chunks = [_chunk(model, model.initial_state(1)) for _ in range(3)]
    metrics = algo.train_step(chunks)
    assert np.isfinite(metrics["total_loss"])
    assert algo.policy_version == 1
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert {k for k, _ in model.named_parameters()} <= changed


def test_algorithm_exposes_model():
    model = make_simple_model()
    assert APPO(model, AlgorithmConfig()).model is model


def test_chunk_to_and_serialization_keep_initial_state():
    model = make_simple_model(obs_dim=OBS, num_actions=A, core="lstm")
    state = model.initial_state(1)
    state = {k: v + 0.5 for k, v in state.items()}
    chunk = _chunk(model, state)
    moved = chunk.to("cpu")
    assert torch.equal(moved.initial_state["h"], state["h"])
    data, compressed = serialize_chunk(chunk)
    restored = deserialize_chunk("a", 0, data, compressed)
    assert set(restored.initial_state) == {"h", "c"}
    for x, y in zip(tree_leaves(restored.initial_state), tree_leaves(state)):
        assert torch.equal(x, y)
    stateless = _chunk(model, None)
    data, compressed = serialize_chunk(stateless)
    assert deserialize_chunk("a", 0, data, compressed).initial_state is None
