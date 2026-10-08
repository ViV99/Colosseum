"""ComposedModel: encoder -> core -> heads as a PolicyModel."""

from __future__ import annotations

import pytest
import torch

from helpers import CORE_KINDS, make_simple_model

OBS, B, T = 8, 3, 5


def test_stateless_state_dict_keys():
    """A stateless ComposedModel keeps the flat encoder/policy/value checkpoint layout."""
    model = make_simple_model(obs_dim=OBS)
    assert set(model.state_dict()) == {
        f"{part}.fc.{param}" for part in ("encoder", "policy", "value") for param in ("weight", "bias")
    }
    assert not model.is_stateful


@pytest.mark.parametrize("kind", CORE_KINDS)
def test_unroll_matches_step_loop_with_resets(kind):
    torch.manual_seed(0)
    model = make_simple_model(obs_dim=OBS, core=kind)
    obs = torch.randn(T, B, OBS)
    dones = torch.zeros(T, B, dtype=torch.bool)
    dones[1, 0] = True
    dones[3, 2] = True
    mask = torch.ones(T, B, 4, dtype=torch.bool)
    mask[:, :, 0] = False

    state = model.initial_state(B)
    logits, values = [], []
    for t in range(T):
        out = model.step(obs[t], state, mask[t])
        logits.append(out.dist.logits)
        values.append(out.value)
        state = model.reset_state(out.state, dones[t])

    unrolled = model.unroll(obs, model.initial_state(B), dones, mask)
    assert unrolled.value.shape == (T * B,)
    assert torch.allclose(unrolled.dist.logits, torch.cat(logits), atol=1e-5)
    assert torch.allclose(unrolled.value, torch.cat(values), atol=1e-5)
    assert not unrolled.dist.log_prob(torch.zeros(T * B, dtype=torch.long)).isfinite().any()


@pytest.mark.parametrize("kind", ["lstm", "window"])
def test_stateful_unroll_requires_state0(kind):
    model = make_simple_model(obs_dim=OBS, core=kind)
    with pytest.raises(ValueError, match="state0 is None"):
        model.unroll(torch.zeros(T, B, OBS), None, torch.zeros(T, B, dtype=torch.bool))


def test_unroll_rejects_empty_sequence():
    model = make_simple_model(obs_dim=OBS, core="gru")
    with pytest.raises(ValueError, match="T >= 1"):
        model.unroll(torch.zeros(0, B, OBS), model.initial_state(B), torch.zeros(0, B, dtype=torch.bool))


@pytest.mark.parametrize("empty", [{}, (), []])
def test_reset_state_without_tensor_leaves_returns_state_unchanged(empty):
    model = make_simple_model(obs_dim=OBS, core="lstm")
    assert model.reset_state(empty, torch.tensor([True, False])) is empty
