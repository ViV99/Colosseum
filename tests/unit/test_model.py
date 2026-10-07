"""Unit tests for the PolicyModel protocol, act() and Distribution.cat."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from colosseum.networks.distributions import CategoricalDist, CompositeDist, DiagGaussianDist
from colosseum.networks.model import ActOutput, PolicyModel, StepOutput, UnrollOutput, act

OBS, A = 3, 4


class CounterModel(PolicyModel):
    """Stateful toy model: the state counts steps since the episode start."""

    def __init__(self) -> None:
        super().__init__()
        self.pi = nn.Linear(OBS + 1, A)
        self.v = nn.Linear(OBS + 1, 1)

    def initial_state(self, batch_size, device="cpu"):
        return {"count": torch.zeros(batch_size, 1, device=device)}

    def step(self, obs, state, action_mask=None):
        x = torch.cat([obs, state["count"]], dim=-1)
        dist = CategoricalDist(self.pi(x))
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        return StepOutput(dist, self.v(x).squeeze(-1), {"count": state["count"] + 1.0})


class StatelessModel(PolicyModel):
    def __init__(self) -> None:
        super().__init__()
        self.pi = nn.Linear(OBS, A)
        self.v = nn.Linear(OBS, 1)

    def step(self, obs, state, action_mask=None):
        dist = CategoricalDist(self.pi(obs))
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        return StepOutput(dist, self.v(obs).squeeze(-1), None)


def _manual_unroll(model, obs, dones, masks=None):
    T, B = obs.shape[:2]
    state = model.initial_state(B)
    logits, values = [], []
    for t in range(T):
        out = model.step(obs[t], state, None if masks is None else masks[t])
        logits.append(out.dist.logits)
        values.append(out.value)
        state = out.state
        if state is not None:
            keep = (~dones[t]).float().unsqueeze(-1)
            state = {"count": state["count"] * keep}
    return torch.cat(logits), torch.cat(values)


@pytest.mark.parametrize("model_cls", [CounterModel, StatelessModel])
def test_default_unroll_matches_step_loop_with_resets(model_cls):
    torch.manual_seed(0)
    model = model_cls()
    T, B = 5, 3
    obs = torch.randn(T, B, OBS)
    dones = torch.zeros(T, B, dtype=torch.bool)
    dones[1, 0] = True
    dones[3, 2] = True
    masks = torch.ones(T, B, A, dtype=torch.bool)
    masks[:, :, 0] = False

    out = model.unroll(obs, model.initial_state(B), dones, masks)
    assert isinstance(out, UnrollOutput)
    assert out.value.shape == (T * B,)
    ref_logits, ref_values = _manual_unroll(model, obs, dones, masks)
    assert torch.allclose(out.dist.logits, ref_logits, atol=1e-6)
    assert torch.allclose(out.value, ref_values, atol=1e-6)
    actions = torch.randint(1, A, (T * B,))
    assert out.dist.log_prob(actions).shape == (T * B,)


def test_unroll_is_time_major():
    torch.manual_seed(0)
    model = CounterModel()
    T, B = 3, 2
    obs = torch.randn(T, B, OBS)
    out = model.unroll(obs, model.initial_state(B), torch.zeros(T, B, dtype=torch.bool))
    state = model.initial_state(B)
    for t in range(T):
        step = model.step(obs[t], state)
        state = step.state
        for b in range(B):
            assert torch.allclose(out.value[t * B + b], step.value[b], atol=1e-6)


def test_reset_state_resets_only_done_rows():
    model = CounterModel()
    state = {"count": torch.tensor([[3.0], [5.0]])}
    out = model.reset_state(state, torch.tensor([True, False]))
    assert out["count"].tolist() == [[0.0], [5.0]]
    assert model.reset_state(None, torch.tensor([True])) is None


def test_is_stateful_and_default_hooks():
    assert CounterModel().is_stateful
    stateless = StatelessModel()
    assert not stateless.is_stateful
    assert stateless.initial_state(4) is None
    assert stateless.update_normalizers(torch.zeros(2, OBS)) is None


def test_act_samples_masked_actions_and_returns_state():
    torch.manual_seed(0)
    model = CounterModel()
    obs = torch.randn(6, OBS)
    mask = torch.zeros(6, A, dtype=torch.bool)
    mask[:, 2] = True
    out = act(model, obs, model.initial_state(6), mask)
    assert isinstance(out, ActOutput)
    assert out.actions.tolist() == [2] * 6
    assert out.log_probs.shape == (6,) and torch.allclose(out.log_probs, torch.zeros(6), atol=1e-6)
    assert out.values.shape == (6,)
    assert out.state["count"].tolist() == [[1.0]] * 6
    assert not out.log_probs.requires_grad


def test_act_deterministic_takes_mode():
    torch.manual_seed(0)
    model = StatelessModel()
    obs = torch.randn(5, OBS)
    out = act(model, obs, None, deterministic=True)
    expected = model.pi(obs).argmax(dim=-1)
    assert torch.equal(out.actions, expected)
    assert out.state is None


def test_distribution_cat_categorical_gaussian_composite():
    torch.manual_seed(0)
    c1, c2 = CategoricalDist(torch.randn(2, A)), CategoricalDist(torch.randn(3, A))
    cc = CategoricalDist.cat([c1, c2])
    a = torch.randint(0, A, (5,))
    assert torch.allclose(cc.log_prob(a), torch.cat([c1.log_prob(a[:2]), c2.log_prob(a[2:])]))

    g1 = DiagGaussianDist(torch.randn(2, 2), torch.zeros(1, 2).expand(2, -1))
    g2 = DiagGaussianDist(torch.randn(1, 2), torch.zeros(1, 2))
    gc = DiagGaussianDist.cat([g1, g2])
    x = torch.randn(3, 2)
    assert torch.allclose(gc.log_prob(x), torch.cat([g1.log_prob(x[:2]), g2.log_prob(x[2:])]))

    k1 = CompositeDist({"d": c1, "s": DiagGaussianDist(torch.randn(2, 1), torch.zeros(2, 1))})
    k2 = CompositeDist({"d": c2, "s": DiagGaussianDist(torch.randn(3, 1), torch.zeros(3, 1))})
    kc = CompositeDist.cat([k1, k2])
    flat = torch.cat([a.float().unsqueeze(-1), torch.randn(5, 1)], dim=-1)
    assert torch.allclose(kc.log_prob(flat), torch.cat([k1.log_prob(flat[:2]), k2.log_prob(flat[2:])]))


def test_masked_categorical_cat_keeps_mask():
    logits = torch.zeros(2, A)
    mask = torch.tensor([[True, False, True, False], [False, True, True, True]])
    d = CategoricalDist(logits, mask=mask)
    cc = CategoricalDist.cat([d, d])
    assert torch.isinf(cc.logits[0, 1]) and torch.isinf(cc.logits[2, 1])
    assert torch.isfinite(cc.entropy()).all()


@pytest.mark.parametrize("empty", [{}, (), [], {"a": None}])
def test_reset_state_without_tensor_leaves_returns_state_unchanged(empty):
    out = CounterModel().reset_state(empty, torch.tensor([True, False]))
    assert out is empty


def test_unroll_of_stateful_model_requires_state0():
    model = CounterModel()
    with pytest.raises(ValueError, match="state0 is None"):
        model.unroll(torch.zeros(2, 2, OBS), None, torch.zeros(2, 2, dtype=torch.bool))


@pytest.mark.parametrize("model_cls", [CounterModel, StatelessModel])
def test_unroll_rejects_empty_sequence(model_cls):
    model = model_cls()
    with pytest.raises(ValueError, match="T >= 1"):
        model.unroll(torch.zeros(0, 2, OBS), model.initial_state(2), torch.zeros(0, 2, dtype=torch.bool))
