"""Contract: the learner reproduces the worker's joint and per-unit log-probs at zero lag on
chunks with BOOT, PAD, truncation and elimination, and bootstraps with its own values (T4.3).

Chunks come from the real RolloutLoop + MatchRunner (in-process), cross the numpy payload
boundary, and are re-evaluated by the real APPO (``unroll`` from the chunks' initial states
with their ``reset_after`` flags).
"""

from __future__ import annotations

import copy

import gymnasium
import numpy as np
import pytest
import torch
import torch.nn as nn

from colosseum.algorithms.appo import APPO
from colosseum.algorithms.vtrace import compute_vtrace_slots
from colosseum.core.config import AlgorithmConfig
from colosseum.core.specs import ActionSpec
from colosseum.core.tree import tree_map, tree_stack
from colosseum.core.types import SLOT_ACT, SLOT_BOOT, SLOT_PAD, WeightPayload
from colosseum.envs.spaces import Units
from colosseum.networks.model import PolicyModel
from colosseum.networks.state import cat_batch, tree_leaves
from game_harness import (
    Collected,
    GameFactory,
    kinds,
    learner_eval,
    lineup,
    make_loop,
    run_until_chunks,
    slot_steps,
)
from game_helpers import CORE_KINDS, GlobalStateGame, Tick, TickGame, make_test_model

TOL = 1e-5

# 3 seats: seat 2 is eliminated (with a reward in that step), seat 1 becomes a dead
# teammate, then the episode is truncated.
ELIMINATION_THEN_TRUNCATION = [
    Tick(acting={0, 1, 2}),
    Tick(acting={0, 1, 2}, rewards={0: 0.5}),
    Tick(acting={0, 1}, rewards={2: -1.0}, terminated={2}),
    Tick(acting={0, 1}),
    Tick(acting={0}, rewards={1: 0.25}),
    Tick(acting={0}),
    Tick(over=True, truncated=True, rewards={0: 1.0, 1: 1.0}),
]
# 3 seats take turns, the episode ends by the rules.
TURNS = [
    Tick(acting={0}), Tick(acting={1}), Tick(acting={2}), Tick(acting={0}, rewards={1: 0.5}),
    Tick(acting={1}), Tick(over=True, rewards={0: 1.0, 1: -1.0, 2: 0.0}),
]
UNITS_SPACE = Units(3, gymnasium.spaces.Discrete(3))


def _unit_mask(k: int, t: int, seat: int) -> dict:
    unit = np.array([(t + seat + i + k) % 3 != 0 for i in range(3)])
    action = np.ones((3, 3), bool)
    action[:, 2] = (t % 2 == 0)
    return {"unit": unit, "action": action}


def _collect(core: str, *, units: bool, n_chunks: int = 24, chunk_length: int = 5, obs_dtype=np.float32,
             mask_fn=_unit_mask):
    game_kwargs = dict(action_space=UNITS_SPACE, mask_fn=mask_fn) if units else {}
    role = TickGame(TURNS, 3, obs_dtype=obs_dtype, **game_kwargs).spec.roles["player"]
    torch.manual_seed(0)
    learner_model = make_test_model(role, core=core)
    col = Collected(weights={"a": [WeightPayload.from_model("a", 0, learner_model)]})
    factory = GameFactory((ELIMINATION_THEN_TRUNCATION, 3), (TURNS, 3), obs_dtype=obs_dtype, **game_kwargs)
    loop, col = make_loop(factory, {"a": lambda: make_test_model(role, core=core)},
                          [lineup("3p", "a", "a", "a"), lineup("3p", "a", "a", "a")],
                          chunk_length=chunk_length, collected=col)
    chunks = run_until_chunks(loop, col, n_chunks)
    loop.close()
    return learner_model, role, chunks


def _max_diff(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    return float((a[mask] - b[mask]).abs().max())


@pytest.mark.parametrize("core", CORE_KINDS)
@pytest.mark.parametrize("units", [False, True], ids=["discrete", "units"])
def test_learner_reproduces_worker_log_probs_at_zero_lag(core, units):
    model, role, chunks = _collect(core, units=units)
    letters = "".join(kinds(c) for c in chunks)
    for needed in "TBRP":
        assert needed in letters, f"no {needed!r} slot in {[kinds(c) for c in chunks]}"
    # env 0 runs ELIMINATION_THEN_TRUNCATION: seat 2's ACT at t=1 is closed by its elimination,
    # and seat 1 (alive, no longer acting) gets a truncation BOOT from its final_obs at t=6.
    slots = {(s.env, s.seat, s.t, s.final, letter) for c in chunks for s, letter in zip(slot_steps(c), kinds(c))}
    assert (0, 2, 1, 0, "T") in slots, "no terminal ACT for the eliminated seat"
    assert (0, 1, 6, 1, "R") in slots, "no truncation BOOT for the dead teammate"
    if core != "none":
        assert any(any(float(x.abs().sum()) > 0 for x in tree_leaves(c.initial_state)) for c in chunks), \
            "no chunk starts mid-episode with a non-zero state"
    view = learner_eval(model, chunks, role)
    assert _max_diff(view.log_probs, view.worker_log_probs, view.is_act) < TOL
    if units:
        act = view.is_act
        assert view.unit_log_probs.shape[1] == 3
        assert _max_diff(view.unit_log_probs, view.worker_unit_log_probs, act) < TOL
        assert any(not bool(c.action_masks["unit"][c.kind == SLOT_ACT].all()) for c in chunks)
    else:
        assert view.unit_log_probs is None and view.worker_unit_log_probs is None


@pytest.mark.parametrize("core", CORE_KINDS)
def test_mutated_chunks_are_detected(core):
    model, role, chunks = _collect(core, units=False)
    shifted = []
    for c in chunks:
        bad = copy.deepcopy(c)
        bad.obs = torch.roll(bad.obs, shifts=1, dims=0)
        shifted.append(bad)
    view = learner_eval(model, shifted, role)
    assert _max_diff(view.log_probs, view.worker_log_probs, view.is_act) > 1e-3
    if core != "none":
        zeroed = []
        for c in chunks:
            bad = copy.deepcopy(c)
            bad.initial_state = model.initial_state(1)
            zeroed.append(bad)
        view = learner_eval(model, zeroed, role)
        assert _max_diff(view.log_probs, view.worker_log_probs, view.is_act) > 1e-4


def _value_bias(model: PolicyModel) -> torch.Tensor:
    return [m for m in model.value.modules() if isinstance(m, nn.Linear)][-1].bias


def test_bootstrap_uses_the_learners_current_values():
    """With lambda = 0 the target of an ACT is ``r + gamma * V(next slot)`` (0 after a terminal
    ACT). Shifting the learner's value head by delta after collection must shift the target of
    every open ACT by gamma * delta, also when the next slot is a BOOT: the bootstrap is the
    learner's current V(boot), never a number recorded by the worker."""
    model, role, chunks = _collect("lstm", units=False)
    gamma, delta = 0.9, 0.75
    S, B = chunks[0].num_slots, len(chunks)
    terminal = torch.stack([c.terminal for c in chunks], dim=1)
    rewards = torch.stack([c.reward for c in chunks], dim=1)

    def values_and_targets(m):
        view = learner_eval(m, chunks, role)
        values = view.values.reshape(S, B)
        vs = compute_vtrace_slots(log_rhos=torch.zeros(S, B), rewards=rewards, values=values,
                                  is_act=view.is_act.reshape(S, B), terminal=terminal, gamma=gamma, lam=0.0).vs
        return view.kind.reshape(S, B), values, vs

    kind, _, before = values_and_targets(model)
    is_act = kind == SLOT_ACT
    shifted = copy.deepcopy(model)
    with torch.no_grad():
        _value_bias(shifted).add_(delta)
    _, shifted_values, shifted_vs = values_and_targets(shifted)
    change = shifted_vs - before
    next_is_boot = torch.zeros_like(is_act)
    next_is_boot[:-1] = kind[1:] == SLOT_BOOT
    open_act, closed = is_act & ~terminal, is_act & terminal
    assert bool((open_act & next_is_boot).any()) and bool(closed.any())
    assert torch.allclose(change[open_act], torch.full_like(change[open_act], gamma * delta), atol=1e-5)
    assert torch.allclose(change[closed], torch.zeros_like(change[closed]), atol=1e-5)
    # APPO's own target path: its value loss is the ACT mean of (V - vs)^2 with these targets.
    algo = APPO(shifted, AlgorithmConfig(gamma=gamma, vtrace_lambda=0.0), ActionSpec.from_space(role.action_space))
    with torch.no_grad():
        value_loss = float(algo.compute_loss(chunks)["value_loss"])
    expected = float(((shifted_values - shifted_vs)[is_act] ** 2).mean())
    assert value_loss == pytest.approx(expected, rel=1e-5, abs=1e-6)


class DtypeSpy(PolicyModel):
    """Delegates to ``inner``; records the observation dtype seen by ``step`` and ``unroll``."""

    def __init__(self, inner: PolicyModel) -> None:
        super().__init__()
        self.inner = inner
        self.seen: list[tuple[str, torch.dtype]] = []

    def initial_state(self, batch_size, device="cpu"):
        return self.inner.initial_state(batch_size, device)

    def step(self, obs, state, action_mask=None):
        self.seen.append(("step", obs.dtype))
        return self.inner.step(obs, state, action_mask)

    def unroll(self, obs, state0, reset_after, action_mask=None, global_state=None, with_value=True):
        self.seen.append(("unroll", obs.dtype))
        return self.inner.unroll(obs, state0, reset_after, action_mask, global_state, with_value)

    def reset_state(self, state, done):
        return self.inner.reset_state(state, done)


def test_uint8_observations_reach_the_model_on_both_sides():
    role = TickGame(TURNS, 3, obs_dtype=np.uint8).spec.roles["player"]
    worker_model = DtypeSpy(make_test_model(role))
    loop, col = make_loop(GameFactory((TURNS, 3), obs_dtype=np.uint8), {"a": lambda: worker_model},
                          [lineup("3p", "a", "a", "a")], chunk_length=4)
    chunks = run_until_chunks(loop, col, 3)
    assert {d for _, d in worker_model.seen} == {torch.uint8}
    assert all(c.obs.dtype == torch.uint8 for c in chunks)
    learner_model = DtypeSpy(copy.deepcopy(worker_model.inner))
    learner_eval(learner_model, chunks, role)
    assert ("unroll", torch.uint8) in learner_model.seen


# ---------------------------------------------------------------------------
# R16: global_state reaches only the value path, end to end
# ---------------------------------------------------------------------------


def _global_state_game() -> GlobalStateGame:
    return GlobalStateGame(length=5, truncate_at=3)


@pytest.mark.parametrize("core", CORE_KINDS)
def test_global_state_changes_values_but_not_log_probs(core):
    """RolloutLoop -> chunk -> APPO with GlobalStateGame: perturbing the chunks' ``global_state``
    leaves the learner's log-probs unchanged (they still reproduce the worker's) and changes
    its values on every ACT and BOOT slot."""
    role = _global_state_game().spec.roles["player"]
    torch.manual_seed(0)
    model = make_test_model(role, core=core)
    col = Collected(weights={"a": [WeightPayload.from_model("a", 0, model)]})
    loop, col = make_loop(_global_state_game, {"a": lambda: make_test_model(role, core=core)},
                          [lineup("2p", "a", "a"), lineup("2p", "a", "a")], chunk_length=6, collected=col)
    chunks = run_until_chunks(loop, col, 12)
    loop.close()
    letters = "".join(kinds(c) for c in chunks)
    assert "R" in letters and "B" in letters and "A" in letters
    assert all(c.global_state is not None for c in chunks)

    perturbed = []
    for c in chunks:
        bad = copy.deepcopy(c)
        bad.global_state = tree_map(lambda t: t + 1.0, bad.global_state)
        perturbed.append(bad)
    view, other = learner_eval(model, chunks, role), learner_eval(model, perturbed, role)

    assert _max_diff(view.log_probs, view.worker_log_probs, view.is_act) < TOL
    assert torch.equal(other.log_probs, view.log_probs)
    scored = view.kind != SLOT_PAD
    assert bool(((other.values - view.values).abs()[scored] > 1e-6).all())


# ---------------------------------------------------------------------------
# Controller ruling (T4.2 review): per_unit policy loss and mean_valid entropy
# under random unit masks, recomputed by hand
# ---------------------------------------------------------------------------


def _random_unit_mask(k: int, t: int, seat: int) -> dict:
    """Pseudo-random Units mask: units switched off, and units on but with an empty action row
    (also invalid deciders), so the number of valid deciders varies from 0 to 3."""
    rng = np.random.default_rng([k, t, seat])
    return {"unit": rng.random(3) < 0.6, "action": rng.random((3, 3)) < 0.6}


def _decider_terms(model: PolicyModel, chunks) -> tuple[torch.Tensor, torch.Tensor]:
    """``unit_valid`` (bool) and ``unit_entropy`` ``[S, B, K]`` of ``model`` on the chunks (direct unroll)."""
    S, B = chunks[0].num_slots, len(chunks)
    actions = tree_map(lambda t: t.reshape(S * B, *t.shape[2:]), tree_stack([c.actions for c in chunks], axis=1))
    with torch.no_grad():
        out = model.unroll(tree_stack([c.obs for c in chunks], axis=1), cat_batch([c.initial_state for c in chunks]),
                           torch.stack([c.reset_after for c in chunks], dim=1).bool(),
                           tree_stack([c.action_masks for c in chunks], axis=1))
        return (out.dist.unit_valid(actions).reshape(S, B, -1),
                out.dist.unit_entropy(actions).float().reshape(S, B, -1))


@pytest.mark.parametrize("normalize", [True, False], ids=["normalized", "raw"])
@pytest.mark.parametrize("core", ["none", "lstm"])
def test_per_unit_policy_loss_and_mean_valid_entropy_under_random_unit_masks(core, normalize):
    """Zero lag: every per-decider ratio is 1, so the per_unit surrogate of a valid decider is
    the (normalized) advantage. Both the policy loss and the entropy are "mean over the valid
    deciders of a slot, then mean over ACT slots" (a slot without a valid decider adds 0)."""
    model, role, chunks = _collect(core, units=True, mask_fn=_random_unit_mask)
    S, B = chunks[0].num_slots, len(chunks)
    config = AlgorithmConfig(normalize_advantages=normalize)
    algo = APPO(model, config, ActionSpec.from_space(role.action_space), device="cpu")
    assert algo.modes == ("per_unit", "geo_mean", "mean_valid")

    kind = torch.stack([c.kind for c in chunks], dim=1)
    is_act = kind == SLOT_ACT
    unit_valid, unit_entropy = _decider_terms(model, chunks)
    masks = torch.stack([c.action_masks["unit"] for c in chunks], dim=1)
    rows = torch.stack([c.action_masks["action"] for c in chunks], dim=1).any(-1)
    assert torch.equal(unit_valid[is_act], (masks & rows)[is_act])     # validity comes from the masks
    valid = unit_valid & is_act.unsqueeze(-1)
    n_valid = valid.sum(-1)
    counts = set(n_valid[is_act].tolist())
    assert {0, 1, 2, 3} <= counts, f"valid-decider counts on ACT slots: {counts}"

    log_probs, values, unit_log_probs = algo.evaluate_chunks(chunks)
    worker_unit = torch.stack([c.behavior_unit_logp for c in chunks], dim=1).reshape(S * B, -1)
    assert _max_diff(unit_log_probs, worker_unit, is_act.reshape(-1)) < TOL          # zero lag
    vt = compute_vtrace_slots(
        log_rhos=torch.zeros(S, B), rewards=torch.stack([c.reward for c in chunks], dim=1),
        values=values.reshape(S, B), is_act=is_act, terminal=torch.stack([c.terminal for c in chunks], dim=1).bool(),
        gamma=config.gamma, rho_bar=config.vtrace_rho_bar, c_bar=config.vtrace_c_bar, lam=config.vtrace_lambda)
    adv = vt.td[is_act]                                     # per_unit: no rho factor
    if normalize:
        adv = (adv - adv.mean()) / (adv.std(unbiased=True) + 1e-8)
    n_act_valid = n_valid[is_act].float()
    per_slot_valid_mean = adv * (n_act_valid > 0)           # mean over valid deciders of a shared advantage
    expected_policy = -per_slot_valid_mean.mean()
    entropy_sum = torch.where(valid, unit_entropy, torch.zeros_like(unit_entropy)).sum(-1)[is_act]
    expected_entropy = (entropy_sum / n_act_valid.clamp(min=1)).mean()

    with torch.no_grad():
        got = algo.compute_loss(chunks)
    assert float(got["policy_loss"]) == pytest.approx(float(expected_policy), abs=1e-5)
    assert float(got["entropy"]) == pytest.approx(float(expected_entropy), abs=1e-5)
    # The check is discriminating: other reductions give clearly different numbers.
    assert abs(float(-(adv * n_act_valid).mean() - expected_policy)) > 1e-3                 # sum over deciders
    # (raw advantages are small and nearly uncorrelated with the decider count, so the pooled
    # mean lands ~5e-4 away: still 10x the 1e-5 match tolerance)
    assert abs(float(-(adv * n_act_valid).sum() / n_act_valid.sum() - expected_policy)) > 10 * 1e-5   # pooled
    assert abs(float(-(adv * n_act_valid / 3).mean() - expected_policy)) > 10 * 1e-5                  # all K
    assert abs(float(entropy_sum.mean() - expected_entropy)) > 1e-3                          # sum over deciders
    assert abs(float(entropy_sum.sum() / n_act_valid.sum() - expected_entropy)) > 1e-3       # pooled mean
    assert abs(float((entropy_sum / 3).mean() - expected_entropy)) > 1e-3                    # mean over all K


# Per-decider log-ratio offsets: e^0.5 > 1 + eps, e^-0.5 < 1 - eps, e^0.1 inside (eps = 0.2).
LOG_RATIO_OFFSETS = torch.tensor([0.5, -0.5, 0.1])


@pytest.mark.parametrize("normalize", [True, False], ids=["normalized", "raw"])
@pytest.mark.parametrize("core", ["none", "lstm"])
def test_per_unit_policy_loss_off_policy_under_random_unit_masks(core, normalize):
    """Off policy: the behavior log-probs of every valid decider are lowered by a known offset
    (the joint one by their sum), so the per-decider ratios lie on both sides of 1 +- eps.
    APPO's per_unit policy loss is ``min(r_i A, clip(r_i, 1 +- eps) A)`` per valid decider with
    the shared V-trace advantage (geo-mean trace, no rho factor), mean over the valid deciders,
    then mean over ACT slots."""
    model, role, chunks = _collect(core, units=True, mask_fn=_random_unit_mask)
    S, B = chunks[0].num_slots, len(chunks)
    config = AlgorithmConfig(normalize_advantages=normalize)
    eps = config.eps_clip
    algo = APPO(model, config, ActionSpec.from_space(role.action_space), device="cpu")
    assert algo.modes == ("per_unit", "geo_mean", "mean_valid")

    is_act = torch.stack([c.kind for c in chunks], dim=1) == SLOT_ACT
    unit_valid, _ = _decider_terms(model, chunks)
    valid = unit_valid & is_act.unsqueeze(-1)                                   # [S, B, K]
    offsets = torch.where(valid, LOG_RATIO_OFFSETS, torch.zeros(()))
    shifted = []
    for b, c in enumerate(chunks):
        bad = copy.deepcopy(c)
        bad.behavior_unit_logp = bad.behavior_unit_logp - offsets[:, b]
        bad.behavior_logp = bad.behavior_logp - offsets[:, b].sum(-1)
        shifted.append(bad)

    _, values, unit_log_probs = algo.evaluate_chunks(shifted)
    behavior_unit = torch.stack([c.behavior_unit_logp for c in shifted], dim=1)
    unit_log_ratio = torch.where(valid, unit_log_probs.reshape(S, B, -1) - behavior_unit, torch.zeros(()))
    assert torch.allclose(unit_log_ratio, offsets, atol=TOL)                     # the learner is still the worker
    ratio = torch.exp(unit_log_ratio)
    assert bool((valid & (ratio > 1 + eps)).any()) and bool((valid & (ratio < 1 - eps)).any())
    assert bool((valid & ((ratio - 1).abs() < eps) & (ratio != 1)).any())

    n_valid = valid.sum(-1)
    vt = compute_vtrace_slots(
        log_rhos=unit_log_ratio.sum(-1) / n_valid.clamp(min=1), rewards=torch.stack([c.reward for c in chunks], dim=1),
        values=values.reshape(S, B), is_act=is_act, terminal=torch.stack([c.terminal for c in chunks], dim=1).bool(),
        gamma=config.gamma, rho_bar=config.vtrace_rho_bar, c_bar=config.vtrace_c_bar, lam=config.vtrace_lambda)
    adv = vt.td                                                                 # per_unit: no rho factor
    if normalize:
        act_adv = vt.td[is_act]
        adv = (adv - act_adv.mean()) / (act_adv.std(unbiased=True) + 1e-8)
    a = adv.unsqueeze(-1)

    def reduce(unit_surr: torch.Tensor) -> float:
        per_slot = torch.where(valid, unit_surr, torch.zeros(())).sum(-1) / n_valid.clamp(min=1)
        return float(-per_slot[is_act].mean())

    expected = reduce(torch.min(ratio * a, torch.clamp(ratio, 1 - eps, 1 + eps) * a))
    with torch.no_grad():
        got = float(algo.compute_loss(shifted)["policy_loss"])
    assert got == pytest.approx(expected, abs=1e-5)
    # Discriminating: no clipping, or the joint path's rho-weighted advantage, give other numbers.
    assert abs(reduce(ratio * a) - expected) > 1e-3
    rho_adv = (vt.td * vt.clipped_rho).unsqueeze(-1)
    if normalize:
        act_rho_adv = rho_adv.squeeze(-1)[is_act]
        rho_adv = (rho_adv - act_rho_adv.mean()) / (act_rho_adv.std(unbiased=True) + 1e-8)
    assert abs(reduce(torch.min(ratio * rho_adv, torch.clamp(ratio, 1 - eps, 1 + eps) * rho_adv)) - expected) > 1e-3
