"""APPO on chunk v2: unit modes, reductions over ACT slots, diagnostics, state (T4.2)."""

from __future__ import annotations

import copy
import itertools
import logging
import math

import gymnasium
import numpy as np
import pytest
import torch

import colosseum.algorithms.appo as appo_module
from colosseum.algorithms.appo import APPO, _select_chunks, resolve_modes
from colosseum.algorithms.vtrace import VTraceOut
from colosseum.core.config import AlgorithmConfig
from colosseum.core.specs import ActionSpec
from colosseum.core.tree import tree_leaves
from colosseum.core.types import SLOT_ACT, TrajectoryChunk
from colosseum.envs.game import RoleSpec
from colosseum.envs.spaces import Units
from colosseum.networks.state import slice_batch
from game_helpers import CORE_KINDS, make_test_model, synthetic_chunk

OBS = gymnasium.spaces.Box(-1.0, 1.0, (5,), np.float32)
DISCRETE = RoleSpec(OBS, gymnasium.spaces.Discrete(4))
UNITS = RoleSpec(OBS, Units(6, gymnasium.spaces.Discrete(3)))
UNITS_ONE = RoleSpec(OBS, Units(1, gymnasium.spaces.Discrete(3)))

DIAGNOSTICS = {
    "approx_kl", "clip_fraction", "clip_fraction_joint", "rho_mean", "rho_clip_frac", "c_clip_frac",
    "log_rho_abs_mean", "log_rho_abs_p95", "log_rho_joint_abs_mean", "log_rho_joint_abs_p95", "ess",
    "deciders_valid_mean", "deciders_valid_max", "boot_frac", "pad_frac", "explained_variance",
}
STATE_KEYS = {"optimizer", "progress", "scaler", "kickstart", "policy_version", "consumed_samples",
              "critic_warmup_done"}


def _patterns() -> list[str]:
    """4 chunks x 6 slots: 19 ACTs, 4 BOOTs (B or R) and 1 PAD."""
    return ["AAATAB", "ATAATP", "AARAAB", "AAAAAB"]


def _chunks(model, role, *, noise=0.3, seed=0, patterns=None):
    return [synthetic_chunk(model, role, p, seed=seed + i, logp_noise=noise)
            for i, p in enumerate(patterns or _patterns())]


def _algo(role, model, **cfg) -> APPO:
    return APPO(model, AlgorithmConfig(**cfg), ActionSpec.from_space(role.action_space), device="cpu")


def test_modes_resolve_auto_and_collapse_for_one_decider():
    units = ActionSpec.from_space(UNITS.action_space)
    single = ActionSpec.from_space(DISCRETE.action_space)
    one_unit = ActionSpec.from_space(UNITS_ONE.action_space)
    # unit_trace auto with Units = joint: the T8.4 units-experiment ruling (docs/benchmarks.md).
    assert resolve_modes(AlgorithmConfig(), units) == ("per_unit", "joint", "mean_valid")
    assert resolve_modes(AlgorithmConfig(ratio_mode="joint"), units) == ("joint", "joint", "sum")
    assert resolve_modes(AlgorithmConfig(unit_trace="geo_mean"), units) == ("per_unit", "geo_mean", "mean_valid")
    assert resolve_modes(AlgorithmConfig(ratio_mode="joint", entropy_reduction="mean_valid", unit_trace="none"),
                         units) == ("joint", "none", "mean_valid")
    for spec in (single, one_unit):
        for r, t, e in itertools.product(("auto", "joint", "per_unit"), ("auto", "joint", "geo_mean", "none"),
                                         ("auto", "mean_valid", "sum")):
            cfg = AlgorithmConfig(ratio_mode=r, unit_trace=t, entropy_reduction=e)
            # One decider: everything collapses to the joint path, except an explicit
            # unit_trace "none" (rho = c = 1 is defined independently of K), which is honoured.
            assert resolve_modes(cfg, spec) == ("joint", "none" if t == "none" else "joint", "sum"), (r, t, e)


def test_explicit_per_unit_with_one_decider_is_logged_once(monkeypatch, caplog):
    single = ActionSpec.from_space(DISCRETE.action_space)
    monkeypatch.setattr(appo_module, "_per_unit_collapse_logged", False)
    with caplog.at_level(logging.INFO, logger=appo_module.__name__):
        resolve_modes(AlgorithmConfig(), single)
        resolve_modes(AlgorithmConfig(unit_trace="geo_mean", entropy_reduction="mean_valid"), single)
        assert caplog.records == []                       # auto and silent collapses: no log
        for _ in range(3):
            resolve_modes(AlgorithmConfig(ratio_mode="per_unit"), single)
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.INFO, "one decider: per_unit equals joint except for the rho factor; using joint")]


@pytest.mark.parametrize("role", [DISCRETE, UNITS], ids=["discrete", "units"])
def test_compute_loss_returns_finite_losses_and_every_diagnostic(role):
    torch.manual_seed(0)
    model = make_test_model(role)
    losses = _algo(role, model).compute_loss(_chunks(model, role))
    assert {"total_loss", "policy_loss", "value_loss", "entropy"} | DIAGNOSTICS <= set(losses)
    for key, value in losses.items():
        assert torch.isfinite(value), key
    assert float(losses["boot_frac"]) == pytest.approx(4 / 24)
    assert float(losses["pad_frac"]) == pytest.approx(1 / 24)
    if role is UNITS:
        assert 1.0 <= float(losses["deciders_valid_mean"]) <= float(losses["deciders_valid_max"]) <= 6.0
    else:
        assert float(losses["deciders_valid_mean"]) == float(losses["deciders_valid_max"]) == 1.0


@pytest.mark.parametrize("core", ["none", "lstm"])
@pytest.mark.parametrize("role", [DISCRETE, UNITS], ids=["discrete", "units"])
def test_zero_lag_diagnostics_are_exact(role, core):
    """Fresh chunks of the same weights: rho = 1 everywhere, nothing is clipped, ESS = 1.

    The bars are above 1 (as in SP1's zero-lag metric test): worker and learner log-probs agree
    only up to float rounding (~1e-7), and a rho of 1 + 1e-7 is clipped by a bar of exactly 1.
    """
    torch.manual_seed(0)
    model = make_test_model(role, core=core)
    chunks = _chunks(model, role, noise=0.0)
    algo = _algo(role, model, vtrace_rho_bar=1.5, vtrace_c_bar=1.5)
    with torch.no_grad():
        losses = {k: float(v) for k, v in algo.compute_loss(chunks).items()}
    assert losses["ess"] == pytest.approx(1.0, abs=1e-6)
    assert losses["clip_fraction"] == losses["clip_fraction_joint"] == 0.0
    assert losses["rho_clip_frac"] == losses["c_clip_frac"] == 0.0
    assert losses["rho_mean"] == pytest.approx(1.0, abs=1e-5)
    assert losses["log_rho_joint_abs_p95"] == pytest.approx(0.0, abs=1e-5)
    assert losses["log_rho_abs_p95"] == pytest.approx(0.0, abs=1e-5)


def test_log_rho_quantiles_and_ess_are_taken_over_act_slots():
    torch.manual_seed(0)
    model = make_test_model(UNITS)
    chunks = _chunks(model, UNITS, noise=0.5)
    algo = _algo(UNITS, copy.deepcopy(model), unit_trace="joint")
    lp, _, unit_lp = algo.evaluate_chunks(chunks)
    S, B = chunks[0].num_slots, len(chunks)
    act = torch.stack([c.kind for c in chunks], dim=1).reshape(-1) == SLOT_ACT
    behavior = torch.stack([c.behavior_logp for c in chunks], dim=1).reshape(-1)
    behavior_unit = torch.stack([c.behavior_unit_logp for c in chunks], dim=1).reshape(S * B, -1)
    valid = torch.stack([c.action_masks["unit"] for c in chunks], dim=1).reshape(S * B, -1) & act.unsqueeze(-1)
    joint = (lp - behavior)[act]
    per_decider = (unit_lp - behavior_unit)[valid]
    rho = torch.exp(joint)
    with torch.no_grad():
        losses = {k: float(v) for k, v in algo.compute_loss(chunks).items()}
    assert losses["log_rho_joint_abs_p95"] == pytest.approx(float(torch.quantile(joint.abs(), 0.95)), abs=1e-5)
    assert losses["log_rho_joint_abs_mean"] == pytest.approx(float(joint.abs().mean()), abs=1e-5)
    assert losses["log_rho_abs_p95"] == pytest.approx(float(torch.quantile(per_decider.abs(), 0.95)), abs=1e-5)
    assert losses["ess"] == pytest.approx(float(rho.sum() ** 2 / (rho.numel() * (rho ** 2).sum())), rel=1e-4)
    assert 0.0 < losses["ess"] < 1.0
    assert losses["rho_clip_frac"] == pytest.approx(float((rho > 1.0).float().mean()))


@pytest.mark.parametrize("role", [DISCRETE, UNITS_ONE], ids=["discrete", "units1"])
def test_one_decider_gives_identical_results_in_every_mode(role):
    torch.manual_seed(0)
    model = make_test_model(role)
    chunks = _chunks(model, role, noise=0.5)
    with torch.no_grad():
        reference = {k: float(v) for k, v in _algo(role, copy.deepcopy(model)).compute_loss(chunks).items()}
    # unit_trace "none" is honoured with one decider (rho = c = 1), so it differs off-policy.
    for r, t, e in itertools.product(("joint", "per_unit"), ("joint", "geo_mean"), ("mean_valid", "sum")):
        algo = _algo(role, copy.deepcopy(model), ratio_mode=r, unit_trace=t, entropy_reduction=e)
        with torch.no_grad():
            got = {k: float(v) for k, v in algo.compute_loss(chunks).items()}
        assert got == pytest.approx(reference, abs=1e-6, nan_ok=True), (r, t, e)


def _patched_vtrace(algo: APPO, rho: torch.Tensor):
    """Replace the clipped rho V-trace returns by ``rho`` (td and vs unchanged)."""
    inner = algo._compute_vtrace

    def patched(**kwargs):
        out = inner(**kwargs)
        return VTraceOut(vs=out.vs, td=out.td, clipped_rho=torch.where(kwargs["is_act"], rho, 0.0))

    algo._compute_vtrace = patched


def test_per_unit_does_not_multiply_the_advantage_by_rho_but_joint_does():
    torch.manual_seed(0)
    model = make_test_model(UNITS)
    chunks = _chunks(model, UNITS, noise=0.5)
    rho = torch.rand(6, len(chunks)) + 0.1
    for mode, changes in (("per_unit", False), ("joint", True)):
        plain = _algo(UNITS, copy.deepcopy(model), ratio_mode=mode).compute_loss(chunks)
        algo = _algo(UNITS, copy.deepcopy(model), ratio_mode=mode)
        _patched_vtrace(algo, rho)
        patched = algo.compute_loss(chunks)
        same = torch.allclose(plain["policy_loss"], patched["policy_loss"], atol=1e-7)
        assert same != changes, mode
        assert torch.allclose(plain["value_loss"], patched["value_loss"])


def _captured_log_rhos(algo: APPO, chunks):
    seen = {}
    inner = algo._compute_vtrace

    def spy(**kwargs):
        seen.update(kwargs)
        return inner(**kwargs)

    algo._compute_vtrace = spy
    algo.compute_loss(chunks)
    return seen


def test_unit_trace_sets_the_scalar_log_rho_of_vtrace():
    torch.manual_seed(0)
    model = make_test_model(UNITS)
    chunks = _chunks(model, UNITS, noise=0.5)
    algo = _algo(UNITS, copy.deepcopy(model))
    _, _, unit_lp = algo.evaluate_chunks(chunks)
    S, B = chunks[0].num_slots, len(chunks)
    behavior = torch.stack([c.behavior_unit_logp for c in chunks], dim=1).reshape(S * B, -1)
    kind = torch.stack([c.kind for c in chunks], dim=1).reshape(-1)
    valid = torch.stack([c.action_masks["unit"] for c in chunks], dim=1).reshape(S * B, -1)
    is_act = (kind == SLOT_ACT).unsqueeze(-1)
    diff = torch.where(valid & is_act, unit_lp - behavior, torch.zeros_like(unit_lp))
    joint = diff.sum(-1).reshape(S, B)
    geo = (diff.sum(-1) / valid.sum(-1).clamp(min=1)).reshape(S, B)
    for mode, expected, bars in (("joint", joint, (1.0, 1.0)), ("geo_mean", geo, (1.0, 1.0)),
                                 ("none", torch.zeros(S, B), (1.0, 1.0))):
        cfg = {"unit_trace": mode, "vtrace_rho_bar": 0.5, "vtrace_c_bar": 0.7}
        seen = _captured_log_rhos(_algo(UNITS, copy.deepcopy(model), **cfg), chunks)
        assert torch.allclose(seen["log_rhos"], expected, atol=1e-5), mode
        if mode == "none":
            assert (seen["rho_bar"], seen["c_bar"]) == bars
        else:
            assert (seen["rho_bar"], seen["c_bar"]) == (0.5, 0.7)


def test_unit_trace_none_is_honoured_with_one_decider():
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    chunks = _chunks(model, DISCRETE, noise=0.5)
    algo = _algo(DISCRETE, model, unit_trace="none", vtrace_rho_bar=0.5, vtrace_c_bar=0.7)
    assert algo.modes == ("joint", "none", "sum")
    seen = _captured_log_rhos(algo, chunks)
    assert torch.equal(seen["log_rhos"], torch.zeros_like(seen["log_rhos"]))
    assert (seen["rho_bar"], seen["c_bar"]) == (1.0, 1.0)


def test_entropy_reduction_sum_vs_mean_valid():
    torch.manual_seed(0)
    model = make_test_model(UNITS)
    chunks = [synthetic_chunk(model, UNITS, p, seed=i, random_units=False) for i, p in enumerate(_patterns())]
    total = _algo(UNITS, copy.deepcopy(model), entropy_reduction="sum").compute_loss(chunks)["entropy"]
    mean = _algo(UNITS, copy.deepcopy(model), entropy_reduction="mean_valid").compute_loss(chunks)["entropy"]
    assert float(mean.detach()) == pytest.approx(float(total.detach()) / 6, rel=1e-5)   # all 6 units valid


@pytest.mark.parametrize("core", ["none", "lstm"])
def test_losses_and_metrics_ignore_non_act_slots(core):
    torch.manual_seed(0)
    model = make_test_model(UNITS, core=core)
    chunks = _chunks(model, UNITS, patterns=["AATP", "ATAB", "AARP"])
    poisoned = []
    for c in chunks:
        bad = copy.deepcopy(c)
        pad = (bad.kind == 2).unsqueeze(-1)
        bad.obs = torch.where(pad, torch.full_like(bad.obs, float("nan")), bad.obs)
        bad.reward = torch.where(bad.kind != SLOT_ACT, torch.full_like(bad.reward, float("nan")), bad.reward)
        poisoned.append(bad)
    algo = _algo(UNITS, model)
    with torch.no_grad():
        clean, dirty = algo.compute_loss(chunks), algo.compute_loss(poisoned)
    for key in clean:
        assert torch.isfinite(dirty[key]), key
        assert float(dirty[key]) == pytest.approx(float(clean[key]), abs=1e-6, nan_ok=True), key


def test_value_targets_bootstrap_from_the_learners_boot_value():
    """Changing only the BOOT slot's observation changes the value loss: the bootstrap is V(boot)."""
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    chunk = synthetic_chunk(model, DISCRETE, "AAAB", seed=1)
    other = copy.deepcopy(chunk)
    other.obs[3] = other.obs[3] + 3.0
    algo = _algo(DISCRETE, model, vtrace_lambda=1.0)
    with torch.no_grad():
        a, b = algo.compute_loss([chunk]), algo.compute_loss([other])
    assert float(a["value_loss"]) != pytest.approx(float(b["value_loss"]))
    terminal = synthetic_chunk(model, DISCRETE, "AATP", seed=1)
    moved = copy.deepcopy(terminal)
    moved.obs[3] = moved.obs[3] + 3.0                          # a PAD after a terminal ACT
    with torch.no_grad():
        assert float(algo.compute_loss([terminal])["value_loss"]) == pytest.approx(
            float(algo.compute_loss([moved])["value_loss"]))


@pytest.mark.parametrize("core", CORE_KINDS)
@pytest.mark.parametrize("role", [DISCRETE, UNITS], ids=["discrete", "units"])
def test_evaluate_chunks_reproduces_behavior_log_probs_at_zero_lag(core, role):
    torch.manual_seed(0)
    model = make_test_model(role, core=core)
    chunks = _chunks(model, role, noise=0.0)
    lp, values, unit = _algo(role, model).evaluate_chunks(chunks)
    S, B = chunks[0].num_slots, len(chunks)
    kind = torch.stack([c.kind for c in chunks], dim=1).reshape(-1)
    act = kind == SLOT_ACT
    behavior = torch.stack([c.behavior_logp for c in chunks], dim=1).reshape(-1)
    assert lp.shape == values.shape == (S * B,)
    assert torch.allclose(lp[act], behavior[act], atol=1e-5)
    if role is UNITS:
        bu = torch.stack([c.behavior_unit_logp for c in chunks], dim=1).reshape(S * B, -1)
        assert torch.allclose(unit[act], bu[act], atol=1e-5)
    else:
        assert unit is None


def test_train_step_counts_act_slots_and_reports_metrics():
    torch.manual_seed(0)
    model = make_test_model(UNITS, core="gru")
    algo = _algo(UNITS, model, minibatch_chunks=2, num_epochs=2)
    before = [p.detach().clone() for p in model.parameters()]
    metrics = algo.train_step(_chunks(model, UNITS))
    assert algo.policy_version == 1
    assert algo.consumed_samples == sum(p.count("A") + p.count("T") for p in _patterns())
    assert DIAGNOSTICS | {"grad_norm", "skipped_updates", "policy_version", "lr", "total_loss"} <= set(metrics)
    assert all(isinstance(v, float) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_normalizers_update_once_from_act_and_reset_boot_slots(monkeypatch):
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    calls = []
    monkeypatch.setattr(model, "update_normalizers", lambda obs, gs=None: calls.append((obs.clone(), gs)))
    chunks = [synthetic_chunk(model, DISCRETE, p, seed=i) for i, p in enumerate(["AARP", "ATAB"])]
    _algo(DISCRETE, model, num_epochs=3, minibatch_chunks=1).train_step(chunks)
    assert len(calls) == 1
    obs, gs = calls[0]
    expected = [chunks[0].obs[0], chunks[1].obs[0], chunks[0].obs[1], chunks[1].obs[1], chunks[0].obs[2],
                chunks[1].obs[2]]                            # [S, B] row-major: slot 2 of chunk 1 is an ACT
    assert gs is None and obs.shape == (6, 5)
    assert torch.equal(obs, torch.stack(expected))


def test_minibatch_slice_equals_a_prepared_minibatch():
    torch.manual_seed(0)
    model = make_test_model(UNITS, core="lstm")
    chunks = _chunks(model, UNITS)
    algo = _algo(UNITS, model)
    idx = torch.tensor([2, 0])
    sliced = _select_chunks(algo._prepare_batch(chunks), idx)
    direct = algo._prepare_batch([chunks[2], chunks[0]])
    with torch.no_grad():
        a, b = algo._loss_from_batch(sliced), algo._loss_from_batch(direct)
    for key in a:
        assert float(a[key]) == pytest.approx(float(b[key]), abs=1e-6, nan_ok=True), key
    assert torch.equal(slice_batch(algo._prepare_batch(chunks)["initial_state"], idx)["h"],
                       direct["initial_state"]["h"])
    with pytest.raises(KeyError):
        _select_chunks({"kind": torch.zeros(2, 2), "bogus": torch.zeros(2, 2)}, torch.tensor([0]))


def test_state_dict_keys_and_resume_reproduce_the_next_update():
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    algo = _algo(DISCRETE, model)
    algo.train_step(_chunks(model, DISCRETE, seed=0))
    state = algo.state_dict()
    assert set(state) == STATE_KEYS
    clone_model = copy.deepcopy(model)
    clone = _algo(DISCRETE, clone_model)
    clone.load_state_dict(state)
    batch = _chunks(model, DISCRETE, seed=10)
    torch.manual_seed(1)
    algo.train_step(batch)
    torch.manual_seed(1)
    clone.train_step(batch)
    for a, b in zip(model.parameters(), clone_model.parameters()):
        assert torch.equal(a, b)
    assert clone.policy_version == algo.policy_version == 2
    assert clone.consumed_samples == algo.consumed_samples


@pytest.mark.parametrize("schedule,progress,factor", [("constant", 0.5, 1.0), ("linear", 0.25, 0.75),
                                                      ("cosine", 0.5, 0.5)])
def test_lr_is_a_function_of_progress(schedule, progress, factor):
    model = make_test_model(DISCRETE)
    algo = _algo(DISCRETE, model, lr_schedule=schedule, learning_rate=1e-3)
    algo.set_progress(progress)
    assert algo.state_dict()["optimizer"]["param_groups"][0]["lr"] == pytest.approx(1e-3 * factor)


def test_payload_roundtrip_keeps_the_loss():
    torch.manual_seed(0)
    model = make_test_model(UNITS)
    chunks = _chunks(model, UNITS)
    algo = _algo(UNITS, model)
    with torch.no_grad():
        a = algo.compute_loss(chunks)
        b = algo.compute_loss([TrajectoryChunk.from_payload(c.to_payload()) for c in chunks])
    assert float(a["total_loss"]) == pytest.approx(float(b["total_loss"]), abs=1e-7)


# ---------------------------------------------------------------------------
# GradScaler state (CPU coverage of the save/restore path) and CUDA successors of SP1's
# gpu tests (AMP, GradScaler, state_dict on CUDA, Dict + uint8 observations, pin_memory)
# ---------------------------------------------------------------------------


def _tensors(tree):
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _tensors(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _tensors(value)


def _assert_scaler_round_trip(algo: APPO, other: APPO) -> None:
    """Give ``algo``'s scaler a non-default state, save it, restore it into ``other``'s fresh scaler."""
    algo._scaler.update(new_scale=1234.0)            # distinctive scale (default init_scale is 65536)
    state = algo.state_dict()
    saved = state["scaler"]
    assert saved is not None and saved["scale"] == 1234.0
    assert saved["_growth_tracker"] > 0              # successful unskipped steps since the last growth
    fresh = other._scaler.state_dict()
    assert fresh["scale"] != saved["scale"] and fresh["_growth_tracker"] != saved["_growth_tracker"]
    other.load_state_dict(state)
    assert other._scaler.state_dict() == saved
    assert other._scaler.get_scale() == 1234.0


def test_grad_scaler_state_round_trips_with_a_cpu_scaler():
    """APPO creates a GradScaler only for AMP on CUDA, so this test installs a CPU one by hand."""
    torch.manual_seed(0)
    model = make_test_model(UNITS)
    chunks = _chunks(model, UNITS)
    algo = _algo(UNITS, model, num_epochs=1, minibatch_chunks=2)
    algo._scaler = torch.amp.GradScaler("cpu")
    algo.train_step(chunks)
    algo.train_step(chunks)
    other = _algo(UNITS, make_test_model(UNITS), num_epochs=1, minibatch_chunks=2)
    other._scaler = torch.amp.GradScaler("cpu")
    _assert_scaler_round_trip(algo, other)


@pytest.mark.gpu
@pytest.mark.parametrize("amp_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("core", ["none", "lstm"])
def test_amp_train_step_on_cuda_with_units(amp_dtype, core):
    torch.manual_seed(0)
    model = make_test_model(UNITS, core=core)
    chunks = _chunks(model, UNITS)
    algo = APPO(model, AlgorithmConfig(use_amp=True, amp_dtype=amp_dtype), ActionSpec.from_space(UNITS.action_space),
                device="cuda", pin_memory=True)
    metrics = algo.train_step(chunks)
    assert np.isfinite(metrics["total_loss"]) and algo.policy_version == 1
    lp, _, unit = algo.evaluate_chunks(chunks)
    assert lp.device.type == "cuda" and unit.shape[1] == 6


@pytest.mark.gpu
def test_grad_scaler_state_round_trips_on_cuda():
    torch.manual_seed(0)
    model = make_test_model(UNITS)
    chunks = _chunks(model, UNITS)
    spec = ActionSpec.from_space(UNITS.action_space)
    cfg = AlgorithmConfig(num_epochs=2, minibatch_chunks=2, use_amp=True, amp_dtype="float16")
    algo = APPO(model, cfg, spec, device="cuda")
    assert algo._scaler is not None
    for _ in range(3):
        algo.train_step(chunks)
    other = APPO(make_test_model(UNITS), cfg, spec, device="cuda")
    _assert_scaler_round_trip(algo, other)


@pytest.mark.gpu
def test_state_dict_round_trip_with_model_on_cuda():
    """Saved tensors are CPU copies; loading puts the optimizer state back on the model's device."""
    torch.manual_seed(0)
    model = make_test_model(UNITS, core="lstm")
    chunks = _chunks(model, UNITS)
    spec = ActionSpec.from_space(UNITS.action_space)
    cfg = AlgorithmConfig(num_epochs=1, minibatch_chunks=0)
    algo = APPO(model, cfg, spec, device="cuda")
    algo.set_progress(0.3)
    algo.train_step(chunks)
    state = algo.state_dict()
    assert set(state) == STATE_KEYS
    assert all(t.device.type == "cpu" for t in _tensors(state))

    resumed = APPO(make_test_model(UNITS, core="lstm"), cfg, spec, device="cuda")
    resumed.model.load_state_dict(algo.model.state_dict())
    resumed.load_state_dict(state)
    assert resumed.policy_version == 1
    assert resumed.consumed_samples == algo.consumed_samples == sum(c.num_acts for c in chunks)
    assert resumed._optimizer.param_groups[0]["lr"] == algo._optimizer.param_groups[0]["lr"]
    for per_param in resumed._optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            assert per_param[name].device.type == "cuda"
    torch.manual_seed(1)
    algo.train_step(chunks)
    torch.manual_seed(1)
    metrics = resumed.train_step(chunks)
    assert metrics["policy_version"] == 2.0
    for a, b in zip(algo.model.parameters(), resumed.model.parameters()):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("amp_dtype", ["float16", "bfloat16"])
def test_amp_train_step_on_cuda_with_dict_uint8_obs_and_global_state(amp_dtype):
    role = RoleSpec(
        gymnasium.spaces.Dict({"pixels": gymnasium.spaces.Box(0, 255, (2, 3), np.uint8),
                               "vec": gymnasium.spaces.Box(-1.0, 1.0, (4,), np.float32)}),
        Units(3, gymnasium.spaces.Discrete(3)),
        global_state_space=gymnasium.spaces.Box(0, 255, (6,), np.uint8),
    )
    torch.manual_seed(0)
    model = make_test_model(role, core="gru")
    chunks = _chunks(model, role)
    assert chunks[0].obs["pixels"].dtype == torch.uint8 and chunks[0].global_state.dtype == torch.uint8
    algo = APPO(model, AlgorithmConfig(use_amp=True, amp_dtype=amp_dtype, num_epochs=2, minibatch_chunks=2),
                ActionSpec.from_space(role.action_space), device="cuda", pin_memory=True)
    batch = algo._prepare_batch(chunks)
    assert batch["obs"]["pixels"].dtype == torch.uint8 and batch["obs"]["pixels"].device.type == "cuda"
    assert batch["global_state"].dtype == torch.uint8 and batch["global_state"].device.type == "cuda"
    metrics = algo.train_step(chunks)
    for key in ("total_loss", "policy_loss", "value_loss", "entropy", "rho_mean", "ess", "explained_variance"):
        assert math.isfinite(metrics[key]), key
    assert math.isfinite(metrics["grad_norm"]) or metrics["skipped_updates"] > 0
    assert algo.policy_version == 1
    for p in algo.model.parameters():
        assert p.dtype == torch.float32 and torch.isfinite(p).all()


@pytest.mark.gpu
@pytest.mark.parametrize("core", ["none", "lstm"])
def test_pin_memory_batch_preparation_on_cuda(core):
    torch.manual_seed(0)
    model = make_test_model(UNITS, core=core)
    chunks = _chunks(model, UNITS)
    spec = ActionSpec.from_space(UNITS.action_space)
    plain = APPO(copy.deepcopy(model), AlgorithmConfig(), spec, device="cuda")
    pinned = APPO(model, AlgorithmConfig(), spec, device="cuda", pin_memory=True)
    batch, plain_batch = pinned._prepare_batch(chunks), plain._prepare_batch(chunks)
    assert batch.keys() == plain_batch.keys()
    for key, value in batch.items():
        if value is None:
            assert plain_batch[key] is None, key
            continue
        leaves = list(_tensors(value)) if key == "initial_state" else tree_leaves(value)
        plain_leaves = list(_tensors(plain_batch[key])) if key == "initial_state" else tree_leaves(plain_batch[key])
        for a, b in zip(leaves, plain_leaves, strict=True):
            assert a.device.type == "cuda" and a.dtype == b.dtype and torch.equal(a, b), key
    metrics = pinned.train_step(chunks)
    assert all(math.isfinite(v) for v in metrics.values())
