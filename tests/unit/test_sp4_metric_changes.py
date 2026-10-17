"""Metric changes of SP4 block 7 (T3.1): the truncated importance-weight mass replaces the clip fractions (exactly
0 with unit_trace none, 0 up to float rounding on-policy, the formula off-policy), the run-level loss counters never
decrease, and `colosseum bc` logs its device."""
from __future__ import annotations

import copy
import logging

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.bc.offline_bc import OfflineBCTrainer
from colosseum.core.config import AlgorithmConfig
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.types import SLOT_ACT
from colosseum.envs.game import RoleSpec
from colosseum.metrics.aggregator import SystemStats
from game_helpers import make_test_model, synthetic_chunk

OBS = gymnasium.spaces.Box(-1.0, 1.0, (5,), np.float32)
DISCRETE = RoleSpec(OBS, gymnasium.spaces.Discrete(4))
PATTERNS = ["AAATAB", "ATAATP", "AARAAB", "AAAAAB"]


def _chunks(model, noise, seed=0):
    return [synthetic_chunk(model, DISCRETE, p, seed=seed + i, logp_noise=noise) for i, p in enumerate(PATTERNS)]


def _losses(model, chunks, **cfg) -> dict[str, float]:
    algo = APPO(model, AlgorithmConfig(**cfg), ActionSpec.from_space(DISCRETE.action_space), device="cpu")
    with torch.no_grad():
        return {k: float(v) for k, v in algo.compute_loss(chunks).items()}


def test_the_clip_fractions_are_gone():
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    losses = _losses(model, _chunks(model, 0.3))
    assert {"rho_trunc_mass", "c_trunc_mass"} <= set(losses)
    assert not {"rho_clip_frac", "c_clip_frac"} & set(losses)


def test_on_policy_the_mass_is_zero_up_to_float_rounding_even_at_bars_of_one():
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    losses = _losses(model, _chunks(model, 0.0), vtrace_rho_bar=1.0, vtrace_c_bar=1.0)
    assert 0.0 <= losses["rho_trunc_mass"] < 1e-6 and 0.0 <= losses["c_trunc_mass"] < 1e-6


def test_off_policy_the_mass_is_the_truncated_share_of_the_weights():
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    chunks = _chunks(model, 0.5)
    algo = APPO(copy.deepcopy(model), AlgorithmConfig(vtrace_rho_bar=1.0, vtrace_c_bar=0.8),
                ActionSpec.from_space(DISCRETE.action_space), device="cpu")
    lp, _, _ = algo.evaluate_chunks(chunks)
    act = torch.stack([c.kind for c in chunks], dim=1).reshape(-1) == SLOT_ACT
    behavior = torch.stack([c.behavior_logp for c in chunks], dim=1).reshape(-1)
    rho = torch.exp(lp - behavior)[act]
    with torch.no_grad():
        losses = {k: float(v) for k, v in algo.compute_loss(chunks).items()}
    assert losses["rho_trunc_mass"] == pytest.approx(float((rho - 1.0).clamp(min=0).sum() / rho.sum()), rel=1e-4)
    assert losses["c_trunc_mass"] == pytest.approx(float((rho - 0.8).clamp(min=0).sum() / rho.sum()), rel=1e-4)
    assert 0.0 < losses["rho_trunc_mass"] < losses["c_trunc_mass"] < 1.0


def test_with_unit_trace_none_the_mass_is_exactly_zero():
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    losses = _losses(model, _chunks(model, 0.5), unit_trace="none", vtrace_rho_bar=0.5, vtrace_c_bar=0.5)
    assert losses["rho_trunc_mass"] == losses["c_trunc_mass"] == 0.0     # rho = c = 1 and V-trace uses bars of 1


def test_run_level_loss_counters_never_decrease():
    clock = [0.0]
    stats = SystemStats(clock=lambda: clock[0])
    stats.on_worker_stats({"worker_id": 0, "dropped_reward_episodes": 3, "dropped_results": 1})
    stats.on_worker_stats({"worker_id": 1, "dropped_reward_episodes": 4})
    clock[0] = 1.0
    first = stats.snapshot(10, {})
    assert (first["dropped_reward_episodes"], first["dropped_match_results"]) == (7, 1)
    clock[0] = 100.0                                    # every worker fell silent (stalled, or the run is ending)
    later = stats.snapshot(20, {})
    assert later["workers_reporting"] == 0 and later["dropped_reward_episodes"] == 7
    stats.on_worker_stats({"worker_id": 2, "dropped_reward_episodes": 1})   # a new worker adds to the total
    assert stats.snapshot(30, {})["dropped_reward_episodes"] == 8


def test_bc_logs_its_device(caplog):
    role = RoleSpec(OBS, gymnasium.spaces.Discrete(4))
    with caplog.at_level(logging.INFO, logger="colosseum.bc.offline_bc"):
        OfflineBCTrainer(make_test_model(role), ActionSpec.from_space(role.action_space), ObsSpec.from_space(OBS),
                         device="cpu")
    assert any(r.getMessage() == "BC trainer: device cpu" for r in caplog.records)
