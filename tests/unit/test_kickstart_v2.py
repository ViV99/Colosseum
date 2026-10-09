"""Kickstart v2: per-decider KL over ACT slots, teacher unrolled without value (T4.2)."""

from __future__ import annotations

import copy
import math

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.networks.state import cat_batch
from colosseum.sp2.algorithms.appo import APPO
from colosseum.sp2.bc.kickstart import KickstartLoss
from colosseum.sp2.core.config import AlgorithmConfig
from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.tree import tree_map, tree_stack
from colosseum.sp2.core.types import SLOT_ACT
from colosseum.sp2.envs.game import RoleSpec
from colosseum.sp2.envs.spaces import Units
from game_helpers import make_test_model, synthetic_chunk

OBS = gymnasium.spaces.Box(-1.0, 1.0, (5,), np.float32)
UNITS = RoleSpec(OBS, Units(4, gymnasium.spaces.Discrete(3)))


def _batch(chunks):
    S, B = chunks[0].num_slots, len(chunks)
    stack = lambda get: tree_stack([get(c) for c in chunks], axis=1)  # noqa: E731
    actions = tree_map(lambda t: t.reshape(S * B, *t.shape[2:]), stack(lambda c: c.actions))
    return dict(obs=stack(lambda c: c.obs), reset_after=stack(lambda c: c.reset_after),
                action_mask=stack(lambda c: c.action_masks), actions=actions,
                is_act=stack(lambda c: c.kind) == SLOT_ACT,
                state0=cat_batch([c.initial_state for c in chunks]))


def _student_dist(model, batch):
    return model.unroll(batch["obs"], batch["state0"], batch["reset_after"], batch["action_mask"],
                        with_value=False).dist


def test_identical_teacher_gives_zero_kl_with_lstm_and_unit_masks():
    torch.manual_seed(0)
    student = make_test_model(UNITS, core="lstm")
    chunks = [synthetic_chunk(student, UNITS, p, seed=i) for i, p in enumerate(["AATP", "AARP"])]
    batch = _batch(chunks)
    loss = KickstartLoss(copy.deepcopy(student)).compute(student_dist=_student_dist(student, batch),
                                                          reduction="sum", **batch)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("reduction", ["sum", "mean_valid"])
def test_kl_is_reduced_over_valid_deciders_then_averaged_over_act_slots(direction, reduction):
    torch.manual_seed(0)
    student, teacher = make_test_model(UNITS), make_test_model(UNITS)
    chunks = [synthetic_chunk(student, UNITS, p, seed=i) for i, p in enumerate(["AATP", "ATAB"])]
    batch = _batch(chunks)
    s_dist = _student_dist(student, batch)
    kick = KickstartLoss(teacher, initial_lambda=0.5, direction=direction)
    loss = kick.compute(student_dist=s_dist, reduction=reduction, **batch)
    with torch.no_grad():
        t_dist = _student_dist(teacher, batch)
    kl = t_dist.unit_kl(s_dist, batch["actions"]) if direction == "forward" else s_dist.unit_kl(
        t_dist, batch["actions"])
    valid = s_dist.unit_valid(batch["actions"])
    per_slot = torch.where(valid, kl, 0.0).sum(-1)
    if reduction == "mean_valid":
        per_slot = per_slot / valid.sum(-1).clamp(min=1)
    act = batch["is_act"].reshape(-1)
    expected = 0.5 * per_slot[act].mean()
    assert float(loss.detach()) == pytest.approx(float(expected.detach()), rel=1e-5)
    assert float(loss.detach()) > 0


def test_teacher_is_unrolled_without_value_and_frozen():
    torch.manual_seed(0)
    student = make_test_model(UNITS)
    teacher = make_test_model(UNITS)
    calls = []
    original = teacher.unroll

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    teacher.unroll = spy
    kick = KickstartLoss(teacher)
    assert all(not p.requires_grad for p in teacher.parameters())
    batch = _batch([synthetic_chunk(student, UNITS, "AATP")])
    kick.compute(student_dist=_student_dist(student, batch), reduction="sum", **batch)
    assert calls and calls[0]["with_value"] is False and "global_state" not in calls[0]


def test_lambda_decays_and_state_round_trips():
    kick = KickstartLoss(make_test_model(UNITS), initial_lambda=1.0, decay_steps=4)
    for _ in range(3):
        kick.step()
    assert kick.current_lambda == pytest.approx(0.25)
    other = KickstartLoss(make_test_model(UNITS), initial_lambda=1.0, decay_steps=4)
    other.load_state_dict(kick.state_dict())
    assert other.current_lambda == pytest.approx(0.25)
    with pytest.raises(ValueError):
        KickstartLoss(make_test_model(UNITS), direction="sideways")


def test_appo_adds_kickstart_with_the_entropy_reduction_and_checks_the_state_layout():
    torch.manual_seed(0)
    student = make_test_model(UNITS, core="gru")
    teacher = make_test_model(UNITS, core="gru")
    spec = ActionSpec.from_space(UNITS.action_space)
    algo = APPO(student, AlgorithmConfig(), spec, kickstart=KickstartLoss(teacher, initial_lambda=0.5))
    chunks = [synthetic_chunk(student, UNITS, "AATP", seed=s) for s in range(2)]
    metrics = algo.train_step(chunks)
    assert metrics["kickstart_loss"] > 0 and metrics["kickstart_lambda"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="state layout"):
        APPO(make_test_model(UNITS, core="lstm"), AlgorithmConfig(), spec,
             kickstart=KickstartLoss(make_test_model(UNITS, core="none")))


@pytest.mark.gpu
@pytest.mark.parametrize("use_amp", [False, True])
def test_kickstart_on_cuda_with_masks_and_an_lstm_core(use_amp):
    torch.manual_seed(0)
    student = make_test_model(UNITS, core="lstm")
    teacher = make_test_model(UNITS, core="lstm")
    chunks = [synthetic_chunk(student, UNITS, p, seed=i) for i, p in enumerate(["AATP", "ATAB", "AARP"])]
    assert all(c.action_masks is not None for c in chunks)
    kick = KickstartLoss(teacher, initial_lambda=1.0, decay_steps=100, direction="forward")
    algo = APPO(student, AlgorithmConfig(num_epochs=1, minibatch_chunks=0, use_amp=use_amp),
                ActionSpec.from_space(UNITS.action_space), device="cuda", kickstart=kick)
    assert algo._use_amp == use_amp
    assert next(teacher.parameters()).device.type == "cuda"
    assert next(algo.model.parameters()).device.type == "cuda"
    metrics = algo.train_step(chunks)
    assert metrics["kickstart_loss"] > 0
    assert metrics["kickstart_lambda"] == pytest.approx(1.0)
    for key in ("total_loss", "policy_loss", "value_loss", "entropy", "kickstart_loss"):
        assert math.isfinite(metrics[key]), key
    assert math.isfinite(metrics["grad_norm"]) or metrics["skipped_updates"] > 0
    assert kick.current_lambda == pytest.approx(0.99)
