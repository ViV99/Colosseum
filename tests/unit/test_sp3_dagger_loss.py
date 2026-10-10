"""The DAgger loss (spec block 6): lambda * mean(-log pi(a_teacher)) over labeled ACT slots, joint for one
decider and mean over valid deciders with Units; unlabeled chunks add nothing; the teacher flag; a
student learns the teacher's action (T4.5)."""
from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.bc.kickstart import KickstartLoss
from colosseum.bc.offline_bc import per_sample_nll
from colosseum.core.config import AlgorithmConfig
from colosseum.core.specs import ActionSpec
from colosseum.core.tree import tree_map, tree_stack
from colosseum.core.types import SLOT_ACT, TrajectoryChunk
from colosseum.envs.game import RoleSpec
from colosseum.envs.spaces import Units
from colosseum.learner.learner import _push_weights
from colosseum.networks.state import cat_batch
from game_helpers import NumpyOnlyQueue, chunk_v2_payload, learner_role, make_test_model, synthetic_chunk

OBS = gymnasium.spaces.Box(-1.0, 1.0, (5,), np.float32)
UNITS = RoleSpec(OBS, Units(4, gymnasium.spaces.Discrete(3)))
ROLE3 = learner_role(3)


def labeled(chunk: TrajectoryChunk, label_tree, every: int = 1) -> TrajectoryChunk:
    """``chunk`` with the teacher action ``label_tree`` (one row) on every ``every``-th ACT slot."""
    S = chunk.num_slots
    is_act = chunk.kind == SLOT_ACT
    picked = is_act & (torch.arange(S) % every == 0)
    rows = tree_map(lambda leaf: torch.as_tensor(leaf).expand(S, *torch.as_tensor(leaf).shape).clone(), label_tree)
    zeros = tree_map(torch.zeros_like, chunk.actions)
    actions = tree_map(lambda t, z: torch.where(picked.reshape(-1, *([1] * (t.dim() - 1))), t.to(z.dtype), z),
                       rows, zeros)
    payload = chunk.to_payload()
    payload["teacher_action"] = tree_map(lambda t: t.numpy(), actions)
    payload["has_teacher"] = picked.numpy()
    return TrajectoryChunk.from_payload(payload)


def student_view(model, chunks):
    S, B = chunks[0].num_slots, len(chunks)
    stack = lambda get: tree_stack([get(c) for c in chunks], axis=1)   # noqa: E731
    obs, reset, masks = stack(lambda c: c.obs), stack(lambda c: c.reset_after), stack(lambda c: c.action_masks)
    with torch.no_grad():                              # values only: no autograd graph in these checks
        dist = model.unroll(obs, cat_batch([c.initial_state for c in chunks]), reset.bool(), masks,
                            with_value=False).dist
    flat = lambda tree: tree_map(lambda t: t.reshape(S * B, *t.shape[2:]), tree)   # noqa: E731
    is_act = stack(lambda c: c.kind) == SLOT_ACT
    return dist, flat(stack(lambda c: c.teacher_action)), stack(lambda c: c.has_teacher), is_act


def test_compute_labels_is_the_joint_nll_over_labeled_act_slots():
    torch.manual_seed(0)
    model = make_test_model(ROLE3)
    chunks = [labeled(synthetic_chunk(model, ROLE3, p, seed=i), np.int64(2), every=2)
              for i, p in enumerate(["AAAATP", "AAAAAB"])]
    dist, actions, has, is_act = student_view(model, chunks)
    kick = KickstartLoss(None, initial_lambda=0.5)
    loss = kick.compute_labels(student_dist=dist, teacher_actions=actions, has_teacher=has, is_act=is_act,
                               reduction="sum")
    rows = (has & is_act).reshape(-1)
    expected = 0.5 * (-dist.log_prob(actions))[rows].mean()
    assert float(loss) == pytest.approx(float(expected), rel=1e-5) and float(loss) > 0


def test_with_units_it_is_the_mean_nll_over_valid_deciders_as_in_bc():
    torch.manual_seed(0)
    model = make_test_model(UNITS)
    label = np.array([1, 2, 0, 1], np.int64)
    chunks = [labeled(synthetic_chunk(model, UNITS, "AATP", seed=i, random_units=False), label) for i in range(2)]
    dist, actions, has, is_act = student_view(model, chunks)
    loss = KickstartLoss(None).compute_labels(student_dist=dist, teacher_actions=actions, has_teacher=has,
                                              is_act=is_act, reduction="mean_valid")
    nll, _valid = per_sample_nll(dist, actions, 4)
    rows = (has & is_act).reshape(-1)
    assert float(loss) == pytest.approx(float(nll[rows].mean()), rel=1e-5)


def test_no_label_gives_zero_and_lambda_zero_gives_zero():
    torch.manual_seed(0)
    model = make_test_model(ROLE3)
    chunk = labeled(synthetic_chunk(model, ROLE3, "AAAB"), np.int64(2), every=100)   # only slot 0 labeled
    dist, actions, has, is_act = student_view(model, [chunk])
    none = torch.zeros_like(has)
    assert float(KickstartLoss(None).compute_labels(student_dist=dist, teacher_actions=actions, has_teacher=none,
                                                    is_act=is_act, reduction="sum")) == 0.0
    spent = KickstartLoss(None, initial_lambda=1.0, decay_steps=1)
    spent.step()
    assert float(spent.compute_labels(student_dist=dist, teacher_actions=actions, has_teacher=has, is_act=is_act,
                                      reduction="sum")) == 0.0


def test_chunks_without_labels_add_nothing_and_stay_out_of_the_denominator():
    torch.manual_seed(0)
    model = make_test_model(ROLE3)
    with_labels = labeled(synthetic_chunk(model, ROLE3, "AAAAAB", seed=1), np.int64(2))
    without = synthetic_chunk(model, ROLE3, "AAAAAB", seed=2)                           # e.g. a parked buffer
    algo = APPO(model, AlgorithmConfig(), ActionSpec.from_space(ROLE3.action_space),
                kickstart=KickstartLoss(None, initial_lambda=1.0))
    alone = algo.compute_loss([with_labels])
    mixed = algo.compute_loss([with_labels, without])
    assert float(mixed["kickstart_loss"]) == pytest.approx(float(alone["kickstart_loss"]), rel=1e-5)
    assert float(mixed["kickstart_label_frac"]) == pytest.approx(0.5)


def test_appo_reports_labels_and_switches_the_teacher_flag_off_at_lambda_zero():
    torch.manual_seed(0)
    model = make_test_model(ROLE3)
    chunks = [labeled(synthetic_chunk(model, ROLE3, "AAAAAB", seed=s), np.int64(2)) for s in range(2)]
    algo = APPO(model, AlgorithmConfig(), ActionSpec.from_space(ROLE3.action_space),
                kickstart=KickstartLoss(None, initial_lambda=0.5, decay_steps=2))
    queue = NumpyOnlyQueue(maxsize=1)
    _push_weights(algo, "a", [queue])
    assert algo.teacher_active and queue.get_nowait().teacher_active is True
    metrics = algo.train_step(chunks)
    assert metrics["kickstart_loss"] > 0 and metrics["kickstart_label_frac"] == pytest.approx(1.0)
    algo.train_step(chunks)                                              # lambda decays to 0
    assert not algo.teacher_active
    _push_weights(algo, "a", [queue])
    assert queue.get_nowait().teacher_active is False
    plain = APPO(make_test_model(ROLE3), AlgorithmConfig(), ActionSpec.from_space(ROLE3.action_space))
    assert not plain.teacher_active and "kickstart_label_frac" not in plain.train_step(chunks)


def test_a_student_learns_the_teachers_action():
    torch.manual_seed(0)
    model = make_test_model(ROLE3)
    chunks = []
    for s in range(4):
        payload = chunk_v2_payload(pattern="A" * 15 + "B", version=s)
        payload["reward"][:] = 0.0
        chunks.append(labeled(TrajectoryChunk.from_payload(payload), np.int64(2)))
    algo = APPO(model, AlgorithmConfig(learning_rate=1e-2, lr_schedule="constant", entropy_coeff=0.0),
                ActionSpec.from_space(ROLE3.action_space), kickstart=KickstartLoss(None, initial_lambda=5.0,
                                                                                 decay_steps=10**6))
    for _ in range(60):
        algo.train_step(chunks)
    with torch.no_grad():
        obs = torch.randn(64, 4)
        probs = model.step(obs, None, torch.ones(64, 3, dtype=torch.bool)).dist.log_prob(torch.full((64,), 2)).exp()
    assert float(probs.mean()) > 0.9
