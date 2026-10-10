"""Neural kickstart teacher per agent, of any architecture (spec block 6, T4.4)."""
from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.bc.kickstart import KickstartLoss
from colosseum.core.config import AlgorithmConfig, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import env_spec, validate_config
from colosseum.core.specs import ActionSpec
from colosseum.envs.game import RoleSpec
from colosseum.learner.factory import build_algorithm, check_teacher_compat, resolve_teacher
from colosseum.networks.dist import make_distribution
from colosseum.networks.model import UnrollOutput
from game_helpers import (
    SP2_CHECKPOINT,
    SP2_TTT_TINY,
    GameTestModel,
    agent_role_of,
    make_test_config,
    make_test_model,
    synthetic_chunk,
    write_ckpt_dir,
)

ROLE = RoleSpec(gymnasium.spaces.Box(-1.0, 1.0, (5,), np.float32), gymnasium.spaces.Discrete(3))
SPEC = ActionSpec.from_space(ROLE.action_space)


class WrongDistModel(GameTestModel):
    """``unroll`` returns a distribution over 5 actions, whatever the role's action space."""

    def unroll(self, obs, state0, reset_after, action_mask=None, global_state=None, with_value=True):
        out = super().unroll(obs, state0, reset_after, action_mask, global_state=global_state, with_value=with_value)
        five = ActionSpec.from_space(gymnasium.spaces.Discrete(5))
        return UnrollOutput(dist=make_distribution(five, torch.zeros(reset_after.numel(), 5)), value=out.value)


def frozen_teacher_config(tmp_path, *, teacher_networks: dict, student_networks: dict | None = None):
    """turns config whose agent_0 learns from frozen agent 'teacher' (a checkpoint of ``teacher_networks``)."""
    base = make_test_config("turns")
    ckpt = write_ckpt_dir(tmp_path / "teacher", base, networks=teacher_networks, seed=2)
    student = {"kickstart": {"teacher": "teacher"}}
    if student_networks is not None:
        student["networks"] = student_networks
    return make_test_config("turns", agents={"agent_0": student, "teacher": {"kind": "frozen", "path": str(ckpt)}},
                            matchmaking={"anchors": []})


def test_a_stateless_teacher_works_with_a_recurrent_student():
    torch.manual_seed(0)
    student, teacher = make_test_model(ROLE, core="lstm"), make_test_model(ROLE, core="none")
    seen = []
    original = teacher.unroll

    def spy(obs, state0, *args, **kwargs):
        seen.append(state0)
        return original(obs, state0, *args, **kwargs)

    teacher.unroll = spy
    algo = APPO(student, AlgorithmConfig(), SPEC, kickstart=KickstartLoss(teacher, initial_lambda=0.5))
    metrics = algo.train_step([synthetic_chunk(student, ROLE, "AATP", seed=s) for s in range(2)])
    assert metrics["kickstart_loss"] > 0 and seen and all(state0 is None for state0 in seen)


def test_a_recurrent_teacher_with_the_students_layout_gets_the_chunk_state():
    torch.manual_seed(0)
    student, teacher = make_test_model(ROLE, core="gru"), make_test_model(ROLE, core="gru")
    algo = APPO(student, AlgorithmConfig(), SPEC, kickstart=KickstartLoss(teacher))
    assert algo.train_step([synthetic_chunk(student, ROLE, "AATP", seed=1)])["kickstart_loss"] > 0


@pytest.mark.parametrize("student_core, teacher_core", [("lstm", "gru"), ("none", "lstm"), ("gru", "lstm")])
def test_a_recurrent_teacher_needs_the_students_state_layout(student_core, teacher_core):
    student, teacher = make_test_model(ROLE, core=student_core), make_test_model(ROLE, core=teacher_core)
    with pytest.raises(ValueError, match="state layout"):
        APPO(student, AlgorithmConfig(), SPEC, kickstart=KickstartLoss(teacher))
    with pytest.raises(ConfigError, match="agent 'a'.*state layout"):
        check_teacher_compat(student, teacher, "agent 'a'")


def test_a_teacher_of_another_architecture_is_built_from_its_meta(tmp_path):
    cfg = frozen_teacher_config(tmp_path, teacher_networks={"model_class": "game_helpers.GameTestModel",
                                                            "kwargs": {"core": "none", "hidden": 32}})
    validate_config(cfg)
    spec = env_spec(cfg)
    teacher = resolve_teacher(cfg, "agent_0", spec)
    _roles, role = agent_role_of(cfg, "agent_0")
    algo = build_algorithm(cfg.get_agent_config("agent_0"), role, spec, device="cpu", teacher=teacher)
    assert algo._kickstart.teacher.inner.encoder.fc.out_features == 32
    assert algo.model.inner.encoder.fc.out_features == 16
    chunks = [synthetic_chunk(algo.model, role, "AAAB", seed=s, agent_id="agent_0") for s in range(2)]
    assert algo.train_step(chunks)["kickstart_loss"] > 0


def test_teachers_are_per_agent_and_of_their_own_architecture(tmp_path):
    base = make_test_config("turns")
    big = write_ckpt_dir(tmp_path / "big", base, networks={"model_class": "game_helpers.GameTestModel",
                                                           "kwargs": {"core": "none", "hidden": 32}})
    small = write_ckpt_dir(tmp_path / "small", base)
    cfg = make_test_config("turns", agents={"a": {"kickstart": {"teacher": "big"}},
                                            "b": {"kickstart": {"teacher": str(small)}},
                                            "big": {"kind": "frozen", "path": str(big)}},
                           matchmaking={"anchors": []})
    spec = env_spec(cfg)
    hidden = {}
    for agent_id in ("a", "b"):
        algo = build_algorithm(cfg.get_agent_config(agent_id), agent_role_of(cfg, agent_id)[1], spec, device="cpu",
                               teacher=resolve_teacher(cfg, agent_id, spec))
        hidden[agent_id] = algo._kickstart.teacher.inner.encoder.fc.out_features
    assert hidden == {"a": 32, "b": 16}


def test_a_teacher_must_play_every_role_of_its_student(tmp_path):
    base = make_test_config("asymmetric")
    ckpt = write_ckpt_dir(tmp_path / "old_hunter", base, agent_id="hunter")
    cfg = make_test_config("asymmetric", agents={
        "hunter": {"roles": ["hunter"]},
        "prey": {"roles": ["prey"], "kickstart": {"teacher": "old_hunter"}},
        "old_hunter": {"kind": "frozen", "path": str(ckpt)},
    })
    with pytest.raises(ConfigError, match="every role"):
        resolve_teacher(cfg, "prey", env_spec(cfg))
    with pytest.raises(ConfigError, match="every role"):                  # validate rejects it before any process
        validate_config(cfg)


@pytest.mark.parametrize("teacher_networks, student_networks, message", [
    ({"model_class": "game_helpers.GameTestModel", "kwargs": {"core": "gru", "hidden": 16}},
     {"kwargs": {"core": "lstm", "hidden": 16}}, "state layout"),
    ({"model_class": "test_sp3_neural_teacher.WrongDistModel", "kwargs": {"core": "none", "hidden": 16}},
     None, "KL"),
])
def test_validate_rejects_teachers_the_student_cannot_learn_from(tmp_path, teacher_networks, student_networks,
                                                                  message):
    cfg = frozen_teacher_config(tmp_path, teacher_networks=teacher_networks, student_networks=student_networks)
    with pytest.raises(ConfigError, match=message):
        validate_config(cfg)


def test_validate_rejects_a_checkpoint_dir_teacher_with_another_recurrent_state_layout(tmp_path):
    base = make_test_config("turns")
    ckpt = write_ckpt_dir(tmp_path / "gru", base, networks={"model_class": "game_helpers.GameTestModel",
                                                            "kwargs": {"core": "gru", "hidden": 16}})
    cfg = make_test_config("turns", networks={"kwargs": {"core": "lstm"}}, kickstart={"teacher": str(ckpt)})
    with pytest.raises(ConfigError, match="agent 'agent_0'.*state layout"):
        validate_config(cfg)


def test_the_sp2_checkpoint_fixture_works_as_a_kickstart_teacher():
    # SP2-format config (translated); the student is wider than the SP2 teacher (built from its meta.json)
    cfg = load_config(SP2_TTT_TINY, {"kickstart.teacher": str(SP2_CHECKPOINT), "networks.kwargs.hidden": 32})
    validate_config(cfg)
    spec = env_spec(cfg)
    teacher = resolve_teacher(cfg, "agent_0", spec)
    assert teacher is not None and teacher.source == str(SP2_CHECKPOINT)
    _roles, role = agent_role_of(cfg, "agent_0")
    algo = build_algorithm(cfg.get_agent_config("agent_0"), role, spec, device="cpu", teacher=teacher)
    chunks = [synthetic_chunk(algo.model, role, "AAAB", seed=s, agent_id="agent_0") for s in range(2)]
    assert algo.train_step(chunks)["kickstart_loss"] > 0
