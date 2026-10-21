"""Learner start state from hub blobs (SP5 T2.4): a frozen teacher's ``FrozenSpec`` and ``init`` weights become a
``TeacherSpec`` / ``InitState`` from the agent's effective config only (the node reads no file)."""
from __future__ import annotations

import numpy as np

from colosseum.learner.factory import InitState, TeacherSpec, init_from_blob, teacher_from_frozen
from colosseum.players.registry import FrozenSpec
from game_helpers import make_test_config, scripted_agent


def _config(**agent):
    return make_test_config("turns", agents={"a": agent, "random": scripted_agent()},
                            kickstart={"lambda": 0.5, "decay_steps": 100, "kl": "reverse"})


def test_a_frozen_teacher_blob_becomes_a_neural_teacher_with_the_agents_kickstart_settings():
    config = _config(kickstart={"lambda": 0.25})
    frozen = FrozenSpec(agent_id="t", roles=("player",), networks={"model_class": "x"},
                        model_state={"w": np.zeros(2, np.float32)}, source="hub blob teacher-abc")
    teacher = teacher_from_frozen(config, "a", frozen)
    assert isinstance(teacher, TeacherSpec) and teacher.kind == "neural" and teacher.frozen is frozen
    assert (teacher.lambda_, teacher.decay_steps, teacher.kl) == (0.25, 100, "reverse")
    assert teacher.source == "hub blob teacher-abc" and teacher.bot is None


def test_init_weights_from_a_blob_follow_the_agents_strictness():
    state = {"w": np.ones(3, np.float32)}
    strict = init_from_blob(_config(), "a", state, "hub init blob")
    assert isinstance(strict, InitState) and strict.strict and strict.report == []
    assert strict.source == "hub init blob" and strict.model_state == state
    assert not init_from_blob(_config(init={"strict": False}), "a", state, "x").strict
