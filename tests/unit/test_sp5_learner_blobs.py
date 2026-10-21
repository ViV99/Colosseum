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


def test_the_node_side_start_state_helpers_cover_every_teacher_kind_and_init():
    from colosseum.net.chunks import WeightsMeta
    from colosseum.node.learner_main import init_from_cached, teacher_for
    from colosseum.node.weights import CachedWeights, blob_key
    from colosseum.players.registry import BotSpec

    config = _config()
    state = {"w": np.ones(2, np.float32)}
    teacher_blob = CachedWeights(blob_key("teacher", "a", "t1"), state, WeightsMeta(
        agent_id="t", kind="teacher", generation=0, policy_version=0, architecture={"model_class": "m"},
        roles=("player",)), 8)
    neural = teacher_for(config, "a", cached=teacher_blob)
    assert neural.kind == "neural" and neural.frozen.roles == ("player",) and neural.frozen.networks == {
        "model_class": "m"} and neural.source == "hub blob teacher-t1" and neural.lambda_ == 0.5
    assert neural.frozen.model_state["w"].flags.writeable                  # a copy, not the cache's view
    scripted = teacher_for(config, "a", bot=BotSpec("colosseum.players.RandomBot"))
    assert scripted.kind == "scripted" and scripted.bot.class_path == "colosseum.players.RandomBot"
    assert (scripted.lambda_, scripted.decay_steps, scripted.kl) == (0.5, 100, "reverse")
    assert teacher_for(config, "a") is None
    init_blob = CachedWeights(blob_key("init", "a", "i1"), state, WeightsMeta(
        agent_id="a", kind="init", generation=0, policy_version=0), 8)
    init = init_from_cached(config, "a", init_blob)
    assert init.strict and init.source == "hub blob init-i1" and np.array_equal(init.model_state["w"], state["w"])
