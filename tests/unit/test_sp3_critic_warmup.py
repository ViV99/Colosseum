"""Critic warm-up (spec block 6): only the value path learns for the first N train steps, and the
policy the workers receive stays bit-identical, normalizer statistics included (T4.3)."""
from __future__ import annotations

import copy

import gymnasium
import numpy as np
import pytest
import torch
import torch.nn as nn

from colosseum.algorithms.appo import APPO
from colosseum.bc.kickstart import KickstartLoss
from colosseum.core.config import AlgorithmConfig
from colosseum.core.errors import ConfigError
from colosseum.core.registry import env_spec, validate_config
from colosseum.core.specs import ActionSpec
from colosseum.core.types import WeightPayload
from colosseum.envs.game import RoleSpec
from colosseum.learner.factory import build_algorithm
from colosseum.networks.base import BaseCriticEncoder, BaseEncoder, EncoderOutput
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import NoCore
from colosseum.networks.model import PolicyModel
from colosseum.networks.normalization import NormalizeObs
from game_helpers import (
    GameTestModel,
    GenericValue,
    TreePolicyHead,
    agent_role_of,
    make_test_config,
    make_test_model,
    synthetic_chunk,
)

ROLE = RoleSpec(gymnasium.spaces.Box(-5.0, 5.0, (4,), np.float32), gymnasium.spaces.Discrete(3),
                gymnasium.spaces.Box(-5.0, 5.0, (2,), np.float32))
SPEC = ActionSpec.from_space(ROLE.action_space)


class NormEncoder(BaseEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.norm = NormalizeObs((4,))
        self.fc = nn.Linear(4, 16)

    @property
    def latent_dim(self) -> int:
        return 16

    def forward(self, obs):
        return EncoderOutput(torch.relu(self.fc(self.norm(obs))), {})


class NormCritic(BaseCriticEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.norm = NormalizeObs((2,), source="global_state")
        self.fc = nn.Linear(2, 8)

    @property
    def output_dim(self) -> int:
        return 8

    def forward(self, global_state):
        return torch.relu(self.fc(self.norm(global_state)))


class NoValueParams(GameTestModel):
    """A model without value_parameters()."""

    def value_parameters(self):
        return PolicyModel.value_parameters(self)


class OldAPPO(APPO):
    """An algorithm class without the critic_warmup_steps argument."""

    def __init__(self, model, config, action_spec, device="cpu", pin_memory=False, kickstart=None):
        super().__init__(model, config, action_spec, device=device, pin_memory=pin_memory, kickstart=kickstart)


def norm_model(seed: int = 0) -> ComposedModel:
    torch.manual_seed(seed)
    return ComposedModel(NormEncoder(), NoCore(16), TreePolicyHead(16, SPEC), GenericValue(24), NormCritic())


def chunks(model) -> list:
    return [synthetic_chunk(model, ROLE, "AAAAAAAB", seed=s) for s in range(4)]


def value_keys(model) -> set[str]:
    ids = {id(p) for p in model.value_parameters()}
    return {name for name, p in model.named_parameters() if id(p) in ids}


def test_value_parameters_of_the_models():
    model = norm_model()
    assert value_keys(model) == {name for name, _ in model.named_parameters()
                                 if name.startswith(("value.", "critic_encoder."))}
    plain = make_test_model(RoleSpec(ROLE.observation_space, ROLE.action_space))
    assert {id(p) for p in plain.value_parameters()} == {id(p) for p in plain.value.parameters()}
    wrapped = GameTestModel(ROLE.observation_space, ROLE.action_space)
    assert {id(p) for p in wrapped.value_parameters()} == {id(p) for p in wrapped.inner.value.parameters()}
    with pytest.raises(NotImplementedError, match="value_parameters"):
        NoValueParams(ROLE.observation_space, ROLE.action_space).value_parameters()


def test_warmup_trains_only_the_value_path_and_keeps_the_workers_policy_bit_identical():
    model = norm_model()
    teacher = copy.deepcopy(model)
    with torch.no_grad():
        for p in teacher.parameters():
            p.add_(0.3)
    kick = KickstartLoss(teacher, initial_lambda=0.5, decay_steps=10)
    algo = APPO(model, AlgorithmConfig(learning_rate=1e-2), SPEC, kickstart=kick, critic_warmup_steps=3)
    batch = chunks(model)
    keys = value_keys(model)
    before = WeightPayload.from_model("a", 0, model).state_dict           # what the workers receive
    for _ in range(3):
        assert algo.critic_warming_up
        metrics = algo.train_step(batch)
        assert metrics["critic_warmup"] == 1.0 and metrics["kickstart_loss"] == 0.0
        assert "policy_loss" in metrics and "entropy" in metrics           # diagnostics stay
    after = WeightPayload.from_model("a", algo.policy_version, model).state_dict
    assert algo.policy_version == 3 and not algo.critic_warming_up
    for key, value in before.items():
        if key not in keys:                                                # policy weights AND normalizer buffers
            assert np.array_equal(value, after[key]), key
    assert any(not np.array_equal(before[k], after[k]) for k in keys)
    value_ids = {id(p) for p in model.value_parameters()}
    adam = {id(p) for p in algo._optimizer.state}
    assert adam and adam <= value_ids                                     # no Adam state for frozen parameters
    assert kick.step_count == 0 and kick.current_lambda == 0.5            # the decay starts after the warm-up
    metrics = algo.train_step(batch)
    final = WeightPayload.from_model("a", algo.policy_version, model).state_dict
    assert metrics["critic_warmup"] == 0.0 and metrics["kickstart_loss"] > 0.0 and kick.step_count == 1
    assert not np.array_equal(after["encoder.fc.weight"], final["encoder.fc.weight"])
    assert not np.array_equal(after["encoder.norm.rms.count"], final["encoder.norm.rms.count"])


def test_the_lr_schedule_runs_as_usual_during_the_warmup():
    model = norm_model()
    warm = APPO(model, AlgorithmConfig(learning_rate=1e-2, lr_schedule="linear"), SPEC, critic_warmup_steps=5)
    warm.set_progress(0.25)
    assert warm._optimizer.param_groups[0]["lr"] == pytest.approx(0.75e-2)


def test_the_warmup_counter_is_trainer_state():
    batch = chunks(norm_model())
    first = APPO(norm_model(), AlgorithmConfig(), SPEC, critic_warmup_steps=3)
    first.train_step(batch)
    state = first.state_dict()
    assert state["critic_warmup_done"] == 1
    resumed = APPO(norm_model(), AlgorithmConfig(), SPEC, critic_warmup_steps=3)
    resumed.load_state_dict(state)
    resumed.train_step(batch)
    assert resumed.critic_warming_up
    resumed.train_step(batch)
    assert not resumed.critic_warming_up
    del state["critic_warmup_done"]                                       # a trainer state written before SP3
    older = APPO(norm_model(), AlgorithmConfig(), SPEC, critic_warmup_steps=3)
    older.load_state_dict(state)
    assert older.critic_warming_up and older.state_dict()["critic_warmup_done"] == 0


def test_appo_rejects_a_warmup_it_cannot_do():
    with pytest.raises(ValueError, match="value_parameters"):
        APPO(NoValueParams(ROLE.observation_space, ROLE.action_space), AlgorithmConfig(), SPEC, critic_warmup_steps=1)
    with pytest.raises(ValueError, match="value_loss_coeff"):
        APPO(norm_model(), AlgorithmConfig(value_loss_coeff=0.0), SPEC, critic_warmup_steps=1)
    assert not APPO(norm_model(), AlgorithmConfig(), SPEC).critic_warming_up


def test_build_algorithm_passes_the_agents_warmup():
    cfg = make_test_config("turns", agents={"a": {"init": {"critic_warmup_steps": 2}}, "b": {}})
    spec = env_spec(cfg)
    for agent_id, warming in (("a", True), ("b", False)):
        algo = build_algorithm(cfg.get_agent_config(agent_id), agent_role_of(cfg, agent_id)[1], spec, device="cpu",
                               teacher=None)
        assert algo.critic_warming_up is warming


@pytest.mark.parametrize("sections, message", [
    ({"networks": {"model_class": "test_sp3_critic_warmup.NoValueParams"}}, "value_parameters"),
    ({"algorithm": {"algorithm_class": "test_sp3_critic_warmup.OldAPPO"}}, "takes no critic_warmup_steps"),
    ({"algorithm": {"value_loss_coeff": 0.0}}, "value_loss_coeff"),
])
def test_validate_rejects_a_warmup_the_agent_cannot_do(sections, message):
    with pytest.raises(ConfigError, match=message):
        validate_config(make_test_config("turns", init={"critic_warmup_steps": 3}, **sections))
    validate_config(make_test_config("turns", **sections))                # fine without a warm-up
