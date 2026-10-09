"""build_model with role injection, critic encoder, make_env / env_spec (SP2 T2.4)."""
import numpy as np
import pytest
import torch

from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError, EnvContractError
from colosseum.core.registry import build_model, env_spec, import_class, make_env
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.tree import tree_index, tree_stack, tree_to_torch
from colosseum.envs.game import GameSpec, RoleSpec, SeatSpec
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import LSTMCore
from colosseum.networks.model import PolicyModel, act
from game_helpers import GlobalStateGame, RandomPolicy, SoloCounterGame, UnitsGame

UNITS_ROLE = UnitsGame(max_units=3).spec.roles["player"]
GS_ROLE = GlobalStateGame().spec.roles["player"]


def _config(networks=None, env="game_helpers.UnitsGame", env_kwargs=None):
    networks = networks or {"encoder_class": "game_helpers.GenericEncoder",
                            "policy_class": "game_helpers.TreePolicyHead",
                            "value_class": "game_helpers.GenericValue", "kwargs": {"hidden": 8}}
    return ColosseumConfig.model_validate({"env": {"env_class": env, "kwargs": env_kwargs or {}},
                                           "networks": networks})


class InjectedModel(RandomPolicy):
    """Names observation_space and action_spec (not action_space): gets exactly those two."""

    def __init__(self, observation_space, action_spec, width: int = 1) -> None:
        super().__init__(RoleSpec(observation_space, UNITS_ROLE.action_space))
        self.got = {"observation_space": observation_space, "action_spec": action_spec, "width": width}


class SpecFromKwargs(RandomPolicy):
    def __init__(self, action_spec, observation_space=None) -> None:
        super().__init__(UNITS_ROLE)
        self.got = action_spec


class NotAModel:
    def __init__(self, **kwargs) -> None:
        pass


class _CountingGame(SoloCounterGame):
    closed = 0

    def close(self) -> None:
        _CountingGame.closed += 1


class _BadTeams(SoloCounterGame):
    def __init__(self) -> None:
        super().__init__()
        self.spec = GameSpec(roles=self.spec.roles, layouts={"solo": (SeatSpec("player", 1),)})


class _NoSpec(SoloCounterGame):
    def __init__(self) -> None:
        super().__init__()
        self.spec = "solo"


def test_composed_model_gets_the_role_spaces():
    model = build_model(_config(), UNITS_ROLE)
    assert isinstance(model, ComposedModel) and model.critic_encoder is None
    assert model.encoder.obs_spec == ObsSpec.from_space(UNITS_ROLE.observation_space)
    assert model.policy.spec == ActionSpec.from_space(UNITS_ROLE.action_space)
    assert model.encoder.latent_dim == 8
    space = UNITS_ROLE.observation_space
    space.seed(0)
    obs = tree_to_torch(tree_stack([space.sample() for _ in range(2)]))
    out = act(model, obs, None, tree_to_torch(ActionSpec.from_space(UNITS_ROLE.action_space).full_mask((2,))))
    assert out.unit_log_probs.shape == (2, 4)


def test_core_and_in_dim():
    cfg = _config({"encoder_class": "game_helpers.GenericEncoder", "policy_class": "game_helpers.TreePolicyHead",
                   "value_class": "game_helpers.GenericValue", "kwargs": {"hidden": 8},
                   "core": {"class": "colosseum.networks.cores.LSTMCore", "kwargs": {"hidden_size": 12}}})
    model = build_model(cfg, UNITS_ROLE)
    assert isinstance(model.core, LSTMCore) and model.core.output_dim == 12
    assert model.value.net[0].in_features == 12 and model.is_stateful


def test_critic_encoder_widens_the_value_input():
    cfg = _config({"encoder_class": "game_helpers.GenericEncoder", "policy_class": "game_helpers.TreePolicyHead",
                   "value_class": "game_helpers.GenericValue", "kwargs": {"hidden": 8},
                   "critic_encoder_class": "game_helpers.GenericCriticEncoder"}, env="game_helpers.GlobalStateGame")
    model = build_model(cfg, GS_ROLE)
    assert model.critic_encoder is not None and model.value.net[0].in_features == 8 + 8
    out = model.unroll(torch.zeros(3, 2, 2), None, torch.zeros(3, 2, dtype=torch.bool),
                       global_state=torch.zeros(3, 2, 4))
    assert out.value.shape == (6,)


def test_critic_encoder_needs_a_global_state():
    cfg = _config({"encoder_class": "game_helpers.GenericEncoder", "policy_class": "game_helpers.TreePolicyHead",
                   "value_class": "game_helpers.GenericValue",
                   "critic_encoder_class": "game_helpers.GenericCriticEncoder"})
    with pytest.raises(ConfigError, match="declares no global_state_space"):
        build_model(cfg, UNITS_ROLE)
    cfg = _config({"encoder_class": "game_helpers.GenericEncoder", "policy_class": "game_helpers.TreePolicyHead",
                   "value_class": "game_helpers.GenericValue", "critic_encoder_class": "game_helpers.GenericValue"})
    with pytest.raises(ConfigError, match="must subclass colosseum.networks.base.BaseCriticEncoder"):
        build_model(cfg, GS_ROLE)


def test_model_class_injection_by_name_only():
    model = build_model(_config({"model_class": "test_build_model_v2.InjectedModel", "kwargs": {"width": 3}}),
                        UNITS_ROLE)
    assert set(model.got) == {"observation_space", "action_spec", "width"} and model.got["width"] == 3
    assert model.got["action_spec"] == ActionSpec.from_space(UNITS_ROLE.action_space)
    assert model.got["observation_space"] is UNITS_ROLE.observation_space


def test_kwargs_win_over_injection():
    model = build_model(_config({"model_class": "test_build_model_v2.SpecFromKwargs",
                                 "kwargs": {"action_spec": "from-kwargs"}}), UNITS_ROLE)
    assert model.got == "from-kwargs"


def test_model_class_must_be_a_policy_model():
    with pytest.raises(ConfigError, match="must subclass colosseum.networks.model.PolicyModel"):
        build_model(_config({"model_class": "test_build_model_v2.NotAModel"}), UNITS_ROLE)
    assert issubclass(import_class("game_helpers.RandomPolicy"), PolicyModel)


def test_import_class_errors():
    with pytest.raises(ValueError, match="module.ClassName"):
        import_class("RandomPolicy")
    with pytest.raises(TypeError, match="not a class"):
        import_class("game_helpers.make_test_model")


def test_make_env_and_env_spec():
    env = make_env(_config(env="game_helpers.UnitsGame", env_kwargs={"max_units": 2}))
    assert isinstance(env, UnitsGame) and env.U == 2
    _CountingGame.closed = 0
    spec = env_spec(_config(env="test_build_model_v2._CountingGame"))
    assert list(spec.layouts) == ["solo"] and _CountingGame.closed == 1
    with pytest.raises(ConfigError, match="Failed to create env 'game_helpers.UnitsGame': TypeError"):
        make_env(_config(env="game_helpers.UnitsGame", env_kwargs={"bogus": 1}))
    with pytest.raises(ConfigError, match="must subclass colosseum.envs.game.MultiAgentEnv"):
        make_env(_config(env="collections.OrderedDict"))
    with pytest.raises(EnvContractError, match="team numbers must be exactly"):
        env_spec(_config(env="test_build_model_v2._BadTeams"))
    with pytest.raises(ConfigError, match="spec must be a colosseum.envs.game.GameSpec"):
        env_spec(_config(env="test_build_model_v2._NoSpec"))


def test_built_models_step_on_env_observations():
    env = UnitsGame(max_units=3)
    first = env.reset(None, "solo")
    obs = tree_to_torch(tree_stack([first.obs[0]]))
    model = build_model(_config(), UNITS_ROLE)
    mask = tree_to_torch(tree_stack([ActionSpec.from_space(UNITS_ROLE.action_space)
                                     .normalize_mask(first.action_masks[0], "seat 0")]))
    out = act(model, obs, None, mask)
    assert out.unit_log_probs[0, 2:].tolist() == [0.0, 0.0]          # units 1, 2 do not exist yet
    assert np.isfinite(out.log_probs.numpy()).all()
    assert tree_index(obs, 0)["grid"].dtype == torch.uint8
