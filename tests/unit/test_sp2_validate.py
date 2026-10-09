"""validate_config (spec block 9) and `python -m colosseum validate` (T6.2)."""
from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch
import yaml
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.errors import ConfigError, EnvContractError
from colosseum.core.registry import validate_config
from colosseum.core.validation import random_legal_action
from colosseum.envs.game import GameSpec, MultiAgentEnv, RoleSpec, StepResult
from colosseum.networks.model import UnrollOutput
from game_helpers import CORE_KINDS, GameTestModel, make_test_config

BOX = gymnasium.spaces.Box(0.0, 1.0, (3,), np.float32)
TWO = gymnasium.spaces.Discrete(2)


class BoundedSolo(MultiAgentEnv):
    """Solo, 3 steps, Box(0, 1) observations; ``bad`` breaks one rule at step 2."""

    spec = GameSpec.solo(BOX, TWO)

    def __init__(self, bad: str = "") -> None:
        self.bad = bad
        self.t = 0

    def _obs(self) -> np.ndarray:
        obs = np.full(3, 0.5, np.float32)
        if self.bad == "out_of_bounds" and self.t == 2:
            obs[0] = 5.0
        return obs

    def reset(self, seed, layout):
        if self.bad == "reset":
            raise RuntimeError("cannot reset")
        self.t = 0
        return StepResult(acting={0}, obs={0: self._obs()}, action_masks={0: np.array([True, True])})

    def step(self, actions):
        self.t += 1
        if self.t >= 3:
            return StepResult(acting=set(), obs={}, rewards={0: 1.0}, episode_over=True)
        return StepResult(acting={0}, obs={0: self._obs()}, rewards={0: 1.0},
                          action_masks={0: np.array([True, True])})


class TextObsSolo(BoundedSolo):
    """A role with an observation space colosseum does not support."""

    spec = GameSpec.solo(gymnasium.spaces.Text(5), TWO)


class ValueAsMatrixModel(GameTestModel):
    """``unroll`` returns values shaped [S, B] instead of [S*B]."""

    def unroll(self, obs, state0, reset_after, action_mask=None, global_state=None, with_value=True):
        out = super().unroll(obs, state0, reset_after, action_mask, global_state, with_value)
        if out.value is None:
            return out
        return UnrollOutput(out.dist, out.value.reshape(reset_after.shape))


class AlwaysValueModel(GameTestModel):
    """``unroll(with_value=False)`` still returns values."""

    def unroll(self, obs, state0, reset_after, action_mask=None, global_state=None, with_value=True):
        return super().unroll(obs, state0, reset_after, action_mask, global_state, True)


class LayerFirstStateModel(GameTestModel):
    """State laid out [layers, B, H]: the batch dimension is not first."""

    def initial_state(self, batch_size, device="cpu"):
        return torch.zeros(2, batch_size, 4, device=device)


def solo_config(**sections):
    data = {"env": {"env_class": "test_sp2_validate.BoundedSolo"},
            "networks": {"model_class": "game_helpers.GameTestModel"}}
    for key, value in sections.items():
        data[key] = {**data.get(key, {}), **value}
    from colosseum.core.config import ColosseumConfig

    return ColosseumConfig.model_validate(data)


@pytest.mark.parametrize("game", ["solo", "turns", "simultaneous", "ffa", "dead_teammate", "units", "asymmetric",
                                  "coop", "global_state"])
def test_toy_game_configs_are_valid(game):
    validate_config(make_test_config(game))


@pytest.mark.parametrize("core", CORE_KINDS)
def test_stateful_cores_pass_the_model_checks(core):
    validate_config(make_test_config("units", networks={"kwargs": {"core": core}}))


def test_a_bounded_env_passes():
    validate_config(solo_config())


def test_observation_outside_the_space_is_an_env_contract_error():
    # The global context order "seat, episode step, layout", as in the tracker's own errors.
    with pytest.raises(EnvContractError, match=r"^validate, seat 0, episode step 2, layout solo: observation is not "
                                               r"in the role's space"):
        validate_config(solo_config(env={"kwargs": {"bad": "out_of_bounds"}}))


def test_reset_failure_is_a_config_error():
    with pytest.raises(ConfigError, match="env.reset"):
        validate_config(solo_config(env={"kwargs": {"bad": "reset"}}))


def test_unsupported_space_is_a_config_error():
    with pytest.raises(ConfigError, match="unsupported observation space Text"):
        validate_config(solo_config(env={"env_class": "test_sp2_validate.TextObsSolo"}))


def test_matchmaking_and_roles_are_checked():
    with pytest.raises(ConfigError, match="unknown layouts"):
        validate_config(make_test_config("turns", matchmaking={"layouts": {"9p": 1.0}}))
    with pytest.raises(ConfigError, match="prey"):
        validate_config(make_test_config("asymmetric", agents={"hunter": {"roles": ["hunter"]}}))


def test_critic_encoder_needs_a_global_state():
    cfg = solo_config(networks={"model_class": None, "encoder_class": "game_helpers.Nope",
                                "policy_class": "game_helpers.Nope", "value_class": "game_helpers.Nope",
                                "critic_encoder_class": "game_helpers.Nope"})
    with pytest.raises(ConfigError, match="critic_encoder_class"):
        validate_config(cfg)


@pytest.mark.parametrize(("model", "message"), [
    ("ValueAsMatrixModel", "time-major values"),
    ("AlwaysValueModel", "with_value=False"),
    ("LayerFirstStateModel", "batch dimension first"),
])
def test_model_protocol_violations(model, message):
    with pytest.raises(ConfigError, match=message):
        validate_config(solo_config(networks={"model_class": f"test_sp2_validate.{model}"}))


def test_kickstart_teacher_must_fit_every_agent(tmp_path):
    teacher = tmp_path / "teacher.pt"
    torch.save(GameTestModel(BOX, TWO).state_dict(), teacher)
    validate_config(solo_config(training={"kickstart_teacher": str(teacher)}))
    wrong = tmp_path / "wrong.pt"
    torch.save(torch.nn.Linear(2, 2).state_dict(), wrong)
    with pytest.raises(ConfigError, match="do not match"):
        validate_config(solo_config(training={"kickstart_teacher": str(wrong)}))


def test_one_global_teacher_cannot_serve_roles_with_different_spaces(tmp_path):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"never read")
    with pytest.raises(ConfigError, match="kickstart_teacher"):
        validate_config(make_test_config("asymmetric", training={"kickstart_teacher": str(teacher)}))


def test_random_legal_action_respects_masks():
    rng = np.random.default_rng(0)
    role = RoleSpec(BOX, gymnasium.spaces.Discrete(4))
    assert {int(random_legal_action(role, np.array([False, True, False, False]), rng)) for _ in range(20)} == {1}
    multi = RoleSpec(BOX, gymnasium.spaces.MultiDiscrete([2, 3]))
    mask = np.array([False, True, True, False, False])
    assert {tuple(random_legal_action(multi, mask, rng).tolist()) for _ in range(20)} == {(1, 0)}
    mixed = RoleSpec(BOX, gymnasium.spaces.Dict({"move": gymnasium.spaces.Discrete(3),
                                                 "aim": gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32)}))
    action = random_legal_action(mixed, {"move": np.array([False, False, True])}, rng)
    assert int(action["move"]) == 2 and action["aim"].shape == (2,) and mixed.action_space.contains(action)


def test_cli_validate_reports_ok_and_one_line_errors(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(make_test_config("asymmetric").model_dump(mode="json", by_alias=True)))
    result = CliRunner().invoke(main, ["validate", "-c", str(good)])
    assert result.exit_code == 0, result.output
    assert "OK: agent 'hunter'" in result.output and "Config is valid." in result.output
    result = CliRunner().invoke(main, ["validate", "-c", str(good), "--set", "rollout.num_worker=3"])
    assert result.exit_code == 1 and result.stderr.startswith("Config error:") and "num_worker" in result.stderr
    result = CliRunner().invoke(main, ["validate", "-c", str(good), "--set", "matchmaking.layouts={9p: 1}"])
    assert result.exit_code == 1 and "unknown layouts" in result.stderr
