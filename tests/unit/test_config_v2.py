"""Config v2: matchmaking, checkpoint keys, roles, unit modes, removed keys, chunk_length >= 2 (SP2 T1.7)."""
import pytest
import yaml
from pydantic import ValidationError

from colosseum.core.config import (
    AlgorithmConfig,
    CheckpointConfig,
    ColosseumConfig,
    EnvConfig,
    MatchmakingConfig,
    NetworkConfig,
    RolloutConfig,
    TrainingConfig,
    load_config,
    parse_override_value,
)
from colosseum.core.errors import ConfigError

BASE = {
    "env": {"env_class": "my_game.game.MyGame"},
    "networks": {"encoder_class": "my_game.models.Encoder", "policy_class": "my_game.models.Policy",
                 "value_class": "my_game.models.Value"},
}


def _write(tmp_path, data):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_defaults():
    cfg = ColosseumConfig.model_validate(BASE)
    assert cfg.algorithm.algorithm_class == "colosseum.algorithms.appo.APPO"
    assert (cfg.algorithm.ratio_mode, cfg.algorithm.unit_trace, cfg.algorithm.entropy_reduction) == \
        ("auto", "auto", "auto")
    assert cfg.env.max_idle_steps == 1000 and cfg.env.kwargs == {}
    assert cfg.networks.critic_encoder_class is None
    m = cfg.matchmaking
    assert (m.mode, m.layouts, m.self_play_ratio, m.pfsp_exponent, m.latest_prob) == ("self_play", {}, 0.5, 1.0, 0.5)
    assert (m.teammates, m.teammate_self_prob, m.shuffle_seats) == ("self", 0.5, True)
    assert (cfg.checkpoint.interval, cfg.checkpoint.pool_size, cfg.checkpoint.save_optimizer) == (1000, 20, True)
    assert cfg.rollout.chunk_length == 256
    assert cfg.get_trainable_agent_ids() == ["agent_0"] and cfg.agent_roles("agent_0") is None


@pytest.mark.parametrize("section, body, key", [
    ("env", {"env_class": "x.Y", "num_players": 2}, "num_players"),
    ("training", {"phase": "self_play"}, "phase"),
    ("self_play", {"pool_size": 5}, "self_play"),
])
def test_removed_keys_are_rejected(tmp_path, section, body, key):
    data = {**BASE, section: body}
    with pytest.raises(ConfigError, match=key):
        load_config(_write(tmp_path, data))


def test_chunk_length_needs_two_slots():
    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        RolloutConfig(chunk_length=1)
    assert RolloutConfig(chunk_length=2).chunk_length == 2


@pytest.mark.parametrize("kwargs", [
    {"mode": "arena"}, {"layouts": {"2p": 0.0}}, {"layouts": {"2p": -1.0}}, {"self_play_ratio": 1.5},
    {"latest_prob": -0.1}, {"teammates": "random"}, {"teammate_self_prob": 2.0}, {"pfsp_exponent": -1.0},
    {"layouts": {"2p": float("nan")}},
])
def test_matchmaking_bounds(kwargs):
    with pytest.raises(ValidationError):
        MatchmakingConfig(**kwargs)


def test_matchmaking_layout_weights():
    m = MatchmakingConfig(mode="league", layouts={"2p": 0.25, "4p": 0.75}, teammates="mixed")
    assert m.layouts == {"2p": 0.25, "4p": 0.75} and m.mode == "league"


def test_checkpoint_bounds():
    with pytest.raises(ValidationError):
        CheckpointConfig(interval=0)
    with pytest.raises(ValidationError):
        CheckpointConfig(pool_size=0)


@pytest.mark.parametrize("field, value", [
    ("ratio_mode", "mean"), ("unit_trace", "sum"), ("entropy_reduction", "mean"),
])
def test_unit_mode_literals(field, value):
    with pytest.raises(ValidationError):
        AlgorithmConfig(**{field: value})
    assert AlgorithmConfig(ratio_mode="per_unit", unit_trace="geo_mean", entropy_reduction="sum").ratio_mode == \
        "per_unit"


def test_env_max_idle_steps_bounds():
    with pytest.raises(ValidationError):
        EnvConfig(env_class="x.Y", max_idle_steps=0)


def test_critic_encoder_is_for_composed_models_only():
    net = NetworkConfig(**BASE["networks"], critic_encoder_class="my_game.models.Critic")
    assert net.critic_encoder_class == "my_game.models.Critic"
    with pytest.raises(ValidationError, match="critic_encoder_class must be omitted"):
        NetworkConfig(model_class="x.Model", critic_encoder_class="x.Critic")


def test_agent_roles():
    cfg = ColosseumConfig.model_validate({**BASE, "agents": {
        "hunter": {"roles": ["hunter"], "algorithm": {"learning_rate": 1e-3}}, "prey": {"roles": ["prey"]},
        "free": None}})
    assert cfg.agent_roles("hunter") == ["hunter"] and cfg.agent_roles("prey") == ["prey"]
    assert cfg.agent_roles("free") is None
    hunter = cfg.get_agent_config("hunter")
    assert hunter.algorithm.learning_rate == 1e-3 and hunter.agents == {}
    with pytest.raises(ConfigError, match="Unknown agent 'ghost'"):
        cfg.agent_roles("ghost")
    with pytest.raises(ConfigError, match="only agent is 'agent_0'"):
        ColosseumConfig.model_validate(BASE).agent_roles("alpha")


@pytest.mark.parametrize("roles, message", [([], "must not be empty"), (["a", "b", "a"], r"\['a'\] more than once")])
def test_bad_agent_roles(roles, message):
    with pytest.raises(ValidationError, match=message):
        ColosseumConfig.model_validate({**BASE, "agents": {"x": {"roles": roles}}})


def test_overrides_reach_the_new_keys(tmp_path):
    overrides = {
        "matchmaking.layouts": parse_override_value("{2p: 1.0}"),
        "matchmaking.layouts.4p": parse_override_value("3"),
        "matchmaking.mode": "league",
        "checkpoint.interval": parse_override_value("50"),
        "env.max_idle_steps": parse_override_value("5"),
        "agents.hunter.roles": parse_override_value("[hunter]"),
    }
    cfg = load_config(_write(tmp_path, BASE), overrides)
    assert cfg.matchmaking.layouts == {"2p": 1.0, "4p": 3.0} and cfg.matchmaking.mode == "league"
    assert cfg.checkpoint.interval == 50 and cfg.env.max_idle_steps == 5
    assert cfg.agent_roles("hunter") == ["hunter"]
    with pytest.raises(ConfigError, match="Unknown config key 'self_play'"):
        load_config(_write(tmp_path, BASE), {"self_play.pool_size": 3})


def test_training_has_no_phase():
    assert "phase" not in TrainingConfig.model_fields
