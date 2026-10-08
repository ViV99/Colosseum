"""Tests for multi-agent support: per-agent config overrides and multi-agent configs."""

from colosseum.core.config import (
    AgentConfig,
    ColosseumConfig,
    load_config,
)
from helpers import example_config

# ---------------------------------------------------------------
# T3.8: Per-agent config tests
# ---------------------------------------------------------------

def test_agent_config_defaults():
    """AgentConfig fields should default to None."""
    ac = AgentConfig()
    assert ac.networks is None
    assert ac.algorithm is None
    assert ac.learner is None


def test_get_trainable_agent_ids_empty():
    """Empty agents dict should return ['agent_0']."""
    cfg = load_config(example_config("tic_tac_toe.yaml"))
    assert cfg.get_trainable_agent_ids() == ["agent_0"]


def test_get_trainable_agent_ids_multi():
    """Non-empty agents dict should return agent IDs."""
    cfg = load_config(example_config("tic_tac_toe_multi.yaml"))
    ids = cfg.get_trainable_agent_ids()
    assert "agent_alpha" in ids
    assert "agent_beta" in ids
    assert len(ids) == 2


def test_get_agent_config_no_override():
    """get_agent_config for unknown agent returns a copy with same values."""
    cfg = load_config(example_config("tic_tac_toe.yaml"))
    result = cfg.get_agent_config("agent_0")
    assert result is not cfg  # should be a copy, not the same object
    assert result.algorithm == cfg.algorithm
    assert result.networks == cfg.networks


def test_get_agent_config_with_override():
    """get_agent_config merges per-agent overrides into a new config."""
    cfg = load_config(example_config("tic_tac_toe_multi.yaml"))

    # Modify the config to have an actual override for agent_alpha
    cfg_dict = cfg.model_dump()
    cfg_dict["agents"]["agent_alpha"]["algorithm"] = {
        "name": "appo",
        "learning_rate": 1e-2,
    }
    cfg = ColosseumConfig.model_validate(cfg_dict)

    alpha_cfg = cfg.get_agent_config("agent_alpha")
    beta_cfg = cfg.get_agent_config("agent_beta")

    # alpha should have overridden LR
    assert alpha_cfg.algorithm.learning_rate == 1e-2
    # beta should keep global LR
    assert beta_cfg.algorithm.learning_rate == cfg.algorithm.learning_rate
    # Both should share the same env config
    assert alpha_cfg.env.env_class == cfg.env.env_class


def test_load_multi_agent_config_roundtrip():
    """Multi-agent config should survive dump/reload."""
    cfg = load_config(example_config("tic_tac_toe_multi.yaml"))
    data = cfg.model_dump()
    cfg2 = ColosseumConfig.model_validate(data)
    assert cfg2.get_trainable_agent_ids() == cfg.get_trainable_agent_ids()
    assert cfg2.training.phase.value == "league"
