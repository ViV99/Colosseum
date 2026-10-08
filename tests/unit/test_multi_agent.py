"""Tests for multi-agent support: per-agent config overrides and multi-agent configs."""

from colosseum.core.config import (
    AgentOverride,
    ColosseumConfig,
    load_config,
)
from helpers import example_config

# ---------------------------------------------------------------
# T3.8: Per-agent config tests
# ---------------------------------------------------------------

def test_agent_config_defaults():
    """AgentOverride fields should default to None."""
    ac = AgentOverride()
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
    """get_agent_config("agent_0") without an agents section returns a copy with same values."""
    cfg = load_config(example_config("tic_tac_toe.yaml"))
    result = cfg.get_agent_config("agent_0")
    assert result is not cfg  # should be a copy, not the same object
    assert result.algorithm == cfg.algorithm
    assert result.networks == cfg.networks


def test_load_multi_agent_config_roundtrip():
    """Multi-agent config should survive dump/reload."""
    cfg = load_config(example_config("tic_tac_toe_multi.yaml"))
    data = cfg.model_dump()
    cfg2 = ColosseumConfig.model_validate(data)
    assert cfg2.get_trainable_agent_ids() == cfg.get_trainable_agent_ids()
    assert cfg2.training.phase.value == "league"
