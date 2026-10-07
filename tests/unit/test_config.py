"""Tests for config validation (Pydantic models)."""

from helpers import example_config
import os
import sys


import pytest
from pydantic import ValidationError

from colosseum.core.config import (
    AlgorithmConfig,
    ColosseumConfig,
    EnvConfig,
    LearnerConfig,
    NetworkConfig,
    RolloutConfig,
    TrainingConfig,
    load_config,
)


def test_invalid_gamma_too_high():
    """gamma > 1 should raise ValidationError."""
    with pytest.raises(ValidationError):
        AlgorithmConfig(gamma=1.5)


def test_invalid_gamma_negative():
    """gamma < 0 should raise ValidationError."""
    with pytest.raises(ValidationError):
        AlgorithmConfig(gamma=-0.1)


def test_invalid_learning_rate():
    """lr <= 0 should raise ValidationError."""
    with pytest.raises(ValidationError):
        AlgorithmConfig(learning_rate=0.0)
    with pytest.raises(ValidationError):
        AlgorithmConfig(learning_rate=-1e-3)


def test_invalid_num_workers():
    """num_workers < 1 should raise ValidationError."""
    with pytest.raises(ValidationError):
        RolloutConfig(num_workers=0)


def test_invalid_num_epochs():
    """num_epochs < 1 should raise ValidationError."""
    with pytest.raises(ValidationError):
        AlgorithmConfig(num_epochs=0)


def test_invalid_eps_clip():
    """eps_clip <= 0 should raise ValidationError."""
    with pytest.raises(ValidationError):
        AlgorithmConfig(eps_clip=0.0)


def test_valid_defaults():
    """Default config should be valid."""
    algo = AlgorithmConfig()
    assert algo.gamma == 0.99
    assert algo.learning_rate == 3e-4
    assert algo.normalize_advantages is True


def test_load_config_roundtrip():
    """load_config from the example YAML should produce a valid config."""
    cfg = load_config(example_config("tic_tac_toe.yaml"))
    assert cfg.env.num_players == 2
    assert cfg.algorithm.name == "appo"
    # Dump and reload should preserve values
    data = cfg.model_dump()
    cfg2 = ColosseumConfig(**data)
    assert cfg2.env.env_class == cfg.env.env_class
    assert cfg2.algorithm.gamma == cfg.algorithm.gamma


def test_missing_env_class():
    """EnvConfig without env_class should raise ValidationError."""
    with pytest.raises(ValidationError):
        EnvConfig()  # env_class is required (no default)


def test_missing_network_classes():
    """NetworkConfig without required fields should raise ValidationError."""
    with pytest.raises(ValidationError):
        NetworkConfig()  # encoder_class, policy_class, value_class are required


def test_training_config_seed():
    """seed field should accept None or int."""
    tc = TrainingConfig()
    assert tc.seed is None
    tc2 = TrainingConfig(seed=42)
    assert tc2.seed == 42


def test_learner_config_weight_push_interval():
    """weight_push_interval must be >= 1."""
    with pytest.raises(ValidationError):
        LearnerConfig(weight_push_interval=0)
    lc = LearnerConfig(weight_push_interval=5)
    assert lc.weight_push_interval == 5
