"""Tests for build_model / validate_config and the networks config schema."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from colosseum.core.config import ColosseumConfig, CoreConfig, EnvConfig, NetworkConfig, load_config
from colosseum.core.errors import ColosseumError, ConfigError
from colosseum.core.registry import build_model, validate_config
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import LSTMCore, NoCore, WindowAttentionCore
from colosseum.networks.model import PolicyModel
from helpers import TinyMonolithicModel, example_config

TTT_ENV = "examples.tic_tac_toe.env.TicTacToeEnv"
TTT_NETS = "examples.tic_tac_toe.networks"


def _cfg(**networks) -> ColosseumConfig:
    return ColosseumConfig(env=EnvConfig(env_class=TTT_ENV), networks=NetworkConfig(**networks))


def _composed(core=None, policy=f"{TTT_NETS}.TicTacToePolicy", value=f"{TTT_NETS}.TicTacToeValue"):
    return _cfg(
        encoder_class=f"{TTT_NETS}.TicTacToeEncoder",
        core=core,
        policy_class=policy,
        value_class=value,
    )


def test_errors_hierarchy():
    assert issubclass(ConfigError, ColosseumError)


def test_core_config_accepts_class_alias_and_forbids_extra():
    cc = CoreConfig.model_validate({"class": "colosseum.networks.cores.LSTMCore", "kwargs": {"hidden_size": 8}})
    assert cc.class_path == "colosseum.networks.cores.LSTMCore"
    with pytest.raises(ValidationError):
        CoreConfig.model_validate({"class": "x.Y", "hidden_size": 8})


def test_network_config_requires_model_or_all_parts():
    with pytest.raises(ValidationError, match="missing: encoder_class, policy_class, value_class"):
        NetworkConfig()
    with pytest.raises(ValidationError, match="missing: value_class"):
        NetworkConfig(encoder_class="a.B", policy_class="a.C")
    with pytest.raises(ValidationError, match="must be omitted"):
        NetworkConfig(model_class="a.M", encoder_class="a.B")
    NetworkConfig(model_class="a.M")


def test_network_config_rejects_removed_recurrent_keys():
    with pytest.raises(ValidationError):
        NetworkConfig.model_validate({
            "encoder_class": "a.B", "policy_class": "a.C", "value_class": "a.D",
            "recurrent_type": "lstm",
        })


def test_agent_config_roundtrip_keeps_core():
    cfg = _composed(core={"class": "colosseum.networks.cores.GRUCore", "kwargs": {"hidden_size": 64}})
    again = cfg.get_agent_config("agent_0")
    assert again.networks.core.class_path == "colosseum.networks.cores.GRUCore"
    assert again.networks.core.kwargs == {"hidden_size": 64}


def test_build_model_default_is_composed_nocore():
    model = build_model(_composed())
    assert isinstance(model, ComposedModel)
    assert isinstance(model.core, NoCore)
    assert not model.is_stateful


def test_build_model_passes_in_dim_to_heads():
    model = build_model(_composed(core={"class": "colosseum.networks.cores.LSTMCore",
                                        "kwargs": {"hidden_size": 32}}))
    assert isinstance(model.core, LSTMCore)
    assert model.core.input_dim == 64
    assert model.policy.net.in_features == 32
    assert model.value.net[0].in_features == 32
    out = model.step(torch.zeros(2, 3, 3, 3), model.initial_state(2))
    assert out.value.shape == (2,)


def test_build_model_with_model_class():
    model = build_model(_cfg(model_class="helpers.TinyMonolithicModel"))
    assert isinstance(model, TinyMonolithicModel)
    with pytest.raises(ConfigError, match="must subclass colosseum.networks.model.PolicyModel"):
        build_model(_cfg(model_class="torch.nn.Linear", kwargs={"in_features": 2, "out_features": 2}))


def test_build_model_rejects_non_core_class():
    with pytest.raises(ConfigError, match="must subclass colosseum.networks.cores.Core"):
        build_model(_composed(core={"class": "torch.nn.Identity"}))


@pytest.mark.parametrize("name", ["tic_tac_toe.yaml", "tic_tac_toe_multi.yaml", "chase.yaml"])
def test_example_configs_build_and_validate(name):
    cfg = load_config(example_config(name))
    for aid in cfg.get_trainable_agent_ids():
        acfg = cfg.get_agent_config(aid)
        assert isinstance(build_model(acfg), PolicyModel)
        validate_config(acfg)


def test_validate_config_accepts_window_attention_core():
    validate_config(_composed(core={"class": "colosseum.networks.cores.WindowAttentionCore",
                                    "kwargs": {"d_model": 16, "window": 4, "num_heads": 2}}))
    model = build_model(_composed(core={"class": "colosseum.networks.cores.WindowAttentionCore"}))
    assert isinstance(model.core, WindowAttentionCore)


def test_validate_config_reports_head_dim_mismatch():
    cfg = _composed(
        core={"class": "colosseum.networks.cores.LSTMCore", "kwargs": {"hidden_size": 32}},
        policy="helpers.SimplePolicy",  # in_dim is passed -> fine
        value="helpers.SimpleValue",
    )
    validate_config(cfg)
    cfg_bad = _cfg(
        encoder_class=f"{TTT_NETS}.TicTacToeEncoder",
        core={"class": "colosseum.networks.cores.LSTMCore", "kwargs": {"hidden_size": 32}},
        policy_class="helpers.FixedInputPolicy",
        value_class=f"{TTT_NETS}.TicTacToeValue",
    )
    with pytest.raises(ConfigError, match="core.output_dim=32"):
        validate_config(cfg_bad)


def test_validate_config_reports_non_distribution_policy():
    with pytest.raises(ConfigError, match="must return a colosseum Distribution"):
        validate_config(_composed(policy="torch.nn.Identity"))


def test_validate_config_reports_value_shape():
    with pytest.raises(ConfigError, match=r"value must have shape \[B\]"):
        validate_config(_composed(value="helpers.BadShapeValue"))


def test_validate_config_reports_bad_env_and_bad_class():
    with pytest.raises(ConfigError, match="Failed to create env"):
        validate_config(ColosseumConfig(env=EnvConfig(env_class="nope.Env"),
                                        networks=NetworkConfig(model_class="helpers.TinyMonolithicModel")))
    with pytest.raises(ConfigError, match="Failed to build model"):
        validate_config(_composed(policy="nope.Policy"))
