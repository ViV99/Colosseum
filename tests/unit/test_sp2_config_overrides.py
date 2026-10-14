"""SP2 config: strict models, deep-merged agent overrides, --set parsing, agent ids (SP1 guarantees, T7.1).

The CLI/launcher tests of SP1's test_config_overrides.py are in test_sp2_entry_points.py."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from pydantic import BaseModel

import colosseum.core.config as config_module
from colosseum.core.config import (
    ColosseumConfig,
    apply_overrides,
    deep_merge,
    load_config,
    parse_override_value,
)
from colosseum.core.errors import ConfigError

TTT = "examples.tic_tac_toe"  # the SP2 example: game.py, models.py
REPO_ROOT = Path(__file__).resolve().parents[2]


def base_data(**extra) -> dict:
    data = {
        "env": {"env_class": f"{TTT}.game.TicTacToeGame"},
        "networks": {
            "encoder_class": f"{TTT}.models.TicTacToeEncoder",
            "policy_class": f"{TTT}.models.TicTacToePolicy",
            "value_class": f"{TTT}.models.TicTacToeValue",
        },
        "algorithm": {"lr_schedule": "constant", "learning_rate": 1e-3},
        "learner": {"batch_chunks": 2, "queue_size": 16},
    }
    data.update(extra)
    return data


def write_yaml(tmp_path, data, name="cfg.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def test_every_config_model_forbids_extra_keys():
    models = [obj for _, obj in inspect.getmembers(config_module, inspect.isclass)
              if issubclass(obj, BaseModel) and obj.__module__ == config_module.__name__]
    assert len(models) >= 12
    for model in models:
        assert model.model_config.get("extra") == "forbid", model.__name__


@pytest.mark.parametrize("data", [
    base_data(rollot={}),
    base_data(rollout={"num_worker": 3}),
    base_data(training={"kickstart_teachr": "x.pt"}),
    base_data(agents={"x": {"algoritm": {}}}),
    base_data(agents={"x": {"algorithm": {"learnin_rate": 1.0}}}),
    base_data(agents={"x": {"training": {}}}),
])
def test_typos_are_rejected(tmp_path, data):
    with pytest.raises(ConfigError):
        load_config(write_yaml(tmp_path, data))


def test_agent_override_deep_merges_onto_global_section(tmp_path):
    data = base_data(agents={
        "alpha": None,
        "beta": {"algorithm": {"learning_rate": 1e-4}, "learner": {"batch_chunks": 8}},
    })
    cfg = load_config(write_yaml(tmp_path, data))
    beta = cfg.get_agent_config("beta")
    assert beta.algorithm.learning_rate == 1e-4
    assert beta.algorithm.lr_schedule == cfg.algorithm.lr_schedule  # kept the global "constant"
    assert beta.learner.batch_chunks == 8 and beta.learner.queue_size == 16
    alpha = cfg.get_agent_config("alpha")
    assert alpha.algorithm.learning_rate == 1e-3
    assert beta.agents == {}


def test_partial_networks_override_merges_core_kwargs(tmp_path):
    data = base_data()
    data["networks"]["core"] = {"class": "colosseum.networks.cores.LSTMCore", "kwargs": {"hidden_size": 16}}
    data["agents"] = {
        "big": {"networks": {"core": {"kwargs": {"hidden_size": 64}}}},
        "flat": {"networks": {"core": None}},
    }
    cfg = load_config(write_yaml(tmp_path, data))
    big = cfg.get_agent_config("big").networks
    assert big.core.class_path == "colosseum.networks.cores.LSTMCore"
    assert big.core.kwargs == {"hidden_size": 64}
    assert cfg.get_agent_config("flat").networks.core is None


def test_unknown_agent_id_raises():
    cfg = ColosseumConfig.model_validate(base_data(agents={"alpha": {}}))
    with pytest.raises(ConfigError, match="Unknown agent"):
        cfg.get_agent_config("alpah")
    single = ColosseumConfig.model_validate(base_data())
    assert single.get_agent_config("agent_0").learner.queue_size == 16
    with pytest.raises(ConfigError):
        single.get_agent_config("other")


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": [1]}}
    out = deep_merge(base, {"a": {"b": 2}})
    assert out == {"a": {"b": 2, "c": [1]}} and base == {"a": {"b": 1, "c": [1]}}


@pytest.mark.parametrize("raw,expected", [
    ("null", None), ("~", None), ("", None),
    ("3", 3), ("-2", -2), ("3.0", 3.0), ("1e-4", 1e-4), ("1.5e3", 1500.0),
    ("true", True), ("false", False),
    ("[1, 2]", [1, 2]), ("{a: 1}", {"a": 1}), ("{}", {}),
    ("abc", "abc"), ("runs/x", "runs/x"), ("2026-10-08", "2026-10-08"),
])
def test_parse_override_value(raw, expected):
    value = parse_override_value(raw)
    assert value == expected and type(value) is type(expected)


@pytest.mark.parametrize("raw,expected", [
    ('"123"', "123"), ("'1e-3'", "1e-3"), ("1e-3", 0.001), ("'null'", "null"), ("  7 ", 7),
])
def test_parse_override_value_quoting(raw, expected):
    """Fix round 1: quoted values stay strings; the number rule applies to unquoted input only."""
    value = parse_override_value(raw)
    assert value == expected and type(value) is type(expected)


@pytest.mark.parametrize("raw", ["[1,", "*a", "a: b: c"])
def test_parse_override_value_malformed_yaml_is_config_error(raw):
    with pytest.raises(ConfigError, match=r"--set rollout\.x=") as info:
        parse_override_value(raw, key="rollout.x")
    assert raw in str(info.value)


@pytest.mark.parametrize("agent_id", ["a.b", "../x", ".hidden"])
def test_bad_agent_ids_are_rejected_on_load(tmp_path, agent_id):
    with pytest.raises(ConfigError, match="Invalid agent id"):
        load_config(write_yaml(tmp_path, base_data(agents={agent_id: None})))


def test_dotted_agent_id_message(tmp_path):
    with pytest.raises(ConfigError, match=r"'\.' is not allowed \(it separates --set path parts\)"):
        load_config(write_yaml(tmp_path, base_data(agents={"a.b": None})))


@pytest.mark.parametrize("agent_id", sorted(config_module.RESERVED_AGENT_IDS))
def test_reserved_agent_ids_are_rejected(agent_id):
    """Agent ids may not equal a global metrics namespace (T6.4 fix round 1)."""
    with pytest.raises(ConfigError, match=rf"Invalid agent id '{agent_id}': reserved for global metrics"):
        config_module.check_agent_id(agent_id)


def test_reserved_agent_id_is_rejected_on_load(tmp_path):
    with pytest.raises(ConfigError, match="reserved for global metrics"):
        load_config(write_yaml(tmp_path, base_data(agents={"system": {}})))


def test_apply_overrides_sets_nested_values_and_creates_agent_sections():
    data = apply_overrides(base_data(), {
        "rollout.num_workers": 8,
        "training.resume_from": None,
        "agents.alpha.algorithm.learning_rate": 1e-4,
        "env.kwargs.size": 5,
        "run.name": "exp",
    })
    cfg = ColosseumConfig.model_validate(data)
    assert cfg.rollout.num_workers == 8 and cfg.training.resume_from is None
    assert cfg.get_agent_config("alpha").algorithm.learning_rate == 1e-4
    assert cfg.env.kwargs == {"size": 5} and cfg.run.name == "exp"


@pytest.mark.parametrize("key", [
    "rollout.num_worker", "rollot.x", "rollout.num_workers.x", "agents.a.trainin.x", "env.env_class.x", "a..b",
])
def test_apply_overrides_unknown_path_raises(key):
    with pytest.raises(ConfigError):
        apply_overrides(base_data(), {key: 1})


@pytest.mark.parametrize("key, whole", [
    ("matchmaking.anchors.random", "--set matchmaking.anchors={random: ...}"),
    ("matchmaking.opponents.latest.0", "--set matchmaking.opponents.latest={0: ...}"),
])
def test_setting_inside_an_anchor_map_or_a_schedule_hints_at_the_whole_value(key, whole):
    with pytest.raises(ConfigError, match="set the whole value") as info:
        apply_overrides(base_data(), {key: 0.2})
    assert whole in str(info.value)


def test_load_config_applies_overrides_before_validation(tmp_path):
    path = write_yaml(tmp_path, base_data())
    cfg = load_config(path, {"learner.batch_chunks": 4, "agents.alpha.learner.queue_size": 8})
    assert cfg.learner.batch_chunks == 4
    assert cfg.get_agent_config("alpha").learner.queue_size == 8
    assert cfg.get_agent_config("alpha").learner.batch_chunks == 4


def test_cli_validate_reports_typo_with_exit_1(tmp_path, monkeypatch):
    from colosseum.cli import main

    monkeypatch.chdir(REPO_ROOT)
    bad = write_yaml(tmp_path, base_data(rollout={"num_worker": 3}))
    result = CliRunner().invoke(main, ["validate", "-c", str(bad)])
    assert result.exit_code == 1
    assert "num_worker" in result.output
    good = write_yaml(tmp_path, base_data(), name="good.yaml")
    ok = CliRunner().invoke(main, ["validate", "-c", str(good), "--set", "rollout.num_workers=2"])
    assert ok.exit_code == 0, ok.output
