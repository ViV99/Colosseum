"""init / kickstart config sections (spec block 6): defaults, aliases, per-agent overrides, and
the translation of SP2's training.kickstart_* (T4.1)."""
from __future__ import annotations

import logging

import pytest
import yaml

from colosseum.core.config import (
    SP2_KICKSTART_KNOBS,
    ColosseumConfig,
    TrainingConfig,
    load_config,
    translate_sp2_kickstart,
)
from colosseum.core.errors import ConfigError

BASE = {
    "env": {"env_class": "my_game.game.MyGame"},
    "networks": {"encoder_class": "my_game.models.Encoder", "policy_class": "my_game.models.Policy",
                 "value_class": "my_game.models.Value"},
}
CONFIG_LOGGER = "colosseum.core.config"
SP2_FIELDS = {"kickstart_teacher", "kickstart_lambda", "kickstart_decay_steps", "kickstart_kl"}


def _write(tmp_path, data, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def _kickstart_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == CONFIG_LOGGER and r.levelno == logging.WARNING
            and "training.kickstart_" in r.getMessage()]


def test_defaults_and_no_training_fields():
    cfg = ColosseumConfig.model_validate(BASE)
    assert (cfg.init.from_, cfg.init.strict, cfg.init.critic_warmup_steps) == (None, True, 0)
    k = cfg.kickstart
    assert (k.teacher, k.lambda_, k.decay_steps, k.kl) == (None, 1.0, 50_000, "forward")
    assert not SP2_FIELDS & set(TrainingConfig.model_fields)
    assert set(SP2_KICKSTART_KNOBS) == SP2_FIELDS


def test_yaml_aliases_load_and_dump(tmp_path):
    cfg = load_config(_write(tmp_path, {**BASE, "init": {"from": "bc.pt", "critic_warmup_steps": 5},
                                        "kickstart": {"teacher": "bot", "lambda": 0.3}}))
    assert (cfg.init.from_, cfg.init.critic_warmup_steps) == ("bc.pt", 5) and cfg.kickstart.lambda_ == 0.3
    dumped = cfg.model_dump(mode="json", by_alias=True)
    assert dumped["init"]["from"] == "bc.pt" and dumped["kickstart"]["lambda"] == 0.3
    assert load_config(_write(tmp_path, dumped, "resolved.yaml")).init == cfg.init


def test_per_agent_sections_deep_merge_onto_the_global_ones():
    cfg = ColosseumConfig.model_validate({
        **BASE, "init": {"from": "global.pt"}, "kickstart": {"teacher": "bot", "lambda": 0.5},
        "agents": {"a": {"init": {"strict": False}, "kickstart": {"lambda": 0.1}},
                   "b": {"kickstart": {"teacher": None}}},
    })
    a, b = cfg.get_agent_config("a"), cfg.get_agent_config("b")
    assert (a.init.from_, a.init.strict) == ("global.pt", False)
    assert (a.kickstart.teacher, a.kickstart.lambda_) == ("bot", 0.1)
    assert b.kickstart.teacher is None and b.init.from_ == "global.pt"


def test_set_reaches_the_new_sections(tmp_path):
    cfg = load_config(_write(tmp_path, {**BASE, "agents": {"a": {}}}),
                      {"init.from": "x.pt", "agents.a.kickstart.lambda": 0.2})
    assert cfg.init.from_ == "x.pt" and cfg.get_agent_config("a").kickstart.lambda_ == 0.2


@pytest.mark.parametrize("data", [
    {"init": {"critic_warmup_steps": -1}},
    {"init": {"frm": "x"}},
    {"kickstart": {"lambda": -0.1}},
    {"kickstart": {"decay_steps": 0}},
    {"kickstart": {"kl": "sideways"}},
    {"agents": {"a": {"kickstart": {"teachr": "x"}}}},
    {"agents": {"a": {"init": {"strict": "maybe"}}}},
])
def test_bad_values_are_config_errors(tmp_path, data):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, {**BASE, **data}))


def test_training_kickstart_knobs_are_translated_once_and_never_stored(tmp_path, caplog):
    data = {**BASE, "training": {"total_timesteps": 100, "kickstart_teacher": "bc.pt", "kickstart_lambda": 0.4,
                                 "kickstart_decay_steps": 10, "kickstart_kl": "reverse"}}
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        cfg = load_config(_write(tmp_path, data))
    assert len(_kickstart_warnings(caplog)) == 1
    k = cfg.kickstart
    assert (k.teacher, k.lambda_, k.decay_steps, k.kl) == ("bc.pt", 0.4, 10, "reverse")
    assert cfg.training.total_timesteps == 100
    dumped = cfg.model_dump(mode="json", by_alias=True)
    assert not SP2_FIELDS & set(dumped["training"])
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        again = load_config(_write(tmp_path, dumped, "resolved.yaml"))
        cfg.get_agent_config("agent_0")
    assert not _kickstart_warnings(caplog) and again.kickstart == cfg.kickstart


def test_partially_set_knobs_keep_the_other_defaults():
    cfg = ColosseumConfig.model_validate({**BASE, "training": {"kickstart_lambda": 0.2}})
    assert (cfg.kickstart.teacher, cfg.kickstart.lambda_, cfg.kickstart.decay_steps) == (None, 0.2, 50_000)
    sp2_resolved = {**BASE, "training": {"kickstart_teacher": None, "kickstart_lambda": 1.0,
                                         "kickstart_decay_steps": 50000, "kickstart_kl": "forward"}}
    assert ColosseumConfig.model_validate(sp2_resolved).kickstart == ColosseumConfig.model_validate(BASE).kickstart


def test_knobs_together_with_a_kickstart_section_are_a_config_error(tmp_path):
    data = {**BASE, "training": {"kickstart_teacher": "a.pt"}, "kickstart": {"teacher": "b.pt"}}
    with pytest.raises(ConfigError, match="kickstart"):
        load_config(_write(tmp_path, data))


def test_set_accepts_the_training_kickstart_paths(tmp_path):
    cfg = load_config(_write(tmp_path, BASE),
                      {"training.kickstart_teacher": "x.pt", "training.kickstart_kl": "reverse"})
    assert (cfg.kickstart.teacher, cfg.kickstart.kl) == ("x.pt", "reverse")


def test_translate_sp2_kickstart_is_pure():
    raw = {"training": {"kickstart_lambda": 0.5, "seed": 1}}
    out = translate_sp2_kickstart(raw)
    assert out == {"training": {"seed": 1}, "kickstart": {"lambda": 0.5}}
    assert raw == {"training": {"kickstart_lambda": 0.5, "seed": 1}}
    assert translate_sp2_kickstart(BASE) is BASE
