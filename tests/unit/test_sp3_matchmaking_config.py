"""Matchmaking config v3 (spec block 5): shares and schedules, anchors, pfsp, per-agent override,
translation of the SP2 knobs (T3.1)."""
from __future__ import annotations

import logging

import pytest
import yaml
from pydantic import ValidationError

from colosseum.core.config import (
    AGENT_MATCHMAKING_KEYS,
    SP2_MATCHMAKING_KNOBS,
    ColosseumConfig,
    MatchmakingConfig,
    load_config,
    merge_matchmaking,
    translate_sp2_matchmaking,
)
from colosseum.core.errors import ConfigError
from colosseum.league.schedule import parse_schedule, schedule_points, schedule_value

BASE = {
    "env": {"env_class": "my_game.game.MyGame"},
    "networks": {"encoder_class": "my_game.models.Encoder", "policy_class": "my_game.models.Policy",
                 "value_class": "my_game.models.Value"},
}
CONFIG_LOGGER = "colosseum.core.config"


def _write(tmp_path, data, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def _knob_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records
            if r.name == CONFIG_LOGGER and r.levelno == logging.WARNING and "SP2 matchmaking knobs" in r.getMessage()]


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def test_a_number_is_a_constant_schedule():
    assert parse_schedule(0.25) == 0.25 and parse_schedule(1) == 1.0
    assert schedule_value(0.25, 10**9) == 0.25
    assert schedule_points(0.25) == [0]


def test_a_schedule_is_piecewise_linear_and_constant_outside_its_points():
    s = parse_schedule({1000: 0.5, 3000: 0.1})
    assert schedule_points(s) == [1000, 3000]
    assert schedule_value(s, 0) == 0.5 and schedule_value(s, 1000) == 0.5
    assert schedule_value(s, 2000) == pytest.approx(0.3)
    assert schedule_value(s, 3000) == pytest.approx(0.1) and schedule_value(s, 10**7) == pytest.approx(0.1)


def test_schedule_keys_accept_strings_like_1e6_and_underscores():
    s = parse_schedule({"0": 1.0, "1e6": 0.0, "2_000_000": 0.5})
    assert schedule_points(s) == [0, 1_000_000, 2_000_000]
    assert schedule_value(s, 500_000) == pytest.approx(0.5)
    assert list(s) == [0, 1_000_000, 2_000_000]  # stored sorted


@pytest.mark.parametrize("raw, message", [
    ({}, "empty"),
    ({-1: 0.5}, "non-negative"),
    ({"1.5": 0.5}, "integer"),
    ({"abc": 0.5}, "not a number"),
    ({0: -0.1}, ">= 0"),
    (-0.5, ">= 0"),
    (float("nan"), "finite"),
    ({0: "x"}, "number"),
    ("0.5", "number"),
    (True, "number"),
    ({0: 0.1, "0": 0.2}, "twice"),
])
def test_schedule_errors_name_the_problem(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_schedule(raw, "matchmaking.opponents.latest")


# ---------------------------------------------------------------------------
# The v3 model
# ---------------------------------------------------------------------------


def test_v3_defaults_and_no_sp2_fields():
    m = ColosseumConfig.model_validate(BASE).matchmaking
    o = m.opponents
    assert (o.latest, o.snapshots, o.rivals, o.anchors) == (0.7, 0.2, 0.0, 0.1)
    assert m.anchors is None
    assert (m.pfsp.weighting, m.pfsp.exponent, m.pfsp.halflife_games) == ("hard", 2.0, 200.0)
    assert (m.layouts, m.teammates, m.teammate_self_prob, m.shuffle_seats, m.matchmaker_class) == \
        ({}, "self", 0.5, True, None)
    assert not set(SP2_MATCHMAKING_KNOBS) & set(MatchmakingConfig.model_fields)


def test_shares_take_schedules_and_anchors_take_lists_or_weights():
    m = MatchmakingConfig.model_validate({
        "opponents": {"latest": {0: 0.9, "1e6": 0.5}, "anchors": {0: 0.1, 1_000_000: 0.5}},
        "anchors": {"greedy": 2.0, "random": {0: 1.0, 500: 0.0}},
    })
    assert m.opponents.latest == {0: 0.9, 1_000_000: 0.5}
    assert m.anchors == {"greedy": 2.0, "random": {0: 1.0, 500: 0.0}}
    assert MatchmakingConfig(anchors=["greedy", "random"]).anchors == ["greedy", "random"]
    assert MatchmakingConfig(anchors=[]).anchors == []


@pytest.mark.parametrize("data", [
    {"opponents": {"latest": -0.1}},
    {"opponents": {"latest": {}}},
    {"opponents": {"bogus": 0.1}},
    {"anchors": ["a", "a"]},
    {"anchors": {"a": -1.0}},
    {"anchors": [""]},
    {"pfsp": {"weighting": "soft"}},
    {"pfsp": {"exponent": -1.0}},
    {"pfsp": {"halflife_games": 0}},
    {"pfsp": {"exponent": float("inf")}},
    {"pfsp": {"exponent": float("nan")}},
    {"pfsp": {"halflife_games": float("inf")}},
    {"layouts": {"2p": 0.0}},
    {"layouts": {"2p": float("inf")}},
    {"teammates": "random"},
    {"teammate_self_prob": 1.5},
    {"matchmaker_class": ""},
])
def test_v3_bounds(data):
    with pytest.raises(ValidationError):
        MatchmakingConfig.model_validate(data)


@pytest.mark.parametrize("knob", SP2_MATCHMAKING_KNOBS)
def test_the_model_itself_rejects_sp2_knobs_and_names_the_replacement(knob):
    with pytest.raises(ValidationError, match="opponents"):
        MatchmakingConfig.model_validate({knob: 0.5})


# ---------------------------------------------------------------------------
# SP2 knob translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("knobs, latest_snapshots_rivals, exponent", [
    ({"mode": "self_play", "latest_prob": 0.8}, (0.8, 0.2, 0.0), 1.0),
    ({"mode": "self_play"}, (0.5, 0.5, 0.0), 1.0),
    ({"latest_prob": 0.3}, (0.3, 0.7, 0.0), 1.0),            # mode defaults to self_play
    ({"mode": "league"}, (0.25, 0.25, 0.5), 1.0),            # self_play_ratio and latest_prob default to 0.5
    ({"mode": "league", "self_play_ratio": 0.0, "latest_prob": 0.8, "pfsp_exponent": 3.0}, (0.0, 0.0, 1.0), 3.0),
    ({"self_play_ratio": 0.2}, (0.5, 0.5, 0.0), 1.0),        # ignored under the default mode self_play
    ({"pfsp_exponent": 0.0}, (0.5, 0.5, 0.0), 0.0),
])
def test_translate_sp2_knobs(knobs, latest_snapshots_rivals, exponent):
    out = translate_sp2_matchmaking({**knobs, "layouts": {"2p": 1.0}, "shuffle_seats": False})
    assert not set(SP2_MATCHMAKING_KNOBS) & set(out)
    assert out["layouts"] == {"2p": 1.0} and out["shuffle_seats"] is False
    got = out["opponents"]
    assert (got["latest"], got["snapshots"], got["rivals"]) == pytest.approx(latest_snapshots_rivals)
    assert got["anchors"] == 0.0
    assert out["pfsp"] == {"weighting": "hard", "exponent": exponent}


def test_translate_without_knobs_returns_a_deep_copy():
    raw = {"opponents": {"latest": 1.0}, "layouts": {"2p": 1.0}}
    out = translate_sp2_matchmaking(raw)
    assert out == raw and out is not raw and out["layouts"] is not raw["layouts"]


@pytest.mark.parametrize("raw, message", [
    ({"mode": "league", "opponents": {"latest": 1.0}}, "opponents"),
    ({"latest_prob": 0.5, "pfsp": {"exponent": 1.0}}, "pfsp"),
    ({"mode": "arena"}, "mode"),
    ({"latest_prob": 1.5}, "latest_prob"),
    ({"self_play_ratio": "x"}, "self_play_ratio"),
    ({"pfsp_exponent": -1}, "pfsp_exponent"),
])
def test_translate_errors(raw, message):
    with pytest.raises(ConfigError, match=message):
        translate_sp2_matchmaking(raw)


def test_loading_sp2_knobs_translates_once_and_never_stores_them(tmp_path, caplog):
    path = _write(tmp_path, {**BASE, "matchmaking": {"mode": "league", "self_play_ratio": 0.4, "latest_prob": 0.75,
                                                     "teammates": "mixed"}})
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        cfg = load_config(path)
    warnings = _knob_warnings(caplog)
    assert len(warnings) == 1 and "opponents" in warnings[0].getMessage()
    o = cfg.matchmaking.opponents
    assert (o.latest, o.snapshots, o.rivals, o.anchors) == pytest.approx((0.3, 0.1, 0.6, 0.0))
    assert cfg.matchmaking.teammates == "mixed" and cfg.matchmaking.pfsp.exponent == 1.0
    dumped = cfg.model_dump(mode="json", by_alias=True)
    assert not set(SP2_MATCHMAKING_KNOBS) & set(dumped["matchmaking"])
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        again = load_config(_write(tmp_path, dumped, "resolved.yaml"))
        cfg.get_agent_config("agent_0")                    # derived configs do not warn either
    assert not _knob_warnings(caplog)
    assert again.matchmaking == cfg.matchmaking


def test_sp2_knobs_together_with_new_keys_are_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="opponents"):
        load_config(_write(tmp_path, {**BASE, "matchmaking": {"mode": "league", "opponents": {"latest": 1.0}}}))


def test_set_accepts_the_sp2_knob_paths(tmp_path):
    cfg = load_config(_write(tmp_path, BASE), {"matchmaking.mode": "league", "matchmaking.self_play_ratio": 0.0})
    assert cfg.matchmaking.opponents.rivals == 1.0
    with pytest.raises(ConfigError, match="Unknown config key"):
        load_config(_write(tmp_path, BASE), {"matchmaking.latest_probb": 0.5})


def test_resolved_schedules_round_trip_through_yaml(tmp_path):
    data = {**BASE, "matchmaking": {"opponents": {"latest": {0: 0.9, "2e6": 0.5}},
                                    "anchors": {"bot": {0: 1.0, 1000: 0.2}}}}
    cfg = load_config(_write(tmp_path, data))
    again = load_config(_write(tmp_path, cfg.model_dump(mode="json", by_alias=True), "resolved.yaml"))
    assert again.matchmaking.opponents.latest == {0: 0.9, 2_000_000: 0.5}
    assert again.matchmaking.anchors == {"bot": {0: 1.0, 1000: 0.2}}


# ---------------------------------------------------------------------------
# Per-agent override
# ---------------------------------------------------------------------------


def _two_agents(**agent_a) -> ColosseumConfig:
    return ColosseumConfig.model_validate({
        **BASE,
        "matchmaking": {"opponents": {"latest": {0: 0.6, 1000: 0.2}, "snapshots": 0.3}, "anchors": ["bot"],
                        "layouts": {"2p": 1.0, "4p": 1.0}},
        "agents": {"a": {"matchmaking": agent_a}, "b": {}},
    })


def test_agent_matchmaking_keys():
    assert AGENT_MATCHMAKING_KEYS == {"opponents", "anchors", "pfsp", "layouts", "teammates", "teammate_self_prob"}


def test_per_agent_override_merges_opponents_and_pfsp_per_key_and_replaces_the_rest():
    cfg = _two_agents(opponents={"latest": 1.0}, anchors={"other": 0.5}, layouts={"4p": 1.0},
                      pfsp={"weighting": "uniform"}, teammates="mixed")
    a, b = cfg.get_agent_config("a").matchmaking, cfg.get_agent_config("b").matchmaking
    assert a.opponents.latest == 1.0 and a.opponents.snapshots == 0.3 and a.opponents.anchors == 0.1
    assert a.anchors == {"other": 0.5} and a.layouts == {"4p": 1.0}
    assert (a.pfsp.weighting, a.pfsp.exponent) == ("uniform", 2.0) and a.teammates == "mixed"
    assert b.opponents.latest == {0: 0.6, 1000: 0.2} and b.anchors == ["bot"] and b.layouts == {"2p": 1.0, "4p": 1.0}


def test_a_per_agent_schedule_replaces_the_global_schedule_whole():
    cfg = _two_agents(opponents={"latest": {500: 0.9}})
    assert cfg.get_agent_config("a").matchmaking.opponents.latest == {500: 0.9}


def test_merge_matchmaking_does_not_mutate_its_inputs():
    base = {"opponents": {"latest": {0: 0.5}}, "anchors": ["x"]}
    override = {"opponents": {"rivals": 0.1}, "anchors": ["y"]}
    out = merge_matchmaking(base, override)
    assert out == {"opponents": {"latest": {0: 0.5}, "rivals": 0.1}, "anchors": ["y"]}
    assert base == {"opponents": {"latest": {0: 0.5}}, "anchors": ["x"]} and override["anchors"] == ["y"]
    out["opponents"]["latest"][0] = 9.0
    assert base["opponents"]["latest"][0] == 0.5


@pytest.mark.parametrize("override, message", [
    ({"shuffle_seats": False}, "global only"),
    ({"matchmaker_class": "x.Y"}, "global only"),
    ({"mode": "league"}, "global matchmaking section"),
    ({"oponents": {}}, "unknown keys"),
    ({"opponents": {"latest": -1.0}}, "matchmaking"),
])
def test_per_agent_override_errors(tmp_path, override, message):
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, {**BASE, "agents": {"a": {"matchmaking": override}}}))


def test_set_reaches_per_agent_matchmaking(tmp_path):
    cfg = load_config(_write(tmp_path, {**BASE, "agents": {"a": {}}}), {"agents.a.matchmaking.opponents.rivals": 0.5})
    assert cfg.get_agent_config("a").matchmaking.opponents.rivals == 0.5
