"""League state (SP4 T3.2, spec block 7.4): RatingBook and PfspStats save and restore every table (forgotten
snapshots included), entries of agents that left the config are dropped with INFO, the coordinator writes and reads
league_state.json, and an unreadable file is a ConfigError."""
from __future__ import annotations

import json
import logging

import gymnasium
import numpy as np
import pytest

from colosseum.coordinator.coordinator import LEAGUE_STATE_FILE
from colosseum.coordinator.ratings import RatingBook
from colosseum.core.errors import ConfigError
from colosseum.envs.game import GameSpec, RoleSpec, SeatSpec
from colosseum.league.pfsp import PfspStats
from game_helpers import make_coordinator, make_match_result, make_test_config, scripted_agent

SPEC = GameSpec(
    roles={"player": RoleSpec(gymnasium.spaces.Box(0, 1, (2,), np.float32), gymnasium.spaces.Discrete(3))},
    layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1)),
             "coop": (SeatSpec("player", 0), SeatSpec("player", 0))},
)


def _results():
    rng = np.random.default_rng(0)
    out = []
    for k in range(300):
        other = ("b", "latest") if k % 3 else ("a", "ckpt_v10")
        if k % 5 == 0:
            other = ("rnd", "fixed")
        ranks = [1.0, 2.0] if rng.random() < 0.6 else ([2.0, 1.0] if rng.random() < 0.8 else [1.0, 1.0])
        out.append(make_match_result([(0, "a", "latest"), (1, *other)], ranks, uid=f"m{k}"))
    for k in range(20):
        out.append(make_match_result([(0, "a", "latest"), (0, "b", "latest")], scores=[float(k % 4)], layout="coop",
                                     uid=f"c{k}"))
    return out


def test_the_rating_book_round_trips():
    book = RatingBook(SPEC, ["a", "b", "rnd"], past_window=50)
    for result in _results():
        book.update(result)
    state = json.loads(json.dumps(book.state_dict()))       # through JSON, as league_state.json
    again = RatingBook(SPEC, ["a", "b", "rnd"], past_window=50)
    assert again.load_state_dict(state, agents=["a", "b", "rnd"]) == []
    assert again.snapshot() == book.snapshot()
    for result in _results()[:60]:                          # the tables go on from the restored state
        book.update(result)
        again.update(result)
    assert again.snapshot() == book.snapshot()


def test_agents_that_left_the_config_are_dropped_with_info(caplog):
    book = RatingBook(SPEC, ["a", "b", "rnd"])
    for result in _results():
        book.update(result)
    fresh = RatingBook(SPEC, ["a", "newcomer"])
    with caplog.at_level(logging.INFO, logger="colosseum.coordinator.ratings"):
        assert fresh.load_state_dict(book.state_dict(), agents=["a", "newcomer"]) == ["b", "rnd"]
    snap = fresh.snapshot()["2p"]
    assert set(snap["games"]["a"]) == {"newcomer"} and snap["games"]["a"]["newcomer"] == 0
    assert snap["elo"]["newcomer"] == 1200.0 and snap["wr_vs_past"]["a"] is not None
    assert fresh.snapshot()["coop"]["cross_play"] == {}      # "a+b" had a dropped member
    assert any("dropped: ['b', 'rnd']" in r.getMessage() for r in caplog.records)
    other_game = {"layouts": {"9p": book.state_dict()["layouts"]["2p"]}}
    with caplog.at_level(logging.INFO, logger="colosseum.coordinator.ratings"):
        fresh.load_state_dict(other_game, agents=["a"])
    assert any("layouts this game does not have are dropped: ['9p']" in r.getMessage() for r in caplog.records)


def test_a_layout_whose_outcome_kind_changed_is_dropped_with_its_own_reason(caplog):
    book = RatingBook(SPEC, ["a", "b", "rnd"])
    for result in _results():
        book.update(result)
    state = book.state_dict()
    state["layouts"]["2p"]["outcome_kind"] = "rank"                # saved by a game whose 2p had three teams
    fresh = RatingBook(SPEC, ["a", "b", "rnd"])
    with caplog.at_level(logging.INFO, logger="colosseum.coordinator.ratings"):
        fresh.load_state_dict(state, agents=["a", "b", "rnd"])
    (line,) = [r.getMessage() for r in caplog.records if "League state" in r.getMessage()]
    assert "outcome kind changed" in line and "2p (saved rank, now wdl)" in line
    assert "does not have" not in line
    assert fresh.snapshot()["2p"]["games"]["a"]["b"] == 0          # not restored


def test_pfsp_statistics_round_trip_with_the_forgotten_snapshots(caplog):
    stats = PfspStats({"a": 20.0, "b": 20.0})
    for result in _results():
        stats.update(result)
    stats.forget(("a", "ckpt_v10"))
    state = json.loads(json.dumps(stats.state_dict()))
    again = PfspStats({"a": 20.0, "b": 20.0})
    assert again.load_state_dict(state, agents=["a", "b", "rnd"]) == []
    assert again.snapshot() == stats.snapshot()
    late = make_match_result([(0, "a", "latest"), (1, "a", "ckpt_v10")], uid="late")
    again.update(late)                                      # a forgotten snapshot never comes back (SP3 #50)
    assert again.games("2p", "a", ("a", "ckpt_v10")) == 0.0
    only_a = PfspStats({"a": 20.0})
    with caplog.at_level(logging.INFO, logger="colosseum.league.pfsp"):
        assert only_a.load_state_dict(state, agents=["a", "b"]) == ["b", "rnd"]
    assert set(only_a.snapshot()["2p"]) == {"a"} and "rnd@fixed" not in only_a.snapshot()["2p"]["a"]
    lines = [r.getMessage() for r in caplog.records if "League state" in r.getMessage()]
    # b is still in the config but no longer trainable (no PFSP owner here); rnd left the config
    assert any("owners that are no longer trainable agents are dropped: ['b']" in line for line in lines)
    assert any("agents not in this config are dropped: ['rnd']" in line for line in lines)


def test_a_snapshot_forgotten_before_the_load_stays_forgotten():
    stats = PfspStats({"a": 20.0})
    for result in _results():
        stats.update(result)
    state = stats.state_dict()
    again = PfspStats({"a": 20.0})
    again.forget(("a", "ckpt_v10"))                         # evicted by the import of the pool, before the load
    again.load_state_dict(state, agents=["a", "b", "rnd"])
    assert "a@ckpt_v10" not in again.snapshot()["2p"]["a"]


def test_the_coordinator_saves_and_loads_league_state(tmp_path):
    config = make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent()})
    coordinator = make_coordinator(config, tmp_path / "ckpt")
    for k in range(40):
        coordinator.report_match_result(make_match_result([(0, "agent_0", "latest"), (1, "rnd", "fixed")],
                                                          [1.0, 2.0] if k % 3 else [2.0, 1.0], uid=f"m{k}"))
    path = tmp_path / LEAGUE_STATE_FILE
    coordinator.save_league_state(path, env_steps=1234)
    saved = json.loads(path.read_text())
    assert (saved["version"], saved["env_steps"]) == (1, 1234) and set(saved) == {"version", "env_steps", "ratings",
                                                                                   "pfsp"}
    fresh = make_coordinator(config, tmp_path / "ckpt2")
    assert fresh.load_league_state(tmp_path / "missing.json") is False
    assert fresh.load_league_state(path) is True
    assert fresh.ratings_snapshot() == coordinator.ratings_snapshot()
    path.write_text("{")
    with pytest.raises(ConfigError, match="league state"):
        fresh.load_league_state(path)
    path.write_text(json.dumps({"version": 99}))
    with pytest.raises(ConfigError, match="version 99"):
        fresh.load_league_state(path)
