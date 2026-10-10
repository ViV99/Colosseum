"""metrics.jsonl schema, episode aggregation by layout/role, system stats, hub cadence (T5.3)."""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from colosseum.core.types import MatchResult, SeatResult, TeamResult
from colosseum.metrics.aggregator import EpisodeAggregator, SystemStats, opponent_type
from colosseum.metrics.console import ConsoleReporter
from colosseum.metrics.hub import MetricsHub, flatten, wr_arena_over_layouts, wr_vs_past_over_layouts
from colosseum.metrics.jsonl import METRIC_KINDS, REQUIRED_KEYS, MetricsWriter, write_json_atomic


def seat(i, team, agent, network="latest", reward=0.0, role="player", eliminated=None) -> SeatResult:
    return SeatResult(seat=i, role=role, team=team, agent_id=agent, network_id=network, reward=reward,
                      eliminated_step=eliminated)


def result(layout, seats, ranks, scores=None, length=5) -> MatchResult:
    kind = "score" if len(ranks) == 1 else "wdl" if len(ranks) == 2 else "rank"
    teams = [TeamResult(team=t, rank=float(r), score=float((scores or {}).get(t, 0.0))) for t, r in ranks.items()]
    return MatchResult(match_id="m", layout=layout, outcome_kind=kind, seats=seats, teams=teams,
                       episode_length=length)


def duel(a, b, rank_a, rank_b, net_b="latest", ra=0.0, rb=0.0, length=5):
    return result("2p", [seat(0, 0, a, reward=ra), seat(1, 1, b, net_b, reward=rb)], {0: rank_a, 1: rank_b},
                  length=length)


RATINGS = {
    "2p": {"elo": {"a": 1216.0, "b": 1184.0}, "win_rates": {"a": {"b": 1.0}, "b": {"a": 0.0}},
           "games": {"a": {"b": 1}, "b": {"a": 1}}, "wr_vs_past": {"a": None, "b": 0.75},
           "past_games": {"a": 0, "b": 4}, "scores": {}, "cross_play": {}, "role_win_rates": {}},
    "4p": {"elo": {"a": 1200.0, "b": 1200.0}, "win_rates": {"a": {"b": 0.0}, "b": {"a": 1.0}},
           "games": {"a": {"b": 3}, "b": {"a": 3}}, "wr_vs_past": {"a": 0.5, "b": 0.25},
           "past_games": {"a": 2, "b": 4}, "scores": {}, "cross_play": {}, "role_win_rates": {}},
}


def test_writer_one_json_line_per_record_with_numpy_and_nan(tmp_path):
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    writer.write("train", agent="a", train_step=np.int64(3), loss=np.float32(0.5), ev=float("nan"))
    with pytest.raises(ValueError):
        writer.write("bogus")
    writer.close()
    record = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    assert record["kind"] == "train" and record["train_step"] == 3 and record["ev"] is None


def test_write_json_atomic(tmp_path):
    write_json_atomic(tmp_path / "ratings.json", {"layouts": {"2p": {"elo": {"a": 1200.0}}}})
    assert json.loads((tmp_path / "ratings.json").read_text()) == {"layouts": {"2p": {"elo": {"a": 1200.0}}}}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ratings.json"]


def test_opponent_type_ignores_teammates():
    r = duel("a", "a", 1, 2, net_b="ckpt_v5")
    assert opponent_type(r, r.seats[0]) == "past"
    r = duel("a", "b", 1, 2)
    assert opponent_type(r, r.seats[0]) == "arena"
    r = duel("a", "a", 1, 1)
    assert opponent_type(r, r.seats[1]) == "latest"
    coop = result("coop2", [seat(0, 0, "a"), seat(1, 0, "b")], {0: 1})
    assert opponent_type(coop, coop.seats[0]) is None  # a teammate of another agent is no opponent
    team = result("2v2", [seat(0, 0, "a"), seat(1, 0, "b"), seat(2, 1, "a"), seat(3, 1, "a")], {0: 1, 1: 2})
    assert opponent_type(team, team.seats[0]) == "latest" and opponent_type(team, team.seats[2]) == "arena"


def test_episode_aggregator_wdl_from_team_ranks_and_layout_breakdown():
    agg = EpisodeAggregator()
    agg.add(duel("a", "a", 1, 2, net_b="ckpt_v5", ra=1.0, rb=-1.0, length=7))
    agg.add(duel("a", "a", 1, 1, length=9))
    agg.add(duel("b", "a", 2, 1, ra=-1.0, rb=1.0, length=5))
    agg.add(result("4p", [seat(i, i, "a", eliminated=(3 if i == 3 else None)) for i in range(4)],
                   {0: 1, 1: 2, 2: 2, 3: 4}, length=10))
    agg.add(result("solo", [seat(0, 0, "s", reward=3.0)], {0: 1}, scores={0: 3.0}, length=10))
    out = agg.flush()
    assert out["a"]["wdl"] == {"latest": [1, 2, 3], "past": [1, 0, 0], "arena": [1, 0, 0], "anchor": [0, 0, 0]}
    assert out["a"]["episodes"] == 8 and out["a"]["seat_counts"] == [3, 3, 1, 1]
    assert out["b"]["wdl"]["arena"] == [0, 0, 1]
    assert out["s"]["return_mean"] == 3.0 and out["s"]["wdl"]["latest"] == [0, 0, 0]
    four = out["a"]["by_layout"]["4p"]["player"]
    assert four["episodes"] == 4 and four["eliminated_frac"] == 0.25
    assert four["length_mean"] == 10.0 and four["wdl"] == {"latest": [1, 0, 3], "past": [0, 0, 0],
                                                           "arena": [0, 0, 0], "anchor": [0, 0, 0]}
    two = out["a"]["by_layout"]["2p"]["player"]  # the checkpoint seat of the first duel is not a's
    assert two["episodes"] == 4 and two["length_mean"] == pytest.approx((7 + 9 + 9 + 5) / 4)
    assert two["wdl"] == {"latest": [0, 2, 0], "past": [1, 0, 0], "arena": [1, 0, 0], "anchor": [0, 0, 0]}
    assert out["b"]["by_layout"]["2p"]["player"]["wdl"]["arena"] == [0, 0, 1]
    assert out["s"]["by_layout"] == {"solo": {"player": {
        "episodes": 1, "return_mean": 3.0, "length_mean": 10.0, "team_score_mean": 3.0, "eliminated_frac": 0.0,
        "wdl": {"latest": [0, 0, 0], "past": [0, 0, 0], "arena": [0, 0, 0], "anchor": [0, 0, 0]}}}}
    assert set(out["a"]["by_layout"]) == {"2p", "4p"}
    assert agg.flush() == {}


def test_episode_aggregator_splits_roles():
    agg = EpisodeAggregator()
    agg.add(result("1v2", [seat(0, 0, "h", role="hunter", reward=2.0), seat(1, 1, "p", role="prey", reward=-1.0),
                           seat(2, 1, "p", role="prey", reward=-3.0)], {0: 1, 1: 2}))
    out = agg.flush()
    assert out["h"]["by_layout"]["1v2"]["hunter"]["return_mean"] == 2.0
    assert out["p"]["by_layout"]["1v2"]["prey"]["return_mean"] == -2.0
    assert out["p"]["by_layout"]["1v2"]["prey"]["episodes"] == 2
    assert out["h"]["by_layout"]["1v2"]["hunter"]["wdl"]["arena"] == [1, 0, 0]
    assert out["p"]["by_layout"]["1v2"]["prey"]["wdl"]["arena"] == [0, 0, 2]
    assert set(out["p"]["by_layout"]["1v2"]) == {"prey"}


def test_system_stats_rates_and_resume_baselines():
    clock = [0.0]
    stats = SystemStats(clock=lambda: clock[0], initial_env_steps=1000, initial_train_steps={"a": 10})
    stats.on_train_step("a", 14)
    stats.on_worker_stats({"worker_id": 0, "parked_buffers": 2, "dropped_reward_episodes": 3})
    stats.on_worker_stats({"worker_id": 1, "parked_buffers": 0, "dropped_reward_episodes": 4})
    clock[0] = 2.0
    snap = stats.snapshot(2000, {"a": 3})
    assert snap["env_steps_per_sec"] == 500.0 and snap["train_steps_per_sec"] == {"a": 2.0}
    assert snap["parked_buffers"] == 2 and snap["workers_reporting"] == 2
    assert snap["dropped_reward_episodes"] == 7          # cumulative per worker, summed over workers
    clock[0] = 20.0
    assert stats.snapshot(2000, {})["workers_reporting"] == 0  # silent workers are forgotten


def test_console_line_and_layout_aggregates():
    line = ConsoleReporter(10_000).format_line(
        "a", train_step=5, env_steps=2500, fps=1234.5, loss=0.123, entropy=None,
        return_mean=0.5, wr_vs_past=0.61, wr_arena=None)
    assert line.startswith("[a] step 5 |") and "wr_vs_past 0.61" in line
    assert wr_vs_past_over_layouts(RATINGS, "b") == pytest.approx((0.75 * 4 + 0.25 * 4) / 8)
    assert wr_vs_past_over_layouts(RATINGS, "a") == 0.5  # 2p has no past games for a
    assert wr_arena_over_layouts(RATINGS, "a") == pytest.approx((1.0 * 1 + 0.0 * 3) / 4)
    assert wr_arena_over_layouts({}, "a") is None


def test_flatten():
    assert flatten("r", {"2p": {"elo": {"a": 1.0}}, "x": None, "l": [1, 2], "ok": True}) == {
        "r/2p/elo/a": 1.0, "r/l/0": 1.0, "r/l/1": 2.0}


def test_hub_cadence_schema_and_ratings_file(tmp_path, caplog):
    clock = [0.0]
    hub = MetricsHub(writer=MetricsWriter(tmp_path / "metrics.jsonl"), ratings_path=tmp_path / "ratings.json",
                     agent_ids=["a", "b"], total_timesteps=1000, log_interval=2, console_interval_sec=10.0,
                     clock=lambda: clock[0])
    for step in range(1, 6):
        hub.on_train_metrics({"agent_id": "a", "train_step": step, "total_loss": 0.1 * step, "note": "x"})
    hub.on_match_result(duel("a", "b", 1, 2, ra=1.0, rb=-1.0, length=7))
    hub.on_worker_stats({"kind": "worker_stats", "worker_id": 0, "env_steps": 100, "parked_buffers": 1})
    assert not hub.maybe_tick(env_steps=100, ratings=RATINGS, queue_depths={"a": 1, "b": 0})
    clock[0] = 11.0
    with caplog.at_level(logging.INFO):
        assert hub.maybe_tick(env_steps=300, ratings=RATINGS, queue_depths={"a": 1, "b": 0})
    assert "[a] step 5 |" in caplog.text and "[b] step 0 |" in caplog.text and "wr_arena 0.25" in caplog.text
    clock[0] = 12.0
    hub.close(env_steps=400, ratings=RATINGS, queue_depths={"a": 0, "b": 0})

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["train_step"] for r in records if r["kind"] == "train"] == [1, 3, 5]
    assert {r["kind"] for r in records} == set(METRIC_KINDS)
    for record in records:
        assert not (REQUIRED_KEYS[record["kind"]] - set(record)), record["kind"]
    ratings_records = [r for r in records if r["kind"] == "ratings"]
    assert ratings_records[-1]["layouts"]["2p"]["elo"] == {"a": 1216.0, "b": 1184.0}
    episodes = [r for r in records if r["kind"] == "episodes"]
    assert {r["agent"] for r in episodes} == {"a", "b"} and "2p" in episodes[0]["by_layout"]
    ratings = json.loads((tmp_path / "ratings.json").read_text())
    assert ratings["env_steps"] == 400 and ratings["layouts"]["4p"]["wr_vs_past"] == {"a": 0.5, "b": 0.25}


def test_hub_forwards_layout_namespaces_to_wandb(tmp_path):
    calls = []

    class FakeWandB:
        def log_train(self, agent_id, metrics, train_step):
            calls.append(("train", agent_id, train_step, dict(metrics)))

        def log_global(self, metrics, env_steps):
            calls.append(("global", env_steps, dict(metrics)))

    hub = MetricsHub(writer=MetricsWriter(tmp_path / "m.jsonl"), ratings_path=tmp_path / "r.json",
                     agent_ids=["a"], total_timesteps=10, log_interval=1, console_interval_sec=0.0,
                     wandb_logger=FakeWandB())
    hub.on_train_metrics({"agent_id": "a", "train_step": 1, "total_loss": 0.5})
    hub.on_match_result(duel("a", "a", 1, 2))
    hub.on_worker_stats({"kind": "worker_stats", "worker_id": 0, "dropped_reward_episodes": 2})
    hub.maybe_tick(env_steps=5, ratings=RATINGS, queue_depths={"a": 0})
    assert calls[0] == ("train", "a", 1, {"total_loss": 0.5})
    row = calls[1][2]
    assert calls[1][0] == "global" and calls[1][1] == 5
    assert row["ratings/2p/elo/a"] == 1216.0 and row["ratings/4p/games/a/b"] == 3.0
    assert row["episodes/a/by_layout/2p/player/episodes"] == 2.0
    assert row["episodes/a/by_layout/2p/player/length_mean"] == 5.0
    assert row["system/dropped_reward_episodes"] == 2.0
    assert not any("/wdl" in key for key in row)  # W/D/L counts stay in metrics.jsonl (SP1 behavior)


def test_hub_keeps_numpy_scalar_metrics_and_closes_on_failure(tmp_path):
    hub = MetricsHub(writer=MetricsWriter(tmp_path / "m.jsonl"), ratings_path=tmp_path / "r.json",
                     agent_ids=["a"], total_timesteps=10, log_interval=1, console_interval_sec=10.0)
    hub.on_train_metrics({"agent_id": "a", "train_step": np.int64(1), "loss": np.float32(0.5),
                          "flag": np.bool_(True), "ok": True})
    hub.close(env_steps=0, ratings=RATINGS, queue_depths={})
    train = json.loads((tmp_path / "m.jsonl").read_text().splitlines()[0])
    assert train["loss"] == 0.5 and "flag" not in train and "ok" not in train

    writer = MetricsWriter(tmp_path / "m2.jsonl")
    failing = MetricsHub(writer=writer, ratings_path=tmp_path / "missing-dir" / "r.json", agent_ids=["a"],
                         total_timesteps=10, log_interval=1, console_interval_sec=10.0)
    with pytest.raises(OSError):
        failing.close(env_steps=0, ratings=RATINGS, queue_depths={})
    assert writer.closed
