"""metrics.jsonl schema, episode/system aggregation, console line, hub cadence (T6.3)."""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from colosseum.core.types import MatchResult, SeatResult
from colosseum.metrics.aggregator import EpisodeAggregator, SystemStats, opponent_type
from colosseum.metrics.console import ConsoleReporter
from colosseum.metrics.hub import MetricsHub, flatten
from colosseum.metrics.jsonl import METRIC_KINDS, REQUIRED_KEYS, MetricsWriter, write_json_atomic


def seat(i, agent, outcome, network="latest", reward=0.0):
    return SeatResult(seat=i, agent_id=agent, network_id=network, outcome=outcome, reward=reward)


def result(*seats, length=5):
    return MatchResult(match_id="m", seats=list(seats), episode_length=length)


RATINGS = {
    "elo": {"a": 1216.0, "b": 1184.0},
    "win_rates": {"a": {"b": 1.0}, "b": {"a": 0.0}},
    "games": {"a": {"b": 1}, "b": {"a": 1}},
    "wr_vs_past": {"a": None, "b": 0.75},
    "past_games": {"a": 0, "b": 4},
}


def test_writer_one_json_line_per_record_with_numpy_and_nan(tmp_path):
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    writer.write("train", agent="a", train_step=np.int64(3), loss=np.float32(0.5), ev=float("nan"))
    with pytest.raises(ValueError):
        writer.write("bogus")
    writer.close()
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "train" and record["train_step"] == 3 and record["ev"] is None
    assert isinstance(record["ts"], float)


def test_write_json_atomic(tmp_path):
    write_json_atomic(tmp_path / "ratings.json", {"elo": {"a": 1200.0}})
    assert json.loads((tmp_path / "ratings.json").read_text()) == {"elo": {"a": 1200.0}}
    assert not list(tmp_path.glob(".*.tmp"))


def test_opponent_type():
    r = result(seat(0, "a", 1.0), seat(1, "a", 0.0, network="ckpt_v5"))
    assert opponent_type(r, r.seats[0]) == "past"
    r = result(seat(0, "a", 1.0), seat(1, "b", 0.0))
    assert opponent_type(r, r.seats[0]) == "arena"
    r = result(seat(0, "a", 1.0), seat(1, "a", 0.0))
    assert opponent_type(r, r.seats[1]) == "latest"
    r = result(seat(0, "a", 1.0))
    assert opponent_type(r, r.seats[0]) is None


def test_episode_aggregator_splits_wdl_by_opponent_type():
    agg = EpisodeAggregator()
    agg.add(result(seat(0, "a", 1.0, reward=1.0), seat(1, "a", 0.0, network="ckpt_v5", reward=-1.0), length=7))
    agg.add(result(seat(0, "a", 0.5), seat(1, "a", 0.5), length=9))
    agg.add(result(seat(0, "b", 0.0, reward=-1.0), seat(1, "a", 1.0, reward=1.0), length=5))
    agg.add(result(seat(0, "s", 0.0, reward=3.0), length=10))
    out = agg.flush()
    assert out["a"]["wdl"] == {"latest": [0, 2, 0], "past": [1, 0, 0], "arena": [1, 0, 0]}
    assert out["a"]["episodes"] == 4 and out["a"]["seat_counts"] == [2, 2]
    assert out["a"]["length_mean"] == pytest.approx((7 + 9 + 9 + 5) / 4)
    assert out["b"]["wdl"]["arena"] == [0, 0, 1]
    assert out["s"]["return_mean"] == 3.0 and out["s"]["wdl"]["latest"] == [0, 0, 0]
    assert agg.flush() == {}


def test_system_stats_rates():
    clock = [0.0]
    stats = SystemStats(clock=lambda: clock[0])
    stats.on_train_step("a", 10)
    stats.on_worker_stats({"worker_id": 0, "parked_buffers": 2})
    stats.on_worker_stats({"worker_id": 1, "parked_buffers": 1})
    clock[0] = 2.0
    first = stats.snapshot(1000, {"a": 3})
    assert first["env_steps_per_sec"] == 500.0 and first["train_steps_per_sec"] == {"a": 5.0}
    assert first["parked_buffers"] == 3 and first["workers_reporting"] == 2
    stats.on_train_step("a", 14)
    clock[0] = 4.0
    second = stats.snapshot(3000, {"a": 0})
    assert second["env_steps_per_sec"] == 1000.0 and second["train_steps_per_sec"] == {"a": 2.0}
    assert stats.snapshot(3000, {})["env_steps_per_sec"] == 0.0  # zero interval -> zero rate


def test_console_line():
    line = ConsoleReporter(10_000).format_line(
        "a", train_step=5, env_steps=2500, fps=1234.5, loss=0.123, entropy=None,
        return_mean=0.5, wr_vs_past=0.61, wr_arena=None)
    assert line.startswith("[a] step 5 |")
    assert "25.0% budget" in line and "loss 0.1230" in line and "entropy -" in line
    assert "wr_vs_past 0.61" in line and "wr_arena" not in line


def test_flatten():
    assert flatten("r", {"elo": {"a": 1.0}, "x": None, "l": [1, 2], "ok": True}) == {
        "r/elo/a": 1.0, "r/l/0": 1.0, "r/l/1": 2.0}


def test_hub_cadence_schema_and_ratings_file(tmp_path, caplog):
    clock = [0.0]
    hub = MetricsHub(writer=MetricsWriter(tmp_path / "metrics.jsonl"), ratings_path=tmp_path / "ratings.json",
                     agent_ids=["a", "b"], total_timesteps=1000, log_interval=2, console_interval_sec=10.0,
                     clock=lambda: clock[0])
    for step in range(1, 6):
        hub.on_train_metrics({"agent_id": "a", "train_step": step, "total_loss": 0.1 * step,
                              "entropy": 2.0, "note": "ignored"})
    hub.on_match_result(result(seat(0, "a", 1.0, reward=1.0), seat(1, "b", 0.0, reward=-1.0), length=7))
    hub.on_worker_stats({"kind": "worker_stats", "worker_id": 0, "env_steps": 100, "parked_buffers": 1})
    assert not hub.maybe_tick(env_steps=100, ratings=RATINGS, queue_depths={"a": 1, "b": 0})
    clock[0] = 11.0
    with caplog.at_level(logging.INFO):
        assert hub.maybe_tick(env_steps=300, ratings=RATINGS, queue_depths={"a": 1, "b": 0})
    assert "[a] step 5 |" in caplog.text and "[b] step 0 |" in caplog.text
    clock[0] = 12.0
    hub.close(env_steps=400, ratings=RATINGS, queue_depths={"a": 0, "b": 0})

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["train_step"] for r in records if r["kind"] == "train"] == [1, 3, 5]
    assert {r["kind"] for r in records} == set(METRIC_KINDS)
    for record in records:
        missing = REQUIRED_KEYS[record["kind"]] - set(record)
        assert not missing, (record["kind"], missing)
    assert "note" not in [r for r in records if r["kind"] == "train"][0]
    ratings = json.loads((tmp_path / "ratings.json").read_text())
    assert ratings["env_steps"] == 400 and ratings["wr_vs_past"] == {"a": None, "b": 0.75}


def test_hub_forwards_to_wandb_logger(tmp_path):
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
    hub.maybe_tick(env_steps=5, ratings={"elo": {"a": 1200.0}}, queue_depths={"a": 0})
    assert calls[0] == ("train", "a", 1, {"total_loss": 0.5})
    assert calls[1][0] == "global" and calls[1][1] == 5 and calls[1][2]["ratings/elo/a"] == 1200.0


def test_report_worker_stats_tags_item_and_never_blocks():
    import queue

    from colosseum.worker.rollout_worker import report_worker_stats

    q = queue.Queue(maxsize=1)
    report_worker_stats(q, 3, {"env_steps": np.int64(40), "parked_buffers": 2})
    assert q.get_nowait() == {"kind": "worker_stats", "worker_id": 3, "env_steps": 40, "parked_buffers": 2}
    q.put_nowait("full")
    report_worker_stats(q, 3, {"env_steps": 1})  # full queue: dropped, no exception
    assert q.get_nowait() == "full"
