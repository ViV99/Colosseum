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
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ratings.json"]  # no tmp file left


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


def test_system_stats_resume_baselines_have_no_rate_spike():
    clock = [0.0]
    stats = SystemStats(clock=lambda: clock[0], initial_env_steps=1_000_000, initial_train_steps={"a": 5000})
    stats.on_train_step("a", 5000)  # the step the learner resumed from: no progress yet
    stats.on_train_step("a", 5010)
    clock[0] = 2.0
    snap = stats.snapshot(1_000_100, {"a": 0})
    assert snap["env_steps_per_sec"] == 50.0
    assert snap["train_steps_per_sec"] == {"a": 5.0}


def test_hub_resume_baselines(tmp_path):
    clock = [0.0]
    hub = MetricsHub(writer=MetricsWriter(tmp_path / "m.jsonl"), ratings_path=tmp_path / "r.json",
                     agent_ids=["a"], total_timesteps=10**7, log_interval=1, console_interval_sec=1.0,
                     clock=lambda: clock[0], initial_env_steps=1_000_000, initial_train_steps={"a": 5000})
    hub.on_train_metrics({"agent_id": "a", "train_step": 5001})
    hub.on_train_metrics({"agent_id": "a", "train_step": 5010})
    clock[0] = 2.0
    hub.close(env_steps=1_000_100, ratings=RATINGS, queue_depths={"a": 0})
    system = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()
              if json.loads(line)["kind"] == "system"]
    assert system[0]["env_steps_per_sec"] == 50.0 and system[0]["train_steps_per_sec"] == {"a": 5.0}


def test_system_stats_forget_silent_workers():
    clock = [0.0]
    stats = SystemStats(clock=lambda: clock[0], worker_timeout_sec=6.0)
    stats.on_worker_stats({"worker_id": 0, "parked_buffers": 4})
    clock[0] = 5.0
    stats.on_worker_stats({"worker_id": 1, "parked_buffers": 1})
    clock[0] = 6.0
    assert stats.snapshot(0, {})["workers_reporting"] == 2  # worker 0 is exactly at the timeout
    clock[0] = 7.0
    snap = stats.snapshot(0, {})
    assert snap["workers_reporting"] == 1 and snap["parked_buffers"] == 1  # worker 0 went silent
    clock[0] = 20.0
    assert stats.snapshot(0, {})["workers_reporting"] == 0
    stats.on_worker_stats({"worker_id": 0, "parked_buffers": 2})  # a worker that reports again counts again
    assert stats.snapshot(0, {})["workers_reporting"] == 1


def test_default_worker_timeout_is_three_stats_intervals():
    from colosseum.metrics.aggregator import WORKER_STATS_INTERVAL_SEC

    clock = [0.0]
    stats = SystemStats(clock=lambda: clock[0])
    stats.on_worker_stats({"worker_id": 0, "parked_buffers": 0})
    clock[0] = 3 * WORKER_STATS_INTERVAL_SEC
    assert stats.snapshot(0, {})["workers_reporting"] == 1
    clock[0] = 3 * WORKER_STATS_INTERVAL_SEC + 0.1
    assert stats.snapshot(0, {})["workers_reporting"] == 0


def test_hub_keeps_numpy_scalar_metrics(tmp_path):
    hub = MetricsHub(writer=MetricsWriter(tmp_path / "m.jsonl"), ratings_path=tmp_path / "r.json",
                     agent_ids=["a"], total_timesteps=10, log_interval=1, console_interval_sec=10.0)
    hub.on_train_metrics({"agent_id": "a", "train_step": np.int64(1), "loss": np.float32(0.5),
                          "n": np.int32(3), "kl": np.float64(0.25), "flag": np.bool_(True), "ok": True})
    hub.close(env_steps=0, ratings=RATINGS, queue_depths={})
    train = json.loads((tmp_path / "m.jsonl").read_text().splitlines()[0])
    assert train["kind"] == "train" and train["train_step"] == 1
    assert train["loss"] == 0.5 and train["n"] == 3.0 and train["kl"] == 0.25
    assert "flag" not in train and "ok" not in train
    assert flatten("x", {"a": np.float32(1.5), "b": np.int64(2), "c": np.bool_(False)}) == {"x/a": 1.5, "x/b": 2.0}


def test_hub_close_closes_the_file_even_if_the_final_records_fail(tmp_path):
    writer = MetricsWriter(tmp_path / "m.jsonl")
    hub = MetricsHub(writer=writer, ratings_path=tmp_path / "missing-dir" / "r.json", agent_ids=["a"],
                     total_timesteps=10, log_interval=1, console_interval_sec=10.0)
    with pytest.raises(OSError):
        hub.close(env_steps=0, ratings=RATINGS, queue_depths={})
    assert writer.closed


def test_launcher_closes_metrics_file_when_final_drain_fails(tmp_path):
    import queue

    from colosseum.core.config import load_config
    from colosseum.launcher import Launcher
    from helpers import example_config, make_test_run_dir

    config = load_config(example_config("tic_tac_toe.yaml"))
    run = make_test_run_dir(config, tmp_path)
    launcher = Launcher(config, run)
    writer = MetricsWriter(run.metrics_path)
    launcher._metrics_writer = writer
    launcher._hub = MetricsHub(writer=writer, ratings_path=run.ratings_path, agent_ids=["agent_0"],
                               total_timesteps=10, log_interval=1, console_interval_sec=10.0)

    class FailingCoordinator:
        def report_match_result(self, result):
            raise RuntimeError("boom")

        def ratings_snapshot(self):
            return RATINGS

    results = queue.Queue()
    results.put(result(seat(0, "agent_0", 1.0)))
    launcher._coordinator = FailingCoordinator()
    launcher._results_queue, launcher._metrics_queue = results, queue.Queue()
    with pytest.raises(RuntimeError, match="boom"):
        launcher._finish_metrics()
    assert writer.closed
