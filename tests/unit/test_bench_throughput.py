"""Pure helpers of scripts/bench_throughput.py: window validation and rates (no processes)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bench_throughput.py"
_spec = importlib.util.spec_from_file_location("bench_throughput", _SCRIPT)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

START, END = 100.0, 160.0


def _samples(times, chunks_per_step=8):
    return [(t, i + 1, (i + 1) * chunks_per_step) for i, t in enumerate(times)]


def test_full_window_is_valid():
    samples = _samples([90.0, 95.0] + [100.0 + 2.0 * i for i in range(31)])
    assert bench.validate_window(samples, START, END) is None


def test_too_few_samples_in_window_is_rejected():
    samples = _samples([90.0, 101.0, 130.0, 159.0])
    error = bench.validate_window(samples, START, END)
    assert error is not None and "samples" in error


def test_no_samples_is_rejected():
    assert bench.validate_window([], START, END) is not None


def test_run_that_stopped_early_is_rejected():
    # learners died 20 s before the end of the window: plenty of samples, short window
    samples = _samples([100.0 + i for i in range(41)])
    error = bench.validate_window(samples, START, END)
    assert error is not None and "last sample" in error


def test_run_that_started_late_is_rejected():
    samples = _samples([130.0 + i for i in range(31)])
    error = bench.validate_window(samples, START, END)
    assert error is not None and "first sample" in error


def _system(times, env_steps_per_s=1000.0, depth=32):
    return [(t, int(env_steps_per_s * (t - START)), depth) for t in times]


def test_rates_use_the_global_env_step_counter_and_record_timestamps():
    train = _samples([100.0 + 2.0 * i for i in range(31)], chunks_per_step=8)
    system = _system([100.5 + i for i in range(60)], env_steps_per_s=640.0)
    r = bench.compute_rates(train, system, START, END)
    assert r["train_steps"] == 30
    assert r["updates_per_s"] == pytest.approx(0.5)
    assert r["chunks_per_update"] == pytest.approx(8.0)  # informational only
    assert r["env_steps_per_s"] == pytest.approx(640.0, rel=1e-3)
    assert r["queue_depth_mean"] == pytest.approx(32.0)
    assert r["bound"] == "learner"


def test_full_batches_with_an_empty_queue_mean_worker_bound():
    # Since T2.4 every batch is full, so chunks/update cannot tell the bottleneck; the queue depth does.
    train = _samples([100.0 + 2.0 * i for i in range(31)], chunks_per_step=8)
    system = _system([100.5 + i for i in range(60)], depth=1)
    r = bench.compute_rates(train, system, START, END)
    assert r["chunks_per_update"] == pytest.approx(8.0)
    assert r["bound"] == "worker"


def test_bound_thresholds_and_unknown_depth():
    assert bench.classify_bound(bench.LEARNER_BOUND_QUEUE_FRACTION * bench.QUEUE_SIZE) == "learner"
    assert bench.classify_bound(bench.LEARNER_BOUND_QUEUE_FRACTION * bench.QUEUE_SIZE - 0.1) == "worker"
    assert bench.classify_bound(None) == "unknown"
    train = _samples([100.0 + 2.0 * i for i in range(31)])
    system = _system([100.5 + i for i in range(60)], depth=-1)  # qsize() unsupported on the platform
    r = bench.compute_rates(train, system, START, END)
    assert r["queue_depth_mean"] is None and r["bound"] == "unknown"


def test_missing_system_records_are_rejected():
    error = bench.validate_window(_system([101.0, 102.0]), START, END, "system")
    assert error is not None and "system samples" in error


def test_load_samples_reads_metrics_jsonl(tmp_path):
    from colosseum.metrics.jsonl import MetricsWriter

    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    writer.write("train", agent="agent_0", train_step=1, chunks_received=8.0, total_loss=0.1)
    writer.write("train", agent="other", train_step=1, chunks_received=8.0)
    writer.write("system", env_steps=512, env_steps_per_sec=0.0, train_steps_per_sec={},
                 queue_depths={"agent_0": 30}, parked_buffers=0, workers_reporting=1)
    writer.write("ratings", env_steps=512, layouts={"2p": {"elo": {}, "win_rates": {}}})
    writer.close()
    train, system = bench.load_samples(tmp_path / "metrics.jsonl")
    assert [s[1:] for s in train] == [(1, 8)]
    assert [s[1:] for s in system] == [(512, 30)]
    assert all(isinstance(s[0], float) for s in train + system)


def test_workload_summary_names_the_layout():
    workload = bench.workload_summary(15.0, 60.0)
    assert set(workload) == {"env", "layout", "envs_per_worker", "chunk_length", "batch_chunks", "queue_size",
                             "warmup_s", "duration_s"}
    assert workload["layout"] == "2p" and workload["queue_size"] == bench.QUEUE_SIZE


def test_benchmark_config_builds_and_validates(tmp_path):
    """Guard: the pinned benchmark config follows the current config schema."""
    from colosseum.core.registry import validate_config

    cfg = bench._make_config(1, str(tmp_path))
    assert cfg.networks.core is None and cfg.matchmaking.layouts == {"2p": 1.0}
    assert cfg.env.env_class == "examples.tic_tac_toe.game.TicTacToeGame"
    validate_config(cfg)
