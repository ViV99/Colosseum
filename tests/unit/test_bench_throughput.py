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


def test_rates_report_chunks_per_update_and_bound():
    samples = _samples([100.0 + 2.0 * i for i in range(31)], chunks_per_step=8)
    r = bench.compute_rates(samples, START, END, num_players=2)
    assert r["train_steps"] == 30
    assert r["updates_per_s"] == pytest.approx(0.5)
    assert r["chunks_per_update"] == pytest.approx(8.0)
    assert r["env_steps_per_s"] == pytest.approx(8 * bench.CHUNK_LENGTH / 2 * 0.5)
    assert r["bound"] == "learner"


def test_partial_batches_mean_worker_bound():
    samples = _samples([100.0 + 2.0 * i for i in range(31)], chunks_per_step=3)
    r = bench.compute_rates(samples, START, END, num_players=2)
    assert r["chunks_per_update"] == pytest.approx(3.0)
    assert r["bound"] == "worker"


def test_benchmark_config_builds_and_validates(tmp_path):
    """Guard: the pinned benchmark config follows the current config schema."""
    from colosseum.core.registry import validate_config

    cfg = bench._make_config(1, str(tmp_path))
    assert cfg.networks.core is None
    validate_config(cfg)
