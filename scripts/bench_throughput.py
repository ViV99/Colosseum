"""Throughput benchmark: tic-tac-toe self-play training for a fixed wall-clock time.

For every worker count the real ``Launcher`` runs (spawned worker and learner
processes) with a pinned tic-tac-toe workload (``_make_config``: every value
that affects the workload is set here, not inherited from the example yaml, so
"before" and "after" runs measure the same work). The learner's metrics are
captured in this process by replacing ``colosseum.launcher.WandBLogger`` with
an in-memory recorder (the monitor loop forwards every learner metrics dict to
it). After a warm-up period the script measures, over ``--duration`` seconds:

- ``updates/s``          learner train steps per second;
- ``env_steps/s``        env steps consumed by the learner per second, i.e.
  ``chunks_received * chunk_length / num_players`` per second. This assumes
  every seat records one transition per env step (true while TicTacToeEnv has
  no ``info["active"]``; T6.3 switches to the global env-step counter).
  Checkpoints are disabled, so every slot plays the latest policy and collects
  data, and a full queue blocks the workers, so consumption equals production;
- ``chunks/update``      chunks per train step. Equal to ``batch_chunks`` means
  every batch was full (learner-bound); below it means the learner waited for
  data (worker-bound).

A run whose measurement window is incomplete (too few learner samples, or
learner metrics starting late / stopping early, e.g. because a learner died)
is reported as an error, gets no row, and makes the script exit non-zero.

Usage::

    .venv/bin/python scripts/bench_throughput.py                 # 1, 2, 4 workers, 60 s each
    .venv/bin/python scripts/bench_throughput.py --workers 1 2 --duration 30 --json out.json

When the launcher stops reporting learner metrics through ``WandBLogger``
(metrics.jsonl, block 6), switch ``_Recorder`` to that source.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import platform
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # `examples.*` for this process and spawned children

ENVS_PER_WORKER = 8
CHUNK_LENGTH = 32
BATCH_CHUNKS = 8
NUM_PLAYERS = 2

# Window validation: minimum learner samples inside the window, and how far the
# first / last sample may be from the window's start / end.
MIN_WINDOW_SAMPLES = 5
WINDOW_EDGE_TOLERANCE_S = 5.0
# chunks/update at or above this fraction of batch_chunks counts as learner-bound.
FULL_BATCH_FRACTION = 0.95

# (monotonic time, train_step, chunks_received) per learner metrics message.
_SAMPLES: list[tuple[float, int, int]] = []


class _Recorder:
    """Stand-in for WandBLogger with the same interface; records learner metrics."""

    def __init__(self, config, run_name=None) -> None:
        pass

    def log_config(self, config) -> None:
        pass

    def log_metrics(self, metrics, step=None) -> None:
        pass

    def log_train_step(self, agent_id, metrics, step) -> None:
        _SAMPLES.append((time.monotonic(), int(step), int(metrics.get("chunks_received", 0))))

    def finish(self) -> None:
        pass


def _make_config(num_workers: int, checkpoint_dir: str):
    """The pinned benchmark workload. Every workload-relevant value is explicit."""
    from colosseum.core.config import ColosseumConfig

    return ColosseumConfig(
        env={
            "env_class": "examples.tic_tac_toe.env.TicTacToeEnv",
            "num_players": NUM_PLAYERS,
            "kwargs": {},
        },
        networks={
            "encoder_class": "examples.tic_tac_toe.networks.TicTacToeEncoder",
            "policy_class": "examples.tic_tac_toe.networks.TicTacToePolicy",
            "value_class": "examples.tic_tac_toe.networks.TicTacToeValue",
            "kwargs": {},
            "core": None,
        },
        algorithm={
            "algorithm_class": "colosseum.algorithms.appo.APPO",
            "gamma": 0.99,
            "vtrace_lambda": 1.0,
            "eps_clip": 0.2,
            "value_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "max_grad_norm": 0.5,
            "num_epochs": 1,
            "minibatch_chunks": 0,
            "vtrace_rho_bar": 1.0,
            "vtrace_c_bar": 1.0,
            "learning_rate": 3.0e-4,
            "lr_schedule": "constant",
            "use_torch_compile": False,
            "normalize_advantages": True,
            "use_amp": False,
        },
        rollout={
            "chunk_length": CHUNK_LENGTH,
            "num_workers": num_workers,
            "envs_per_worker": ENVS_PER_WORKER,
            "weight_sync_interval_sec": 2.0,
            "vec_env": "sync",
            "match_refresh_interval_sec": 0.0,
            "torch_threads": 1,
        },
        learner={
            "device": "cpu",
            "queue_size": 4 * BATCH_CHUNKS,
            "batch_chunks": BATCH_CHUNKS,
            "weight_push_interval": 5,
            "pin_memory": False,
            "torch_threads": None,  # auto: (cpu_count - num_workers * 1) // 1
        },
        training={
            "phase": "self_play",
            "total_timesteps": 10**12,  # stopped by the timer, not the budget
            "seed": 0,
        },
        self_play={
            "checkpoint_interval": 10**9,  # no checkpoints: every slot collects
            "pool_size": 10,
            "latest_prob": 0.5,
        },
        checkpoint={"dir": checkpoint_dir, "save_optimizer": True},
        metrics={"use_wandb": False, "log_interval": 1},
        transport={"mode": "local"},
    )


def _window(samples: list[tuple[float, int, int]], start: float, end: float) -> list[tuple[float, int, int]]:
    return [s for s in samples if start <= s[0] <= end]


def validate_window(samples: list[tuple[float, int, int]], start: float, end: float) -> str | None:
    """Return an error message if the measurement window is incomplete, else None."""
    window = _window(samples, start, end)
    if len(window) < MIN_WINDOW_SAMPLES:
        return f"only {len(window)} learner samples in the window (need >= {MIN_WINDOW_SAMPLES})"
    first, last = window[0][0], window[-1][0]
    if first - start > WINDOW_EDGE_TOLERANCE_S:
        return f"first sample {first - start:.1f}s after the window start (learner started late or stalled)"
    if end - last > WINDOW_EDGE_TOLERANCE_S:
        return f"last sample {end - last:.1f}s before the window end (learner stopped early or stalled)"
    return None


def compute_rates(samples: list[tuple[float, int, int]], start: float, end: float, num_players: int) -> dict:
    """Rates over the window; call only after ``validate_window`` returned None."""
    window = _window(samples, start, end)
    (t0, step0, chunks0), (t1, step1, chunks1) = window[0], window[-1]
    dt = max(t1 - t0, 1e-9)
    steps = step1 - step0
    chunks = chunks1 - chunks0
    chunks_per_update = chunks / steps if steps else 0.0
    return {
        "updates_per_s": steps / dt,
        "env_steps_per_s": chunks * CHUNK_LENGTH / num_players / dt,
        "chunks_per_update": chunks_per_update,
        "bound": "learner" if chunks_per_update >= FULL_BATCH_FRACTION * BATCH_CHUNKS else "worker",
        "train_steps": steps,
    }


def run_one(num_workers: int, duration: float, warmup: float) -> dict:
    """Run one configuration; raises RuntimeError if the window is incomplete."""
    import colosseum.launcher as launcher_mod

    _SAMPLES.clear()
    launcher_mod.WandBLogger = _Recorder
    with tempfile.TemporaryDirectory(prefix="bench-") as tmp:
        config = _make_config(num_workers, tmp)
        launcher = launcher_mod.Launcher(config)
        started = time.monotonic()
        timer = threading.Timer(warmup + duration, launcher._stop_event.set)
        timer.start()
        try:
            launcher.launch()
        finally:
            timer.cancel()
    start, end = started + warmup, started + warmup + duration
    error = validate_window(_SAMPLES, start, end)
    if error is not None:
        raise RuntimeError(error)
    result = compute_rates(_SAMPLES, start, end, config.env.num_players)
    result["workers"] = num_workers
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--duration", type=float, default=60.0, help="measured seconds per run")
    parser.add_argument("--warmup", type=float, default=15.0, help="ignored seconds after start")
    parser.add_argument("--json", type=str, default=None, help="also write results to this JSON file")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    logging.basicConfig(level=logging.WARNING)

    import torch

    machine = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads_default": torch.get_num_threads(),
    }
    workload = {
        "env": "tic_tac_toe",
        "num_players": NUM_PLAYERS,
        "envs_per_worker": ENVS_PER_WORKER,
        "chunk_length": CHUNK_LENGTH,
        "batch_chunks": BATCH_CHUNKS,
        "warmup_s": args.warmup,
        "duration_s": args.duration,
    }
    print(f"machine: {machine}", flush=True)
    print(f"workload: {workload}", flush=True)

    results = []
    failed = []
    for n in args.workers:
        try:
            r = run_one(n, args.duration, args.warmup)
        except RuntimeError as exc:
            failed.append(n)
            print(f"ERROR workers={n}: invalid measurement: {exc}; no row reported", file=sys.stderr, flush=True)
            continue
        results.append(r)
        print(
            f"workers={n}: updates/s={r['updates_per_s']:.2f} env_steps/s={r['env_steps_per_s']:.0f} "
            f"chunks/update={r['chunks_per_update']:.2f} ({r['bound']}-bound, "
            f"train_steps in window={r['train_steps']})",
            flush=True,
        )

    print()
    print("| workers | updates/s | env steps/s | chunks/update | bound |")
    print("|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['workers']} | {r['updates_per_s']:.2f} | {r['env_steps_per_s']:.0f} "
            f"| {r['chunks_per_update']:.2f} | {r['bound']} |"
        )

    if failed:
        print(f"\nERROR: invalid measurement for workers={failed}; no JSON written", file=sys.stderr)
        return 1

    rates = [r["env_steps_per_s"] for r in results]
    monotonic = all(b > a for a, b in zip(rates, rates[1:]))
    print(f"\nenv steps/s increases monotonically with workers: {'yes' if monotonic else 'no'}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"machine": machine, "workload": workload, "results": results}, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
