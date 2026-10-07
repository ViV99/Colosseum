"""Throughput benchmark: tic-tac-toe self-play training for a fixed wall-clock time.

For every worker count the real ``Launcher`` runs (spawned worker and learner
processes) with the tic-tac-toe example config. The learner's metrics are
captured in this process by replacing ``colosseum.launcher.WandBLogger`` with
an in-memory recorder (the monitor loop forwards every learner metrics dict to
it). After a warm-up period the script measures, over ``--duration`` seconds:

- ``updates/s``   learner train steps per second;
- ``env_steps/s`` env steps consumed by the learner per second, i.e.
  ``chunks_received * chunk_length / num_players`` per second. Checkpoints are
  disabled, so every slot plays the latest policy and collects data, and a
  full queue blocks the workers, so consumption equals production.

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

CONFIG = REPO_ROOT / "configs" / "examples" / "tic_tac_toe.yaml"
ENVS_PER_WORKER = 8
CHUNK_LENGTH = 32
BATCH_CHUNKS = 8

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
    from colosseum.core.config import ColosseumConfig, load_config

    data = load_config(CONFIG).model_dump()
    data["rollout"].update(
        num_workers=num_workers,
        envs_per_worker=ENVS_PER_WORKER,
        chunk_length=CHUNK_LENGTH,
        match_refresh_interval_sec=0.0,
    )
    data["learner"].update(batch_chunks=BATCH_CHUNKS, queue_size=4 * BATCH_CHUNKS, device="cpu")
    data["training"]["total_timesteps"] = 10**12  # stopped by the timer, not the budget
    data["self_play"]["checkpoint_interval"] = 10**9  # no checkpoints: every slot collects
    data["metrics"].update(use_wandb=False, log_interval=1)
    data["checkpoint"]["dir"] = checkpoint_dir
    return ColosseumConfig(**data)


def _rates(samples: list[tuple[float, int, int]], start: float, end: float, num_players: int) -> dict:
    window = [s for s in samples if start <= s[0] <= end]
    if len(window) < 2:
        return {"updates_per_s": 0.0, "env_steps_per_s": 0.0, "train_steps": 0}
    (t0, step0, chunks0), (t1, step1, chunks1) = window[0], window[-1]
    dt = max(t1 - t0, 1e-9)
    return {
        "updates_per_s": (step1 - step0) / dt,
        "env_steps_per_s": (chunks1 - chunks0) * CHUNK_LENGTH / num_players / dt,
        "train_steps": step1 - step0,
    }


def run_one(num_workers: int, duration: float, warmup: float) -> dict:
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
    result = _rates(_SAMPLES, started + warmup, started + warmup + duration, config.env.num_players)
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
    print(f"machine: {machine}", flush=True)
    print(
        f"config: {CONFIG.relative_to(REPO_ROOT)}, envs_per_worker={ENVS_PER_WORKER}, "
        f"chunk_length={CHUNK_LENGTH}, batch_chunks={BATCH_CHUNKS}, "
        f"warmup={args.warmup:.0f}s, duration={args.duration:.0f}s",
        flush=True,
    )

    results = []
    for n in args.workers:
        r = run_one(n, args.duration, args.warmup)
        results.append(r)
        print(
            f"workers={n}: updates/s={r['updates_per_s']:.2f} env_steps/s={r['env_steps_per_s']:.0f} "
            f"(train_steps in window={r['train_steps']})",
            flush=True,
        )

    print()
    print("| workers | updates/s | env steps/s |")
    print("|---|---|---|")
    for r in results:
        print(f"| {r['workers']} | {r['updates_per_s']:.2f} | {r['env_steps_per_s']:.0f} |")
    rates = [r["env_steps_per_s"] for r in results]
    monotonic = all(b > a for a, b in zip(rates, rates[1:]))
    print(f"\nenv steps/s increases monotonically with workers: {'yes' if monotonic else 'no'}")

    if args.json:
        Path(args.json).write_text(json.dumps({"machine": machine, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
