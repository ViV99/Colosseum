"""Throughput benchmark: tic-tac-toe self-play training for a fixed wall-clock time.

For every worker count the real ``train`` path runs (SP5: a ``Hub`` and its local node, which spawns the
worker and learner processes; for a ``--tree`` without the hub, the SP4 ``Launcher``) with a pinned
tic-tac-toe workload (``_make_config``: every value that affects the workload is set here, not inherited
from the example yaml, so "before" and "after" runs measure the same work). After the run the script reads
the run's ``metrics.jsonl`` (T6.3): ``train`` records of ``agent_0`` (one per
learner train step, ``metrics.log_interval=1``) and ``system`` records (one per
``metrics.console_interval_sec``). Rates use the clock of ``--clock`` (below). After a
warm-up period the script measures, over ``--duration`` seconds:

- ``updates/s``          learner train steps per second (``train`` records);
- ``env_steps/s``        env steps per second from the global env-step counter
  (``env_steps`` of the ``system`` records), so turn-based envs (only the acting seat
  decides) are counted correctly;
- ``chunks/update``      chunks per train step (informational: since T2.4 every
  batch has exactly ``batch_chunks`` chunks);
- ``queue depth``        mean depth of the learner's trajectory queue over the
  ``system`` records in the window. At or above ``LEARNER_BOUND_QUEUE_FRACTION``
  of the queue capacity the workers wait for the learner (learner-bound); below
  it the learner waits for data (worker-bound). ``unknown`` where the platform
  cannot report queue sizes.

Checkpoints are disabled, so every slot plays the latest policy and collects data.

Clocks (``--clock``): ``counters`` (default since SP4) samples the hub's (launcher's) global env-step counter and
the newest ``train`` / ``system`` records on ``time.monotonic()`` in this process (``CounterSampler``);
``ts`` uses the records' wall-clock ``ts`` as before SP4. On machines whose wall clock jumps (WSL2: GPU
checks report §8) only ``counters`` works. ``--tree PATH`` measures another checkout (for example a ``main``
worktree) with its own ``_make_config``; run it with ``PYTHONPATH=PATH/src``.

A run whose measurement window is incomplete (too few learner or system samples,
or samples starting late / stopping early, e.g. because a learner died) is
reported as an error, gets no row, and makes the script exit non-zero.

Usage::

    .venv/bin/python scripts/bench_throughput.py                 # 1, 2, 4 workers, 60 s each
    .venv/bin/python scripts/bench_throughput.py --workers 1 2 --duration 30 --json out.json
    PYTHONPATH=../main/src .venv/bin/python scripts/bench_throughput.py --tree ../main   # another checkout
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
LAYOUT = "2p"  # tic-tac-toe has one layout; agent_0 plays both seats
QUEUE_SIZE = 4 * BATCH_CHUNKS  # learner.queue_size
AGENT_ID = "agent_0"
# Cadence of the ``system`` records in metrics.jsonl (measurement resolution, not workload).
SYSTEM_RECORD_INTERVAL_S = 1.0

# Window validation: minimum learner samples inside the window, and how far the
# first / last sample may be from the window's start / end.
MIN_WINDOW_SAMPLES = 5
WINDOW_EDGE_TOLERANCE_S = 5.0
# Mean learner queue depth at or above this fraction of QUEUE_SIZE counts as learner-bound.
LEARNER_BOUND_QUEUE_FRACTION = 0.5
CLOCKS = ("counters", "ts")
# Counter clock: how often the sampler reads the env-step counter and metrics.jsonl (seconds of time.monotonic()).
SAMPLE_INTERVAL_S = 0.25


class CounterSampler:
    """Measurement by counters on ``time.monotonic()`` (GPU checks report §8: WSL2 wall clocks jump by ~9 s,
    so windows over the records' ``ts`` fail). Every ``interval`` seconds it records:

    - ``system``: ``(t, env_steps, queue_depth)``: the hub's (launcher's) global env-step counter and the learner
      queue depth of the newest ``system`` record (-1 = unknown);
    - ``train``: ``(t, train_step, chunks_received)`` of the newest ``train`` record of ``agent_id``, once
      per new train step.

    The lists have the shapes ``load_samples`` returns, so ``validate_window`` / ``compute_rates`` apply.
    """

    def __init__(self, env_steps_fn, metrics_path: str | Path, agent_id: str = AGENT_ID,
                 interval: float = SAMPLE_INTERVAL_S, clock=time.monotonic) -> None:
        self.train: list[tuple[float, int, int]] = []
        self.system: list[tuple[float, int, int]] = []
        self._env_steps_fn = env_steps_fn
        self._path = Path(metrics_path)
        self._agent_id = agent_id
        self._interval = float(interval)
        self._clock = clock
        self._offset = 0
        self._last_train: tuple[int, int] | None = None
        self._depth = -1
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bench-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def sample(self) -> None:
        now = self._clock()
        self._read_metrics()
        self.system.append((now, int(self._env_steps_fn()), self._depth))
        if self._last_train is not None and (not self.train or self.train[-1][1] != self._last_train[0]):
            self.train.append((now, *self._last_train))

    def _read_metrics(self) -> None:
        """New complete lines of metrics.jsonl (a partial last line waits for the next sample)."""
        try:
            with self._path.open("rb") as fh:
                fh.seek(self._offset)
                chunk = fh.read()
        except OSError:
            return
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        self._offset += end + 1
        for line in chunk[:end].split(b"\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["kind"] == "train" and r["agent"] == self._agent_id:
                self._last_train = (int(r["train_step"]), int(r.get("chunks_received") or 0))
            elif r["kind"] == "system":
                self._depth = int(r["queue_depths"].get(self._agent_id, -1))

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.sample()


def load_tree_config(tree: str | Path):
    """``_make_config`` of the benchmark script of another checkout (``--tree``, e.g. a ``main`` worktree),
    whose ``src`` must be the ``colosseum`` this process imported (``PYTHONPATH=<tree>/src``)."""
    import importlib.util

    import colosseum

    tree = Path(tree).resolve()
    if not Path(colosseum.__file__).resolve().is_relative_to(tree / "src"):
        raise SystemExit(f"--tree {tree}: colosseum is imported from {Path(colosseum.__file__).parent}; run with "
                         f"PYTHONPATH={tree / 'src'} so the measured code is the tree's")
    spec = importlib.util.spec_from_file_location("bench_throughput_tree", tree / "scripts" / "bench_throughput.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))         # the tree's examples.* for this process and spawned children
    return module._make_config


def load_samples(metrics_path: str | Path, agent_id: str = AGENT_ID) -> tuple[list, list]:
    """``metrics.jsonl`` -> (train, system) samples of ``agent_id``.

    train:  ``(ts, train_step, chunks_received)`` per ``train`` record;
    system: ``(ts, env_steps, queue_depth)`` per ``system`` record (depth -1 = unknown).
    """
    train, system = [], []
    for line in Path(metrics_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["kind"] == "train" and r["agent"] == agent_id:
            train.append((float(r["ts"]), int(r["train_step"]), int(r.get("chunks_received") or 0)))
        elif r["kind"] == "system":
            system.append((float(r["ts"]), int(r["env_steps"]), int(r["queue_depths"].get(agent_id, -1))))
    return train, system


def _make_config(num_workers: int, run_parent: str):
    """The pinned benchmark workload. Every workload-relevant value is explicit.

    The run (logs, resolved config, checkpoints) goes to ``<run_parent>/bench-w<num_workers>``.
    """
    from colosseum.core.config import ColosseumConfig

    return ColosseumConfig(
        env={
            "env_class": "examples.tic_tac_toe.game.TicTacToeGame",
            "kwargs": {},
            "max_idle_steps": 1000,
        },
        networks={
            "encoder_class": "examples.tic_tac_toe.models.TicTacToeEncoder",
            "policy_class": "examples.tic_tac_toe.models.TicTacToePolicy",
            "value_class": "examples.tic_tac_toe.models.TicTacToeValue",
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
            "ratio_mode": "auto",
            "unit_trace": "auto",
            "entropy_reduction": "auto",
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
            "queue_size": QUEUE_SIZE,
            "batch_chunks": BATCH_CHUNKS,
            "weight_push_interval": 5,
            "pin_memory": False,
            "torch_threads": None,  # auto: (cpu_count - num_workers * 1) // 1
            # SP5: no lag cut (default), the recovery cadence and placement of the one machine's learner pinned
            "max_policy_lag": None,
            "recovery_interval_sec": 300.0,
            "placement": {"node": None, "tags": [], "strict": False},
        },
        training={
            "total_timesteps": 10**12,  # stopped by the timer, not the budget
            "seed": 0,
        },
        matchmaking={
            "opponents": {"latest": 1.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.0},  # SP2 workload: latest only
            "anchors": [],
            "pfsp": {"weighting": "hard", "exponent": 1.0, "halflife_games": 200.0},
            "layouts": {LAYOUT: 1.0},
            "teammates": "self",
            "teammate_self_prob": 0.5,
            "shuffle_seats": True,  # one agent, every seat latest+collect: shuffling is a no-op
            "matchmaker_class": None,
        },
        checkpoint={
            "interval": 10**9,  # no checkpoints: every seat collects
            "keep_last": 10,
            "keep_every": 0,
            "keep_top_k": 0,
            "top_k_quantile": 0.1,
            "top_k_min_games": 200,
            "max_snapshots": 100,
            "save_optimizer": True,
        },
        init={"from": None, "strict": True, "critic_warmup_steps": 0},
        kickstart={"teacher": None, "lambda": 1.0, "decay_steps": 50_000, "kl": "forward"},
        run={"dir": run_parent, "name": f"bench-w{num_workers}"},
        metrics={"use_wandb": False, "log_interval": 1, "console_interval_sec": SYSTEM_RECORD_INTERVAL_S},
        # SP4 criterion 3.6: the journal is on while throughput is measured.
        match_log={"enabled": True, "detail_window_mb": 512.0, "min_free_disk_mb": 2048.0,
                   "compact_interval_sec": 60.0},
        ratings={"anchor": None, "refit_interval_sec": 60.0, "bootstrap_samples": 200},
        evaluation={"enabled": False, "agents": None, "field": None, "layouts": None, "every": 1, "blocks": 25,
                    "num_envs": 8, "deterministic": False, "max_backlog": 4},   # criterion 3.6: pulse off
        # SP5 (spec 9.1): every key pinned. The hub and the local node talk over loopback; heartbeat_sec 0.25: the
        # hub's env-step counter (sampled by CounterSampler) moves in heartbeats, quantized < 1 % of the window (T2.5).
        distributed={
            "port": 47100, "data_port": 47102, "insecure": False, "heartbeat_sec": 0.25, "node_lost_after_sec": 10.0,
            "learner_relocate_after_sec": 120.0, "hub_lost_timeout_sec": 1800.0, "hub_lost_collect_sec": 300.0,
            "shutdown_grace_sec": 20.0, "connect_timeout_sec": 5.0, "block_lease_sec": 600.0, "relay": True,
            "results_batch": 256, "results_flush_sec": 0.25, "node_result_buffer_mb": 256.0, "max_message_mb": 64,
            "bundle": {"include": [], "exclude": [], "max_file_mb": 10.0}, "requirements": None,
        },
    )


def _window(samples: list[tuple[float, int, int]], start: float, end: float) -> list[tuple[float, int, int]]:
    return [s for s in samples if start <= s[0] <= end]


def validate_window(samples: list[tuple[float, int, int]], start: float, end: float,
                    source: str = "learner") -> str | None:
    """Return an error message if the measurement window of ``source`` samples is incomplete, else None."""
    window = _window(samples, start, end)
    if len(window) < MIN_WINDOW_SAMPLES:
        return f"only {len(window)} {source} samples in the window (need >= {MIN_WINDOW_SAMPLES})"
    first, last = window[0][0], window[-1][0]
    if first - start > WINDOW_EDGE_TOLERANCE_S:
        return f"first sample {first - start:.1f}s after the window start ({source} started late or stalled)"
    if end - last > WINDOW_EDGE_TOLERANCE_S:
        return f"last sample {end - last:.1f}s before the window end ({source} stopped early or stalled)"
    return None


def classify_bound(queue_depth_mean: float | None, queue_size: int = QUEUE_SIZE) -> str:
    """'learner' (queue mostly full), 'worker' (queue mostly empty) or 'unknown' (no depth)."""
    if queue_depth_mean is None:
        return "unknown"
    return "learner" if queue_depth_mean >= LEARNER_BOUND_QUEUE_FRACTION * queue_size else "worker"


def compute_rates(train: list[tuple[float, int, int]], system: list[tuple[float, int, int]],
                  start: float, end: float) -> dict:
    """Rates over the window; call only after ``validate_window`` returned None for both sample lists."""
    window = _window(train, start, end)
    (t0, step0, chunks0), (t1, step1, chunks1) = window[0], window[-1]
    steps = step1 - step0
    chunks = chunks1 - chunks0
    sys_window = _window(system, start, end)
    (s0, env0, _), (s1, env1, _) = sys_window[0], sys_window[-1]
    depths = [d for _, _, d in sys_window]
    queue_depth_mean = sum(depths) / len(depths) if all(d >= 0 for d in depths) else None
    return {
        "updates_per_s": steps / max(t1 - t0, 1e-9),
        "env_steps_per_s": (env1 - env0) / max(s1 - s0, 1e-9),
        "chunks_per_update": chunks / steps if steps else 0.0,
        "queue_depth_mean": queue_depth_mean,
        "bound": classify_bound(queue_depth_mean),
        "train_steps": steps,
    }


def _runner(config, run_dir):
    """``(runner, stop)``: SP5's ``Hub`` with its local node (``train``'s path; ``stop`` = ``request_stop``), or, for a
    measured tree without ``colosseum.hub.hub`` (``--tree`` of ``main``), the SP4 ``Launcher`` (``stop`` sets its
    stop event). Both expose ``env_steps_done`` (the global budget counter) and ``launch()``."""
    try:
        from colosseum.hub.hub import Hub, HubOptions, LocalNodeOptions
    except ImportError:                       # documented --tree fallback: the measured checkout predates the hub
        from colosseum.launcher import Launcher

        launcher = Launcher(config, run_dir)
        return launcher, launcher._stop_event.set
    hub = Hub(config, run_dir, options=HubOptions(mode="train", local_node=LocalNodeOptions.from_config(config,
                                                                                                        run_dir)))
    return hub, hub.request_stop


def run_one(num_workers: int, duration: float, warmup: float, *, clock: str = "counters",
            make_config=None) -> dict:
    """Run one configuration; raises RuntimeError if the window is incomplete."""
    from colosseum.core.run_dir import RunDir

    if clock not in CLOCKS:
        raise ValueError(f"clock {clock!r} is not one of {CLOCKS}")
    make_config = make_config or _make_config
    with tempfile.TemporaryDirectory(prefix="bench-") as tmp:
        config = make_config(num_workers, tmp)
        run_dir = RunDir.create(config)
        runner, stop = _runner(config, run_dir)
        sampler = CounterSampler(lambda: runner.env_steps_done, run_dir.metrics_path) if clock == "counters" else None
        started = time.monotonic() if sampler is not None else time.time()  # ts mode: metrics.jsonl ``ts`` is Unix time
        timer = threading.Timer(warmup + duration, stop)
        timer.start()
        if sampler is not None:
            sampler.start()
        try:
            runner.launch()
        finally:
            timer.cancel()
            if sampler is not None:
                sampler.stop()
        train, system = (sampler.train, sampler.system) if sampler is not None else load_samples(run_dir.metrics_path)
    start, end = started + warmup, started + warmup + duration
    error = validate_window(train, start, end, "learner") or validate_window(system, start, end, "system")
    if error is not None:
        raise RuntimeError(error)
    result = compute_rates(train, system, start, end)
    result["workers"] = num_workers
    return result


def workload_summary(warmup: float, duration: float) -> dict:
    """The workload as printed and written to the JSON output (comparable across runs)."""
    return {
        "env": "tic_tac_toe",
        "layout": LAYOUT,
        "envs_per_worker": ENVS_PER_WORKER,
        "chunk_length": CHUNK_LENGTH,
        "batch_chunks": BATCH_CHUNKS,
        "queue_size": QUEUE_SIZE,
        "warmup_s": warmup,
        "duration_s": duration,
    }


def _fmt_depth(result: dict) -> str:
    depth = result["queue_depth_mean"]
    return "-" if depth is None else f"{depth:.1f}/{QUEUE_SIZE}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--duration", type=float, default=60.0, help="measured seconds per run")
    parser.add_argument("--warmup", type=float, default=15.0, help="ignored seconds after start")
    parser.add_argument("--json", type=str, default=None, help="also write results to this JSON file")
    parser.add_argument("--clock", choices=CLOCKS, default="counters",
                        help="counters: time.monotonic() samples of the counters (default); ts: records' ts")
    parser.add_argument("--tree", type=str, default=None,
                        help="measure another checkout with its own _make_config (run with PYTHONPATH=<tree>/src)")
    args = parser.parse_args()
    make_config = load_tree_config(args.tree) if args.tree else _make_config
    tree = str(Path(args.tree).resolve()) if args.tree else str(REPO_ROOT)

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
    workload = workload_summary(args.warmup, args.duration)
    print(f"machine: {machine}", flush=True)
    print(f"workload: {workload}", flush=True)
    print(f"clock: {args.clock}; tree: {tree}", flush=True)

    results = []
    failed = []
    for n in args.workers:
        try:
            r = run_one(n, args.duration, args.warmup, clock=args.clock, make_config=make_config)
        except RuntimeError as exc:
            failed.append(n)
            print(f"ERROR workers={n}: invalid measurement: {exc}; no row reported", file=sys.stderr, flush=True)
            continue
        results.append(r)
        print(
            f"workers={n}: updates/s={r['updates_per_s']:.2f} env_steps/s={r['env_steps_per_s']:.0f} "
            f"chunks/update={r['chunks_per_update']:.2f} queue depth={_fmt_depth(r)} ({r['bound']}-bound, "
            f"train_steps in window={r['train_steps']})",
            flush=True,
        )

    print()
    print("| workers | updates/s | env steps/s | chunks/update | queue depth | bound |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['workers']} | {r['updates_per_s']:.2f} | {r['env_steps_per_s']:.0f} "
            f"| {r['chunks_per_update']:.2f} | {_fmt_depth(r)} | {r['bound']} |"
        )

    if failed:
        print(f"\nERROR: invalid measurement for workers={failed}; no JSON written", file=sys.stderr)
        return 1

    rates = [r["env_steps_per_s"] for r in results]
    monotonic = all(b > a for a, b in zip(rates, rates[1:]))
    print(f"\nenv steps/s increases monotonically with workers: {'yes' if monotonic else 'no'}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"machine": machine, "workload": workload, "clock": args.clock, "tree": tree,
                        "results": results}, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
