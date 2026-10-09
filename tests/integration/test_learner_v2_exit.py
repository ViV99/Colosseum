"""SP2 learner exit vs. weight queues holding large numpy payloads (T4.4; port of SP1's T2.2 tests).

Numpy weight payloads are pickled in full, so a payload larger than the pipe buffer keeps
the queue's feeder thread blocked until a worker reads it:
- workers stopped (``stop_event`` set): the learner must not wait for them at exit;
- learner stopped by itself (``consumed_samples`` budget) while workers still run: the
  feeder must flush the whole message, or a live worker blocks forever on a truncated one.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time

from colosseum.core.config import LearnerConfig
from colosseum.learner.learner import learner_process
from game_helpers import chunk_v2_payload, learner_appo

BIG_NUM_ACTIONS = 8192      # policy head 16 x 8192 floats: ~0.5 MB per weight payload
WEIGHT_QUEUE_SIZE = 1       # newest-wins mailbox per (agent, worker), as the launcher uses
ONE_BATCH = 6               # total_timesteps: one batch of 2 chunks x 3 ACT slots, then the learner stops


def _big_appo():
    return learner_appo(num_actions=BIG_NUM_ACTIONS)


def _payload(version: int = 0) -> dict:
    return chunk_v2_payload(S=4, version=version, num_actions=BIG_NUM_ACTIONS)


def _start_learner(ctx, weights, stop, total_timesteps: int):
    traj = ctx.Queue()
    for version in range(2):
        traj.put(_payload(version))
    proc = ctx.Process(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=_big_appo, trajectory_queue=traj, weight_queues=[weights],
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"),
        stop_event=stop, total_timesteps=total_timesteps,
    ))
    proc.start()
    return proc, traj


def _cleanup(proc, *queues) -> None:
    if proc.is_alive():
        proc.kill()
        proc.join()
    for q in queues:
        q.cancel_join_thread()


def test_learner_exits_when_workers_stopped_and_weights_are_unread():
    ctx = mp.get_context("spawn")
    unread = ctx.Queue(maxsize=WEIGHT_QUEUE_SIZE)
    stop = ctx.Event()
    stop.set()
    proc, traj = _start_learner(ctx, unread, stop, total_timesteps=ONE_BATCH)
    try:
        proc.join(timeout=60)
        assert not proc.is_alive(), "learner hung at exit flushing an unread weight payload"
        assert proc.exitcode == 0
    finally:
        _cleanup(proc, unread, traj)


def test_live_slow_reader_gets_final_weights_after_learner_budget_exit():
    ctx = mp.get_context("spawn")
    weights = ctx.Queue(maxsize=WEIGHT_QUEUE_SIZE)
    stop = ctx.Event()
    received: list[int] = []

    def slow_reader() -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and 1 not in received:
            try:
                received.append(weights.get(timeout=1.0).policy_version)
            except Exception:  # queue.Empty
                continue
            time.sleep(1.0)

    proc, traj = _start_learner(ctx, weights, stop, total_timesteps=ONE_BATCH)
    reader = threading.Thread(target=slow_reader, daemon=True)
    reader.start()
    try:
        reader.join(timeout=90)
        assert not reader.is_alive(), f"reader blocked on a truncated weight payload (got {received})"
        assert received[-1] == 1, f"final weights (version 1) never arrived: {received}"
        proc.join(timeout=30)
        assert proc.exitcode == 0
    finally:
        _cleanup(proc, weights, traj)


def test_learner_budget_exit_does_not_deadlock_with_a_worker_blocked_on_chunks():
    ctx = mp.get_context("spawn")
    traj = ctx.Queue(maxsize=2)
    weights = ctx.Queue(maxsize=WEIGHT_QUEUE_SIZE)
    stop = ctx.Event()
    proc = ctx.Process(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=_big_appo, trajectory_queue=traj, weight_queues=[weights],
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"),
        stop_event=stop, total_timesteps=ONE_BATCH,
    ))
    received: list[int] = []
    worker_done = threading.Event()

    def worker() -> None:
        while not worker_done.is_set():
            try:
                traj.put(_payload(), timeout=0.5)
            except Exception:  # queue.Full: keep retrying, like send_chunk
                continue
            while True:
                try:
                    received.append(weights.get_nowait().policy_version)
                except Exception:  # queue.Empty
                    break

    proc.start()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        proc.join(timeout=60)
        assert not proc.is_alive(), f"learner and worker deadlocked (worker got versions {received})"
        assert proc.exitcode == 0
    finally:
        worker_done.set()
        thread.join(timeout=10)
        _cleanup(proc, weights, traj)
