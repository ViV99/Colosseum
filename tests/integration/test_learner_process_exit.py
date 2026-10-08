"""Learner exit vs. weight queues holding large numpy payloads (T2.2).

Numpy weight payloads are pickled in full (unlike torch tensors, which travel as
small fd handles), so a payload larger than the pipe buffer keeps the queue's
feeder thread blocked until a worker reads it:

- workers stopped (``stop_event`` set): nobody will read; the learner must not
  wait for them at exit;
- learner stopped by itself (``consumed_samples`` budget, no shared counter)
  while workers still run: the feeder must flush the whole message, or a live
  worker blocks forever on a truncated one.
"""

import multiprocessing as mp
import threading
import time

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig, LearnerConfig
from colosseum.launcher import _WEIGHT_QUEUE_SIZE
from colosseum.learner.learner import learner_process
from dataflow_helpers import TinyModel, chunk_payload

BIG_NUM_ACTIONS = 8192  # policy head 4 x 8192 floats: ~160 KB per weight payload
ONE_BATCH = 8  # total_timesteps: one batch of 2 chunks x T=4, then the learner stops itself


def _big_appo() -> APPO:
    return APPO(TinyModel(num_actions=BIG_NUM_ACTIONS), AlgorithmConfig(), device="cpu")


def _start_learner(ctx, weights, stop, total_timesteps: int):
    traj = ctx.Queue()
    for version in range(2):
        traj.put(chunk_payload(T=4, version=version))
    proc = ctx.Process(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=_big_appo, trajectory_queue=traj,
        weight_queues=[weights],
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
    unread_weights = ctx.Queue(maxsize=_WEIGHT_QUEUE_SIZE)
    stop = ctx.Event()
    stop.set()  # workers are stopping: nobody reads the weight queue any more
    proc, traj = _start_learner(ctx, unread_weights, stop, total_timesteps=ONE_BATCH)
    try:
        proc.join(timeout=60)
        assert not proc.is_alive(), "learner hung at exit flushing an unread weight payload"
        assert proc.exitcode == 0
    finally:
        _cleanup(proc, unread_weights, traj)


def test_live_slow_reader_gets_final_weights_after_learner_budget_exit():
    """Budget reached, stop_event NOT set: the last payload must arrive whole."""
    ctx = mp.get_context("spawn")
    weights = ctx.Queue(maxsize=_WEIGHT_QUEUE_SIZE)
    stop = ctx.Event()
    received: list[int] = []

    def slow_reader() -> None:  # a worker syncing weights every second
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
    """A worker reads weights only between chunk puts (like RolloutLoop).

    When the learner reaches its budget it stops consuming chunks, so the worker
    blocks on the full trajectory queue and never reads the pending weight
    payload; the learner must keep draining chunks while its weights flush.
    """
    ctx = mp.get_context("spawn")
    traj = ctx.Queue(maxsize=2)
    weights = ctx.Queue(maxsize=_WEIGHT_QUEUE_SIZE)
    stop = ctx.Event()
    proc = ctx.Process(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=_big_appo, trajectory_queue=traj,
        weight_queues=[weights],
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"),
        stop_event=stop, total_timesteps=ONE_BATCH,
    ))
    received: list[int] = []
    worker_done = threading.Event()

    def worker() -> None:
        while not worker_done.is_set():
            try:
                traj.put(chunk_payload(T=4), timeout=0.5)
            except Exception:  # queue.Full: keep retrying, like send_chunk
                continue
            while True:  # weight sync after each step
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


def test_weight_flush_timeout_scales_with_weight_sync_interval():
    """A live worker syncs every weight_sync_interval; the exit flush must outlast that."""
    from colosseum.learner.learner import _weight_flush_timeout

    assert _weight_flush_timeout(0.5) == 60.0
    assert _weight_flush_timeout(5.0) == 60.0
    assert _weight_flush_timeout(60.0) == 180.0
    assert _weight_flush_timeout(90.0) == 270.0
