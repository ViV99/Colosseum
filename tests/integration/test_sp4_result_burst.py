"""A burst of match results larger than one pipe-full is not dropped (SP4 FX1; the cause measured by the drops
diagnostic): the main process reads the results queue in one short burst per 0.2 s monitor pass, and the queue's
64 KiB pipe holds only about 100-140 results. Results put one by one pile up in the producer's feeder buffer, which
the queue's semaphore counts, until ``put_nowait`` raises ``Full``. Batches (``ResultBatcher``) keep the queue's
item count far below its cap.

Two spawned producers put the same burst at once, each on its own queue of the launcher's size, and both queues are
read by the launcher's ``QueueReader`` every 0.2 s, as the monitor loop does. The control puts one result per item,
as workers did before FX1, and must overflow: that shows the burst is one the old path drops. The other goes
through ``ResultBatcher`` and must lose nothing and keep the order."""
from __future__ import annotations

import multiprocessing as mp
import queue
import time

from colosseum.core.ipc import QueueReader
from colosseum.core.types import MatchResult, SeatResult, TeamResult
from colosseum.launcher import _RESULTS_QUEUE_SIZE
from colosseum.worker.rollout_worker import ResultBatcher

N = 5000                 # about 40 pipe-fulls of results, five times the queue's 1000 slots
PASS_SEC = 0.2           # the monitor loop's sleep between passes
MODES = ("one_by_one", "batched")


def _result(i: int) -> MatchResult:
    return MatchResult(f"w0_e{i % 16}_ep{i}", "2p", "wdl",
                       [SeatResult(0, "player", 0, "agent_0", "latest", 1.0, version_start=i, version_end=i + 1),
                        SeatResult(1, "player", 1, "agent_0", "latest", -1.0, version_start=i, version_end=i + 1)],
                       [TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], 9, uid=f"RUN/train/w0_e0_ep{i}")


def _produce(results_queue, done, batched: bool) -> None:
    """Child: put N results as fast as possible, then report the drops; the queue's feeder writes the rest while
    the parent reads (exit joins it)."""
    results = [_result(i) for i in range(N)]       # built first: the burst is the puts only
    dropped = 0
    if batched:
        batcher = ResultBatcher(results_queue, worker_id=0)
        for result in results:
            batcher.add(result)
            batcher.flush_if_due()                 # the worker checks once per loop step
        batcher.flush()
        dropped = batcher.dropped
    else:
        for result in results:
            try:
                results_queue.put_nowait(result)
            except queue.Full:
                dropped += 1
    done.put(dropped)


def _count(items: list, mode: str) -> int:
    return len(items) if mode == "one_by_one" else sum(len(batch) for batch in items)


def test_a_burst_larger_than_a_pipe_full_is_dropped_one_by_one_but_not_in_batches():
    ctx = mp.get_context("spawn")
    queues = {mode: ctx.Queue(maxsize=_RESULTS_QUEUE_SIZE) for mode in MODES}
    done = {mode: ctx.Queue() for mode in MODES}
    readers = {mode: QueueReader(queues[mode], mode, lambda: []) for mode in MODES}
    procs = {mode: ctx.Process(target=_produce, args=(queues[mode], done[mode], mode == "batched"), daemon=True)
             for mode in MODES}
    received: dict[str, list] = {mode: [] for mode in MODES}
    dropped: dict[str, int] = {}
    for proc in procs.values():
        proc.start()
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            for mode in MODES:
                received[mode].extend(readers[mode].drain())
                if mode not in dropped:
                    try:
                        dropped[mode] = done[mode].get_nowait()
                    except queue.Empty:
                        pass
            if all(mode in dropped and _count(received[mode], mode) + dropped[mode] == N for mode in MODES):
                break
            # Once both producers reported, every drop is decided (put_nowait fails at once): the rest is read
            # without the pass delay, which keeps the test fast.
            time.sleep(PASS_SEC if len(dropped) < len(MODES) else 0.01)
    finally:
        for proc in procs.values():
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
    assert all(proc.exitcode == 0 for proc in procs.values()), {m: p.exitcode for m, p in procs.items()}
    assert set(dropped) == set(MODES), (dropped, {m: _count(received[m], m) for m in MODES})
    assert _count(received["one_by_one"], "one_by_one") + dropped["one_by_one"] == N
    assert dropped["one_by_one"] > 0, "the burst no longer overflows a per-result queue of this size: raise N"
    batches = received["batched"]
    assert dropped["batched"] == 0 and all(isinstance(batch, list) for batch in batches)
    assert [r.uid for batch in batches for r in batch] == [f"RUN/train/w0_e0_ep{i}" for i in range(N)]
    assert len(batches) < _RESULTS_QUEUE_SIZE / 10
