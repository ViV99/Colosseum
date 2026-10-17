"""Batching keeps the results queue's item count far below its cap (SP4 FX1): a burst of results five times the
cap is put without a drop and in order, while per-item puts of the same burst against the same cap overflow it.

Two spawned producers put the same 5 000 results at once, each on its own queue of the launcher's size
(``_RESULTS_QUEUE_SIZE`` = 1 000 items), and both queues are read by the launcher's ``QueueReader`` every 0.2 s, as
the monitor loop does. The control puts one result per item and must drop; the other goes through
``ResultBatcher`` (about 20 items) and must lose nothing.

What this does not show: the mechanism of the drops before FX1. The control drops because an instant burst of
5 000 items meets a 1 000-item cap, whatever the reader does (the old cap was 10 000, against which this burst does
not overflow). The pre-FX1 drops came from sustained production against a drain that runs once per 0.2 s pass
through a 64 KiB pipe; that evidence, and the check that batching removes them in real runs, are measurements:
``docs/benchmarks.md``, section «После SP4» (the drops diagnostic and the FX1 check)."""
from __future__ import annotations

import multiprocessing as mp
import queue
import time

from colosseum.core.ipc import QueueReader
from colosseum.core.types import MatchResult, SeatResult, TeamResult
from colosseum.launcher import _RESULTS_QUEUE_SIZE
from colosseum.worker.rollout_worker import ResultBatcher

N = 5000                 # five times the queue's 1000 slots
PASS_SEC = 0.2           # the monitor loop's sleep between passes
MODES = ("per_item", "batched")


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
    return len(items) if mode == "per_item" else sum(len(batch) for batch in items)


def test_a_burst_above_the_queue_cap_is_dropped_per_item_but_not_in_batches():
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
    assert _count(received["per_item"], "per_item") + dropped["per_item"] == N
    assert dropped["per_item"] > 0, "per-item puts of the burst no longer overflow a queue of this cap: raise N"
    batches = received["batched"]
    assert dropped["batched"] == 0 and all(isinstance(batch, list) for batch in batches)
    assert [r.uid for batch in batches for r in batch] == [f"RUN/train/w0_e0_ep{i}" for i in range(N)]
    assert len(batches) < _RESULTS_QUEUE_SIZE / 10
