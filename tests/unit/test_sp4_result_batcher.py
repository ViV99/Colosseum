"""Worker-side result batches (SP4 FX1): a worker puts its match results on the results queue as lists, one per
RESULTS_BATCH_SEC or RESULTS_BATCH_MAX results; a full queue drops the whole batch, counts every result in it and
warns once; the worker never blocks."""
from __future__ import annotations

import logging
import queue

from colosseum.core.types import MatchResult
from colosseum.worker.rollout_worker import RESULTS_BATCH_MAX, RESULTS_BATCH_SEC, ResultBatcher

LOGGER = "colosseum.worker.rollout_worker"


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0                                              # exact sums of 0.05 steps

    def __call__(self) -> float:
        return self.t


def _result(i: int) -> MatchResult:
    return MatchResult(f"w0_e0_ep{i}", "solo", "score", [], [], 3)


def _batcher(q, clock=None) -> tuple[ResultBatcher, _Clock]:
    clock = clock or _Clock()
    return ResultBatcher(q, worker_id=7, clock=clock), clock


def _ids(batch: list[MatchResult]) -> list[str]:
    return [r.match_id for r in batch]


def test_the_constants_are_the_controller_ruling():
    assert (RESULTS_BATCH_SEC, RESULTS_BATCH_MAX) == (0.1, 256)


def test_a_batch_is_put_when_the_size_cap_is_reached():
    q: queue.Queue = queue.Queue()
    batcher, _ = _batcher(q)
    for i in range(RESULTS_BATCH_MAX - 1):
        batcher.add(_result(i))
    assert q.empty()                                              # the clock never moved: only the cap can put
    batcher.add(_result(RESULTS_BATCH_MAX - 1))
    batch = q.get_nowait()
    assert isinstance(batch, list) and _ids(batch) == [f"w0_e0_ep{i}" for i in range(RESULTS_BATCH_MAX)]
    assert q.empty() and batcher.dropped == 0
    batcher.flush()
    assert q.empty()                                              # nothing pending: no empty list is put


def test_a_due_batch_is_put_without_a_new_result():
    q: queue.Queue = queue.Queue()
    batcher, clock = _batcher(q)
    batcher.add(_result(0))
    batcher.add(_result(1))
    clock.t += RESULTS_BATCH_SEC / 2
    batcher.flush_if_due()
    assert q.empty()
    clock.t += RESULTS_BATCH_SEC / 2                              # 0.1 s since the batcher's start: due
    batcher.flush_if_due()
    assert _ids(q.get_nowait()) == ["w0_e0_ep0", "w0_e0_ep1"]
    clock.t += 10.0
    batcher.flush_if_due()                                        # due, but nothing pending
    assert q.empty()


def test_the_age_counts_from_the_last_put():
    q: queue.Queue = queue.Queue()
    batcher, clock = _batcher(q)
    batcher.add(_result(0))
    clock.t += RESULTS_BATCH_SEC
    batcher.flush_if_due()                                        # put at t0 + 0.1
    batcher.add(_result(1))
    clock.t += RESULTS_BATCH_SEC / 2                              # 0.05 s after the last put: not due
    batcher.flush_if_due()
    assert _ids(q.get_nowait()) == ["w0_e0_ep0"] and q.empty()
    clock.t += RESULTS_BATCH_SEC / 2
    batcher.flush_if_due()
    assert _ids(q.get_nowait()) == ["w0_e0_ep1"]


def test_flush_sends_the_remainder_in_order():
    q: queue.Queue = queue.Queue()
    batcher, _ = _batcher(q)
    for i in range(RESULTS_BATCH_MAX + 3):
        batcher.add(_result(i))
    batcher.flush()
    first, rest = q.get_nowait(), q.get_nowait()
    assert len(first) == RESULTS_BATCH_MAX and _ids(rest) == [f"w0_e0_ep{RESULTS_BATCH_MAX + i}" for i in range(3)]
    assert q.empty()


def test_a_full_queue_drops_the_whole_batch_counts_every_result_and_warns_once(caplog):
    q: queue.Queue = queue.Queue(maxsize=1)
    batcher, _ = _batcher(q)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for i in range(5):
            batcher.add(_result(i))
        batcher.flush()                                           # fills the queue
        for i in range(5, 8):
            batcher.add(_result(i))
        batcher.flush()                                           # dropped: 3
        for i in range(8, 8 + RESULTS_BATCH_MAX):
            batcher.add(_result(i))                               # the cap puts, dropped: 256 more
        batcher.add(_result(999))
        batcher.flush()                                           # dropped: 1 more
    assert batcher.dropped == 3 + RESULTS_BATCH_MAX + 1
    assert _ids(q.get_nowait()) == [f"w0_e0_ep{i}" for i in range(5)] and q.empty()
    warnings = [r for r in caplog.records if "results queue is full" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING   # the first drop only
    message = warnings[0].getMessage()
    assert "Worker 7" in message and "3 match results" in message and "dropped_match_results" in message
