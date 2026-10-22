"""The node's books (SP5 spec 4.2, 4.3; rulings R8, R11; pre-flight rulings P-PF2-4, P-PF2-12): ``ResultBuffer``
(its unacked result batches) and ``CellTracker`` (its cells, grants waiting for their weights, the cache keys their
lineups reference). Pure: no I/O, no threads; times come from the ``clock`` given (monotonic). Owned by the node
agent's main loop (``colosseum.node.agent`` re-exports both)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["CellTracker", "ResultBuffer"]




class ResultBuffer:
    """The node's unacked result batches (spec 4.3). Results are gathered into batches of at most ``batch`` results or
    every ``flush_sec`` (``due``), numbered per session from 1; ``make(seq, results)`` builds the message once (a
    ``ResultBatch``) and ``size_of(message)`` its encoded size. A batch is kept until its ack, also while the hub is
    away; ``limit_bytes`` bounds the kept bytes: a new batch that does not fit is dropped whole and counted in
    ``dropped`` (one WARNING; overview deviation: the newest is dropped, the oldest keep being resent).
    ``unacked()`` lists the kept messages in seq order; ``resend_all()`` the same, marking them sent now (after every
    reconnect); ``overdue(after_sec)`` the kept batches sent ``after_sec`` ago or earlier, marking them sent again
    (P-PF2-4 a: a refused or lost batch is resent on a live session too); ``give_up()`` counts every kept and pending
    result as dropped (the stop, P-PF2-4 b)."""

    def __init__(self, *, batch: int, flush_sec: float, limit_bytes: int, make: Callable[[int, list], Any],
                 size_of: Callable[[Any], int], clock: Callable[[], float]) -> None:
        self._batch, self._flush_sec, self._limit = max(1, int(batch)), float(flush_sec), int(limit_bytes)
        self._make, self._size_of, self._clock = make, size_of, clock
        self._pending: list = []
        self._since = clock()
        self._next_seq = 1
        self._kept: dict[int, list] = {}          # seq -> [message, size, sent_at, results]
        self._bytes = 0
        self.dropped = 0
        self._warned = False

    def add(self, results: Iterable) -> list[Any]:
        """Gather ``results``; returns the messages cut now (to send)."""
        out = []
        for r in results:
            if not self._pending:
                self._since = self._clock()
            self._pending.append(r)
            if len(self._pending) >= self._batch:
                out.extend(self._cut())
        return out

    def due(self) -> list[Any]:
        """The pending results as one message once ``flush_sec`` passed since the first of them."""
        if self._pending and self._clock() - self._since >= self._flush_sec:
            return self._cut()
        return []

    def flush(self) -> list[Any]:
        return self._cut() if self._pending else []

    def _cut(self) -> list[Any]:
        batch, self._pending = self._pending, []
        message = self._make(self._next_seq, batch)
        size = int(self._size_of(message))
        if self._bytes + size > self._limit:
            self.dropped += len(batch)
            if not self._warned:
                self._warned = True
                logger.warning(f"node result buffer full ({self._bytes / 2**20:.1f} MB unacked); dropped {len(batch)} "
                               f"match results (dropped_match_results.node_buffer; further drops are only counted)")
            return []
        self._kept[self._next_seq] = [message, size, self._clock(), len(batch)]
        self._next_seq += 1
        self._bytes += size
        return [message]

    def ack(self, seqs: Iterable[int]) -> int:
        n = 0
        for seq in seqs:
            kept = self._kept.pop(int(seq), None)
            if kept is not None:
                self._bytes -= kept[1]
                n += 1
        return n

    def unacked(self) -> list[Any]:
        return [self._kept[seq][0] for seq in sorted(self._kept)]

    def resend_all(self) -> list[Any]:
        now = self._clock()
        for seq in self._kept:
            self._kept[seq][2] = now
        return self.unacked()

    def overdue(self, after_sec: float) -> list[Any]:
        now = self._clock()
        out = []
        for seq in sorted(self._kept):
            kept = self._kept[seq]
            if now - kept[2] >= after_sec:
                kept[2] = now
                out.append(kept[0])
        return out

    def give_up(self) -> int:
        """Count every unacked and pending result as dropped and forget them; returns how many (one WARNING)."""
        n = sum(k[3] for k in self._kept.values()) + len(self._pending)
        self._kept.clear()
        self._pending = []
        self._bytes = 0
        if n:
            self.dropped += n
            logger.warning(f"node: gave up {n} match results the hub did not acknowledge before the stop "
                           f"(dropped_match_results.node_buffer)")
        return n

    @property
    def pending(self) -> int:
        """Results gathered but not cut into a batch yet."""
        return len(self._pending)

    @property
    def last_acked(self) -> int:
        """Every seq below the smallest kept one is acked (``Hello.last_acked_seq``)."""
        return (min(self._kept) if self._kept else self._next_seq) - 1

    @property
    def kept_bytes(self) -> int:
        return self._bytes


@dataclass
class _Cell:
    worker: int
    env: int
    incarnation: int
    epoch: int = 0
    deadline: float | None = None       # node monotonic; None: no lineup yet / due now; inf: whole run
    waiting: Any = None                 # a grant waiting for its weights: (lineup, keys)
    waiting_since: float = 0.0
    ttl: float = 0.0                    # of the waiting grant
    keys: dict = field(default_factory=dict)   # seat -> key of the lineup the worker has or has been sent
    requested: bool = False             # in a LineupRequest without an answer yet
    retry_at: float = 0.0
    floor: int = 0                      # epochs <= floor belonged to a dead incarnation of this cell: never taken


class CellTracker:
    """The node's cells (spec 4.2, R11): which need a lineup (new incarnation, TTL expired, refused grant), grants
    that wait for their weights, and the cache keys every live or waiting lineup references (``referenced``: a key
    that is not referenced may leave the cache). ``retry_sec``: how long a requested cell without an answer, or one
    whose fetch failed, waits before it is requested again.

    Grants carry no incarnation. A new incarnation must never play an epoch a dead incarnation of its cell played or
    asked to renew (its match uids ``w_e_g<epoch>_ep<K>`` and its cell seed would repeat, spec 4.3 / 7.6), and the
    hub's answer to the dead incarnation's request may arrive at any time. So ``drop_worker`` keeps, per cell, the
    highest epoch the dropped incarnation held or was waiting on (a floor), and a later incarnation's cell refuses
    every grant with an epoch at or below it: epochs are unique and increasing within a run (spec 4.2) and the hub
    gives a new incarnation a new epoch, so the rule does not depend on timing. Besides, a cell takes a grant only
    while it is requested, or a late one of the epoch it has. As an optimisation a worker restarted while its cells
    were requested asks again only after ``retry_sec`` (the dead incarnation's answer usually comes first and is then
    refused by the "requested" rule instead of being taken and replaced)."""

    def __init__(self, *, retry_sec: float, clock: Callable[[], float]) -> None:
        self._cells: dict[tuple[int, int], _Cell] = {}
        self._retry = float(retry_sec)
        self._clock = clock
        self._hold: dict[int, float] = {}       # worker -> no request before (its dead incarnation's was unanswered)
        self._floors: dict[tuple[int, int], int] = {}   # cell -> highest epoch of its dead incarnations

    def add_worker(self, worker: int, envs: int, incarnation: int) -> None:
        hold = self._hold.pop(worker, 0.0)
        for e in range(envs):
            self._cells[(worker, e)] = _Cell(worker, e, incarnation, retry_at=hold,
                                             floor=self._floors.get((worker, e), 0))

    def drop_worker(self, worker: int) -> None:
        keys = [k for k in self._cells if k[0] == worker]
        if any(self._cells[k].requested for k in keys):
            self._hold[worker] = self._clock() + self._retry
        for key in keys:
            c = self._cells.pop(key)
            self._floors[key] = max(self._floors.get(key, 0), c.floor, c.epoch)

    def due(self) -> list[tuple[int, int, int, int]]:
        """Cells to request now, as ``(worker, env, epoch, incarnation)``; marks them requested."""
        now = self._clock()
        out = []
        for c in self._cells.values():
            if c.waiting is not None or now < c.retry_at:
                continue
            if c.deadline is None or now >= c.deadline:
                c.requested = True
                c.retry_at = now + self._retry
                out.append((c.worker, c.env, c.epoch, c.incarnation))
        return sorted(out)

    def granted(self, worker: int, env: int, epoch: int, ttl_sec: float, lineup: Any, keys: Mapping) -> bool:
        """A grant: kept as waiting until delivered; False for a cell this node no longer has (a dead incarnation),
        an epoch at or below the cell's floor, or a cell that asked for nothing (class docstring)."""
        c = self._cells.get((worker, env))
        if c is None or int(epoch) <= c.floor or not (c.requested or (c.epoch and int(epoch) == c.epoch)):
            return False
        now = self._clock()
        c.requested = False
        c.retry_at = 0.0                         # answered: the retry wait no longer holds it back, its TTL does
        c.epoch = int(epoch)
        c.deadline = float("inf") if ttl_sec <= 0 else now + float(ttl_sec)
        c.waiting = (lineup, dict(keys))
        c.waiting_since = now
        c.ttl = float(ttl_sec)
        return True

    def waiting(self) -> list[tuple[int, int, Any, dict]]:
        return [(c.worker, c.env, c.waiting[0], c.waiting[1]) for c in self._cells.values() if c.waiting is not None]

    def deliver(self, worker: int, env: int) -> tuple[Any, dict]:
        """The waiting grant's weights are cached: it goes to the worker now."""
        c = self._cells[(worker, env)]
        lineup, keys = c.waiting
        c.waiting = None
        c.keys = keys
        return lineup, keys

    def failed(self, worker: int, env: int) -> None:
        """Fetching the waiting grant's weights failed: drop it, keep the old lineup, ask again after ``retry_sec``."""
        c = self._cells.get((worker, env))
        if c is None:
            return
        c.waiting = None
        c.deadline = None
        c.retry_at = self._clock() + self._retry

    def stale_waiting(self, wait_sec: float) -> list[tuple[int, int, dict]]:
        """``(worker, env, keys)`` of the grants waiting ``max(ttl, wait_sec)`` or longer (P-PF2-12)."""
        now = self._clock()
        return sorted((c.worker, c.env, c.waiting[1]) for c in self._cells.values()
                      if c.waiting is not None and now - c.waiting_since >= max(c.ttl, wait_sec))

    def referenced(self) -> set:
        keys = set()
        for c in self._cells.values():
            keys.update(c.keys.values())
            if c.waiting is not None:
                keys.update(c.waiting[1].values())
        return keys

    def live(self) -> list[tuple[int, int, int, int]]:
        """``(worker, env, epoch, incarnation)`` of the cells with an epoch (``Hello.live_cells``)."""
        return sorted((c.worker, c.env, c.epoch, c.incarnation) for c in self._cells.values() if c.epoch)

    def epoch(self, worker: int, env: int) -> int:
        c = self._cells.get((worker, env))
        return 0 if c is None else c.epoch
