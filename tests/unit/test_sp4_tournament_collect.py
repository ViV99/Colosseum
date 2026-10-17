"""The tournament engine's result loop (``engine._collect``, SP4 final fix wave B-M2) on fake queues and a fake
supervisor: a dead worker is noticed at the next message, the other workers are stopped at once, and the blocks
they finish meanwhile are still journaled, for at most ``STOP_GRACE_SEC``."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import pytest

from colosseum.arena.tournament import engine
from colosseum.arena.tournament.engine import TournamentError
from colosseum.utils.process import ChildFailure

GAP = object()       # one empty read of the results queue


class _Results:
    """The results queue: ``items`` in order (``GAP``: one ``queue.Empty``), then ``stream`` (if set) again and
    again for ``stream_sec`` seconds, then empty."""

    def __init__(self, items, stream=None, stream_sec=0.0) -> None:
        self.items, self.stream, self.stream_end = list(items), stream, time.monotonic() + stream_sec

    def get(self, timeout):
        item = self.items.pop(0) if self.items else (self.stream if time.monotonic() < self.stream_end else None)
        if item is None or item is GAP:
            time.sleep(min(timeout, 0.01))
            raise queue.Empty
        return item


class _Supervisor:
    """Worker 0 is dead (killed: no typed error); worker 1 keeps running."""

    def first_failure(self, nonzero_only=False):
        return ChildFailure("tournament-worker-0", -9, Path("logs/tournament-worker-0.log"))

    def alive(self):
        return ["tournament-worker-1"]

    def log_path(self, name):
        return Path(f"logs/{name}.log")


def test_a_dead_worker_stops_the_others_first_and_their_finished_blocks_are_still_journaled():
    stop = threading.Event()
    pending = {0, 1, 2}
    handled: list[tuple[int, bool]] = []      # (task, was the stop already set)

    def handle(msg):
        handled.append((msg[1], stop.is_set()))
        pending.discard(msg[1])

    results = _Results([("done", 1, {}), GAP, ("done", 2, {})])
    with pytest.raises(TournamentError, match="tournament-worker-0 died"):
        engine._collect(results, _Supervisor(), stop, pending, handle, None)
    assert handled == [(1, True), (2, True)]          # noticed before the first message; both peers journaled
    assert pending == {0}


def test_the_search_for_the_dead_workers_error_ends_after_the_grace_period(monkeypatch):
    monkeypatch.setattr(engine, "STOP_GRACE_SEC", 0.3)
    stop = threading.Event()
    handled: list[int] = []
    results = _Results([], stream=("done", 1, {}), stream_sec=4.0)   # a peer keeps streaming (one task, again)
    started = time.monotonic()
    with pytest.raises(TournamentError, match="tournament-worker-0 died"):
        engine._collect(results, _Supervisor(), stop, {0, 1}, lambda msg: handled.append(msg[1]), None)
    assert time.monotonic() - started < 1.5 and handled and stop.is_set()


def test_a_typed_error_found_after_the_failure_keeps_its_class():
    from colosseum.core.errors import PlayerError

    stop = threading.Event()
    handled: list[int] = []
    error = ("error", 0, 0, "PlayerError", "tournament, worker 0, env 0, seat 1: bot failure", "match m of block b")
    results = _Results([("done", 1, {}), error])
    with pytest.raises(PlayerError, match="bot failure") as caught:
        engine._collect(results, _Supervisor(), stop, {0, 1}, lambda msg: handled.append(msg[1]), None)
    assert handled == [1] and "logs/tournament-worker-0.log" in str(caught.value)
