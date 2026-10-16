"""The journal writer thread (SP4 T1.3, spec block 1 "Запись"): it journals what is submitted and feeds the
in-memory table, counts drops (queue full, write errors, left at stop) with one WARNING / ERROR each, compacts on
its cadence with the evicted entities, never raises into the training process, and stops within its timeout. Also
the run-level counters of the system record and the coordinator's evict listener."""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from functools import partial

import numpy as np

from colosseum.arena.comparisons import ComparisonTable
from colosseum.arena.matchlog import (
    MATCH_LOG_FILE,
    WRITE_BATCH,
    AppendStats,
    CompactStats,
    LogSizes,
    MatchLog,
    MatchLogWriter,
    RunInfo,
    WriterStats,
)
from colosseum.metrics.aggregator import SystemStats
from game_helpers import make_coordinator, make_test_config, synthetic_journal_results

RUN = RunInfo(run_id="R", name="run", parent_run_id=None, game="ttt", config_hash="h", checkpoint_interval=10)


def _wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeLog:
    """A MatchLog stand-in: records calls; ``fail_appends`` failing appends first; ``delay`` per append."""

    def __init__(self, fail_appends: int = 0, delay: float = 0.0, file_bytes: int = 0) -> None:
        self.appended: list = []
        self.compactions: list[tuple[int, frozenset]] = []
        self.fail_appends = fail_appends
        self.delay = delay
        self.file_bytes = file_bytes
        self.closed = False

    def append(self, results):
        time.sleep(self.delay)
        if self.fail_appends:
            self.fail_appends -= 1
            raise sqlite3.OperationalError("disk I/O error")
        self.appended.extend(results)
        return AppendStats(detailed=len(results), accepted=[True] * len(results))

    def rating_versions(self, result):
        return [None] * len(result.seats)

    def compact(self, *, window_bytes, evicted=(), should_stop=lambda: False, **_):
        self.compactions.append((window_bytes, frozenset(evicted)))
        return CompactStats()

    def sizes(self):
        return LogSizes(file_bytes=self.file_bytes, wal_bytes=0, detail_bytes=0, detail_matches=0, other_matches=0,
                        counts_rows=0, free_disk_bytes=10**12, degraded=False)

    def close(self):
        self.closed = True


def test_the_writer_journals_everything_and_feeds_the_table(tmp_path):
    table = ComparisonTable()
    factory = partial(MatchLog, tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: 10**12)
    writer = MatchLogWriter(factory, compact_interval_sec=3600.0, window_bytes=10**9, table=table)
    writer.start()
    results = synthetic_journal_results(3000)
    for i in range(0, 3000, 250):
        assert writer.submit(results[i:i + 250]) == 0
    stats = writer.stop(timeout=7.0)
    assert (stats.written, stats.dropped, stats.write_errors, stats.pending) == (3000, 0, 0, 0)
    assert sum(s.n for _, _, s in table.pairs("2p", ["train"])) == 3000
    assert stats.sizes is not None and stats.sizes.detail_matches == 3000
    check = MatchLog(tmp_path / MATCH_LOG_FILE, RUN)
    assert check.comparison_table(("train",)).as_dict() == table.as_dict()
    check.close()


def test_a_full_writer_queue_drops_counts_and_warns_once(caplog):
    log = FakeLog()
    writer = MatchLogWriter(lambda: log, queue_size=10, compact_interval_sec=3600.0, window_bytes=1)
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.matchlog"):
        assert writer.submit(synthetic_journal_results(25)) == 15        # not started yet: the queue fills up
        assert writer.submit(synthetic_journal_results(5, first=25)) == 5
        writer.start()
        stats = writer.stop(timeout=5.0)
    assert (stats.written, stats.dropped) == (10, 20) and len(log.appended) == 10
    full = [r for r in caplog.records if "queue is full" in r.getMessage()]
    assert len(full) == 1 and full[0].levelno == logging.WARNING


def test_stop_counts_what_it_could_not_write_in_time(caplog):
    log = FakeLog(delay=0.3)
    writer = MatchLogWriter(lambda: log, compact_interval_sec=3600.0, window_bytes=1)
    results = synthetic_journal_results(8 * WRITE_BATCH)
    writer.submit(results)
    writer.start()
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.matchlog"):
        started = time.monotonic()
        stats = writer.stop(timeout=1.0)
        took = time.monotonic() - started
    assert took < 2.0
    assert stats.dropped > 0 and stats.written + stats.dropped == len(results)
    assert any("stopped with" in r.getMessage() for r in caplog.records)


def test_a_write_error_is_logged_and_counted_and_the_writer_goes_on(caplog):
    log = FakeLog(fail_appends=1)
    writer = MatchLogWriter(lambda: log, compact_interval_sec=3600.0, window_bytes=1)
    writer.start()
    with caplog.at_level(logging.ERROR, logger="colosseum.arena.matchlog"):
        writer.submit(synthetic_journal_results(7))
        assert _wait(lambda: writer.stats.write_errors == 1)
        writer.submit(synthetic_journal_results(5, first=7))
        stats = writer.stop(timeout=5.0)
    assert (stats.written, stats.dropped, stats.write_errors) == (5, 7, 1)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and isinstance(errors[0].exc_info, tuple) and "training goes on" in errors[0].getMessage()


def test_compaction_runs_on_its_cadence_with_the_evicted_entities():
    log = FakeLog()
    now = [0.0]
    writer = MatchLogWriter(lambda: log, compact_interval_sec=60.0, window_bytes=4096, clock=lambda: now[0])
    writer.start()
    writer.note_evicted(["main@v10"])
    time.sleep(0.3)
    assert log.compactions == []                        # the cadence has not elapsed
    now[0] = 61.0
    assert _wait(lambda: len(log.compactions) == 1)
    assert log.compactions[0] == (4096, frozenset({"main@v10"}))
    now[0] = 122.0
    assert _wait(lambda: len(log.compactions) == 2) and log.compactions[1] == (4096, frozenset())
    assert writer.stop(timeout=5.0).compactions == 2 and log.closed


def test_untouched_parts_beyond_the_window_warn_once(caplog):
    log = FakeLog(file_bytes=200 * 2**20)
    now = [0.0]
    writer = MatchLogWriter(lambda: log, compact_interval_sec=1.0, window_bytes=2**20, clock=lambda: now[0])
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.matchlog"):
        writer.start()
        for step in (2.0, 4.0):
            now[0] = step
            assert _wait(lambda step=step: len(log.compactions) == int(step // 2))
        writer.stop(timeout=5.0)
    assert len([r for r in caplog.records if "compaction does not touch" in r.getMessage()]) == 1


def test_a_journal_that_cannot_be_opened_never_stops_training(caplog):
    def broken():
        raise sqlite3.OperationalError("unable to open database file")

    writer = MatchLogWriter(broken, compact_interval_sec=3600.0, window_bytes=1)
    with caplog.at_level(logging.ERROR, logger="colosseum.arena.matchlog"):
        writer.start()
        writer.submit(synthetic_journal_results(4))
        assert _wait(lambda: writer.stats.dropped == 4)
        assert writer.submit(synthetic_journal_results(3, first=4)) == 3      # a dead writer drops at once
        stats = writer.stop(timeout=5.0)
    assert stats.written == 0 and stats.dropped == 7 and stats.write_errors == 1    # the failed open, once
    assert any("opening the journal failed" in r.getMessage() for r in caplog.records)


def test_a_batch_still_in_flight_at_the_stop_deadline_is_counted_as_dropped(caplog):
    log = FakeLog(delay=2.0)
    writer = MatchLogWriter(lambda: log, compact_interval_sec=3600.0, window_bytes=1)
    results = synthetic_journal_results(3 * WRITE_BATCH)
    writer.submit(results)
    writer.start()
    assert _wait(lambda: writer.stats.pending == len(results) and writer._queue.qsize() == 2 * WRITE_BATCH)
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.matchlog"):
        stats = writer.stop(timeout=1.0)
    assert (stats.written, stats.dropped, stats.pending) == (0, len(results), 0)
    assert any("did not stop within" in r.getMessage() for r in caplog.records)
    assert _wait(lambda: not writer._thread.is_alive())                  # the late append finished
    assert len(log.appended) == WRITE_BATCH and writer.stats == stats       # ... and changed nothing


def test_a_writer_thread_that_dies_counts_everything_as_dropped(caplog):
    log = FakeLog()

    def clock() -> float:
        if threading.current_thread().name == "match-log-writer":
            raise RuntimeError("clock broke")
        return 0.0

    writer = MatchLogWriter(lambda: log, compact_interval_sec=3600.0, window_bytes=1, clock=clock)
    writer.submit(synthetic_journal_results(5))
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.matchlog"):
        writer.start()
        assert _wait(lambda: writer.stats.dropped == 5)
        assert writer.submit(synthetic_journal_results(3, first=5)) == 3
        stats = writer.stop(timeout=5.0)
    assert (stats.written, stats.dropped, stats.write_errors, stats.pending) == (0, 8, 1, 0) and log.closed
    died = [r for r in caplog.records if "writer thread died" in r.getMessage()]
    assert len(died) == 1 and died[0].levelno == logging.ERROR and isinstance(died[0].exc_info, tuple)
    assert any("is not running" in r.getMessage() for r in caplog.records)
    assert not any("queue is full" in r.getMessage() for r in caplog.records)


def test_writer_stats_keep_one_shape():
    empty = WriterStats().as_dict()
    sized = WriterStats(sizes=FakeLog().sizes()).as_dict()
    assert set(empty) == set(sized) and empty["file_bytes"] is None and empty["evicted_folded"] == 0


def test_run_level_counters_never_decrease_and_include_the_writer():
    clock = [0.0]
    stats = SystemStats(clock=lambda: clock[0])
    stats.on_worker_stats({"worker_id": 0, "dropped_results": 3})
    stats.on_worker_stats({"worker_id": 1, "dropped_results": 2})
    clock[0] = 1.0
    assert stats.snapshot(0, {}, match_log={"dropped": 4})["dropped_match_results"] == 9
    clock[0] = 60.0                                      # both workers fell silent: their counts stay
    snap = stats.snapshot(0, {})
    assert snap["workers_reporting"] == 0 and snap["dropped_match_results"] == 5 and snap["match_log"] is None
    stats.on_worker_stats({"worker_id": 0, "dropped_results": 1})   # a stale lower report never decreases it
    assert stats.cumulative("dropped_results") == 5


def test_the_coordinator_tells_evict_listeners(tmp_path):
    config = make_test_config("solo", checkpoint={"interval": 10, "keep_last": 1, "keep_every": 0})
    coordinator = make_coordinator(config, tmp_path / "ckpt")
    heard: list[tuple[str, str]] = []
    coordinator.add_evict_listener(lambda agent, ckpt: heard.append((agent, ckpt)))
    state = {"w": np.zeros(2, np.float32)}
    for version in (10, 20, 30):
        coordinator.save_checkpoint_payload({"agent_id": "agent_0", "policy_version": version, "model_state": state})
    assert heard == [("agent_0", "ckpt_v10"), ("agent_0", "ckpt_v20")]


def test_a_failing_evict_listener_never_stops_the_run(tmp_path, caplog):
    config = make_test_config("solo", checkpoint={"interval": 10, "keep_last": 1, "keep_every": 0})
    coordinator = make_coordinator(config, tmp_path / "ckpt")
    heard: list[str] = []

    def broken(agent: str, ckpt: str) -> None:
        raise RuntimeError("listener broke")

    coordinator.add_evict_listener(broken)
    coordinator.add_evict_listener(lambda agent, ckpt: heard.append(ckpt))
    state = {"w": np.zeros(2, np.float32)}
    with caplog.at_level(logging.ERROR, logger="colosseum.coordinator.coordinator"):
        for version in (10, 20, 30):
            payload = {"agent_id": "agent_0", "policy_version": version, "model_state": state}
            coordinator.save_checkpoint_payload(payload)
    assert heard == ["ckpt_v10", "ckpt_v20"]                         # the other listener still hears every eviction
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 2 and isinstance(errors[0].exc_info, tuple) and not errors[1].exc_info


def test_stop_without_start_and_twice_is_harmless():
    writer = MatchLogWriter(FakeLog, compact_interval_sec=1.0, window_bytes=1)
    assert writer.stop(timeout=1.0).written == 0
    thread_count = threading.active_count()
    writer2 = MatchLogWriter(FakeLog, compact_interval_sec=1.0, window_bytes=1)
    writer2.start()
    writer2.stop(timeout=5.0)
    writer2.stop(timeout=5.0)
    assert threading.active_count() <= thread_count
