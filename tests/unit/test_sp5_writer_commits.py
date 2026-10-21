"""The journal writer's tokens and commits (SP5 T2.5, ruling R8): a tokened submit is written whole in one transaction
and committed in order with its rows; a resend after a hub restart is all duplicates; a tokened submit that does not
fit is refused whole and not counted; a write error or a dead writer gives failed commits and drops only untokened
results; groups never split between appends; tokens left at the stop are counted as dropped."""
from __future__ import annotations

import sqlite3
import time
from functools import partial

from colosseum.arena.matchlog import MATCH_LOG_FILE, WRITE_BATCH, AppendStats, MatchLog, MatchLogWriter, RunInfo
from game_helpers import synthetic_journal_results

RUN = RunInfo(run_id="R", name="run", parent_run_id=None, game="ttt", config_hash="h", checkpoint_interval=10)


def _wait(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class CommitFakeLog:
    def __init__(self, fail_appends=0):
        self.batches = []
        self.fail_appends = fail_appends

    def append(self, results):
        if self.fail_appends:
            self.fail_appends -= 1
            raise sqlite3.OperationalError("disk I/O error")
        self.batches.append(list(results))
        return AppendStats(detailed=len(results), accepted=[True] * len(results), rows=[None] * len(results))

    def rating_versions(self, result):
        return [None] * len(result.seats)

    def compact(self, **_):
        from colosseum.arena.matchlog import CompactStats
        return CompactStats()

    def sizes(self):
        from colosseum.arena.matchlog import LogSizes
        return LogSizes(0, 0, 0, 0, 0, 0, 10**12, False)

    def close(self):
        pass


def _commits(writer, n):
    got = []
    assert _wait(lambda: (got.extend(writer.take_commits()) or True) and len(got) >= n)
    return got


def test_tokened_submits_are_committed_whole_in_order_with_their_rows(tmp_path):
    factory = partial(MatchLog, tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: 10**12)
    writer = MatchLogWriter(factory, compact_interval_sec=3600.0, window_bytes=10**9)
    writer.start()
    results = synthetic_journal_results(9)
    groups = [results[:3], results[3:5], results[5:]]
    for seq, group in enumerate(groups, 1):
        assert writer.submit(group, token=("n", "s", seq)) == 0
    commits = _commits(writer, 3)
    assert [c.token for c in commits] == [("n", "s", 1), ("n", "s", 2), ("n", "s", 3)]
    assert [len(c.accepted) for c in commits] == [3, 2, 4] and all(not c.failed and c.duplicates == 0 for c in commits)
    rows = [c.last_row for c in commits]
    assert rows == sorted(rows) and rows[-1] == 9
    assert writer.submit(groups[0], token=("n", "s2", 1)) == 0               # a resend after a hub restart
    (again,) = _commits(writer, 1)
    assert (again.accepted, again.duplicates, again.last_row, again.failed) == ([], 3, None, False)
    stats = writer.stop(5.0)
    assert (stats.written, stats.duplicates, stats.dropped) == (9, 3, 0)


def test_a_tokened_submit_that_does_not_fit_is_refused_whole_and_not_counted():
    writer = MatchLogWriter(CommitFakeLog, queue_size=5, compact_interval_sec=3600.0, window_bytes=1)
    assert writer.submit(synthetic_journal_results(4), token="a") == 0
    assert writer.submit(synthetic_journal_results(2, first=4), token="b") == 2
    assert writer.stats.dropped == 0 and writer.stats.pending == 4
    assert writer.submit(synthetic_journal_results(3, first=6)) == 2           # untokened: SP4 rule
    assert writer.stats.dropped == 2
    writer.start()
    assert [c.token for c in _commits(writer, 1)] == ["a"]
    writer.stop(5.0)


def test_a_write_error_fails_the_tokened_groups_and_drops_only_the_untokened():
    log = CommitFakeLog(fail_appends=1)
    writer = MatchLogWriter(lambda: log, compact_interval_sec=3600.0, window_bytes=1)
    writer.submit(synthetic_journal_results(3), token="t")
    writer.submit(synthetic_journal_results(2, first=3))
    writer.start()
    (commit,) = _commits(writer, 1)
    assert commit.failed and commit.token == "t" and commit.accepted == []
    stats = writer.stop(5.0)
    assert (stats.dropped, stats.write_errors, stats.written) == (2, 1, 0)


def test_a_dead_writer_fails_queued_tokens_and_refuses_new_ones():
    def broken():
        raise sqlite3.OperationalError("unable to open database file")

    writer = MatchLogWriter(broken, compact_interval_sec=3600.0, window_bytes=1)
    writer.submit(synthetic_journal_results(3), token="t")
    writer.start()
    (commit,) = _commits(writer, 1)
    assert commit.failed and writer.dead and writer.stats.dropped == 0
    assert writer.submit(synthetic_journal_results(2), token="u") == 2 and writer.stats.dropped == 0
    writer.stop(5.0)


def test_a_tokened_group_is_never_split_between_two_appends():
    log = CommitFakeLog()
    writer = MatchLogWriter(lambda: log, compact_interval_sec=3600.0, window_bytes=1)
    results = synthetic_journal_results(4 * 1500)
    for k in range(4):
        writer.submit(results[k * 1500:(k + 1) * 1500], token=k)
    writer.start()
    _commits(writer, 4)
    writer.stop(5.0)
    for k in range(4):
        uids = {r.uid for r in results[k * 1500:(k + 1) * 1500]}
        assert sum(1 for b in log.batches if uids & {r.uid for r in b}) == 1
    assert max(len(b) for b in log.batches) <= WRITE_BATCH                  # 1500 + 1500 would exceed it


def test_queued_tokened_results_left_at_the_stop_are_counted_as_dropped():
    class Slow(CommitFakeLog):
        def append(self, results):
            time.sleep(1.5)
            return super().append(results)

    writer = MatchLogWriter(Slow, compact_interval_sec=3600.0, window_bytes=1)
    for k in range(3):
        writer.submit(synthetic_journal_results(WRITE_BATCH, first=k * WRITE_BATCH), token=k)
    writer.start()
    time.sleep(0.3)
    stats = writer.stop(timeout=1.0)
    assert stats.written + stats.dropped == 3 * WRITE_BATCH and stats.dropped >= 2 * WRITE_BATCH


def test_a_batch_stays_within_write_batch_and_a_larger_group_goes_alone():
    """Review round 1: whole groups in queue order while the total stays within WRITE_BATCH; the group that would
    exceed it opens the next batch; a tokened group larger than WRITE_BATCH is a batch of its own."""
    log = CommitFakeLog()
    writer = MatchLogWriter(lambda: log, compact_interval_sec=3600.0, window_bytes=1)
    results = synthetic_journal_results(6000)
    writer.submit(results[:3000])                                           # untokened: groups of 2000 and 1000
    writer.submit(results[3000:3500], token="small")
    writer.submit(results[3500:6000], token="large")
    writer.start()
    assert [c.token for c in _commits(writer, 2)] == ["small", "large"]
    writer.stop(5.0)
    assert [len(b) for b in log.batches] == [WRITE_BATCH, 1500, 2500]
    assert [r.uid for b in log.batches for r in b] == [r.uid for r in results]          # queue order kept


def test_a_tokened_group_larger_than_the_queue_is_taken_whole_only_when_the_queue_is_empty():
    """Ruling R23: an empty queue takes any tokened group whole (it could never fit otherwise); a non-empty one
    refuses a group that does not fit, whole and uncounted."""
    writer = MatchLogWriter(CommitFakeLog, queue_size=2, compact_interval_sec=3600.0, window_bytes=1)
    assert writer.submit(synthetic_journal_results(3), token="big") == 0
    assert writer.submit(synthetic_journal_results(3, first=3), token="refused") == 3
    assert writer.submit(synthetic_journal_results(1, first=6), token="refused-too") == 1   # 3 + 1 > 2
    assert writer.stats.dropped == 0 and writer.stats.pending == 3
    writer.start()
    (commit,) = _commits(writer, 1)
    assert (commit.token, len(commit.accepted), commit.failed) == ("big", 3, False)
    assert writer.submit(synthetic_journal_results(3, first=3), token="again") == 0       # empty again
    assert [c.token for c in _commits(writer, 1)] == ["again"]
    assert writer.stop(5.0).dropped == 0


def test_a_thread_that_dies_while_writing_fails_its_tokened_groups_and_drops_the_untokened(caplog):
    """The writer thread dies mid-loop (here: the journal reports one flag too few): the groups it was writing get
    failed commits (tokened) or are dropped (untokened); the writer is dead and refuses later tokened submits."""
    class Short(CommitFakeLog):
        def append(self, results):
            stats = super().append(results)
            stats.accepted.pop()
            return stats

    writer = MatchLogWriter(Short, compact_interval_sec=3600.0, window_bytes=1)
    writer.submit(synthetic_journal_results(2))
    writer.submit(synthetic_journal_results(3, first=2), token="t")          # its flags come up one short
    writer.start()
    (commit,) = _commits(writer, 1)
    assert commit.failed and commit.token == "t" and writer.dead
    assert writer.submit(synthetic_journal_results(1, first=5), token="u") == 1
    stats = writer.stop(5.0)
    assert (stats.dropped, stats.write_errors, stats.written, stats.pending) == (2, 1, 0, 0)
    assert any("writer thread died" in r.getMessage() for r in caplog.records)
