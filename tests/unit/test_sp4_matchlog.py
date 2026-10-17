"""The match log library (SP4 T1.2, spec block 1 and section 6 "журнал"): schema and SQLite settings, append in
one transaction, duplicates, the comparison table from detail and counts, compaction within the window (tables
unchanged, team index kept, tournament / pulse untouched, evicted entities first), the disk guard, the resume
import, the read-only reader while a writer writes, two writers, corrupt files and other schema versions."""
from __future__ import annotations

import logging
import os
import sqlite3
import threading

import pytest

from colosseum.arena.comparisons import ComparisonTable
from colosseum.arena.matchlog import (
    MATCH_LOG_FILE,
    SCHEMA_VERSION,
    MatchLog,
    MatchLogError,
    MatchLogReader,
    RunInfo,
    estimated_size,
)
from game_helpers import make_match_result, synthetic_journal_results

RUN = RunInfo(run_id="R", name="run", parent_run_id=None, game="ttt", config_hash="h", checkpoint_interval=10)
FREE = 10**12


def _log(tmp_path, run=RUN, name=MATCH_LOG_FILE, free=lambda _: FREE, **kwargs) -> MatchLog:
    return MatchLog(tmp_path / name, run, disk_free_fn=free, **kwargs)


def _memory_table(log: MatchLog, results) -> ComparisonTable:
    table = ComparisonTable()
    for result in results:
        table.add_result(result, log.rating_versions(result))
    return table


def _close(a: dict, b: dict, tol: float = 1e-9) -> bool:
    """Two ``ComparisonTable.as_dict()`` results agree: same keys, counts equal, sums within ``tol``."""
    if a.keys() != b.keys():
        return False
    for key in a:
        for part in ("pairs", "scores"):
            if a[key][part].keys() != b[key][part].keys():
                return False
            for cell, (n, s1, s2) in a[key][part].items():
                m, t1, t2 = b[key][part][cell]
                if n != m or abs(s1 - t1) > tol * max(1.0, abs(t1)) or abs(s2 - t2) > tol * max(1.0, abs(t2)):
                    return False
    return True


def _sql(path, query, args=()):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(query, args).fetchall()
    finally:
        conn.close()


def test_schema_tables_indexes_and_sqlite_settings(tmp_path):
    log = _log(tmp_path)
    log.close()
    path = tmp_path / MATCH_LOG_FILE
    tables = {r[0] for r in _sql(path, "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == {"meta", "entities", "runs", "matches", "seats", "teams", "counts"}
    indexes = {r[0] for r in _sql(path, "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE "
                                        "'sqlite_autoindex%'")}
    assert {"matches_by_source", "seats_by_agent", "seats_by_entity"} <= indexes
    assert _sql(path, "PRAGMA journal_mode") == [("wal",)]
    assert _sql(path, "PRAGMA auto_vacuum") == [(2,)]                 # INCREMENTAL
    assert _sql(path, "SELECT value FROM meta WHERE key = 'schema_version'") == [(str(SCHEMA_VERSION),)]
    assert _sql(path, "SELECT run_id, checkpoint_interval FROM runs") == [("R", 10)]


def test_append_writes_detail_with_rating_versions_and_ignores_duplicates(tmp_path):
    log = _log(tmp_path)
    results = synthetic_journal_results(50)
    stats = log.append(results)
    assert (stats.detailed, stats.duplicates, stats.counted) == (50, 0, 0) and all(stats.accepted)
    again = log.append(results[:5] + synthetic_journal_results(1, first=50))
    assert (again.detailed, again.duplicates) == (1, 5) and again.accepted == [False] * 5 + [True]
    sizes = log.sizes()
    assert sizes.detail_matches == 51 and sizes.detail_bytes == sum(estimated_size(r) for r in results) + \
        estimated_size(synthetic_journal_results(1, first=50)[0])
    with log.open_readonly(log.path) as reader:
        row = reader.matches(limit=1)[0]                                           # newest first
    assert row.uid == "R/train/w0_e2_ep50" and row.worker == 0 and row.env_steps == 500
    main = next(s for s in row.seats if s.agent_id == "main" and s.network_id == "latest")
    assert main.rating_version == (main.version_start // 10) * 10 and main.entity == f"main@v{main.rating_version}"
    log.close()


def test_append_is_one_transaction(tmp_path):
    log = _log(tmp_path)
    bad = synthetic_journal_results(3)
    bad[2].source = "eval"
    with pytest.raises(ValueError, match="source"):
        log.append(bad)
    bad[2].source, bad[1].uid = "train", ""
    with pytest.raises(ValueError, match="uid"):
        log.append(bad)
    assert log.sizes().detail_matches == 0 and log.comparison_table().layouts() == []
    log.close()


def test_the_table_from_the_journal_equals_the_in_memory_table(tmp_path):
    log = _log(tmp_path)
    results = synthetic_journal_results(300) + [
        make_match_result([(0, "a", "fixed"), (0, "b", "fixed")], scores=[2.0], layout="coop", uid="coop1"),
        make_match_result([(t, f"p{t}", "fixed") for t in range(3)], [3.0, 1.0, 2.0], layout="3p", uid="ffa1")]
    log.append(results)
    assert log.comparison_table(("train",)).as_dict() == _memory_table(log, results).as_dict()
    log.close()


def test_compaction_keeps_the_window_and_the_table(tmp_path):
    log = _log(tmp_path)
    results = synthetic_journal_results(4000)
    for i in range(0, len(results), 500):
        log.append(results[i:i + 500])
    window = log.sizes().detail_bytes // 5
    before = log.comparison_table(("train",)).as_dict()
    stats = log.compact(window_bytes=window, max_batch=700)
    sizes = log.sizes()
    assert stats.folded > 0 and stats.batches >= stats.folded // 700 and not stats.interrupted
    assert sizes.detail_bytes <= window and sizes.detail_bytes > window - 2 * estimated_size(results[0])
    assert sizes.counts_rows > 0 and sizes.wal_bytes == 0               # wal_checkpoint(TRUNCATE)
    after = log.comparison_table(("train",)).as_dict()
    assert _close(after, before) and _close(after, _memory_table(log, results).as_dict())
    oldest = _sql(log.path, "SELECT MIN(id) FROM matches")[0][0]
    assert oldest == stats.folded + 1                                   # ids 1..4000: the oldest were folded
    log.close()


def test_compaction_gives_space_back_and_never_touches_tournament_or_pulse(tmp_path):
    log = _log(tmp_path)
    log.append(synthetic_journal_results(200, source="pulse", prefix="R/pulse/"))
    log.append(synthetic_journal_results(100, source="tournament", prefix="T1/"), tournament="T1")
    for i in range(10):
        log.append(synthetic_journal_results(1000, first=1000 * i))
    file_before = log.sizes().file_bytes
    log.compact(window_bytes=20_000)
    sizes = log.sizes()
    assert sizes.other_matches == 300 and sizes.file_bytes < file_before / 2
    assert sizes.detail_bytes <= 20_000 and sizes.detail_matches > 0      # the window is kept, not emptied
    reader = MatchLogReader(log.path)
    assert len(reader.matches(source="pulse", limit=1000)) == 200
    assert len(reader.matches(tournament="T1", limit=1000)) == 100
    assert reader.comparison_table(("tournament",), tournament="T1").layouts() == ["2p"]
    reader.close()
    log.close()


def test_matches_of_evicted_entities_are_folded_first(tmp_path):
    log = _log(tmp_path)
    results = synthetic_journal_results(600)
    log.append(results)
    before = log.comparison_table().as_dict()
    stats = log.compact(window_bytes=10**9, evicted=["main@v10", "nobody@v5"])
    with_v10 = sum(any(s.network_id == "ckpt_v10" or (s.network_id == "latest" and 10 <= (s.version_start or -1) < 20)
                       for s in r.seats) for r in results)
    assert stats.evicted_folded == stats.folded == with_v10 > 0
    reader = MatchLogReader(log.path)
    assert reader.matches(agent="main", version=10, limit=10) == []
    assert len(reader.matches(agent="main", version=20, limit=10)) == 10
    assert _close(reader.comparison_table().as_dict(), before)
    reader.close()
    log.close()


def test_compaction_stops_between_batches(tmp_path):
    log = _log(tmp_path)
    log.append(synthetic_journal_results(1000))
    calls = []
    stats = log.compact(window_bytes=0, max_batch=100, should_stop=lambda: calls.append(1) or len(calls) > 3)
    assert stats.interrupted and stats.batches == 3 and stats.folded == 300
    log.close()


def test_the_disk_guard_switches_to_counts_and_back(tmp_path, caplog):
    free = [FREE]
    log = _log(tmp_path, free=lambda _: free[0], min_free_disk_mb=100)
    results = synthetic_journal_results(300)
    with caplog.at_level(logging.INFO, logger="colosseum.arena.matchlog"):
        assert log.append(results[:100]).detailed == 100
        free[0] = 50 * 2**20
        low = log.append(results[100:200])
        log.append(results[200:250])
        assert (low.counted, low.detailed, low.degraded) == (100, 0, True) and log.sizes().degraded
        tournament = log.append(synthetic_journal_results(10, source="tournament", prefix="T/"), tournament="T")
        assert tournament.detailed == 10                                # a tournament writes as usual
        free[0] = FREE
        assert log.append(results[250:]).detailed == 50 and not log.sizes().degraded
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    infos = [r for r in caplog.records if r.levelno == logging.INFO and "resumes" in r.getMessage()]
    assert len(warnings) == 1 and "min_free_disk_mb" in warnings[0].getMessage() and len(infos) == 1
    assert log.sizes().detail_matches == 150
    assert _close(log.comparison_table(("train",)).as_dict(), _memory_table(log, results).as_dict())
    log.close()


def test_import_parent_folds_its_detail_and_copies_counts_tournament_and_pulse(tmp_path):
    parent = _log(tmp_path, name="parent.db")
    train = synthetic_journal_results(800)
    parent.append(train)
    parent.compact(window_bytes=parent.sizes().detail_bytes // 2)       # half detail, half counts
    parent.append(synthetic_journal_results(40, source="pulse", prefix="R/pulse/"))
    parent.append(synthetic_journal_results(20, source="tournament", prefix="T1/"), tournament="T1")
    expected = parent.comparison_table(("train", "pulse")).as_dict()
    parent.close()
    child_run = RunInfo("C", "child", "R", "ttt", "h", 10)
    child = _log(tmp_path, run=child_run, name="child.db")
    stats = child.import_parent(tmp_path / "parent.db")
    assert stats.copied == 60 and stats.folded > 0 and stats.counts_rows > 0 and not stats.skipped
    sizes = child.sizes()
    assert sizes.detail_matches == 0 and sizes.other_matches == 60      # no training detail copied
    assert _close(child.comparison_table(("train", "pulse")).as_dict(), expected)
    reader = MatchLogReader(tmp_path / "child.db")
    assert [r.run_id for r in reader.runs()] == ["C", "R"] and reader.runs()[0].parent_run_id == "R"
    assert len(reader.matches(tournament="T1", limit=100)) == 20
    reader.close()
    child.close()


def _parent_journal(tmp_path):
    parent = _log(tmp_path, name="parent.db")
    parent.append(synthetic_journal_results(300))
    parent.compact(window_bytes=parent.sizes().detail_bytes // 2)
    parent.append(synthetic_journal_results(10, source="pulse", prefix="R/pulse/"))
    parent.append(synthetic_journal_results(5, source="tournament", prefix="T1/"), tournament="T1")
    expected = parent.comparison_table(("train", "pulse")).as_dict()
    parent.close()
    return tmp_path / "parent.db", expected


def test_import_parent_twice_changes_nothing(tmp_path, caplog):
    path, expected = _parent_journal(tmp_path)
    child = _log(tmp_path, run=RunInfo("C", "child", "R", "ttt", "h", 10), name="child.db")
    child.import_parent(path)
    with caplog.at_level(logging.INFO, logger="colosseum.arena.matchlog"):
        again = child.import_parent(path)
    assert again.skipped and (again.counts_rows, again.folded, again.copied) == (0, 0, 0)
    assert any("already imported" in r.getMessage() for r in caplog.records)
    assert _close(child.comparison_table(("train", "pulse")).as_dict(), expected)
    assert child.sizes().other_matches == 15
    child.close()


def test_a_failed_import_leaves_the_journal_unchanged(tmp_path, monkeypatch):
    path, expected = _parent_journal(tmp_path)
    child = _log(tmp_path, run=RunInfo("C", "child", "R", "ttt", "h", 10), name="child.db")

    def fail(self, parent, train):
        raise RuntimeError("disk gone mid-import")

    with monkeypatch.context() as patch:                # after the runs, counts and folds, before the copy
        patch.setattr(MatchLog, "_copy_detail", fail)
        with pytest.raises(RuntimeError, match="mid-import"):
            child.import_parent(path)
    assert child.comparison_table(("train", "pulse")).layouts() == [] and child.sizes().counts_rows == 0
    with MatchLogReader(tmp_path / "child.db") as reader:
        assert [r.run_id for r in reader.runs()] == ["C"]
    assert not child.import_parent(path).skipped                       # no marker was left: the retry imports
    assert _close(child.comparison_table(("train", "pulse")).as_dict(), expected)
    child.close()


def test_the_reader_reads_while_a_writer_writes(tmp_path):
    log = _log(tmp_path)
    log.append(synthetic_journal_results(10))
    reader = MatchLogReader(log.path)
    errors: list[BaseException] = []

    def write():
        try:
            writer = _log(tmp_path)
            for i in range(20):
                writer.append(synthetic_journal_results(100, first=10 + 100 * i))
            writer.close()
        except BaseException as e:   # noqa: BLE001 - reported below
            errors.append(e)

    thread = threading.Thread(target=write)
    thread.start()
    seen = []
    while thread.is_alive():
        seen.append(sum(s.n for _, _, s in reader.comparison_table(("train",)).pairs("2p", ["train"])))
    thread.join()
    assert not errors and seen == sorted(seen)                         # monotone: every read is consistent
    assert reader.sizes().detail_matches == 2010
    reader.close()
    log.close()


def test_two_writers_lose_no_record(tmp_path):
    training = _log(tmp_path)
    errors: list[BaseException] = []

    def tournament():
        try:
            writer = MatchLog(tmp_path / MATCH_LOG_FILE, None, disk_free_fn=lambda _: FREE)
            for i in range(30):                                     # one block per transaction
                block = synthetic_journal_results(10, source="tournament", prefix=f"T/{i}/")
                for r in block:
                    r.block = f"b{i}"
                writer.append(block, tournament="T")
            writer.close()
        except BaseException as e:   # noqa: BLE001 - reported below
            errors.append(e)

    thread = threading.Thread(target=tournament)
    thread.start()
    for i in range(30):
        training.append(synthetic_journal_results(100, first=100 * i))
    thread.join()
    assert not errors
    sizes = training.sizes()
    assert (sizes.detail_matches, sizes.other_matches) == (3000, 300)
    reader = MatchLogReader(training.path)
    assert reader.blocks("T") == {f"b{i}" for i in range(30)} and reader.blocks("other") == set()
    reader.close()
    training.close()


def test_matches_filters(tmp_path):
    log = _log(tmp_path)
    results = synthetic_journal_results(200)
    log.append(results)
    reader = MatchLogReader(log.path)
    rows = reader.matches(limit=5)
    assert [r.uid for r in rows] == [r.uid for r in results[::-1][:5]]                   # newest first
    assert all(r.env_steps <= 500 for r in reader.matches(env_steps=(0, 500), limit=1000))
    assert len(reader.matches(env_steps=(0, 500), limit=1000)) == 51
    wins = reader.matches(agent="random", outcome="win", limit=1000)
    assert wins and all(any(s.agent_id == "random" for s in r.seats) for r in wins)
    random_team = [next(s.team for s in r.seats if s.agent_id == "random") for r in wins]
    assert all(dict((t, rank) for t, rank, _ in r.teams)[team] == 1.0 and
               sorted(rank for _, rank, _ in r.teams) == [1.0, 2.0] for r, team in zip(wins, random_team))
    assert reader.matches(agent="nobody") == [] and reader.matches(layout="9p") == []
    with pytest.raises(ValueError, match="agent"):
        reader.matches(outcome="win")
    with pytest.raises(ValueError, match="version filter needs an agent"):
        reader.matches(version=10)
    reader.close()
    log.close()


def test_rows_round_trip_results_and_bootstrap_units(tmp_path):
    """Contract amendments A6-A8: ``MatchRow.to_dict`` / ``to_result``, ``limit=None``, ``match_results`` in row
    order, the reader as a context manager, and ``bootstrap_units`` (counts rows + detailed contributions)."""
    log = _log(tmp_path)
    train = synthetic_journal_results(40)
    pulse = [make_match_result([(0, "main", "ckpt_v10", "player", 10), (1, "random", "fixed")], uid=f"p/{i}",
                               source="pulse", block=f"pulse-x/P/2p/main@v10/random/{i // 2}", seed=i)
             for i in range(6)]
    log.append(train + pulse)
    log.compact(window_bytes=estimated_size(train[0]) * 10)
    with MatchLogReader(log.path) as reader:
        rows = reader.matches(limit=None)
        assert len(rows) == len(reader.matches(source="train", limit=None)) + 6 and rows[0].uid == "p/5"
        row = reader.matches(source="pulse", limit=1)[0]
        d = row.to_dict()
        assert d["uid"] == "p/5" and d["block"].endswith("/2") and d["replay"] is None and "ts" not in d
        assert d["seats"][0]["rating_version"] == 10 and d["seats"][0]["entity"] == "main@v10"
        result, versions = row.to_result()
        assert (result.uid, result.source, result.seed, versions) == ("p/5", "pulse", 5, [10, None])
        assert [r.uid for r in reader.match_results(source="pulse")] == [f"p/{i}" for i in range(6)]
        counts, details = reader.bootstrap_units("2p", ("train", "pulse"))
        assert counts and all(stat.n > 0 for _, _, stat in counts)
        blocks = [b for b, _ in details if b is not None]
        assert len(blocks) == 6 and all(len(contribs) == 1 for _, contribs in details)
        assert sum(stat.n for _, _, stat in counts) + len(details) == len(train) + len(pulse)
        assert reader.bootstrap_units("9p", ("train",)) == ([], [])
    log.close()


def test_a_missing_journal_reads_as_empty_and_other_schemas_are_refused(tmp_path):
    empty = MatchLogReader(tmp_path / "missing.db")
    assert empty.comparison_table().layouts() == [] and empty.matches() == [] and empty.runs() == []
    empty.close()
    log = _log(tmp_path)
    log.close()
    conn = sqlite3.connect(tmp_path / MATCH_LOG_FILE)
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(MatchLogError, match="schema version"):
        _log(tmp_path)
    with pytest.raises(MatchLogError, match="schema version"):
        MatchLogReader(tmp_path / MATCH_LOG_FILE)


def test_a_corrupt_file_is_moved_away_and_a_new_journal_starts(tmp_path, caplog):
    (tmp_path / MATCH_LOG_FILE).write_bytes(b"this is not an sqlite database" * 100)
    with caplog.at_level(logging.ERROR, logger="colosseum.arena.matchlog"):
        log = _log(tmp_path)
    assert (tmp_path / f"{MATCH_LOG_FILE}.corrupt-1").exists()
    assert any("not a readable SQLite database" in r.getMessage() for r in caplog.records)
    assert log.append(synthetic_journal_results(3)).detailed == 3
    log.close()


def test_the_size_estimate_tracks_the_pages_sqlite_uses(tmp_path):
    """The window is enforced on ``estimated_size``; it must stay close to the real page use (calibrated within
    7 % on 1v1, 2v2 and FFA matches; the band here is 25 %)."""
    log = _log(tmp_path)
    for i in range(6):
        log.append(synthetic_journal_results(1000, first=1000 * i, prefix="0123456789abcdef0123456789abcdef/train/"))
    log.close()
    path = tmp_path / MATCH_LOG_FILE
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    pages, free, size = (conn.execute(f"PRAGMA {p}").fetchone()[0] for p in ("page_count", "freelist_count",
                                                                            "page_size"))
    estimate = int(conn.execute("SELECT value FROM meta WHERE key = 'detail_train_bytes'").fetchone()[0])
    conn.close()
    assert 0.75 < (pages - free) * size / estimate < 1.25


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root reads files without permission")
def test_an_unreadable_journal_is_not_moved_away(tmp_path):
    """Controller ruling C3: only "not a database" / "malformed" files are moved aside; any other SQLite error
    (here: no permission) propagates (the launcher turns it into a ConfigError, T1.3) and the file stays."""
    log = _log(tmp_path)
    log.append(synthetic_journal_results(3))
    log.close()
    path = tmp_path / MATCH_LOG_FILE
    path.chmod(0)
    try:
        with pytest.raises(sqlite3.Error) as error:
            _log(tmp_path)
        assert not isinstance(error.value, MatchLogError)
        assert not list(tmp_path.glob(f"{MATCH_LOG_FILE}.corrupt-*"))
    finally:
        path.chmod(0o644)
    with MatchLogReader(path) as reader:
        assert reader.sizes().detail_matches == 3


def test_a_failed_append_surfaces_its_error_and_leaves_no_stale_ids(tmp_path):
    """SQLite rolls a transaction back by itself on SQLITE_FULL: the original error must surface (not a failed
    ROLLBACK) and the ids of the rolled-back transaction must not survive in the cache."""
    log = _log(tmp_path)
    log.append(synthetic_journal_results(5))
    pages = log._conn.execute("PRAGMA page_count").fetchone()[0]
    log._conn.execute(f"PRAGMA max_page_count = {pages}")
    big = [make_match_result([(0, f"q{i}", "fixed"), (1, f"r{i}", "fixed")], uid=f"full/{i}/" + "x" * 300)
           for i in range(3000)]
    with pytest.raises(sqlite3.OperationalError, match="full"):
        log.append(big)
    log._conn.execute("PRAGMA max_page_count = 1073741823")
    # "q0" was cached during the failed batch: a stale cached id would mislabel (or lose) this seat
    assert log.append([make_match_result([(0, "a0", "fixed"), (1, "q0", "fixed")], uid="L2")]).detailed == 1
    with MatchLogReader(log.path) as reader:
        row = reader.matches(limit=1)[0]
        assert row.uid == "L2" and [(s.agent_id, s.entity) for s in row.seats] == [("a0", "a0"), ("q0", "q0")]
        assert reader.sizes().detail_matches == 6
    log.close()


def test_a_read_across_a_compaction_is_retried(tmp_path, monkeypatch):
    """A compaction that commits between a reader's counts query and its detail queries moves matches out of the
    detail it is about to read; the reader sees ``compact_generation`` change and reads again."""
    log = _log(tmp_path)
    results = synthetic_journal_results(400)
    log.append(results)
    full = log.comparison_table(("train",)).as_dict()
    reader = MatchLogReader(log.path)
    original = reader._iter_detail
    calls = []

    def iter_detail(where, args):
        if not calls:
            log.compact(window_bytes=log.sizes().detail_bytes // 2)
        calls.append(1)
        return original(where, args)

    monkeypatch.setattr(reader, "_iter_detail", iter_detail)
    assert _close(reader.comparison_table(("train",)).as_dict(), full) and len(calls) == 2
    calls.clear()
    counts, details = reader.bootstrap_units("2p", ("train",))
    assert len(calls) == 2 and sum(stat.n for _, _, stat in counts) + len(details) == len(results)
    reader.close()
    log.close()


def test_a_read_that_overlaps_every_attempt_warns_and_returns(tmp_path, monkeypatch, caplog):
    """The bounded retry's fallback: a compaction batch commits during every attempt, so the reader gives up after
    ``CONSISTENT_READ_ATTEMPTS`` reads and returns the last one with a WARNING."""
    import colosseum.arena.matchlog as matchlog

    log = _log(tmp_path)
    log.append(synthetic_journal_results(60))
    reader = MatchLogReader(log.path)
    original = reader._iter_detail
    calls, sleeps = [], []

    def iter_detail(where, args):
        calls.append(1)
        log.compact(window_bytes=log.sizes().detail_bytes - 1, max_batch=1)      # one batch per attempt
        return original(where, args)

    monkeypatch.setattr(reader, "_iter_detail", iter_detail)
    monkeypatch.setattr(matchlog.time, "sleep", sleeps.append)
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.matchlog"):
        table = reader.comparison_table(("train",))
    attempts = matchlog.CONSISTENT_READ_ATTEMPTS
    assert len(calls) == attempts and table.layouts() == ["2p"]
    assert sleeps == [matchlog.CONSISTENT_READ_BACKOFF_SEC * k for k in range(1, attempts)]
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert f"overlapped {attempts} compaction batches" in warning.getMessage()
    assert "may undercount" in warning.getMessage()
    reader.close()
    log.close()


def test_compaction_binds_at_most_999_parameters(tmp_path):
    log = _log(tmp_path)
    log.append(synthetic_journal_results(2500))
    before = log.comparison_table(("train",)).as_dict()
    log._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)       # SQLite < 3.32
    stats = log.compact(window_bytes=0)
    assert stats.folded == 2500 and stats.batches == 1 and log.sizes().detail_matches == 0
    assert _close(log.comparison_table(("train",)).as_dict(), before)
    log.close()


def test_paths_with_uri_characters_open_read_only(tmp_path):
    directory = tmp_path / "odd ?#% dir"
    log = MatchLog(directory / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: FREE)
    log.append(synthetic_journal_results(4))
    with MatchLogReader(directory / MATCH_LOG_FILE) as reader:
        assert reader.sizes().detail_matches == 4 and len(reader.matches()) == 4
    log.close()


def test_a_failed_run_registration_closes_the_connection(tmp_path, monkeypatch):
    import colosseum.arena.matchlog as matchlog

    opened = []

    def connect(path, *, readonly):
        opened.append(original(path, readonly=readonly))
        return opened[-1]

    def register(self, run):
        raise sqlite3.OperationalError("database is locked")

    original = matchlog._connect
    monkeypatch.setattr(matchlog, "_connect", connect)
    monkeypatch.setattr(MatchLog, "_register_run", register)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        _log(tmp_path)
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")
