"""The journal's schema version 2 (SP5 T4.3; spec 4.3 and 5.4): the ``node`` column, the in-place migration of an
SP4 journal (also when a writer opens it again, and never touching a parent read by ``import_parent``), row ids
that are never reused (the league-state watermark stays exact after compaction), the reader of both versions, and
``results_after`` (the league replay of an in-place resume)."""
from __future__ import annotations

import dataclasses
import shutil
import sqlite3

import pytest

from colosseum.arena.matchlog import (
    MATCH_LOG_FILE,
    SCHEMA_VERSION,
    MatchLog,
    MatchLogError,
    MatchLogReader,
    RunInfo,
)
from game_helpers import copy_sp4_run, make_match_result, synthetic_journal_results

RUN = RunInfo(run_id="R", name="run", parent_run_id=None, game="ttt", config_hash="h", checkpoint_interval=10)
FREE = 10**12


def _sql(path, query, args=()):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(query, args).fetchall()
    finally:
        conn.close()


def _v1_journal(tmp_path):
    """An SP4 journal (schema version 1) written by SP4 code: the T0.1 run-dir fixture's."""
    path = copy_sp4_run(tmp_path) / MATCH_LOG_FILE
    assert _sql(path, "SELECT value FROM meta WHERE key = 'schema_version'") == [("1",)]
    return path


def test_new_journals_have_version_2_with_a_node_column(tmp_path):
    log = MatchLog(tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: FREE)
    results = synthetic_journal_results(3)
    results[1] = dataclasses.replace(results[1], node=4)
    log.append(results)
    log.close()
    path = tmp_path / MATCH_LOG_FILE
    assert SCHEMA_VERSION == 2
    assert _sql(path, "SELECT value FROM meta WHERE key = 'schema_version'") == [("2",)]
    assert _sql(path, "SELECT id, node FROM matches ORDER BY id") == [(1, None), (2, 4), (3, None)]
    with MatchLogReader(path) as reader:
        rows = reader.matches(limit=None)
        assert [r.node for r in reversed(rows)] == [None, 4, None]
        assert rows[1].to_dict()["node"] == 4 and rows[1].to_result()[0].node == 4


def test_an_sp4_journal_is_migrated_in_place_once(tmp_path, caplog):
    path = _v1_journal(tmp_path)
    before = _sql(path, "SELECT id, uid FROM matches ORDER BY id")
    top = before[-1][0]
    with caplog.at_level("INFO", logger="colosseum.arena.matchlog"):
        MatchLog(path, RUN, disk_free_fn=lambda _: FREE).close()
    assert "schema version 1 migrated to 2" in caplog.text
    assert _sql(path, "SELECT value FROM meta WHERE key = 'schema_version'") == [("2",)]
    assert _sql(path, "SELECT value FROM meta WHERE key = 'next_match_id'") == [(str(top + 1),)]
    assert _sql(path, "SELECT id, uid FROM matches ORDER BY id") == before
    assert _sql(path, "SELECT COUNT(*) FROM matches WHERE node IS NOT NULL") == [(0,)]
    caplog.clear()
    with caplog.at_level("INFO", logger="colosseum.arena.matchlog"):
        MatchLog(path, RUN, disk_free_fn=lambda _: FREE).close()          # a second open migrates nothing
    assert "migrated" not in caplog.text


def test_the_reader_reads_both_versions_and_refuses_others(tmp_path):
    path = _v1_journal(tmp_path)
    with MatchLogReader(path) as reader:                                  # read-only: no migration
        rows = reader.matches(limit=5)
        assert rows and all(r.node is None for r in rows)
    assert _sql(path, "SELECT value FROM meta WHERE key = 'schema_version'") == [("1",)]
    other = tmp_path / "other.db"
    shutil.copy(path, other)
    conn = sqlite3.connect(other)
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(MatchLogError, match="schema version 99"):
        MatchLogReader(other)
    with pytest.raises(MatchLogError, match="schema version 99"):
        MatchLog(other, RUN, disk_free_fn=lambda _: FREE)


def test_row_ids_are_never_reused_after_compaction_deleted_the_newest(tmp_path):
    """The R8 watermark (league_state.json journal_row) means 'every row up to here is accounted': an id must
    never come back, even when compaction folded the newest rows (matches of an evicted snapshot)."""
    log = MatchLog(tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: FREE)
    log.append(synthetic_journal_results(4))
    newest = make_match_result([(0, "main", "latest", "player", 31), (1, "main", "ckpt_v30")], uid="R/train/x",
                               env_steps=99)
    log.append([newest])
    assert log.last_row() == 5
    log.compact(window_bytes=10**9, evicted=["main@v30"])                 # folds the newest row into counts
    assert log.last_row() == 4
    log.append([make_match_result([(0, "main", "latest"), (1, "random", "fixed")], uid="R/train/y")])
    assert log.last_row() == 6                                             # not 5 again
    log.close()


def test_import_of_an_sp4_parent_leaves_the_parent_untouched(tmp_path):
    parent = _v1_journal(tmp_path)
    child = MatchLog(tmp_path / "child.db", dataclasses.replace(RUN, run_id="C", parent_run_id="P"),
                     disk_free_fn=lambda _: FREE)
    stats = child.import_parent(parent)
    child.append([make_match_result([(0, "agent_0", "latest"), (1, "agent_0", "ckpt_v20")], uid="C/train/z")])
    child.close()
    assert stats.folded > 0
    assert _sql(parent, "SELECT value FROM meta WHERE key = 'schema_version'") == [("1",)]
    assert _sql(tmp_path / "child.db", "SELECT value FROM meta WHERE key = 'schema_version'") == [("2",)]
    ids = [r[0] for r in _sql(tmp_path / "child.db", "SELECT id FROM matches ORDER BY id")]
    assert ids == sorted(set(ids))


def test_results_after_returns_the_unaccounted_rows_in_commit_order(tmp_path):
    log = MatchLog(tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: FREE)
    train = synthetic_journal_results(6)
    log.append(train[:3])
    log.append([make_match_result([(0, "main", "ckpt_v10"), (1, "random", "fixed")], uid="R/pulse/p0",
                                  source="pulse", block="b0")])
    log.append(train[3:], tournament=None)
    log.append([make_match_result([(0, "a#1", "latest"), (1, "b#2", "latest")], uid="T/x", source="tournament",
                                  block="t0")], tournament="t")
    log.close()
    with MatchLogReader(tmp_path / MATCH_LOG_FILE) as reader:
        rows = list(reader.results_after(2))
        assert [r for r, _ in rows] == [3, 5, 6, 7]                         # train rows only, ascending
        assert [res.uid for _, res in rows] == [r.uid for r in train[2:]]
        assert all(res.source == "train" for _, res in rows)
        assert [r for r, _ in reader.results_after(None, batch=2)] == [1, 2, 3, 5, 6, 7]
        steps = sorted(r.env_steps for r in train)
        assert [res.env_steps for _, res in reader.results_after(None, min_env_steps=steps[3])] == steps[4:]
        assert list(reader.results_after(7)) == []
        assert reader.max_env_steps() == steps[-1]
