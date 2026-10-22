"""The match log in real training runs (SP4 T1.3, spec block 1 and section 6): every match a worker finished is
journaled exactly once or counted as dropped, seats carry their weights and rating versions, uids carry the run
id, the system record reports the journal, a run-dir resume continues the journal (the parent's detail folded into
counts), and match_log.enabled: false writes no journal."""
from __future__ import annotations

import ast
import json
import re
import sqlite3

from cli_runner import TTT_CONFIG, run_train
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader
from colosseum.core.config import load_config

BUDGET = 8000          # longer than one worker-stats interval (2 s)


def _worker_stats(run) -> dict:
    line = next(line for line in run.log("worker-0").splitlines() if "Worker 0: finished." in line)
    return ast.literal_eval(line.split("finished. ", 1)[1])


def _accounting(run) -> dict[str, int]:
    m = re.search(r"Match results: (\d+) received, (\d+) journaled, (\d+) duplicates, (\d+) dropped by the journal "
                  r"writer, (\d+) dropped by workers", run.log("main"))
    assert m, run.log("main")[-3000:]
    return dict(zip(("received", "journaled", "duplicates", "writer", "workers"), map(int, m.groups()), strict=True))


def _journal_matches(path) -> int:
    conn = sqlite3.connect(path)
    try:
        detailed = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        counted = conn.execute("SELECT COALESCE(SUM(n), 0) FROM counts WHERE kind = 0").fetchone()[0]
    finally:
        conn.close()
    return detailed + counted                          # tic-tac-toe: one team pair per match


def test_every_match_is_journaled_once_or_counted(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="journal",
                    overrides={"training.total_timesteps": str(BUDGET), "match_log.compact_interval_sec": "1",
                               "match_log.detail_window_mb": "0.05", "checkpoint.keep_last": "1"})
    assert run.returncode == 0, run.stderr[-3000:]
    worker, accounting = _worker_stats(run), _accounting(run)
    assert worker["episodes"] > 100
    assert worker["episodes"] == accounting["received"] + worker["dropped_results"]
    assert accounting["workers"] == worker["dropped_results"] and accounting["duplicates"] == 0
    assert accounting["received"] == accounting["journaled"] + accounting["writer"]
    assert _journal_matches(run.root / MATCH_LOG_FILE) == accounting["journaled"]
    system = run.records("system")[-1]
    assert system["dropped_match_results"] == accounting["writer"] + accounting["workers"]
    assert system["match_log"]["written"] == accounting["journaled"] and system["match_log"]["compactions"] >= 1
    # a 50 KB window: compaction folded matches into counts (the final record also counts the matches written after
    # the last pass; the disk bound itself is criterion 3.5, a slow test of part 3)
    assert system["match_log"]["folded"] > 0 and system["match_log"]["counts_rows"] > 0
    # keep_last 1: snapshots are evicted about one checkpoint interval after their save, while their bucket's matches
    # are the newest in the journal; the evict listener (Launcher._on_snapshot_evicted) makes the next pass fold them
    assert system["match_log"]["evicted_folded"] > 0


def test_journaled_seats_carry_versions_and_uids_the_run_id(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="versions", overrides={"training.total_timesteps": str(BUDGET)})
    assert run.returncode == 0, run.stderr[-3000:]
    run_id = json.loads((run.root / "run.json").read_text())["run_id"]
    reader = MatchLogReader(run.root / MATCH_LOG_FILE)
    rows = reader.matches(limit=100_000)
    assert rows and all(r.uid.startswith(f"{run_id}/train/n0/w0_e") and r.source == "train" for r in rows)
    steps = [r.env_steps for r in sorted(rows, key=lambda r: r.id)]
    assert steps == sorted(steps) and steps[-1] > 0                   # stamped on arrival, in arrival order
    interval = load_config(run.root / "config.resolved.yaml").checkpoint.interval   # 20 (cli_runner.TINY)
    latest = [s for r in rows for s in r.seats if s.network_id == "latest"]
    assert latest and all(s.version_start is not None and s.version_end >= s.version_start for s in latest)
    assert all(s.rating_version == (s.version_start // interval) * interval for s in latest)
    snapshots = [s for r in rows for s in r.seats if s.network_id.startswith("ckpt_v")]
    assert all(s.rating_version == int(s.network_id.removeprefix("ckpt_v")) == s.version_start for s in snapshots)
    assert [i.run_id for i in reader.runs()] == [run_id] and reader.runs()[0].checkpoint_interval == interval
    reader.close()


def test_a_run_dir_resume_continues_the_journal(tmp_path):
    first = run_train(TTT_CONFIG, tmp_path, name="first", overrides={"training.total_timesteps": "4000"})
    assert first.returncode == 0, first.stderr[-3000:]
    parent_id = json.loads((first.root / "run.json").read_text())["run_id"]
    parent = MatchLogReader(first.root / MATCH_LOG_FILE)
    parent_total = sum(s.n for _, _, s in parent.comparison_table().pairs("2p", ["train"]))
    parent.close()
    second = run_train(TTT_CONFIG, tmp_path, name="second",
                       overrides={"training.resume_from": str(first.root), "training.total_timesteps": "8000"})
    assert second.returncode == 0, second.stderr[-3000:]
    assert f"Match log: imported {first.root / MATCH_LOG_FILE}" in second.log("main")
    child = MatchLogReader(second.root / MATCH_LOG_FILE)
    runs = child.runs()
    assert [r.run_id for r in runs][1:] == [parent_id] and runs[0].parent_run_id == parent_id
    child_total = sum(s.n for _, _, s in child.comparison_table().pairs("2p", ["train"]))
    assert child_total > parent_total and child.sizes().counts_rows > 0
    assert all(r.uid.startswith(runs[0].run_id) for r in child.matches(limit=100_000))   # no parent detail copied
    child.close()


def test_a_disabled_journal_writes_nothing(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="nolog", overrides={"match_log.enabled": "false"})
    assert run.returncode == 0, run.stderr[-3000:]
    assert not (run.root / MATCH_LOG_FILE).exists() and "Match log: off" in run.log("main")
    assert run.records("system")[-1]["match_log"] is None
