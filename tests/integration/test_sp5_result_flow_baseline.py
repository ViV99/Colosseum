"""The result flow of a training run, read only from run-dir artefacts (SP5 T0.1, spec §6 "Контрактные тесты до
переноса", block 4.3): every match a worker finished is in the journal exactly once or counted in
``dropped_match_results``; uids are unique, carry the run id and name the worker, env and episode; the journal saw no
duplicate. Written on today's code; T2.5 runs it on the hub + local node unchanged except ``_finished_episodes``
(the worker logs' "finished" line), and T4.3 / T7.1 extend it (node column, ``by_reason``)."""
from __future__ import annotations

import ast
import re
import sqlite3

from cli_runner import TTT_CONFIG, run_train
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader
from colosseum.core.run_dir import read_run_id

BUDGET = 8000
UID_RE = re.compile(r"^(?P<run>[0-9a-f]{32})/train/(?:n(?P<n>\d+)/)?w(?P<w>\d+)_e(?P<e>\d+)_(?:g(?P<g>\d+)_)?"
                    r"ep(?P<k>\d+)$")


def _finished_episodes(run) -> int:
    """Episodes every worker finished, from each worker log's last "finished" line."""
    total = 0
    for log in sorted((run.root / "logs").glob("worker-*.log")):
        lines = [line for line in log.read_text().splitlines() if ": finished. " in line]
        if lines:
            total += int(ast.literal_eval(lines[-1].split("finished. ", 1)[1])["episodes"])
    return total


def _journal_rows(path) -> int:
    """Training matches in the journal (detailed rows + compacted pair counts; other sources are not counted)."""
    conn = sqlite3.connect(path)
    try:
        detailed = conn.execute("SELECT COUNT(*) FROM matches m JOIN entities e ON e.id = m.source "
                                "WHERE e.kind = 'source' AND e.name = 'train'").fetchone()[0]
        counted = conn.execute("SELECT COALESCE(SUM(c.n), 0) FROM counts c JOIN entities e ON e.id = c.source "
                               "WHERE c.kind = 0 AND e.kind = 'source' AND e.name = 'train'").fetchone()[0]
    finally:
        conn.close()
    return detailed + counted                     # tic-tac-toe: one team pair per match


def test_every_finished_match_is_journaled_once_or_counted(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="flow", overrides={"training.total_timesteps": str(BUDGET),
                                                                   "match_log.detail_window_mb": "512"})
    assert run.returncode == 0, run.stderr[-3000:]
    finished = _finished_episodes(run)
    system = run.records("system")[-1]
    assert finished > 100
    assert _journal_rows(run.root / MATCH_LOG_FILE) + system["dropped_match_results"] == finished
    assert system["match_log"]["duplicates"] == 0
    run_id = read_run_id(run.root)
    with MatchLogReader(run.root / MATCH_LOG_FILE) as reader:
        rows = reader.matches(source="train", limit=None)
    uids = [r.uid for r in rows]
    assert len(uids) == len(set(uids)) == _journal_rows(run.root / MATCH_LOG_FILE)
    parsed = [UID_RE.match(uid) for uid in uids]
    assert all(m is not None and m["run"] == run_id for m in parsed), uids[:5]
    episodes = {(m["n"], m["w"], m["e"], m["g"], m["k"]) for m in parsed}
    assert len(episodes) == len(uids)                             # one row per (node, worker, env, epoch, episode)
