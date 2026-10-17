"""Slow disk-bound test (SP4 criterion 3.5): a tic_tac_toe run far longer than a 1 MB detail window keeps
``matches.db`` within the window + what compaction never touches + a slack (T8.3 ruling), its counts are
complete (the BT fit from the compacted journal equals the run's final live fit over the in-memory
comparison table, which saw every written result: the same per-entity games exactly, ratings within 0.5 Elo),
a compaction of a copy of the final journal down to the window keeps the detail within 1.1 x the window and
the fit unchanged, and every agent keeps at most ``max_snapshots`` snapshots during the whole run (sampled
every 0.5 s; a save places the new snapshot before retention renames the evicted one away, a few ms, so a
listing over the cap counts only when two more listings 0.1 s apart are over it too). 2 rollout workers."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time

import pytest

from cli_runner import TTT_CONFIG, run_train
from colosseum.arena import bt
from colosseum.arena.live import rating_inputs
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
from colosseum.core.config import load_config

pytestmark = [pytest.mark.slow, pytest.mark.timeout(900)]

MB = 2**20
WINDOW_MB = 1.0
MAX_SNAPSHOTS = 5
# T8.3 ruling: peak excess 6.40 MB (peak 7.50 MB, untouched 0.10 MB, window 1 MB) -> max(4, ceil(2 x 6.40)) = 13 MB.
# The excess is SQLite's WAL (about 4 MB: autocheckpoint at 1000 pages, the file is truncated only after a
# compaction pass) plus the matches of one compaction interval (~1 MB/s here).
SLACK_MB = 13.0
RATING_TOL_ELO = 0.5
COMPACT_TOL_ELO = 1e-3          # the same sufficient statistics, folded into counts: only rounding may differ
OVERRIDES = {
    "training.total_timesteps": 400_000, "rollout.num_workers": 2, "rollout.envs_per_worker": 16,
    # the run lasts ~20 s: with the config's 30 s lineup refresh every match would be latest-vs-latest self-play
    # (snapshots did not exist at the first lineups), which BT does not rate, and the comparison below is empty
    "rollout.match_refresh_interval_sec": 1.0,
    "match_log.detail_window_mb": WINDOW_MB, "match_log.compact_interval_sec": 2.0,
    "checkpoint.interval": 20, "checkpoint.keep_last": 3, "checkpoint.keep_every": 2,
    "checkpoint.max_snapshots": MAX_SNAPSHOTS, "ratings.refit_interval_sec": 5.0,
}


def _snapshot_counts(checkpoints) -> dict[str, int]:
    """``ckpt_v*`` dirs per agent dir under ``checkpoints`` (a listing that races a rename gives what it got)."""
    counts: dict[str, int] = {}
    try:
        for agent_dir in checkpoints.iterdir():
            counts[agent_dir.name] = sum(1 for d in agent_dir.iterdir() if re.fullmatch(r"ckpt_v\d+", d.name))
    except OSError:
        pass
    return counts


def _compacted_copy(db, target):
    """A copy of the journal ``db`` (SQLite backup API) compacted down to the window: (sizes, comparison table)."""
    target.parent.mkdir(parents=True)
    src = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    log = MatchLog(target, None)
    try:
        log.compact(window_bytes=int(WINDOW_MB * MB))
        return log.sizes(), log.comparison_table(sources=("train", "pulse"))
    finally:
        log.close()


def test_journal_and_snapshots_stay_bounded_and_counts_stay_complete(tmp_path):
    db = tmp_path / "runs" / "disk" / MATCH_LOG_FILE
    checkpoints = db.parent / "checkpoints"
    peak = {"bytes": 0, "over_cap_listings": 0}
    peak_snapshots: dict[str, int] = {}
    stop = threading.Event()

    def poll():
        while not stop.wait(0.5):
            total = sum(p.stat().st_size for p in (db, db.with_name(db.name + "-wal")) if p.exists())
            peak["bytes"] = max(peak["bytes"], total)
            counts = _snapshot_counts(checkpoints)
            for _ in range(2):                     # over the cap: a save in progress, unless it persists
                if max(counts.values(), default=0) <= MAX_SNAPSHOTS:
                    break
                peak["over_cap_listings"] += 1
                stop.wait(0.1)
                again = _snapshot_counts(checkpoints)
                counts = {agent: min(n, again.get(agent, n)) for agent, n in counts.items()}
            for agent, n in counts.items():
                peak_snapshots[agent] = max(peak_snapshots.get(agent, 0), n)

    sampler = threading.Thread(target=poll, daemon=True)
    sampler.start()
    start = time.monotonic()
    try:
        run = run_train(TTT_CONFIG, tmp_path, name="disk", overrides=OVERRIDES, timeout=800, tiny=False)
    finally:
        stop.set()
        sampler.join(timeout=5)
    seconds = time.monotonic() - start
    assert run.returncode == 0, run.stderr[-3000:]
    with MatchLog.open_readonly(db) as reader:
        sizes = reader.sizes()
        table = reader.comparison_table(sources=("train", "pulse"))
        detail = len(reader.matches(source="train", limit=None))
    total_pairs = sum(stat.n for a, b, stat in table.pairs("2p", ("train",)))
    untouched = sizes.file_bytes - sizes.detail_bytes
    excess = peak["bytes"] - WINDOW_MB * MB - untouched
    system = run.records("system")
    dropped = max((r.get("dropped_match_results", 0) for r in system), default=0)
    compactions = max(((r.get("match_log") or {}).get("compactions") or 0 for r in system), default=0)
    print(f"[disk] {seconds:.0f}s; peak {peak['bytes'] / MB:.2f} MB; file {sizes.file_bytes / MB:.2f} MB, wal "
          f"{sizes.wal_bytes / MB:.2f} MB, detail {sizes.detail_bytes / MB:.2f} MB, untouched {untouched / MB:.2f} "
          f"MB, peak excess {excess / MB:.2f} MB; counts rows {sizes.counts_rows}; matches {total_pairs}, detail "
          f"kept {detail}; compaction passes {compactions}; dropped match results {dropped}")
    assert total_pairs >= 3 * detail, "the run never outgrew the window: nothing was compacted"
    # The writer compacts every compact_interval_sec and not at the stop (it only flushes its queue within the
    # shutdown grace), so the run ends with the window plus the matches of up to one interval: the slack's share.
    assert sizes.detail_bytes <= (WINDOW_MB + SLACK_MB) * MB
    assert peak["bytes"] <= WINDOW_MB * MB + untouched + SLACK_MB * MB

    config = load_config(run.root / "config.resolved.yaml")
    options, fixed, random_bot = rating_inputs(config)
    from_journal = bt.fit(table, "2p", sources=("train", "pulse"), options=options, fixed_agents=fixed,
                          random_bot=random_bot)
    final_bt = json.loads((run.root / "ratings.json").read_text())["bt"]["2p"]
    rows = [e for label, group in final_bt.items() if not label.startswith("@")
            for e in group]                                  # A14: rows per group label, '@' keys are extras (P5)
    live = {e["entity"]: e["rating"] for e in rows}
    live_games = {e["entity"]: e["games"] for e in rows}
    rated_games = sum(from_journal.games.values())
    print(f"[disk] rated: {len(live)} entities live, {len(from_journal.rating)} from the journal, "
          f"{rated_games} entity-games (self-play excluded)")
    assert len(live) >= 2, "no rated entities: the comparison of the two fits would be empty"
    assert set(live) == set(from_journal.rating)
    # counts complete, exactly: every entity rated on the same number of games in the live table and the journal
    assert live_games == {e: int(from_journal.games.get(e, 0)) for e in from_journal.rating}
    worst = max(abs(live[e] - from_journal.rating[e]) for e in live)
    print(f"[disk] {len(live)} entities; max |live - journal| = {worst:.4f} Elo")
    assert worst <= RATING_TOL_ELO

    # the window itself (the brief's 1.1 x check, where it is valid: no writer, compaction run to the end)
    sizes_c, table_c = _compacted_copy(db, tmp_path / "compacted" / MATCH_LOG_FILE)
    compacted = bt.fit(table_c, "2p", sources=("train", "pulse"), options=options, fixed_agents=fixed,
                       random_bot=random_bot)
    moved = max(abs(compacted.rating[e] - from_journal.rating[e]) for e in from_journal.rating)
    print(f"[disk] compacted copy: detail {sizes_c.detail_bytes / MB:.2f} MB (window {WINDOW_MB} MB), counts rows "
          f"{sizes_c.counts_rows}; max |compacted - journal| = {moved:.6f} Elo")
    assert sizes_c.detail_bytes <= 1.1 * WINDOW_MB * MB
    assert set(compacted.rating) == set(from_journal.rating) and compacted.games == from_journal.games
    assert moved <= COMPACT_TOL_ELO

    print(f"[disk] most snapshots per agent during the run: {peak_snapshots}; listings over the cap "
          f"{peak['over_cap_listings']} (a save in progress)")
    assert peak_snapshots, "the poll never saw a snapshot"
    assert max(peak_snapshots.values()) <= MAX_SNAPSHOTS, peak_snapshots
    for agent_dir in (run.root / "checkpoints").iterdir():
        kept = [d for d in agent_dir.iterdir() if re.fullmatch(r"ckpt_v\d+", d.name)]
        assert len(kept) <= MAX_SNAPSHOTS, (agent_dir.name, sorted(d.name for d in kept))
