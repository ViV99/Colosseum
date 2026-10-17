"""``colosseum tournament`` (SP4 T5.4, spec block 3): select on a run with --pin, the journal and the reports,
resume by whole blocks, the settings check, snapshot links that survive eviction, the env check, several --run
with aliases, --output without a run, scripted teammates in score layouts, SP3 run dirs and checkpoints, usage
errors, absolute paths in tournament.yaml (resume from another cwd), an unconfirmed select winner with --pin.
Every invocation runs in-process (CliRunner) and spawns at most two workers."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import colosseum.arena.tournament.run as run_module
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
from colosseum.arena.tournament.players import entity_hash
from colosseum.cli import main
from game_helpers import copy_sp3_run, scripted_agent, write_fake_run, write_test_config

AGENTS = {"main": {}, "rnd": scripted_agent(), "ones": scripted_agent("game_helpers.ConstantBot", value=1)}
# TurnTakingGame pays the acting seat's action to the other seat: always 0 wins, always 1 gives everything away
STRONG, UNIFORM, WEAK = [20.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-20.0, 20.0, 0.0]


def invoke(*args):
    return CliRunner().invoke(main, ["tournament", *map(str, args)])


def fake_run(tmp_path, name="run", versions=(10, 20, 30), game="turns", **kw):
    return write_fake_run(tmp_path / name, game, agents=AGENTS, versions=versions,
                          biases={10: WEAK, 20: UNIFORM, 30: STRONG} if game == "turns" else None, **kw)


def report(root, name) -> dict:
    return json.loads((root / "tournaments" / name / "report.json").read_text())


def no_play(monkeypatch):
    """Every later block must come from the journal: playing one is a test failure (``append`` deduplicates
    uids, so a replayed block would leave the journal unchanged and go unnoticed otherwise)."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("a block was played again instead of being read from the journal")

    monkeypatch.setattr(run_module, "play_blocks", refuse)


def journal(db, name):
    reader = MatchLog.open_readonly(db)
    try:
        return reader.blocks(name), reader.match_results(source="tournament", tournament=name)
    finally:
        reader.close()


def test_select_on_a_run_writes_the_journal_and_reports_and_pins_the_confirmed_winner(tmp_path, monkeypatch):
    root, _cfg = fake_run(tmp_path)
    result = invoke("--run", root, "--mode", "select", "--field", "rnd,ones", "--halving-blocks", 2,
                    "--confirm-blocks", 16, "--finalists", 2, "--num-workers", 2, "--num-envs", 2, "--seed", 3,
                    "--name", "sel", "--pin")
    assert result.exit_code == 0, result.output
    data = report(root, "sel")
    sel = data["selection"]["player"]
    assert sel["winner"] == "main@v30" and sel["confirmed"] is True
    assert json.loads((root / "checkpoints" / "main" / "ckpt_v30" / "meta.json").read_text())["pinned"] is True
    assert "Pinned main@v30" in result.output
    tdir = root / "tournaments" / "sel"
    for name in ("tournament.yaml", "report.json", "report.txt", "report.html"):
        assert (tdir / name).is_file()
    assert (tdir / "logs" / "tournament.log").is_file()
    blocks, results = journal(root / MATCH_LOG_FILE, "sel")
    assert len(blocks) == data["games"]["blocks"] and len(results) == data["games"]["matches"]
    assert len({r.uid for r in results}) == len(results) and {r.source for r in results} == {"tournament"}
    assert all(r.uid.startswith("sel/") and r.env_steps is None for r in results)
    no_play(monkeypatch)
    again = invoke("--run", root, "--mode", "select", "--field", "rnd,ones", "--halving-blocks", 2,
                   "--confirm-blocks", 16, "--finalists", 2, "--num-workers", 1, "--num-envs", 2, "--seed", 3,
                   "--name", "sel")
    assert again.exit_code == 0, again.output
    assert f"Blocks: 0 played, {len(blocks)} from the journal" in again.output
    assert journal(root / MATCH_LOG_FILE, "sel")[0] == blocks
    assert len(journal(root / MATCH_LOG_FILE, "sel")[1]) == len(results)


def test_a_pinned_winner_evicted_during_the_tournament_is_reported_after_the_report(tmp_path, monkeypatch):
    root, _cfg = fake_run(tmp_path)
    original = run_module.BlockRunner._store

    def evicting(self, block_id, results):
        shutil.rmtree(root / "checkpoints" / "main" / "ckpt_v30", ignore_errors=True)   # the run evicts it
        original(self, block_id, results)

    monkeypatch.setattr(run_module.BlockRunner, "_store", evicting)
    result = invoke("--run", root, "--mode", "select", "--field", "rnd,ones", "--halving-blocks", 2,
                    "--confirm-blocks", 16, "--finalists", 2, "--num-envs", 2, "--seed", 3, "--name", "ev", "--pin")
    assert result.exit_code == 1, result.output
    copy = root / "tournaments" / "ev" / "snapshots" / f"0-{root.name}" / "main" / "ckpt_v30"
    assert "winner main@v30 (confirmed" in result.output                 # the report comes first
    assert "Pin error" in result.output and "evicted" in result.output and str(copy) in result.output
    assert (root / "tournaments" / "ev" / "report.json").is_file() and (copy / "model.pt").is_file()
    assert "Pinned" not in result.output


def test_an_unconfirmed_select_winner_is_not_pinned_and_the_command_still_succeeds(tmp_path, caplog):
    root, _cfg = fake_run(tmp_path)
    with caplog.at_level(logging.WARNING, logger="colosseum"):
        result = invoke("--run", root, "--mode", "select", "--field", "rnd", "--halving-blocks", 1,
                        "--confirm-blocks", 2, "--finalists", 1, "--num-envs", 2, "--name", "lone", "--pin")
    assert result.exit_code == 0, result.output                         # confirmed: false is not an error
    sel = report(root, "lone")["selection"]["player"]
    assert sel["confirmed"] is False
    for version in (10, 20, 30):
        meta = json.loads((root / "checkpoints" / "main" / f"ckpt_v{version}" / "meta.json").read_text())
        assert meta.get("pinned") is not True
    assert "Pinned" not in result.output
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "--pin" in r.getMessage()]
    assert warnings and "colosseum pin" in warnings[0] and sel["winner"] in warnings[0]


def test_an_interrupted_tournament_resumes_with_whole_blocks_only(tmp_path, monkeypatch):
    root, _cfg = fake_run(tmp_path)
    args = ["--run", root, "--snapshots", "last:2", "--field", "rnd", "--blocks", 3, "--num-envs", 2, "--name", "g"]
    original = run_module.BlockRunner._store
    stored = []

    def failing(self, block_id, results):
        if len(stored) == 2:
            raise RuntimeError("interrupted on purpose")
        stored.append(block_id)
        original(self, block_id, results)

    monkeypatch.setattr(run_module.BlockRunner, "_store", failing)
    first = invoke(*args)
    assert first.exit_code != 0 and isinstance(first.exception, RuntimeError)
    assert len(journal(root / MATCH_LOG_FILE, "g")[0]) == 2
    monkeypatch.setattr(run_module.BlockRunner, "_store", original)
    second = invoke(*args)
    assert second.exit_code == 0, second.output
    assert "4 played, 2 from the journal" in second.output              # 2 candidates x 1 field x 3 blocks = 6
    blocks, results = journal(root / MATCH_LOG_FILE, "g")
    assert len(blocks) == 6 and len(results) == 12
    changed = invoke(*args[:-2], "--blocks", 4, "--name", "g")
    assert changed.exit_code == 1 and "blocks" in changed.output and "tournament.yaml" in changed.output


def test_tournament_snapshots_survive_eviction_by_the_run(tmp_path, monkeypatch):
    root, _cfg = fake_run(tmp_path)
    args = ["--run", root, "--field", "rnd", "--blocks", 2, "--num-envs", 2, "--name", "links"]
    original = run_module.BlockRunner._store
    stored = []

    def failing(self, block_id, results):
        if len(stored) == 1:          # main@v10 has played one of its two blocks
            raise RuntimeError("interrupted on purpose")
        stored.append(block_id)
        original(self, block_id, results)

    monkeypatch.setattr(run_module.BlockRunner, "_store", failing)
    assert isinstance(invoke(*args).exception, RuntimeError)
    monkeypatch.setattr(run_module.BlockRunner, "_store", original)
    linked = root / "tournaments" / "links" / "snapshots" / f"0-{root.name}" / "main" / "ckpt_v10" / "model.pt"
    assert linked.is_file() and linked.stat().st_nlink >= 2              # a hard link (same file system)
    shutil.rmtree(root / "checkpoints" / "main" / "ckpt_v10")             # the run evicts it before the rest plays
    again = invoke(*args)
    assert again.exit_code == 0, again.output
    assert "Blocks: 5 played, 1 from the journal" in again.output
    blocks, results = journal(root / MATCH_LOG_FILE, "links")
    v10 = {r.block for r in results if any(s.network_id == "ckpt_v10" for s in r.seats)}
    assert len(v10) == 2 and v10 - set(stored)                           # its second block played from snapshots/
    assert linked.is_file()


def test_a_new_tournament_never_reuses_the_journal_blocks_of_a_removed_one(tmp_path):
    root, _cfg = fake_run(tmp_path, versions=(10,))
    args = ["--run", root, "--field", "rnd", "--blocks", 2, "--num-envs", 2, "--name", "t"]
    assert invoke(*args, "--seed", 1).exit_code == 0
    shutil.rmtree(root / "tournaments" / "t")
    again = invoke(*args, "--seed", 99, "--deterministic")
    assert again.exit_code == 1 and "already holds 2 blocks" in again.output and "another --name" in again.output
    assert "remove it" not in again.output
    assert not (root / "tournaments" / "t" / "tournament.yaml").exists()


def test_tournament_yaml_stores_absolute_paths_and_resumes_from_another_cwd(tmp_path, monkeypatch):
    root, _cfg = fake_run(tmp_path, versions=(10, 30))
    monkeypatch.chdir(tmp_path)
    rel = Path("run") / "checkpoints" / "main" / "ckpt_v10"
    first = invoke("--run", "run", "--snapshots", "last:1", "-a", f"prev={rel}", "--field", "rnd", "--blocks", 1,
                   "--num-envs", 2, "--name", "cwd")
    assert first.exit_code == 0, first.output
    stored = yaml.safe_load((root / "tournaments" / "cwd" / "tournament.yaml").read_text())
    assert [r["path"] for r in stored["request"]["runs"]] == [str(root.resolve())]
    assert stored["request"]["candidates"] == [f"prev={(tmp_path / rel).resolve()}"]
    for player in stored["players"]:
        if player["source"]["kind"] == "model":
            assert Path(player["source"]["weights"]).is_absolute() and Path(player["source"]["origin"]).is_absolute()
        if player["run"] is not None:
            assert Path(player["run"]).is_absolute()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    no_play(monkeypatch)
    again = invoke("--run", root, "--snapshots", "last:1", "-a", f"prev={tmp_path / rel}", "--field", "rnd",
                   "--blocks", 1, "--num-envs", 2, "--name", "cwd")
    assert again.exit_code == 0, again.output
    assert "Blocks: 0 played, 2 from the journal" in again.output


def test_the_env_of_the_tournament_config_must_be_the_runs(tmp_path):
    root, _cfg = fake_run(tmp_path)
    other = write_test_config(tmp_path / "other.yaml", "turns", env={"kwargs": {"length": 8}}, agents=AGENTS)
    result = invoke("-c", other, "--run", root, "--field", "rnd", "--blocks", 1)
    assert result.exit_code == 1 and "env" in result.output and str(root) in result.output


def test_several_runs_with_aliases_and_snapshot_references_in_the_field(tmp_path):
    first, _ = fake_run(tmp_path, "a", versions=(10, 20, 30))
    second, _ = fake_run(tmp_path, "b", versions=(5, 7))
    # --agent / --snapshots belong to the --run before them, also in the --opt=value form
    result = invoke("--run", first, "--snapshots", "last:1", f"--run=old={second}", "--agent=main", "--snapshots=all",
                    "--field", "rnd,main@v10", "--blocks", 1, "--num-envs", 2, "--name", "multi")
    assert result.exit_code == 0, result.output
    data = report(first, "multi")
    assert [c["name"] for c in data["candidates"]] == ["main@v30", "old/main@v5", "old/main@v7"]
    assert [f["name"] for f in data["field"]] == ["rnd", "main@v10"]
    assert (first / MATCH_LOG_FILE).is_file() and not (second / MATCH_LOG_FILE).exists()


def test_output_dir_without_a_run_and_scripted_and_frozen_teammates_in_a_score_layout(tmp_path):
    root, cfg_ = fake_run(tmp_path, game="coop", versions=(1,))
    cfg = write_test_config(tmp_path / "coop.yaml", "coop", agents=AGENTS)
    ckpt = root / "checkpoints" / "main" / "ckpt_v1"
    out = tmp_path / "out"
    result = invoke("-c", cfg, "-a", f"prev={ckpt}", "--field", "ones", "--blocks", 2, "--num-envs", 2, "--output", out,
                    "--name", "coop")
    assert result.exit_code == 0, result.output
    data = json.loads((out / "tournaments" / "coop" / "report.json").read_text())
    (table,) = data["tables"]
    assert table["metric"] == "score" and table["columns"] == ["(self)", "ones"]
    assert table["rows"][0]["name"] == f"prev#{entity_hash(ckpt)}"
    assert (out / MATCH_LOG_FILE).is_file()
    # a frozen teammate (spec section 6, SP3 item): a checkpoint given with -a and named in --field
    mate = invoke("-c", cfg, "-a", f"prev={ckpt}", "-a", f"mate={ckpt}", "--field", "mate", "--blocks", 2,
                  "--num-envs", 2, "--output", out, "--name", "coop-frozen")
    assert mate.exit_code == 0, mate.output
    data = json.loads((out / "tournaments" / "coop-frozen" / "report.json").read_text())
    (table,) = data["tables"]
    assert table["columns"][0] == "(self)" and table["columns"][1].startswith("mate")
    assert data["field"][0]["kind"] != "scripted"


def test_sp3_run_dirs_play_as_runs_and_their_checkpoints_as_frozen_agents(tmp_path):
    sp3 = copy_sp3_run(tmp_path)
    agent = next(p.name for p in (sp3 / "checkpoints").iterdir() if p.is_dir())
    ckpt = sorted((sp3 / "checkpoints" / agent).glob("ckpt_v*"))[0]
    top = invoke("--run", sp3, "--snapshots", "top:1", "-a", f"old={ckpt}", "--field", "old", "--name", "top")
    assert top.exit_code == 1 and "match log" in top.output              # before this test writes a journal there
    result = invoke("--run", sp3, "--snapshots", "last:2", "-a", f"old={ckpt}", "--field", "old", "--blocks", 1,
                    "--num-envs", 2, "--name", "sp3")
    assert result.exit_code == 0, result.output
    data = report(sp3, "sp3")
    assert all(c["name"].startswith(f"{agent}@v") for c in data["candidates"]) and len(data["candidates"]) == 2
    assert data["field"][0]["name"].startswith("old#")


@pytest.mark.parametrize("args, message", [
    (["--agent", "main", "--run", "RUN"], "--agent"),
    (["--run", "RUN", "--field", "rnd", "--pin"], "--pin"),
    (["--run", "RUN", "--field", "rnd", "--output", "OUT"], "--output"),
    (["-a", "x=nowhere.pt"], "--run"),
    (["--run", "RUN", "--snapshots", "all", "--snapshots", "last:1"], "twice"),
    (["-c", "CFG", "--field", "rnd"], "--output"),
])
def test_usage_errors_exit_with_2(tmp_path, args, message):
    root, _cfg = fake_run(tmp_path)
    subst = {"RUN": str(root), "OUT": str(tmp_path / "out"), "CFG": str(root / "config.resolved.yaml")}
    args = [subst.get(a, a) for a in args]
    result = invoke(*args)
    assert result.exit_code == 2 and message in result.output


def test_one_team_games_need_no_field_and_two_team_games_still_do(tmp_path):
    """B-I1 (spec block 3: a score block is the candidate's homogeneous team; the field only adds cross-play): a
    solo game whose config has no scripted / frozen agent selects among its snapshots with an empty field; a
    two-team game's empty field stays the same ConfigError."""
    root, _cfg = write_fake_run(tmp_path / "solo", "solo", versions=(10, 20, 30))
    result = invoke("--run", root, "--mode", "select", "--halving-blocks", 1, "--confirm-blocks", 2, "--finalists",
                    2, "--num-envs", 2, "--name", "solo")
    assert result.exit_code == 0, result.output
    data = report(root, "solo")
    assert data["field"] == [] and data["selection"]["player"]["winner"].startswith("main@v")
    assert {t["metric"] for t in data["tables"]} == {"score"} and all(t["columns"] == ["(self)"]
                                                                     for t in data["tables"])
    two, _cfg = write_fake_run(tmp_path / "turns", "turns", versions=(10,))
    refused = invoke("--run", two, "--blocks", 1, "--num-envs", 2, "--name", "empty")
    assert refused.exit_code == 1
    assert "the field is empty: name opponents with --field (agents of the config, -a players or snapshots)" \
        in refused.output


def test_a_retried_start_never_plays_the_snapshot_links_of_a_failed_one(tmp_path):
    """B-M6: a start that failed after linking its snapshots, retried with the same --name and another second --run
    of the same dir name, links the new run's weights (a new tournament empties snapshots/ first); a complete link
    dir that does not hold the snapshot's own files is replaced, not trusted by its path."""
    import os

    from colosseum.arena.tournament.players import RunSelection
    from colosseum.arena.tournament.run import TournamentDir
    from colosseum.coordinator.checkpoint_manager import read_snapshots

    root, _cfg = fake_run(tmp_path, versions=(30,))
    first, _ = fake_run(tmp_path / "x", "other", versions=(5,))
    second, _ = fake_run(tmp_path / "y", "other", versions=(1, 5))          # other weights (seed by position)
    args = ["--run", root, "--field", "rnd", "--blocks", 1, "--num-envs", 2, "--name", "retry"]
    failed = invoke("--run", root, f"--run=old={first}", "--field", "nobody", "--blocks", 1, "--name", "retry")
    assert failed.exit_code == 1 and "nobody" in failed.output
    link = root / "tournaments" / "retry" / "snapshots" / "1-other" / "main" / "ckpt_v5" / "model.pt"
    assert os.path.samefile(link, first / "checkpoints" / "main" / "ckpt_v5" / "model.pt")   # linked, then failed
    again = invoke("--run", root, f"--run=old={second}", *args[2:])
    assert again.exit_code == 0, again.output
    assert os.path.samefile(link, second / "checkpoints" / "main" / "ckpt_v5" / "model.pt")
    # a complete link dir with other files is relinked
    tdir = TournamentDir(tmp_path / "t")
    (info,) = read_snapshots(first / "checkpoints" / "main")
    dst = tdir.link(1, RunSelection(first), info)
    other = read_snapshots(second / "checkpoints" / "main")[-1]
    assert tdir.link(1, RunSelection(second), other) == dst
    assert os.path.samefile(dst / "model.pt", other.path / "model.pt")


def test_journal_and_eviction_errors_are_config_errors_with_their_context(tmp_path, monkeypatch):
    """B-M3: a journal of another schema version, a journal write that fails (a long lock by another process) and a
    snapshot the run evicts while the tournament starts are ConfigErrors naming the journal / the snapshot, not
    tracebacks; after the eviction a rerun works."""
    import sqlite3

    import colosseum.arena.tournament.players as players_module
    from colosseum.coordinator.checkpoint_manager import read_snapshots

    root, _cfg = fake_run(tmp_path, versions=(10, 20))
    db = root / MATCH_LOG_FILE
    args = ["--run", root, "--field", "rnd", "--blocks", 1, "--num-envs", 2]
    MatchLog(db, None).close()
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    other = invoke(*args, "--name", "schema")
    assert other.exit_code == 1 and isinstance(other.exception, SystemExit), other.output
    assert "Config error" in other.output and str(db) in other.output and "schema version 99" in other.output
    db.unlink()

    def locked(self, results, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(MatchLog, "append", locked)
    busy = invoke(*args, "--name", "busy")
    assert busy.exit_code == 1 and isinstance(busy.exception, SystemExit), busy.output
    assert "Config error" in busy.output and str(db) in busy.output and "database is locked" in busy.output
    monkeypatch.undo()

    def evicting(agent_dir):
        infos = read_snapshots(agent_dir)
        shutil.rmtree(Path(agent_dir) / "ckpt_v10")              # the run evicts it right after the listing
        return infos

    monkeypatch.setattr(players_module, "read_snapshots", evicting)
    race = invoke(*args, "--name", "race")
    assert race.exit_code == 1 and isinstance(race.exception, SystemExit), race.output
    assert "ckpt_v10" in race.output and "evicted by the run while the tournament started; rerun" in race.output
    monkeypatch.undo()
    rerun = invoke(*args, "--name", "race")
    assert rerun.exit_code == 0, rerun.output
    assert [c["name"] for c in report(root, "race")["candidates"]] == ["main@v20"]
