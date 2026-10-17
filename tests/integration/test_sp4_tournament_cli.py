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


def journal(db, name):
    reader = MatchLog.open_readonly(db)
    try:
        return reader.blocks(name), reader.match_results(source="tournament", tournament=name)
    finally:
        reader.close()


def test_select_on_a_run_writes_the_journal_and_reports_and_pins_the_confirmed_winner(tmp_path):
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
    for name in ("tournament.yaml", "report.json", "report.txt"):
        assert (tdir / name).is_file()
    assert (tdir / "logs" / "tournament.log").is_file()
    blocks, results = journal(root / MATCH_LOG_FILE, "sel")
    assert len(blocks) == data["games"]["blocks"] and len(results) == data["games"]["matches"]
    assert len({r.uid for r in results}) == len(results) and {r.source for r in results} == {"tournament"}
    assert all(r.uid.startswith("sel/") and r.env_steps is None for r in results)
    again = invoke("--run", root, "--mode", "select", "--field", "rnd,ones", "--halving-blocks", 2,
                   "--confirm-blocks", 16, "--finalists", 2, "--num-workers", 1, "--num-envs", 2, "--seed", 3,
                   "--name", "sel")
    assert again.exit_code == 0, again.output
    assert "0 played" in again.output                                   # every block came from the journal
    assert journal(root / MATCH_LOG_FILE, "sel")[0] == blocks


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


def test_tournament_snapshots_survive_eviction_by_the_run(tmp_path):
    root, _cfg = fake_run(tmp_path)
    args = ["--run", root, "--field", "rnd", "--blocks", 1, "--num-envs", 2, "--name", "links"]
    assert invoke(*args).exit_code == 0
    linked = root / "tournaments" / "links" / "snapshots" / f"0-{root.name}" / "main" / "ckpt_v10" / "model.pt"
    assert linked.is_file() and linked.stat().st_nlink >= 2              # a hard link (same file system)
    shutil.rmtree(root / "checkpoints" / "main" / "ckpt_v10")             # the run evicts it
    again = invoke(*args)
    assert again.exit_code == 0, again.output
    assert linked.is_file() and "0 played" in again.output


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
    again = invoke("--run", root, "--snapshots", "last:1", "-a", f"prev={tmp_path / rel}", "--field", "rnd",
                   "--blocks", 1, "--num-envs", 2, "--name", "cwd")
    assert again.exit_code == 0, again.output
    assert "0 played" in again.output


def test_the_env_of_the_tournament_config_must_be_the_runs(tmp_path):
    root, _cfg = fake_run(tmp_path)
    other = write_test_config(tmp_path / "other.yaml", "turns", env={"kwargs": {"length": 8}}, agents=AGENTS)
    result = invoke("-c", other, "--run", root, "--field", "rnd", "--blocks", 1)
    assert result.exit_code == 1 and "env" in result.output and str(root) in result.output


def test_several_runs_with_aliases_and_snapshot_references_in_the_field(tmp_path):
    first, _ = fake_run(tmp_path, "a", versions=(10, 30))
    second, _ = fake_run(tmp_path, "b", versions=(5,))
    result = invoke("--run", first, "--snapshots", "last:1", "--run", f"old={second}", "--field", "rnd,main@v10",
                    "--blocks", 1, "--num-envs", 2, "--name", "multi")
    assert result.exit_code == 0, result.output
    data = report(first, "multi")
    assert [c["name"] for c in data["candidates"]] == ["main@v30", "old/main@v5"]
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
])
def test_usage_errors_exit_with_2(tmp_path, args, message):
    root, _cfg = fake_run(tmp_path)
    args = [str(root) if a == "RUN" else str(tmp_path / "out") if a == "OUT" else a for a in args]
    result = invoke(*args)
    assert result.exit_code == 2 and message in result.output
