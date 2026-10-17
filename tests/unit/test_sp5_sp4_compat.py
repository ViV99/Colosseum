"""SP4 compatibility baseline (SP5 T0.1, spec §6 and criterion 6): the SP4 run fixture is an SP4 run dir — journal of
schema version 1 with training, pulse and tournament matches, league and pulse state, a pinned snapshot, run identity
and state — whose snapshots load read-only, whose journal reads without a change, and that `colosseum report` renders.
Created on SP4 code; it stays green through SP5 (T4.3 adds the in-place resume of a copy)."""
from __future__ import annotations

import json

from click.testing import CliRunner

from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader
from colosseum.cli import main
from colosseum.coordinator.checkpoint_manager import check_model_state, load_checkpoint_dir
from colosseum.core.config import NetworkConfig, load_config
from colosseum.core.registry import build_network, env_spec
from colosseum.core.run_dir import read_run_id
from game_helpers import SP4_RUN, SP4_TTT_TINY, copy_sp4_run, sp4_manifest, tree_digest

SP4_META_KEYS = {"agent_id", "checkpoint_id", "config_hash", "env_steps", "final", "networks", "pinned",
                 "policy_version", "role_signature", "roles", "run_id", "timestamp"}


def test_the_run_fixture_is_an_sp4_run_dir():
    manifest = sp4_manifest()
    assert sorted(p.name for p in SP4_RUN.iterdir()) == manifest["files"]
    assert not any(p.name == "logs" for p in SP4_RUN.rglob("*"))           # machine paths stay out
    assert read_run_id(SP4_RUN) == manifest["run_id"]
    agent_dir = SP4_RUN / "checkpoints" / manifest["agent"]
    versions = sorted(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.iterdir())
    assert versions == manifest["versions"] and manifest["final_version"] == max(versions)
    for version in versions:
        meta = json.loads((agent_dir / f"ckpt_v{version}" / "meta.json").read_text())
        assert set(meta) == SP4_META_KEYS and meta["run_id"] == manifest["run_id"]
        assert meta["pinned"] is (version in manifest["pinned"]) and meta["final"] is (version == max(versions))
    assert json.loads((SP4_RUN / "league_state.json").read_text())["version"] == 1
    assert (SP4_RUN / "pulse_state.json").is_file()
    assert json.loads((SP4_RUN / "run_state.json").read_text())["status"] == "finished"
    assert {"bt", "layouts", "env_steps"} <= set(json.loads((SP4_RUN / "ratings.json").read_text()))
    assert (SP4_RUN / "tournaments" / manifest["tournament"] / "report.json").is_file()
    resolved = load_config(SP4_RUN / "config.resolved.yaml")
    assert resolved.get_trainable_agent_ids() == [manifest["agent"]] and resolved.evaluation.enabled


def test_the_sp4_journal_reads_on_a_copy(tmp_path):
    """SQLite in WAL mode creates -wal / -shm files next to a journal it opens: journals are read on copies only."""
    manifest = sp4_manifest()
    copy = copy_sp4_run(tmp_path)
    with MatchLogReader(copy / MATCH_LOG_FILE) as reader:
        counts = {s: len(reader.matches(source=s, limit=None)) for s in ("train", "pulse", "tournament")}
        assert [r.run_id for r in reader.runs()] == [manifest["run_id"]]
    assert counts == manifest["journal"]["by_source"] and manifest["journal"]["schema_version"] == 1
    assert not list(SP4_RUN.glob(f"{MATCH_LOG_FILE}-*"))


def test_every_sp4_snapshot_loads_read_only_with_its_architecture():
    manifest = sp4_manifest()
    before = tree_digest(SP4_RUN)
    spec = env_spec(load_config(SP4_TTT_TINY))
    for version in manifest["versions"]:
        ckpt = load_checkpoint_dir(SP4_RUN / "checkpoints" / manifest["agent"] / f"ckpt_v{version}")
        model = build_network(NetworkConfig.model_validate(ckpt["meta"]["networks"]), spec.roles["player"])
        check_model_state(model, ckpt["model_state"], f"ckpt_v{version}")
    assert tree_digest(SP4_RUN) == before


def test_the_report_of_a_copy_renders(tmp_path):
    copy = copy_sp4_run(tmp_path)
    out = tmp_path / "report.html"
    result = CliRunner().invoke(main, ["report", str(copy), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.stat().st_size > 10_000
