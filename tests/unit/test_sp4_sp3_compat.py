"""SP3 compatibility baseline (SP4 T0.1, spec section 6 and criterion 3.8): the SP3 example configs (copies)
validate and load silently, the SP3 run fixture is an SP3 run dir whose snapshots load read-only and play in eval
(by path and as a frozen agent), and match records are constructible without the SP4 fields.

Created on SP3 code; it must stay green through SP4. Later tasks extend the SP3 baseline elsewhere: T1.3 and T3.2
(`test_sp4_sp3_resume.py`), T4.1 (imported SP3 snapshots are not final), T5.4 (SP3 snapshots in a tournament).
"""
from __future__ import annotations

import dataclasses
import json
import logging

import pytest
import yaml
from click.testing import CliRunner

from cli_runner import REPO_ROOT
from colosseum.cli import main
from colosseum.coordinator.checkpoint_manager import check_model_state, load_checkpoint_dir
from colosseum.core.config import NetworkConfig, load_config
from colosseum.core.registry import build_network, env_spec
from colosseum.core.types import Lineup, MatchResult, SeatAssignment, SeatResult, TeamResult
from game_helpers import SP3_CONFIG_NAMES, SP3_CONFIGS, SP3_RUN, SP3_TTT_TINY, sp3_manifest, tree_digest

SP3_META_KEYS = {"agent_id", "checkpoint_id", "config_hash", "env_steps", "final", "networks",
                 "policy_version", "role_signature", "roles", "timestamp"}


def test_the_fixture_has_a_copy_of_every_sp3_example_config():
    assert sorted(p.stem for p in SP3_CONFIGS.glob("*.yaml")) == sorted(SP3_CONFIG_NAMES)


@pytest.mark.parametrize("name", SP3_CONFIG_NAMES)
def test_sp3_config_copies_validate_without_edits(name, monkeypatch):
    if name == "space_miners":
        # importorskip also silences Box2D's SWIG import-time DeprecationWarnings.
        pytest.importorskip("Box2D", reason="space_miners needs Box2D (pip install -e '.[examples]')")
    monkeypatch.chdir(REPO_ROOT)
    result = CliRunner().invoke(main, ["validate", "-c", str(SP3_CONFIGS / f"{name}.yaml")])
    assert result.exit_code == 0, result.output
    assert result.output.rstrip().endswith("Config is valid."), result.output


@pytest.mark.parametrize("name", SP3_CONFIG_NAMES)
def test_sp3_config_copies_load_without_warnings(name, caplog):
    with caplog.at_level(logging.WARNING):
        load_config(SP3_CONFIGS / f"{name}.yaml")
    assert [r.getMessage() for r in caplog.records if r.name.startswith("colosseum")] == []


def test_the_run_fixture_is_an_sp3_run_dir():
    manifest = sp3_manifest()
    assert sorted(p.name for p in SP3_RUN.iterdir()) == ["checkpoints", "config.resolved.yaml", "metrics.jsonl",
                                                       "ratings.json"]
    agent_dir = SP3_RUN / "checkpoints" / manifest["agent"]
    versions = sorted(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.iterdir())
    assert versions == manifest["versions"] and len(versions) >= 3
    assert manifest["final_version"] == max(versions)
    without_trainer = set(versions) - set(manifest["with_trainer_state"])
    assert without_trainer, "the fixture needs keep_every snapshots without trainer_state.pt"
    for version in versions:
        meta = json.loads((agent_dir / f"ckpt_v{version}" / "meta.json").read_text())
        assert set(meta) == SP3_META_KEYS                     # no run_id / pinned (SP4 adds them in T4.1)
        assert meta["final"] is (version == manifest["final_version"])
    ratings = json.loads((SP3_RUN / "ratings.json").read_text())
    assert set(ratings) == {"env_steps", "layouts"} and manifest["layout"] in ratings["layouts"]
    assert manifest["anchor"] in ratings["layouts"][manifest["layout"]]["elo"]
    kinds = {json.loads(line)["kind"] for line in (SP3_RUN / "metrics.jsonl").read_text().splitlines() if line}
    assert kinds == {"train", "episodes", "ratings", "system"}
    resolved = load_config(SP3_RUN / "config.resolved.yaml")
    assert resolved.get_trainable_agent_ids() == [manifest["agent"]]
    assert resolved.fixed_agent_ids() == [manifest["anchor"]]


def test_every_sp3_snapshot_loads_read_only_with_its_architecture():
    manifest = sp3_manifest()
    before = tree_digest(SP3_RUN)
    spec = env_spec(load_config(SP3_TTT_TINY))
    for version in manifest["versions"]:
        ckpt = load_checkpoint_dir(SP3_RUN / "checkpoints" / manifest["agent"] / f"ckpt_v{version}")
        assert ckpt["meta"]["policy_version"] == version and ckpt["roles"] == ["player"]
        model = build_network(NetworkConfig.model_validate(ckpt["meta"]["networks"]), spec.roles["player"])
        check_model_state(model, ckpt["model_state"], f"ckpt_v{version}")
    assert tree_digest(SP3_RUN) == before


def test_eval_plays_sp3_snapshots_by_path_and_as_a_frozen_agent(tmp_path):
    manifest = sp3_manifest()
    before = tree_digest(SP3_RUN)
    agent_dir = SP3_RUN / "checkpoints" / manifest["agent"]
    first, last = agent_dir / f"ckpt_v{manifest['versions'][0]}", agent_dir / f"ckpt_v{manifest['final_version']}"
    out = tmp_path / "paths.json"
    result = CliRunner().invoke(main, ["eval", "-c", str(SP3_TTT_TINY), "-a", f"old={first}", "-a", f"new={last}",
                                       "-n", "2", "--num-envs", "2", "--seed", "0", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["agents"] == ["old", "new"]
    data = yaml.safe_load(SP3_TTT_TINY.read_text())
    data["agents"]["old"] = {"kind": "frozen", "path": str(first)}
    cfg = tmp_path / "with_frozen.yaml"
    cfg.write_text(yaml.safe_dump(data, sort_keys=False))
    out = tmp_path / "frozen.json"
    result = CliRunner().invoke(main, ["eval", "-c", str(cfg), "-a", "old", "-a", "random", "-n", "2",
                                       "--num-envs", "2", "--seed", "0", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["layouts"]["2p"]["n"] == 2
    assert tree_digest(SP3_RUN) == before


def test_match_records_are_constructible_without_the_sp4_fields():
    seats = [SeatResult(0, "player", 0, "a", "latest", 1.0), SeatResult(1, "player", 1, "b", "fixed", -1.0, None, "")]
    result = MatchResult("m0", "2p", "wdl", seats, [TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], 9)
    assert (result.match_id, result.layout, result.episode_length) == ("m0", "2p", 9)
    lineup = Lineup("2p", [SeatAssignment("a"), SeatAssignment("b", "fixed", False, "anchors")])
    assert lineup.seats[1].source == "anchors"
    # SP3 positional order is kept: new fields only ever come after the SP3 ones.
    assert [f.name for f in dataclasses.fields(MatchResult)][:6] == [
        "match_id", "layout", "outcome_kind", "seats", "teams", "episode_length"]
    assert [f.name for f in dataclasses.fields(SeatResult)][:8] == [
        "seat", "role", "team", "agent_id", "network_id", "reward", "eliminated_step", "source"]
    assert [f.name for f in dataclasses.fields(Lineup)][:2] == ["layout", "seats"]


# ---------------------------------------------------------------------------
# SP4 final fix wave, ruling A-M7: an SP3 config with keep_last > 100 (or SP2 pool_size > 100) still validates
# ---------------------------------------------------------------------------


def _sp3_ttt_with_checkpoint(tmp_path, **checkpoint) -> str:
    raw = yaml.safe_load((SP3_CONFIGS / "tic_tac_toe.yaml").read_text())
    raw["checkpoint"] = {"interval": raw["checkpoint"]["interval"], **checkpoint}
    path = tmp_path / "sp3_ttt.yaml"
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def _cap_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "colosseum.core.config" and "max_snapshots" in
            r.getMessage()]


def test_an_sp3_config_with_keep_last_150_validates_and_caps_at_keep_last(tmp_path, monkeypatch, caplog):
    path = _sp3_ttt_with_checkpoint(tmp_path, keep_last=150)
    with caplog.at_level(logging.INFO, logger="colosseum.core.config"):
        config = load_config(path)
    assert config.checkpoint.keep_last == 150 and config.checkpoint.max_snapshots == 150
    (line,) = _cap_lines(caplog)
    assert "keep_last (150)" in line and "max_snapshots" in line
    assert config.get_agent_config("agent_0").checkpoint.max_snapshots == 150      # re-validated: no second line
    assert len(_cap_lines(caplog)) == 1
    monkeypatch.chdir(REPO_ROOT)
    result = CliRunner().invoke(main, ["validate", "-c", path])
    assert result.exit_code == 0, result.output
    assert "at most 150 per agent" in result.output and result.output.rstrip().endswith("Config is valid.")


def test_an_sp2_pool_size_over_100_is_capped_at_keep_last_too(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="colosseum.core.config"):
        config = load_config(_sp3_ttt_with_checkpoint(tmp_path, pool_size=150))
    assert config.checkpoint.keep_last == 150 and config.checkpoint.max_snapshots == 150
    assert len(_cap_lines(caplog)) == 1


@pytest.mark.parametrize("checkpoint, cap", [
    ({"keep_last": 50}, 100),                               # the default cap stays when it holds keep_last
    ({"keep_last": 150, "max_snapshots": None}, None),      # an explicit "no cap"
    ({"keep_last": 150, "max_snapshots": 200}, 200),
])
def test_the_cap_follows_keep_last_only_when_max_snapshots_is_not_set(tmp_path, caplog, checkpoint, cap):
    with caplog.at_level(logging.INFO, logger="colosseum.core.config"):
        config = load_config(_sp3_ttt_with_checkpoint(tmp_path, **checkpoint))
    assert config.checkpoint.max_snapshots == cap and _cap_lines(caplog) == []


def test_an_explicit_max_snapshots_below_keep_last_is_still_a_config_error(tmp_path):
    from colosseum.core.errors import ConfigError

    with pytest.raises(ConfigError, match=r"max_snapshots \(120\) is smaller than checkpoint.keep_last \(150\)"):
        load_config(_sp3_ttt_with_checkpoint(tmp_path, keep_last=150, max_snapshots=120))
