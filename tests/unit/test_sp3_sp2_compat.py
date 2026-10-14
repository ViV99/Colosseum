"""SP2 compatibility baseline (SP3 T0.1): the SP2 example configs (copies) validate and re-read cleanly,
and the SP2 checkpoint fixture loads for eval and resume.

Created on SP2 code; it must stay green through SP3. Later tasks add their own compatibility checks:
T1.2 / T1.5 (the checkpoint as a frozen agent), T2.1 (run-dir resume imports the pool), T3.1 (SP2
matchmaking knobs), T4.1 (training.kickstart_*), T4.2 (the checkpoint as init), T4.4 (as a teacher).
"""
from __future__ import annotations

import json
import logging

import pytest
import yaml
from click.testing import CliRunner

from cli_runner import REPO_ROOT
from colosseum.cli import main
from colosseum.coordinator.checkpoint_manager import load_checkpoint_dir
from colosseum.core.config import load_config
from colosseum.core.registry import env_spec
from colosseum.core.roles import role_signature
from colosseum.launcher import Launcher, setup_run
from game_helpers import (
    SP2_CHECKPOINT,
    SP2_CHECKPOINT_ENV_STEPS,
    SP2_CHECKPOINT_VERSION,
    SP2_CONFIG_NAMES,
    SP2_CONFIGS,
    SP2_TTT_TINY,
    make_test_run_dir,
)

SP2_META_KEYS = {"agent_id", "checkpoint_id", "config_hash", "env_steps", "final", "networks",
                 "policy_version", "role_signature", "roles", "timestamp"}


def test_the_fixture_has_a_copy_of_every_sp2_example_config():
    assert sorted(p.stem for p in SP2_CONFIGS.glob("*.yaml")) == sorted(SP2_CONFIG_NAMES)


@pytest.mark.parametrize("name", SP2_CONFIG_NAMES)
def test_sp2_config_copies_validate_without_edits(name, monkeypatch):
    if name == "space_miners":
        # importorskip also silences Box2D's SWIG import-time DeprecationWarnings.
        pytest.importorskip("Box2D", reason="space_miners needs Box2D (pip install -e '.[examples]')")
    monkeypatch.chdir(REPO_ROOT)
    result = CliRunner().invoke(main, ["validate", "-c", str(SP2_CONFIGS / f"{name}.yaml")])
    assert result.exit_code == 0, result.output
    assert "Config is valid." in result.output


@pytest.mark.parametrize("name", SP2_CONFIG_NAMES)
def test_the_resolved_form_of_an_sp2_config_rereads_without_warnings(name, tmp_path, caplog):
    """What a run writes as config.resolved.yaml (model_dump by alias) loads again silently and equal."""
    config = load_config(SP2_CONFIGS / f"{name}.yaml")
    resolved = tmp_path / "config.resolved.yaml"
    resolved.write_text(yaml.safe_dump(config.model_dump(mode="json", by_alias=True), sort_keys=False))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        again = load_config(resolved)
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert again == config


def test_the_checkpoint_fixture_is_an_sp2_checkpoint_of_the_tiny_config():
    assert sorted(p.name for p in SP2_CHECKPOINT.iterdir()) == ["meta.json", "model.pt", "trainer_state.pt"]
    meta = load_checkpoint_dir(SP2_CHECKPOINT)["meta"]
    assert set(meta) == SP2_META_KEYS
    assert meta["policy_version"] == SP2_CHECKPOINT_VERSION and meta["env_steps"] == SP2_CHECKPOINT_ENV_STEPS
    assert meta["final"] is True and meta["roles"] == ["player"]
    spec = env_spec(load_config(SP2_TTT_TINY))
    assert meta["role_signature"] == role_signature(spec.roles["player"])


def test_eval_plays_the_sp2_checkpoint_dir_and_leaves_it_untouched(tmp_path):
    before = sorted(p.name for p in SP2_CHECKPOINT.parent.iterdir())
    out = tmp_path / "result.json"
    result = CliRunner().invoke(main, ["eval", "-c", str(SP2_TTT_TINY), "-a", f"old={SP2_CHECKPOINT}", "-n", "2",
                                       "--num-envs", "2", "--seed", "0", "-o", str(out)])
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["agents"] == ["old"] and report["layouts"]["2p"]["n"] == 2
    assert sorted(p.name for p in SP2_CHECKPOINT.parent.iterdir()) == before


def test_resume_from_the_sp2_checkpoint_dir_restores_version_trainer_state_and_env_steps(tmp_path):
    config = load_config(SP2_TTT_TINY, {"training.resume_from": str(SP2_CHECKPOINT)})
    launcher = Launcher(config, make_test_run_dir(config, tmp_path, name="resumed"))
    setup = setup_run(config, validate=False)
    states = launcher._resolve_resume(setup.agent_configs, setup.role_specs)
    assert states["agent_0"]["policy_version"] == SP2_CHECKPOINT_VERSION
    assert states["agent_0"]["trainer_state"] is not None
    assert launcher.env_steps_done == SP2_CHECKPOINT_ENV_STEPS
