"""Example configs, the chase example's action handling and the deployment manifests (T8.5; ported
from SP1's ``tests/unit/test_examples.py``)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.config import load_config
from colosseum.core.registry import build_model, env_spec
from examples.composite_action.game import DIRS, GRID, ChaseGame

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((REPO_ROOT / "configs" / "examples").glob("*.yaml"))


def test_chase_accepts_its_own_sampled_actions():
    env = ChaseGame()
    env.reset(0, "2p")
    space = env.spec.roles["player"].action_space
    space.seed(0)
    for _ in range(3):
        env.step({0: space.sample(), 1: space.sample()})        # direction: int64 scalar, speed: shape (1,)
    sampled = {"direction": np.array([3]), "speed": np.array([0.25], dtype=np.float32)}
    before = env._pos.copy()
    env.step({0: sampled, 1: sampled})
    np.testing.assert_allclose(env._pos, np.clip(before + DIRS[3] * 0.25, 0.0, GRID), rtol=0, atol=1e-6)


def test_chase_rejects_multi_element_action_components():
    env = ChaseGame()
    env.reset(0, "2p")
    with pytest.raises(ValueError, match=r"player 0: action component 'speed'.*shape \(2,\)"):
        env.step({0: {"direction": 0, "speed": np.array([0.1, 0.2])}, 1: {"direction": 0, "speed": 0.1}})


def test_example_configs_exist():
    names = {p.stem for p in CONFIGS}
    assert {"chase", "space_miners", "tic_tac_toe"} <= names


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_every_example_config_validates(path, monkeypatch):
    if path.name == "space_miners.yaml":
        # importorskip also silences Box2D's SWIG import-time DeprecationWarnings.
        pytest.importorskip("Box2D", reason="space_miners needs Box2D (pip install -e '.[examples]')")
    monkeypatch.chdir(REPO_ROOT)
    result = CliRunner().invoke(main, ["validate", "-c", str(path)])          # the full validate_config
    assert result.exit_code == 0, result.output
    assert result.output.rstrip().endswith("Config is valid."), result.output


def test_attention_example_builds_a_stateful_model():
    cfg = load_config(REPO_ROOT / "configs" / "examples" / "tic_tac_toe_attention.yaml")
    assert cfg.networks.core.class_path == "colosseum.networks.cores.WindowAttentionCore"
    role = env_spec(cfg).roles["player"]
    model = build_model(cfg.get_agent_config("agent_0"), role)
    assert model.is_stateful


def test_example_configs_use_run_section_not_checkpoint_dir():
    for path in CONFIGS:
        cfg = load_config(path)
        assert cfg.run.dir == "runs" and cfg.run.name is None, path.name


@pytest.mark.parametrize("manifest", ["docker-compose.yaml", "k8s/learner.yaml"])
def test_deployment_learner_keeps_run_dir_on_the_mounted_volume(manifest):
    docs = list(yaml.safe_load_all((REPO_ROOT / "deployment" / manifest).read_text()))
    commands = [
        c["command"] for d in docs if d
        for c in ([d["services"]["learner"]] if "services" in d else
                  d.get("spec", {}).get("template", {}).get("spec", {}).get("containers", []))
        if "run-learner" in c.get("command", [])
    ]
    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[cmd.index("--set") + 1] == "run.dir=/app/checkpoints"
