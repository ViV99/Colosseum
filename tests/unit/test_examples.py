"""Example envs and configs (T8.1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.config import load_config
from colosseum.core.registry import build_model
from examples.composite_action.env import ChaseEnv
from examples.tic_tac_toe.env import TicTacToeEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((REPO_ROOT / "configs" / "examples").glob("*.yaml"))


def test_tic_tac_toe_reports_active_player_and_legal_moves():
    env = TicTacToeEnv()
    obs, infos = env.reset()
    assert infos[0]["active"] is True and infos[1]["active"] is False
    assert infos[0]["action_mask"].dtype == bool and infos[0]["action_mask"].all()
    _, _, term, _, infos = env.step({0: 4, 1: 0})  # player 1's action is ignored
    assert infos[0]["active"] is False and infos[1]["active"] is True
    expected = np.ones(9, bool)
    expected[4] = False
    assert np.array_equal(infos[1]["action_mask"], expected)
    assert np.array_equal(infos[0]["action_mask"], expected)
    assert not term[0]


def test_tic_tac_toe_full_board_mask_is_all_true():
    env = TicTacToeEnv()
    env.reset()
    # X O X / X O O / O X X  -> draw after 9 legal moves
    for cell in [0, 1, 2, 4, 3, 5, 7, 6, 8]:
        current = env._current_player
        _, rewards, term, _, infos = env.step({current: cell, 1 - current: 0})
    assert term[0] and term[1] and rewards == {0: 0.0, 1: 0.0}
    assert infos[0]["action_mask"].all() and infos[1]["action_mask"].all()


def test_chase_accepts_its_own_sampled_actions_and_scalars():
    env = ChaseEnv()
    env.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(3):
        env.step({0: env.action_space.sample(), 1: env.action_space.sample()})  # speed: shape (1,) array
    env.step({0: {"direction": 1, "speed": 0.5}, 1: {"direction": np.int64(2), "speed": np.float32(1.0)}})


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_every_example_config_validates(path, monkeypatch):
    if path.name == "space_miners.yaml":
        # importorskip also silences Box2D's SWIG import-time DeprecationWarnings.
        pytest.importorskip("Box2D", reason="space_miners needs Box2D (pip install -e '.[examples]')")
    monkeypatch.chdir(REPO_ROOT)
    result = CliRunner().invoke(main, ["validate", "-c", str(path)])
    assert result.exit_code == 0, result.output


def test_attention_example_builds_a_stateful_model():
    cfg = load_config(REPO_ROOT / "configs" / "examples" / "tic_tac_toe_attention.yaml")
    assert cfg.networks.core.class_path == "colosseum.networks.cores.WindowAttentionCore"
    model = build_model(cfg.get_agent_config("agent_0"))
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
