"""Short `colosseum train` runs of the reference examples (T8.5; spec criterion 6: validate and a
smoke train, no learning threshold)."""
from __future__ import annotations

import pytest

from cli_runner import run_train
from demo_checks import example_config


def test_chase_trains_briefly(tmp_path):
    run = run_train(example_config("chase"), tmp_path, "chase", overrides={"training.total_timesteps": "2000"})
    assert run.returncode == 0, run.stderr[-3000:]
    assert run.records("train")
    assert any((run.root / "checkpoints" / "agent_0").glob("ckpt_v*"))


def test_space_miners_trains_briefly(tmp_path):
    pytest.importorskip("Box2D")
    run = run_train(example_config("space_miners"), tmp_path, "miners",
                    overrides={"training.total_timesteps": "1500", "env.kwargs": "{max_ticks: 60}"})
    assert run.returncode == 0, run.stderr[-3000:]
    assert run.records("train")
    assert "deciders_valid_mean" in run.records("train")[-1]
    assert any((run.root / "checkpoints" / "agent_0").glob("ckpt_v*"))
