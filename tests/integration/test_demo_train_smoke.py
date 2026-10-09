"""Short `colosseum train` runs of the demo games (T8.1): the whole pipeline on a Units game."""
from __future__ import annotations

from cli_runner import run_train
from demo_checks import example_config


def test_unit_harvest_trains_briefly_and_logs_unit_diagnostics(tmp_path):
    run = run_train(example_config("unit_harvest"), tmp_path, "uh", overrides={"training.total_timesteps": "3000"})
    assert run.returncode == 0, run.stderr[-3000:]
    train = run.records("train")
    assert train, "no train records"
    for key in ("clip_fraction", "clip_fraction_joint", "ess", "log_rho_abs_p95", "deciders_valid_mean"):
        assert key in train[-1], key
    assert train[-1]["deciders_valid_mean"] > 1.0             # several workers decide per act slot
    assert any((run.root / "checkpoints" / "agent_0").glob("ckpt_v*"))
