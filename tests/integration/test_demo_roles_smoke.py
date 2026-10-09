"""Short `colosseum train` runs of the demo games with several agents (T8.2): roles and teammates."""
from __future__ import annotations

from cli_runner import run_train
from demo_checks import example_config


def test_predator_prey_trains_one_agent_per_role(tmp_path):
    run = run_train(example_config("predator_prey"), tmp_path, "pp", overrides={"training.total_timesteps": "3000"})
    assert run.returncode == 0, run.stderr[-3000:]
    agents = {r["agent"] for r in run.records("train")}
    assert agents == {"hunter", "prey"}
    for agent in ("hunter", "prey"):
        assert any((run.root / "checkpoints" / agent).glob("ckpt_v*")), agent
    assert "1v2" in run.ratings()["layouts"]


def test_coop_buttons_with_mixed_teammates_forms_a_cross_play_table(tmp_path):
    run = run_train(example_config("coop_buttons"), tmp_path, "coop", overrides={"training.total_timesteps": "4000"})
    assert run.returncode == 0, run.stderr[-3000:]
    assert {r["agent"] for r in run.records("train")} == {"coop_a", "coop_b"}
    cross = run.ratings()["layouts"]["coop2"]["cross_play"]
    assert cross, "no cross-play entries after training with teammates: mixed"
