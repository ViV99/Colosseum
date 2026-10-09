"""Short `colosseum train` runs of the T8.2 demo games: roles, teammates and a centralized critic."""
from __future__ import annotations

import math

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
    assert "coop_a+coop_b" in cross, f"no mixed team after training with teammates: mixed: {cross}"
    assert cross["coop_a+coop_b"]["n"] > 0


def test_team_tag_trains_with_dict_obs_uint8_global_state_and_a_critic_encoder(tmp_path):
    run = run_train(example_config("team_tag"), tmp_path, "tag", overrides={"training.total_timesteps": "3000"})
    assert run.returncode == 0, run.stderr[-3000:]          # exit 0 = the env-step budget was reached
    assert run.records("system")[-1]["env_steps"] >= 3000
    train = run.records("train")
    assert train and {r["agent"] for r in train} == {"agent_0"}
    assert all(math.isfinite(r["value_loss"]) for r in train)
