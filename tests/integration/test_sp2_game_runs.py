"""`python -m colosseum train` on toy games: self-play, asymmetric league, FFA layouts,
cooperative mixed teams, resume with role signatures (T5.4)."""
from __future__ import annotations

import json
from pathlib import Path

from cli_runner import TrainRun, run_train
from game_helpers import write_test_config


def train(tmp_path: Path, game: str, name: str, overrides: dict[str, str] | None = None, **sections) -> TrainRun:
    config = write_test_config(tmp_path / f"{name}.yaml", game, **sections)
    return run_train(config, tmp_path, name=name, overrides=overrides)


def metas(run: TrainRun, agent_id: str) -> list[dict]:
    agent_dir = run.root / "checkpoints" / agent_id
    found = [json.loads((d / "meta.json").read_text()) for d in agent_dir.glob("ckpt_v*")]
    return sorted(found, key=lambda m: m["policy_version"])


def train_steps(run: TrainRun, agent_id: str) -> list[int]:
    return [r["train_step"] for r in run.records("train") if r["agent"] == agent_id]


def test_self_play_reaches_the_budget_and_checkpoints_carry_roles(tmp_path):
    run = train(tmp_path, "turns", "selfplay")
    assert run.returncode == 0, run.stderr[-3000:]
    assert "Training budget reached" in run.log("main")
    assert max(train_steps(run, "agent_0")) >= 1
    final = metas(run, "agent_0")[-1]
    assert final["final"] is True and final["roles"] and isinstance(final["role_signature"], str)
    ratings = run.ratings()
    assert set(ratings) == {"env_steps", "layouts"} and ratings["layouts"]
    assert all("layouts" in r for r in run.records("ratings"))
    assert sum(r["episodes"] for r in run.records("episodes")) > 0


def test_asymmetric_two_agent_league_trains_both_roles(tmp_path):
    run = train(tmp_path, "asymmetric", "asymmetric", matchmaking={"mode": "league", "self_play_ratio": 0.5})
    assert run.returncode == 0, run.stderr[-3000:]
    for agent_id in ("hunter", "prey"):
        assert train_steps(run, agent_id), f"{agent_id} never trained"
        assert metas(run, agent_id)[-1]["roles"] == [agent_id]
    assert metas(run, "hunter")[-1]["role_signature"] != metas(run, "prey")[-1]["role_signature"]
    layouts = run.ratings()["layouts"]
    assert any(table["games"]["hunter"]["prey"] > 0 for table in layouts.values()), layouts


def test_ffa_with_two_and_four_player_layouts(tmp_path):
    run = train(tmp_path, "ffa", "ffa", overrides={"training.total_timesteps": "4000"},
                agents={"alpha": {}, "beta": {}},
                matchmaking={"mode": "league", "self_play_ratio": 0.5, "layouts": {"2p": 0.5, "4p": 0.5}})
    assert run.returncode == 0, run.stderr[-3000:]
    played = {layout for r in run.records("episodes") for layout in r["by_layout"]}
    assert {"2p", "4p"} <= played, played
    layouts = run.ratings()["layouts"]
    assert layouts["2p"]["outcome_kind"] == "wdl" and layouts["4p"]["outcome_kind"] == "rank"
    assert layouts["2p"]["games"]["alpha"]["beta"] + layouts["4p"]["games"]["alpha"]["beta"] > 0


def test_cooperative_game_with_mixed_teammates_fills_the_cross_play_table(tmp_path):
    run = train(tmp_path, "coop", "coop", agents={"a": {}, "b": {}},
                matchmaking={"teammates": "mixed", "teammate_self_prob": 0.5})
    assert run.returncode == 0, run.stderr[-3000:]
    (table,) = [t for t in run.ratings()["layouts"].values() if t["outcome_kind"] == "score"]
    assert "a+b" in table["cross_play"], table["cross_play"]
    assert set(table["scores"]) == {"a", "b"}


def test_resume_continues_versions_and_checks_the_role_signature(tmp_path):
    config = write_test_config(tmp_path / "turns.yaml", "turns")
    first = run_train(config, tmp_path, name="first",
                      overrides={"training.total_timesteps": "1500", "checkpoint.interval": "5"})
    assert first.returncode == 0, first.stderr[-3000:]
    final = metas(first, "agent_0")[-1]
    version, done = final["policy_version"], final["env_steps"]
    # The first run overshoots its budget (workers flush their env steps about every 0.5 s and step
    # until they see the stop; in 3 of 12 measured runs by 1500 steps or more), so the resumed budget
    # counts from where it really stopped. 2000 more steps cannot pass before the learner trains: while
    # it consumes nothing, the full trajectory queue stops the worker after about 500 env steps.
    # That margin depends on the TINY config (learner.queue_size, rollout.chunk_length, envs_per_worker).
    resumed_budget = str(done + 2000)

    second = run_train(config, tmp_path, name="second",
                       overrides={"training.total_timesteps": resumed_budget, "training.resume_from": str(first.root)})
    assert second.returncode == 0, second.stderr[-3000:]
    assert f"(policy_version {version})" in second.log("main")
    assert f"env-step counter continues from {done}" in second.log("main")
    assert train_steps(second, "agent_0")[0] == version + 1
    assert min(m["policy_version"] for m in metas(second, "agent_0")) > version

    ckpt = first.root / "checkpoints" / "agent_0" / f"ckpt_v{version}"
    (ckpt / "meta.json").write_text(json.dumps({**final, "role_signature": "another-game"}))
    third = run_train(config, tmp_path, name="third",
                      overrides={"training.total_timesteps": resumed_budget, "training.resume_from": str(first.root)})
    assert third.returncode == 1
    assert "Config error" in third.stderr and "role signature" in third.stderr and str(ckpt) in third.stderr
    assert "Traceback" not in third.stderr
