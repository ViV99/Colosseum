"""A short run writes metrics.jsonl with all four kinds, ratings.json and console progress (T6.3)."""
from __future__ import annotations

from cli_runner import TTT_CONFIG, run_train
from colosseum.metrics.jsonl import REQUIRED_KEYS


def test_metrics_jsonl_ratings_json_and_console(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="metrics-run")
    assert run.returncode == 0, run.stderr[-3000:]
    records = run.records()
    kinds = {r["kind"] for r in records}
    assert kinds == {"train", "episodes", "ratings", "system"}, kinds
    for record in records:
        assert not (REQUIRED_KEYS[record["kind"]] - set(record)), record
    assert max(r["train_step"] for r in run.records("train") if r["agent"] == "agent_0") >= 1
    episodes = run.records("episodes")
    assert sum(r["episodes"] for r in episodes) > 0
    assert set(episodes[0]["wdl"]) == {"latest", "past", "arena"}
    systems = run.records("system")
    assert systems[-1]["env_steps"] >= 3000
    assert any(r["workers_reporting"] >= 1 for r in systems)
    ratings = run.ratings()
    assert set(ratings) >= {"env_steps", "elo", "win_rates", "games", "wr_vs_past", "past_games"}
    assert "[agent_0] step" in run.stderr  # console progress (main logs INFO to stderr)


def test_launch_logs_to_wandb_on_per_agent_axes_and_finishes(tmp_path, monkeypatch):
    """The launcher wires WandBLogger into the hub (T6.4): run name and resolved config at init,
    train rows on ``<agent>/train_step`` and global rows on ``env_steps`` without ``step=``."""
    import sys
    import types

    from colosseum.core.config import ColosseumConfig, load_config
    from colosseum.launcher import Launcher
    from helpers import example_config, make_test_run_dir

    calls: dict = {"init": None, "logged": [], "finished": False}

    def log(row, **kwargs):
        assert "step" not in kwargs
        calls["logged"].append(dict(row))

    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: calls.update(init=kw) or types.SimpleNamespace(
        finish=lambda: calls.update(finished=True))
    fake.define_metric = lambda name, step_metric=None: None
    fake.log = log
    monkeypatch.setitem(sys.modules, "wandb", fake)

    data = load_config(example_config("tic_tac_toe.yaml")).model_dump()
    data["training"]["total_timesteps"] = 400
    data["rollout"].update(num_workers=1, envs_per_worker=2, chunk_length=8)
    data["learner"].update(batch_chunks=2, queue_size=16, device="cpu")
    data["metrics"].update(use_wandb=True, log_interval=1)
    config = ColosseumConfig(**data)
    run = make_test_run_dir(config, tmp_path)
    Launcher(config, run).launch()

    assert calls["init"]["name"] == (run.run_name or run.root.name)
    assert calls["init"]["config"]["training"]["total_timesteps"] == 400
    assert any("agent_0/train_step" in row for row in calls["logged"])
    assert any("env_steps" in row and any(k.startswith("system/") for k in row) for row in calls["logged"])
    assert calls["finished"]
