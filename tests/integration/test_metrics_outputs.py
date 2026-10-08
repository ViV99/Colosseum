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

    from fake_wandb import FakeWandb

    from colosseum.core.config import ColosseumConfig, load_config
    from colosseum.launcher import Launcher
    from colosseum.metrics.jsonl import GLOBAL_KINDS
    from helpers import example_config, make_test_run_dir

    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    data = load_config(example_config("tic_tac_toe.yaml")).model_dump()
    data["training"]["total_timesteps"] = 400
    data["rollout"].update(num_workers=1, envs_per_worker=2, chunk_length=8)
    data["learner"].update(batch_chunks=2, queue_size=16, device="cpu")
    data["metrics"].update(use_wandb=True, log_interval=1)
    config = ColosseumConfig(**data)
    run = make_test_run_dir(config, tmp_path)
    Launcher(config, run).launch()

    assert fake.init_kwargs["name"] == (run.run_name or run.root.name)
    assert fake.init_kwargs["config"]["training"]["total_timesteps"] == 400
    assert ("agent_0/*", "agent_0/train_step") in fake.defined
    assert ("system/*", "env_steps") in fake.defined
    assert fake.logged and all(kwargs == {} for kwargs in fake.log_kwargs)  # never step= (R5-02)
    global_prefixes = tuple(f"{kind}/" for kind in GLOBAL_KINDS)
    for row in fake.logged:  # every row carries exactly one step key, the one of its namespace
        if any(k.startswith("agent_0/") for k in row):
            assert "agent_0/train_step" in row and "env_steps" not in row, row
            assert all(k.startswith("agent_0/") for k in row), row
        else:
            assert "env_steps" in row, row
            assert all(k == "env_steps" or k.startswith(global_prefixes) for k in row), row
    assert any("agent_0/train_step" in row for row in fake.logged)
    assert any(any(k.startswith("system/") for k in row) for row in fake.logged)
    assert fake.finished
