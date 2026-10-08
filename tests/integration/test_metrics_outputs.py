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
    assert set(ratings) >= {"env_steps", "elo", "win_rates", "games", "wr_vs_past"}
    assert "[agent_0] step" in run.stderr  # console progress (main logs INFO to stderr)
