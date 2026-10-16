"""Live ratings in a real training run (SP4 T2.2, spec block 2): ratings.json gets the bt section (the RandomBot
anchor at 0, snapshot buckets of the trainable agent), the ratings records get the compact bt summary, the console
shows `bt <rating> ± <ci>`."""
from __future__ import annotations

from cli_runner import run_train
from game_helpers import SP3_TTT_TINY


def test_a_run_publishes_live_ratings(tmp_path):
    run = run_train(SP3_TTT_TINY, tmp_path, name="live", tiny=False,
                    overrides={"ratings.refit_interval_sec": "1", "training.total_timesteps": "6000"})
    assert run.returncode == 0, run.stderr[-3000:]
    section = run.ratings()["bt"]["2p"]
    rows = section["player"]
    by_entity = {r["entity"]: r for r in rows}
    assert by_entity["random"]["anchor"] and by_entity["random"]["rating"] == 0.0
    assert section["@anchors"]["player/0"]["rule"] == "random_bot"
    buckets = sorted(r["version"] for r in rows if r["lineage"] == "main")
    assert buckets and buckets[0] == 0 and all(v % 20 == 0 for v in buckets)       # checkpoint.interval 20
    assert all(r["ci_low"] <= r["rating"] <= r["ci_high"] for r in rows)
    records = [r for r in run.records("ratings") if "bt" in r]
    assert records and records[-1]["bt"]["main"]["2p"]["latest"]["entity"] == f"main@v{buckets[-1]}"
    assert "| bt " in run.stderr and "Live ratings: Bradley-Terry refit every 1 s" in run.log("main")
