"""learner.max_policy_lag in a real run (SP5 T3.3, spec 4.7, 8): every trained batch stays within the bound, every
train record carries the lag percentiles and the stale count, and the bound does not starve the learner (the run
reaches its budget)."""
from __future__ import annotations

from cli_runner import TINY, TTT_CONFIG, run_train


def test_max_policy_lag_bounds_the_lag_of_every_trained_batch(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="lag",
                    overrides={"learner.max_policy_lag": 2, "learner.weight_push_interval": 1})
    assert run.returncode == 0, run.stderr[-3000:]
    train = run.records("train")
    assert train, "no train records"
    for record in train:
        assert {"policy_lag_p50", "policy_lag_p95", "policy_lag_max", "dropped_stale_chunks"} <= set(record)
        assert record["policy_lag_max"] <= 2
        assert record["policy_lag_p50"] <= record["policy_lag_p95"] <= record["policy_lag_max"]
    stale = [r["dropped_stale_chunks"] for r in train]
    assert stale[-1] >= 0 and stale == sorted(stale)              # cumulative for the learner incarnation
    systems = run.records("system")
    assert systems and systems[-1]["env_steps"] >= int(TINY["training.total_timesteps"])   # no starvation
