"""The tic-tac-toe example trains end to end on the SP2 CLI (smoke; learning is checked by T8.3)."""
from __future__ import annotations

from cli_runner import TTT_CONFIG, run_train


def test_tic_tac_toe_trains_to_the_budget(tmp_path):
    # The tiny run trains for a few seconds after start-up: checkpoints every 5 train steps and
    # lineups redrawn every 0.25 s, so matches against past checkpoints are played and rated. SP5: a snapshot game
    # takes more hops (the snapshot's upload and save, a lineup refresh, its download into the node's cache, the
    # match, two result flushes), so the budget is 16 000 env steps (was 6 000, about 2 s: past_games 0 under load).
    overrides = {"training.total_timesteps": "16000", "checkpoint.interval": "5",
                 "rollout.match_refresh_interval_sec": "0.25"}
    run = run_train(TTT_CONFIG, tmp_path, name="ttt", overrides=overrides)
    assert run.returncode == 0, run.stderr[-3000:]
    assert "Training budget reached" in run.log("main")
    assert max(r["train_step"] for r in run.records("train") if r["agent"] == "agent_0") >= 1
    layout = run.ratings()["layouts"]["2p"]
    assert "agent_0" in layout["past_games"]
    assert layout["past_games"]["agent_0"] > 0
    assert 0.0 <= layout["wr_vs_past"]["agent_0"] <= 1.0
