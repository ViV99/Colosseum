"""League state across a run-dir resume (SP4 T3.2, spec block 7.4 and criterion 3.2): league_state.json is written
by a run and restored by a run-dir resume (the ratings continue), while a resume from a checkpoint dir starts with
empty statistics as in SP3."""
from __future__ import annotations

import json

from cli_runner import newest_checkpoint, run_train
from game_helpers import SP3_TTT_TINY


def _games(ratings: dict) -> int:
    return ratings["layouts"]["2p"]["games"]["main"]["random"]


def test_a_run_dir_resume_continues_the_league_state(tmp_path):
    first = run_train(SP3_TTT_TINY, tmp_path, name="first", tiny=False)
    assert first.returncode == 0, first.stderr[-3000:]
    state = json.loads((first.root / "league_state.json").read_text())
    assert state["version"] == 1 and "main" in state["pfsp"]["table"]["2p"]
    before = _games(first.ratings())
    assert before > 0
    budget = state["env_steps"] + 3000
    second = run_train(SP3_TTT_TINY, tmp_path, name="second", tiny=False,
                       overrides={"training.resume_from": str(first.root), "training.total_timesteps": str(budget)})
    assert second.returncode == 0, second.stderr[-3000:]
    assert "league state (ratings, PFSP statistics) restored from" in second.log("main")
    assert _games(second.ratings()) > before                # counted on from the restored tables
    third = run_train(SP3_TTT_TINY, tmp_path, name="third", tiny=False,
                      overrides={"training.resume_from": str(newest_checkpoint(first.root, "main")),
                                 "training.total_timesteps": str(budget)})
    assert third.returncode == 0, third.stderr[-3000:]
    assert "league state" not in third.log("main")          # an explicit checkpoint resume: SP3 behaviour
    assert _games(third.ratings()) < before                 # its own 3000 steps only (the first run had 6000)
