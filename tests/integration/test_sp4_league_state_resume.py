"""League state across a run-dir resume (SP4 T3.2, spec block 7.4 and criterion 3.2): league_state.json is written
by a run and restored by a run-dir resume (the ratings continue), while a resume from a checkpoint dir starts with
empty statistics as in SP3.

Counts are compared with the matches each run journaled itself (``matches.db``, rows of its own run id), never
between runs: a local run overshoots its budget by about a second of env steps, so how many games a run plays
depends on the machine's load."""
from __future__ import annotations

import json

from cli_runner import newest_checkpoint, run_train
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader
from colosseum.core.run_dir import read_run_id
from game_helpers import SP3_TTT_TINY


def _games(ratings: dict) -> int:
    return ratings["layouts"]["2p"]["games"]["main"]["random"]


def _own_games(run) -> tuple[int, int]:
    """``(main vs random training matches the run journaled under its own run id, its dropped match results)``:
    the coordinator saw at least the first and at most their sum (a result the writer dropped is counted)."""
    run_id = read_run_id(run.root)
    with MatchLogReader(run.root / MATCH_LOG_FILE) as reader:
        rows = reader.matches(agent="main", layout="2p", source="train", limit=None)
    journaled = sum(1 for r in rows if r.run_id == run_id and {s.agent_id for s in r.seats} == {"main", "random"})
    dropped = max((r["dropped_match_results"] for r in run.records("system")), default=0)
    return journaled, dropped


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
    own, dropped = _own_games(second)
    assert own > 0
    assert own <= _games(second.ratings()) - before <= own + dropped     # counted on from the restored tables
    third = run_train(SP3_TTT_TINY, tmp_path, name="third", tiny=False,
                      overrides={"training.resume_from": str(newest_checkpoint(first.root, "main")),
                                 "training.total_timesteps": str(budget)})
    assert third.returncode == 0, third.stderr[-3000:]
    assert "league state" not in third.log("main")          # an explicit checkpoint resume: SP3 behaviour
    with MatchLogReader(third.root / MATCH_LOG_FILE) as reader:
        (only,) = reader.runs()                             # a checkpoint-dir resume imports no parent journal
    assert only.parent_run_id is None
    own, dropped = _own_games(third)
    assert own > 0
    assert own <= _games(third.ratings()) <= own + dropped  # its own games only: nothing restored
