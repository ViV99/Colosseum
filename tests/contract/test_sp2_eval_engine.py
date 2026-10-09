"""In-process eval API on the real MatchRunner: explicit models and lineups (T6.1)."""
from __future__ import annotations

from collections import Counter

import pytest
import torch

from colosseum.sp2.eval import play_lineups, schedule_lineups, summarize
from game_helpers import EliminationFFA, TurnTakingGame, make_test_model

pytestmark = pytest.mark.usefixtures("restore_global_rng")


def models_for(spec, names, core="none", seed=0):
    torch.manual_seed(seed)
    role = spec.roles[next(iter(spec.roles))]
    return {name: make_test_model(role, core=core) for name in names}


def seat_agents(result) -> tuple[str, ...]:
    return tuple(s.agent_id for s in sorted(result.seats, key=lambda s: s.seat))


def turn_layout():
    spec = TurnTakingGame().spec
    return spec, next(iter(spec.layouts))


def test_every_lineup_is_played_exactly_once():
    spec, layout = turn_layout()
    players = {"a": list(spec.roles), "b": list(spec.roles)}
    lineups = schedule_lineups(spec, layout, players, 6)
    results = play_lineups(env_fn=TurnTakingGame, models=models_for(spec, players), lineups=lineups,
                           num_envs=4, seed=0)
    assert len(results) == len(lineups)
    assert Counter(seat_agents(r) for r in results) == Counter(tuple(s.agent_id for s in lu.seats) for lu in lineups)
    assert all(r.layout == layout and r.outcome_kind == "wdl" for r in results)
    report = summarize(spec, results, agents=["a", "b"], num_matches=6).to_dict()
    assert report["layouts"][layout]["pairs"][0]["n"] == 6


def test_more_envs_than_lineups_and_reproducible_with_a_seed():
    spec, layout = turn_layout()
    players = {"a": list(spec.roles)}
    lineups = schedule_lineups(spec, layout, players, 3)
    models = models_for(spec, players)
    before = torch.get_rng_state()
    first = play_lineups(env_fn=TurnTakingGame, models=models, lineups=lineups, num_envs=8, seed=5)
    assert torch.equal(torch.get_rng_state(), before)  # the caller's RNG is untouched
    second = play_lineups(env_fn=TurnTakingGame, models=models, lineups=lineups, num_envs=8, seed=5)
    assert len(first) == 3
    key = sorted((r.match_id, tuple(s.reward for s in r.seats), r.episode_length) for r in first)
    assert key == sorted((r.match_id, tuple(s.reward for s in r.seats), r.episode_length) for r in second)


def test_ffa_layouts_with_eliminations_report_ranks():
    env = EliminationFFA(max_players=4)
    spec = env.spec
    players = {"a": list(spec.roles), "b": list(spec.roles)}
    lineups = schedule_lineups(spec, "4p", players, 2) + schedule_lineups(spec, "2p", players, 2)
    results = play_lineups(env_fn=lambda: EliminationFFA(max_players=4), models=models_for(spec, players),
                           lineups=lineups, num_envs=2, seed=0)
    assert Counter(r.layout for r in results) == {"4p": 2, "2p": 2}
    assert {r.outcome_kind for r in results if r.layout == "4p"} == {"rank"}
    report = summarize(spec, results).to_dict()["layouts"]
    assert report["4p"]["agents"]["a"]["n"] == 4 and report["2p"]["outcome_kind"] == "wdl"


def test_stateful_models_play_and_keep_their_train_flags():
    spec, layout = turn_layout()
    players = {"a": list(spec.roles), "b": list(spec.roles)}
    models = models_for(spec, players, core="lstm")
    models["a"].train()
    models["b"].eval()
    results = play_lineups(env_fn=TurnTakingGame, models=models, lineups=schedule_lineups(spec, layout, players, 2),
                           num_envs=2, seed=1)
    assert len(results) == 2
    assert models["a"].training and not models["b"].training


def test_bad_arguments():
    spec, layout = turn_layout()
    lineups = schedule_lineups(spec, layout, {"a": list(spec.roles)}, 1)
    with pytest.raises(ValueError, match="unknown agents"):
        play_lineups(env_fn=TurnTakingGame, models={}, lineups=lineups)
    with pytest.raises(ValueError, match="num_envs"):
        play_lineups(env_fn=TurnTakingGame, models=models_for(spec, ["a"]), lineups=lineups, num_envs=0)
    assert play_lineups(env_fn=TurnTakingGame, models={}, lineups=[]) == []


def test_one_agent_in_an_ffa_layout_reports_ranks_per_seat():
    spec = EliminationFFA(max_players=4).spec
    players = {"a": list(spec.roles)}
    results = play_lineups(env_fn=lambda: EliminationFFA(max_players=4), models=models_for(spec, players),
                           lineups=schedule_lineups(spec, "3p", players, 4), num_envs=2, seed=0)
    section = summarize(spec, results).to_dict()["layouts"]["3p"]
    assert section["agents"] == {} and section["n"] == 4
    (row,) = section["solo"]
    assert set(row["per_seat"]) == {"0", "1", "2"} and all(c["n"] == 4 for c in row["per_seat"].values())


def test_deterministic_play_ignores_the_sampling_seed():
    spec, layout = turn_layout()
    players = {"a": list(spec.roles), "b": list(spec.roles)}
    models = models_for(spec, players, seed=3)
    lineups = schedule_lineups(spec, layout, players, 4)

    def outcomes(seed):
        results = play_lineups(env_fn=TurnTakingGame, models=models, lineups=lineups, num_envs=2, seed=seed,
                               deterministic=True)
        return Counter((seat_agents(r), tuple(s.reward for s in sorted(r.seats, key=lambda s: s.seat)))
                       for r in results)

    first = outcomes(1)
    assert first == outcomes(2)
    assert len(first) == 2  # one outcome per seat order: greedy actions in a deterministic game
