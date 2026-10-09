"""predator_prey demo game (T8.2): 1 vs 2, roles with different spaces, prey elimination."""
from __future__ import annotations

import numpy as np

from demo_checks import check_example_config, random_matches
from examples.predator_prey.game import PredatorPreyGame, chase_action, flee_action


def _place(env: PredatorPreyGame, positions: list[list[int]]) -> None:
    env.reset(0, "1v2")
    env._pos = np.array(positions, dtype=np.int64)


def test_roles_have_different_spaces():
    spec = PredatorPreyGame().spec
    hunter, prey = spec.roles["hunter"], spec.roles["prey"]
    assert hunter.observation_space.shape == (9,) and prey.observation_space.shape == (7,)
    assert hunter.action_space.n == 5 and prey.action_space.n == 9
    assert [(s.role, s.team) for s in spec.layouts["1v2"]] == [("hunter", 0), ("prey", 1), ("prey", 1)]
    assert spec.outcome_kind("1v2") == "wdl"


def test_reset_keeps_the_prey_out_of_reach():
    env = PredatorPreyGame()
    for seed in range(30):
        res = env.reset(seed, "1v2")
        assert res.acting == {0, 1, 2}
        assert (np.abs(env._pos[1:] - env._pos[0]).max(axis=1) > env.catch_radius).all()
        assert res.obs[0].shape == (9,) and res.obs[1].shape == (7,)
        assert res.action_masks[0].shape == (5,) and res.action_masks[1].shape == (9,)


def test_a_caught_prey_is_eliminated_with_its_penalty():
    env = PredatorPreyGame(size=7)
    _place(env, [[3, 3], [5, 3], [0, 0]])
    res = env.step({0: 4, 1: 0, 2: 0})                       # hunter steps right, prey 1 is adjacent
    assert res.terminated == {1} and res.rewards == {1: -1.0, 0: 0.5}
    assert res.acting == {0, 2} and not res.episode_over


def test_catching_both_prey_ends_the_match_for_the_hunter():
    env = PredatorPreyGame(size=7)
    _place(env, [[3, 3], [5, 3], [5, 4]])
    res = env.step({0: 4, 1: 0, 2: 0})
    assert res.terminated == {1, 2} and res.episode_over and not res.truncated
    assert res.outcome.team_rank == {0: 1.0, 1: 2.0}
    assert res.rewards == {1: -1.0, 2: -1.0, 0: 1.0}


def test_a_surviving_prey_wins_at_the_time_limit():
    env = PredatorPreyGame(size=7, max_steps=1)
    _place(env, [[0, 0], [6, 6], [6, 0]])
    res = env.step({0: 0, 1: 0, 2: 0})
    assert res.episode_over and not res.truncated and not res.terminated
    assert res.outcome.team_rank == {0: 2.0, 1: 1.0}
    assert res.rewards == {0: -1.0, 1: 1.0, 2: 1.0}


def _hunter_win_rate(hunter, prey, episodes: int = 200) -> float:
    env, rng, wins = PredatorPreyGame(), np.random.default_rng(0), 0
    for ep in range(episodes):
        res = env.reset(ep, "1v2")
        while not res.episode_over:
            res = env.step({s: (hunter if s == 0 else prey)(res.obs[s], res.action_masks[s], rng)
                            for s in res.acting})
        wins += res.outcome.team_rank[0] == 1.0
    return wins / episodes


def _random(obs, mask, rng):
    return int(rng.choice(np.flatnonzero(mask)))


def test_random_play_is_balanced_and_the_references_win_their_roles():
    assert 0.35 <= _hunter_win_rate(_random, _random) <= 0.75
    assert _hunter_win_rate(lambda o, m, r: chase_action(o), _random) >= 0.9
    assert _hunter_win_rate(_random, lambda o, m, r: flee_action(o, m)) <= 0.15


def test_the_example_config_validates():
    check_example_config("predator_prey")


def test_random_matches_run_through_the_match_runner():
    results, seen = random_matches(PredatorPreyGame, min_episodes=12)
    for r in results:
        assert r.layout == "1v2" and r.outcome_kind == "wdl"
        assert [s.role for s in sorted(r.seats, key=lambda s: s.seat)] == ["hunter", "prey", "prey"]
    assert seen.terminated > 0
