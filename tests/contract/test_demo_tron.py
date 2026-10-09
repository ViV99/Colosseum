"""tron demo game (T8.1): FFA with elimination, 2p/3p/4p layouts, ranks by elimination order."""
from __future__ import annotations

import numpy as np
import pytest

from demo_checks import check_example_config, random_matches
from examples.tron.game import TronGame, safe_action


def _place(env: TronGame, layout: str, heads: list[list[int]], headings: list[int]) -> None:
    env.reset(0, layout)
    env._walls[:] = False
    for seat, (head, heading) in enumerate(zip(heads, headings)):
        env._head[seat], env._heading[seat] = head, heading
        env._walls[head[0], head[1]] = True


def test_layouts_and_teams():
    spec = TronGame().spec
    assert sorted(spec.layouts) == ["2p", "3p", "4p"]
    assert spec.teams("3p") == [[0], [1], [2]]
    assert spec.outcome_kind("2p") == "wdl" and spec.outcome_kind("4p") == "rank"


def test_every_layout_starts_on_distinct_cells():
    env = TronGame()
    for layout, n in (("2p", 2), ("3p", 3), ("4p", 4)):
        for seed in range(10):
            res = env.reset(seed, layout)
            assert res.acting == set(range(n))
            assert len({tuple(h) for h in env._head[:n]}) == n
            assert all(o.dtype == np.uint8 and o.shape == (2, 7, 7) for o in res.obs.values())


def test_the_window_is_rotated_to_the_heading():
    env = TronGame(size=10, view_radius=3)
    _place(env, "2p", [[5, 5], [0, 0]], [1, 0])           # seat 0 heads right
    env._walls[5, 6] = True                                # the cell straight ahead
    assert env._obs(0)[0, 2, 3] == 1                       # straight ahead = one row up from the centre
    env._walls[5, 6] = False
    env._walls[4, 5] = True                                # above the head = on its left
    assert env._obs(0)[0, 3, 2] == 1


def test_crashes_eliminate_and_ranks_follow_the_elimination_order():
    env = TronGame(size=10)
    _place(env, "3p", [[0, 5], [5, 5], [7, 5]], [0, 1, 1])
    res = env.step({0: 0, 1: 0, 2: 0})                     # seat 0 drives off the top edge
    assert res.terminated == {0} and res.rewards == {0: -1.0}
    assert res.acting == {1, 2} and not res.episode_over and set(res.obs) == {1, 2}
    env._head[1], env._head[2] = [5, 9], [7, 9]           # both at the right edge, heading right
    res = env.step({1: 0, 2: 0})
    assert res.terminated == {1, 2} and res.episode_over and res.acting == set()
    assert res.outcome.team_rank == {0: 3.0, 1: 1.5, 2: 1.5}
    assert res.rewards == {1: pytest.approx(0.5), 2: pytest.approx(0.5)}


def test_a_head_on_crash_in_2p_is_a_draw():
    env = TronGame(size=10)
    _place(env, "2p", [[5, 3], [5, 5]], [1, 3])            # both move into (5, 4)
    res = env.step({0: 0, 1: 0})
    assert res.terminated == {0, 1} and res.episode_over
    assert res.outcome.team_rank == {0: 1.5, 1: 1.5} and res.rewards == {0: 0.0, 1: 0.0}


def test_the_last_cycle_wins_2p():
    env = TronGame(size=10)
    _place(env, "2p", [[0, 5], [5, 5]], [0, 1])
    res = env.step({0: 0, 1: 0})
    assert res.terminated == {0} and res.episode_over
    assert res.outcome.team_rank == {0: 2.0, 1: 1.0} and res.rewards == {0: -1.0, 1: 1.0}


def test_the_safe_reference_outlives_random_cycles():
    env, rng, first = TronGame(), np.random.default_rng(0), 0
    for ep in range(200):
        res = env.reset(ep, "4p")
        while not res.episode_over:
            res = env.step({s: safe_action(res.obs[s], rng) if s == 0 else int(rng.integers(3)) for s in res.acting})
        first += res.outcome.team_rank[0] == 1.0
    assert first / 200 >= 0.6


def test_the_example_config_validates():
    check_example_config("tron")


def test_random_matches_run_through_the_match_runner():
    results, seen = random_matches(TronGame, min_episodes=24)
    assert {r.layout for r in results} == {"2p", "3p", "4p"}
    for r in results:
        n = int(r.layout[0])
        assert r.outcome_kind == ("wdl" if n == 2 else "rank") and len(r.teams) == n
        assert sorted(t.team for t in r.teams) == list(range(n))
        assert sum(t.rank for t in r.teams) == pytest.approx(n * (n + 1) / 2)   # places are shared, never lost
    assert seen.terminated > 0


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"view_radius": 0}, "view_radius"),
    ({"size": 5}, "size"),
    ({"players": (2, 5)}, "players"),
])
def test_bad_constructor_arguments_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TronGame(**kwargs)


def test_the_smallest_view_radius_works():
    assert TronGame(view_radius=1).reset(0, "2p").obs[0].shape == (2, 3, 3)
