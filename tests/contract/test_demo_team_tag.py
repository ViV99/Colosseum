"""team_tag demo game (T8.2): 2v2, local window, global_state, frozen teammates stay live."""
from __future__ import annotations

import numpy as np
import pytest

from demo_checks import check_example_config, random_matches
from examples.team_tag.game import TAG, TEAM, TeamTagGame, chase_action

STAY = {0: 0, 1: 0, 2: 0, 3: 0}


def _place(env: TeamTagGame, positions: list[list[int]]) -> None:
    env.reset(0, "2v2")
    env._pos = np.array(positions, dtype=np.int64)


def test_layout_teams_and_global_state_space():
    env = TeamTagGame(size=7)
    assert list(env.spec.layouts) == ["2v2"] and env.spec.teams("2v2") == [[0, 1], [2, 3]]
    role = env.spec.roles["player"]
    assert role.global_state_space.shape == (4, 7, 7) and role.global_state_space.dtype == np.uint8
    assert TeamTagGame(with_global_state=False).spec.roles["player"].global_state_space is None


def test_reset_gives_windows_masks_and_global_state_to_every_acting_seat():
    env = TeamTagGame(size=7, view_radius=2)
    res = env.reset(0, "2v2")
    assert res.acting == {0, 1, 2, 3}
    assert set(res.global_state) == {0, 1, 2, 3}
    for seat in range(4):
        assert res.obs[seat]["window"].shape == (4, 5, 5) and res.obs[seat]["window"].dtype == np.uint8
        assert res.global_state[seat].shape == (4, 7, 7) and res.global_state[seat][3].sum() == 1
        assert not res.action_masks[seat][TAG]                  # nobody within reach at the start
    assert TeamTagGame(with_global_state=False).reset(0, "2v2").global_state is None


def test_team_one_sees_the_board_mirrored():
    env = TeamTagGame(size=7)
    _place(env, [[0, 1], [0, 5], [6, 1], [6, 5]])
    assert env._obs(0)["vec"][0] == 0.0 and env._obs(2)["vec"][0] == 0.0   # both "start on the left"
    env.step({0: 0, 1: 0, 2: 4, 3: 0})                         # seat 2 steps "right" in its own view
    assert env._pos[2].tolist() == [5, 1]


def test_a_tag_freezes_the_enemy_who_keeps_getting_team_rewards():
    env = TeamTagGame(size=7, tag_reward=0.2, max_steps=10)
    _place(env, [[3, 3], [0, 0], [4, 3], [6, 6]])
    assert env._mask(0)[TAG] and env._mask(2)[TAG]
    res = env.step({0: TAG, 1: 0, 2: 0, 3: 0})
    assert res.acting == {0, 1, 3} and not res.terminated       # frozen, not terminated
    assert res.rewards == {0: pytest.approx(0.2), 1: pytest.approx(0.2), 2: pytest.approx(-0.2),
                           3: pytest.approx(-0.2)}
    res = env.step({0: 0, 1: 0, 3: 0})
    assert 2 in res.rewards and 2 not in res.acting               # the frozen seat is still live


def test_mutual_tags_freeze_both():
    env = TeamTagGame(size=7)
    _place(env, [[3, 3], [0, 0], [4, 4], [6, 6]])
    res = env.step({0: TAG, 1: 0, 2: TAG, 3: 0})
    assert res.acting == {1, 3}


def test_freezing_a_whole_team_ends_the_match_by_rule():
    env = TeamTagGame(size=7)
    _place(env, [[3, 3], [3, 4], [4, 3], [4, 4]])
    res = env.step({0: TAG, 1: 0, 2: 0, 3: 0})                   # seat 0 reaches both enemies
    assert res.episode_over and not res.truncated and res.acting == set() and not res.terminated
    assert res.outcome.team_score == {0: 2.0, 1: 0.0}
    assert res.rewards[0] == pytest.approx(1.0 + 0.4) and res.rewards[3] == pytest.approx(-1.0 - 0.4)


def test_time_limit_is_a_draw_on_equal_numbers():
    env = TeamTagGame(size=7, max_steps=2)
    _place(env, [[0, 0], [0, 6], [6, 0], [6, 6]])
    env.step(STAY)
    res = env.step(STAY)
    assert res.episode_over and not res.truncated
    assert res.outcome.team_score == {0: 2.0, 1: 2.0} and res.rewards == {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}


def test_the_chasing_reference_beats_random_teams():
    env, rng, wins = TeamTagGame(), np.random.default_rng(0), 0
    for ep in range(100):
        res = env.reset(ep, "2v2")
        while not res.episode_over:
            res = env.step({s: chase_action(res.obs[s], res.action_masks[s]) if TEAM[s] == 0
                            else int(rng.choice(np.flatnonzero(res.action_masks[s]))) for s in res.acting})
        wins += res.outcome.team_score[0] > res.outcome.team_score[1]
    assert wins >= 75


def test_the_example_config_validates():
    check_example_config("team_tag")


def test_random_matches_run_through_the_match_runner():
    results, _ = random_matches(TeamTagGame, min_episodes=8)
    for r in results:
        assert r.layout == "2v2" and r.outcome_kind == "wdl"
        assert sorted((s.seat, s.team) for s in r.seats) == [(0, 0), (1, 0), (2, 1), (3, 1)]
        assert all(s.eliminated_step is None for s in r.seats)       # frozen bots are never eliminated
