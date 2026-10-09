"""coop_buttons demo game (T8.2): one team, joint presses, score outcome."""
from __future__ import annotations

import numpy as np
import pytest

from demo_checks import check_example_config, random_matches
from examples.coop_buttons.game import PRESS, CoopButtonsGame, measure_baselines


def _place(env: CoopButtonsGame, pos: list[list[int]], buttons: list[list[int]]) -> None:
    env.reset(0, "coop2")
    env._pos = np.array(pos, dtype=np.int64)
    env._button = np.array(buttons, dtype=np.int64)


def test_one_team_layout():
    spec = CoopButtonsGame().spec
    assert list(spec.layouts) == ["coop2"] and spec.teams("coop2") == [[0, 1]]
    assert spec.outcome_kind("coop2") == "score"


def test_a_joint_press_scores_and_moves_the_buttons():
    env = CoopButtonsGame(button_reward=0.05)
    _place(env, [[1, 1], [3, 3]], [[1, 1], [3, 3]])
    res = env.step({0: PRESS, 1: PRESS})
    assert env._score == 1 and res.acting == {0, 1}
    assert res.rewards[0] >= 1.0 and res.rewards[1] >= 1.0
    assert not (env._button == np.array([[1, 1], [3, 3]])).all()


def test_a_lonely_press_scores_nothing():
    env = CoopButtonsGame(button_reward=0.05)
    _place(env, [[1, 1], [3, 2]], [[1, 1], [3, 3]])
    res = env.step({0: PRESS, 1: PRESS})
    assert env._score == 0
    assert res.rewards == {0: pytest.approx(0.05), 1: 0.0}            # only the button bonus


def test_the_match_ends_by_rule_with_the_team_score():
    env = CoopButtonsGame(max_steps=2)
    _place(env, [[1, 1], [3, 3]], [[1, 1], [3, 3]])
    env.step({0: PRESS, 1: PRESS})
    res = env.step({0: 0, 1: 0})
    assert res.episode_over and not res.truncated and res.outcome.team_score == {0: 1.0}


def test_the_oracle_scores_and_random_play_does_not():
    base = measure_baselines(episodes=100)
    assert base["oracle"] >= 5.0 and base["random"] <= 0.1


def test_the_example_config_validates():
    check_example_config("coop_buttons")


def test_random_matches_run_through_the_match_runner():
    results, _ = random_matches(CoopButtonsGame, min_episodes=8)
    for r in results:
        assert r.layout == "coop2" and r.outcome_kind == "score" and len(r.teams) == 1


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"max_steps": 0}, "max_steps"),
    ({"size": 2}, "size"),
])
def test_bad_constructor_arguments_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CoopButtonsGame(**kwargs)
