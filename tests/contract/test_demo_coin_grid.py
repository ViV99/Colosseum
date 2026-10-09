"""coin_grid demo game (T8.1): solo, Dict observation with a uint8 leaf, masks, truncation."""
from __future__ import annotations

import numpy as np
import pytest

from demo_checks import check_example_config, random_matches
from examples.coin_grid.game import CoinGridGame, greedy_action


def test_reset_gives_a_uint8_grid_and_a_float_vector():
    env = CoinGridGame(size=5, num_coins=3, max_steps=10)
    res = env.reset(0, "solo")
    obs = res.obs[0]
    assert res.acting == {0}
    assert list(obs) == ["grid", "vec"]
    assert obs["grid"].dtype == np.uint8 and obs["grid"].shape == (2, 5, 5)
    assert obs["grid"][0].sum() == 1 and obs["grid"][1].sum() == 3
    assert obs["vec"].dtype == np.float32 and obs["vec"][0] == 1.0
    assert env.spec.roles["player"].observation_space.contains(obs)


def test_moves_off_the_grid_are_masked():
    env = CoinGridGame(size=5)
    env.reset(0, "solo")
    env._pos = np.array([0, 0])
    assert env._mask().tolist() == [True, False, True, False, True]
    env._pos = np.array([4, 4])
    assert env._mask().tolist() == [True, True, False, True, False]


def test_a_collected_coin_pays_one_and_reappears_elsewhere():
    env = CoinGridGame(size=5, num_coins=1)
    env.reset(0, "solo")
    env._coins[:] = False
    env._pos = np.array([2, 2])
    env._coins[2, 3] = True
    res = env.step({0: 4})                                   # right
    assert res.rewards == {0: 1.0}
    assert env._coins.sum() == 1 and not env._coins[2, 3]


def test_the_step_limit_truncates_with_final_obs():
    env = CoinGridGame(size=5, num_coins=2, max_steps=4)
    res = env.reset(1, "solo")
    for _ in range(4):
        assert not res.episode_over
        res = env.step({0: 0})
    assert res.episode_over and res.truncated and res.acting == set()
    assert set(res.final_obs) == {0}
    assert res.final_obs[0]["grid"].dtype == np.uint8 and res.final_obs[0]["vec"][0] == 0.0


def test_the_same_seed_gives_the_same_episode():
    a, b = CoinGridGame().reset(7, "solo"), CoinGridGame().reset(7, "solo")
    assert np.array_equal(a.obs[0]["grid"], b.obs[0]["grid"])


def _mean_score(policy, episodes: int = 100) -> float:
    env, rng, total = CoinGridGame(), np.random.default_rng(0), 0.0
    for ep in range(episodes):
        res = env.reset(ep, "solo")
        while not res.episode_over:
            res = env.step({0: policy(res.obs[0], res.action_masks[0], rng)})
            total += res.rewards.get(0, 0.0)
    return total / episodes


def test_the_scripted_reference_collects_far_more_than_random_play():
    greedy = _mean_score(lambda obs, mask, rng: greedy_action(obs))
    random = _mean_score(lambda obs, mask, rng: int(rng.choice(np.flatnonzero(mask))))
    assert greedy >= 5 * random > 0


def test_the_example_config_validates():
    check_example_config("coin_grid")


def test_random_matches_run_through_the_match_runner():
    results, _ = random_matches(CoinGridGame, min_episodes=12)
    assert {r.layout for r in results} == {"solo"}
    for r in results:
        assert r.outcome_kind == "score" and len(r.teams) == 1 and r.episode_length == 50
        assert r.teams[0].score == pytest.approx(r.seats[0].reward)   # default: mean of the team's returns


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"max_steps": 0}, "max_steps"),
    ({"size": 2}, "size"),
    ({"num_coins": 0}, "num_coins"),
])
def test_bad_constructor_arguments_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CoinGridGame(**kwargs)
