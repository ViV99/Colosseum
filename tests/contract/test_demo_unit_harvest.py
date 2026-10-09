"""unit_harvest demo game (T8.1): a base plus Units workers, births and deaths, entity lists."""
from __future__ import annotations

import numpy as np
import pytest

from colosseum.core.specs import ActionSpec
from demo_checks import check_example_config, random_matches
from examples.unit_harvest.game import UnitHarvestGame, scripted_action

K128 = {"max_units": 128, "size": 16, "num_resources": 16, "initial_workers": 32, "max_steps": 80}


def _stay(env: UnitHarvestGame, base: int = 0) -> dict:
    return {"base": base, "workers": np.zeros(env.max_units, np.int64)}


def test_action_space_keeps_its_order_and_counts_deciders():
    env = UnitHarvestGame(max_units=8)
    space = env.spec.roles["player"].action_space
    assert list(space.spaces) == ["base", "workers"]
    spec = ActionSpec.from_space(space)
    assert spec.has_units and spec.num_deciders == 1 + 8


def test_reset_gives_entity_lists_and_masks():
    env = UnitHarvestGame()
    res = env.reset(0, "2p")
    assert res.acting == {0, 1}
    obs, mask = res.obs[0], res.action_masks[0]
    assert obs["units"].shape == (8, 10) and obs["unit_mask"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert obs["resources"].shape == (6, 2) and obs["resource_mask"].all()
    assert mask["base"].tolist() == [True, False]                       # no stock to build yet
    assert mask["workers"]["unit"].tolist() == [True, True] + [False] * 6
    assert mask["workers"]["action"].shape == (8, 5) and mask["workers"]["action"][:, 0].all()
    assert env.spec.roles["player"].observation_space.contains(obs)


def test_both_sides_see_the_same_mirrored_board():
    for seed in range(20):
        res = UnitHarvestGame().reset(seed, "2p")
        a, b = res.obs[0], res.obs[1]
        np.testing.assert_allclose(np.sort(a["resources"], axis=0), np.sort(b["resources"], axis=0))
        np.testing.assert_allclose(a["units"][:, :3], b["units"][:, :3])


def test_seat_one_actions_are_mirrored_back():
    env = UnitHarvestGame()
    env.reset(0, "2p")
    env._pos[1, 0] = [5, 3]
    act = _stay(env)
    act["workers"][0] = 4                                   # "right" as seat 1 sees the board
    env.step({0: _stay(env), 1: act})
    assert env._pos[1, 0].tolist() == [4, 3]


def test_the_base_builds_a_worker_into_the_lowest_free_slot():
    env = UnitHarvestGame(build_cost=1)
    env.reset(0, "2p")
    env._stock[0] = 1
    res = env.step({0: _stay(env, base=1), 1: _stay(env)})
    assert res.obs[0]["unit_mask"].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert res.action_masks[0]["workers"]["unit"][2]
    assert env._stock[0] == 0 and env._pos[0, 2].tolist() == env._bases[0].tolist()


def test_workers_of_both_sides_in_one_cell_die():
    env = UnitHarvestGame()
    env.reset(0, "2p")
    env._pos[0, 0], env._pos[1, 0] = [3, 3], [4, 3]
    act = _stay(env)
    act["workers"][0] = 4                                   # side 0 steps right onto the enemy
    res = env.step({0: act, 1: _stay(env)})
    assert not env._alive[0, 0] and not env._alive[1, 0]
    assert res.obs[0]["unit_mask"][0] == 0 and res.obs[1]["unit_mask"][0] == 0
    assert not res.action_masks[0]["workers"]["unit"][0]


def test_a_carrier_on_its_base_deposits():
    env = UnitHarvestGame(deposit_reward=0.1)
    env.reset(0, "2p")
    env._carry[0, 0] = True                                  # slot 0 stands on its base
    res = env.step({0: _stay(env), 1: _stay(env)})
    assert env._score.tolist() == [1, 0] and env._stock.tolist() == [1, 0]
    assert res.rewards == {0: pytest.approx(0.1), 1: 0.0}
    assert res.action_masks[0]["base"].tolist() == [True, True]


def test_the_match_ends_by_rule_with_team_scores():
    env = UnitHarvestGame(max_steps=3)
    env.reset(0, "2p")
    env._score[:] = [2, 1]
    for _ in range(3):
        res = env.step({0: _stay(env), 1: _stay(env)})
    assert res.episode_over and not res.truncated and res.acting == set()
    assert res.outcome.team_score == {0: 2.0, 1: 1.0}
    assert res.rewards == {0: pytest.approx(1.0), 1: pytest.approx(-1.0)}


def test_k128_configuration():
    env = UnitHarvestGame(**K128)
    res = env.reset(0, "2p")
    assert res.obs[0]["units"].shape == (128, 10) and res.obs[0]["unit_mask"].sum() == 32
    assert ActionSpec.from_space(env.spec.roles["player"].action_space).num_deciders == 129


def test_the_scripted_reference_beats_idle_play():
    env = UnitHarvestGame()
    res = env.reset(0, "2p")
    while not res.episode_over:
        res = env.step({0: scripted_action(res.obs[0]), 1: _stay(env)})
    assert res.outcome.team_score[0] > 10 and res.outcome.team_score[1] == 0


def test_the_example_config_validates():
    check_example_config("unit_harvest")


def test_random_matches_run_through_the_match_runner():
    results, _ = random_matches(UnitHarvestGame, min_episodes=8)
    for r in results:
        assert r.layout == "2p" and r.outcome_kind == "wdl" and len(r.teams) == 2
        rank = {t.team: t.rank for t in r.teams}
        score = {t.team: t.score for t in r.teams}
        assert (rank[0] < rank[1]) == (score[0] > score[1])


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"size": 4, "num_resources": 10}, "does not fit"),
    ({"max_steps": 0}, "max_steps"),
    ({"num_resources": 3}, "even"),
    ({"initial_workers": 0}, "initial_workers"),
])
def test_bad_constructor_arguments_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        UnitHarvestGame(**kwargs)


def test_resources_filling_every_free_cell_still_reset():
    env = UnitHarvestGame(size=4, num_resources=8)          # 4 free cells per half
    env.reset(0, "2p")
    assert len({tuple(xy) for xy in env._resources.tolist()}) == 8
