"""Reference examples on the SP2 contract (T8.5): chase (tree action) and space_miners (Units with a
Box component, entity list). No learning thresholds: env semantics, validate, random matches."""
from __future__ import annotations

import numpy as np
import pytest

from colosseum.core.specs import ActionSpec
from demo_checks import check_example_config, random_matches
from examples.composite_action.game import DIRS, GRID, ChaseGame


def test_chase_action_is_a_tree_in_natural_order():
    space = ChaseGame().spec.roles["player"].action_space
    assert list(space.spaces) == ["direction", "speed"]
    spec = ActionSpec.from_space(space)
    assert [g.kind for g in spec.groups] == ["discrete", "box"] and not spec.has_units


def test_chase_accepts_scalar_and_array_components():
    env = ChaseGame()
    env.reset(0, "2p")
    before = env._pos.copy()
    env.step({0: {"direction": np.int64(1), "speed": np.array([0.5], np.float32)},
              1: {"direction": 2, "speed": np.float32(1.0)}})
    expected = np.clip(before + np.stack([DIRS[1] * 0.5, DIRS[2] * 1.0]), 0.0, GRID)
    np.testing.assert_allclose(env._pos, expected, atol=1e-6)


def test_chase_ends_by_rule_after_twenty_steps():
    env = ChaseGame()
    res = env.reset(0, "2p")
    steps = 0
    while not res.episode_over:
        res = env.step({p: {"direction": 0, "speed": np.zeros(1, np.float32)} for p in res.acting})
        steps += 1
    assert steps == 20 and not res.truncated and set(res.rewards) == {0, 1}
    assert res.rewards[0] == -res.rewards[1]


def test_chase_config_validates():
    check_example_config("chase")


def test_chase_random_matches():
    results, _ = random_matches(ChaseGame, min_episodes=8)
    assert all(r.layout == "2p" and r.outcome_kind == "wdl" for r in results)


def test_space_miners_spaces_and_masks():
    pytest.importorskip("Box2D")
    from examples.space_miners.game import MAX_SHIPS, SpaceMinersGame

    env = SpaceMinersGame(max_ticks=20)
    role = env.spec.roles["player"]
    assert list(role.action_space.per_unit.spaces) == ["accel", "push"]
    spec = ActionSpec.from_space(role.action_space)
    assert spec.has_units and spec.num_deciders == MAX_SHIPS
    res = env.reset(0, "2p")
    obs, mask = res.obs[0], res.action_masks[0]
    assert obs["asteroids"].shape == (24, 7) and obs["asteroid_mask"].sum() >= 3
    assert mask["unit"].tolist() == [True] * MAX_SHIPS and mask["action"].shape == (MAX_SHIPS, 2)
    assert mask["action"][:, 0].all()                          # "no push" is always legal
    assert role.observation_space.contains(obs)


def test_space_miners_push_is_legal_only_within_reach():
    pytest.importorskip("Box2D")
    from examples.space_miners.game import SpaceMinersGame
    from examples.space_miners.game_engine import ASTEROID_RADIUS_UNITS, PPM, PUSH_RADIUS_UNITS, SHIP_RADIUS_UNITS

    env = SpaceMinersGame(max_ticks=20)
    env.reset(0, "2p")
    game = env._game
    ship, rock = game.players[0].ships[0], game.asteroids[0]
    reach = (SHIP_RADIUS_UNITS + ASTEROID_RADIUS_UNITS[rock.size] + PUSH_RADIUS_UNITS) / PPM
    for other in game.players[0].ships[1:]:                    # keep the other ships far from every asteroid
        other.body.position = (-1000.0, -1000.0)
    sx, sy = ship.body.position
    rock.body.position = (sx + reach - 0.05, sy)
    assert env._mask(0)["action"][:, 1].tolist() == [True, False, False]
    rock.body.position = (sx + reach + 0.05, sy)
    others = [a for a in game.asteroids if a is not rock]
    for a in others:
        a.body.position = (-2000.0, 2000.0)
    assert env._mask(0)["action"][:, 1].tolist() == [False, False, False]


def test_space_miners_players_see_mirrored_starts():
    pytest.importorskip("Box2D")
    from examples.space_miners.game import SpaceMinersGame

    res = SpaceMinersGame(max_ticks=20).reset(3, "2p")
    np.testing.assert_allclose(res.obs[0]["ships"], res.obs[1]["ships"], atol=1e-6)


def test_space_miners_match_ends_by_rule_with_scores():
    pytest.importorskip("Box2D")
    from examples.space_miners.game import MAX_SHIPS, SpaceMinersGame

    env = SpaceMinersGame(max_ticks=5)
    res = env.reset(0, "2p")
    idle = {"accel": np.zeros((MAX_SHIPS, 2), np.float32), "push": np.zeros(MAX_SHIPS, np.int64)}
    for _ in range(5):
        res = env.step({0: idle, 1: idle})
    assert res.episode_over and not res.truncated
    assert set(res.outcome.team_score) == {0, 1} and set(res.outcome.team_rank) == {0, 1}


@pytest.mark.parametrize("kwargs, match", [
    ({"preset": "Round 3"}, r"unknown preset 'Round 3'.*'Round 1', 'Round 2', 'Final Round'"),
    ({"max_ticks": 0}, r"max_ticks must be >= 1"),
    ({"max_asteroids": 20}, r"max_asteroids must be >= 21"),
])
def test_space_miners_rejects_bad_parameters(kwargs, match):
    pytest.importorskip("Box2D")
    from examples.space_miners.game import SpaceMinersGame

    with pytest.raises(ValueError, match=match):
        SpaceMinersGame(**kwargs)


def test_space_miners_config_validates():
    pytest.importorskip("Box2D")
    check_example_config("space_miners")


def test_space_miners_random_matches():
    pytest.importorskip("Box2D")
    import functools

    from examples.space_miners.game import SpaceMinersGame

    results, _ = random_matches(functools.partial(SpaceMinersGame, max_ticks=30), min_episodes=4)
    assert all(r.layout == "2p" and r.episode_length == 30 for r in results)
