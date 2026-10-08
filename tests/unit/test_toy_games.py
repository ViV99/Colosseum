"""The toy games of tests/game_helpers.py have valid specs and well-formed first results (SP2 T1.4)."""
import numpy as np
import pytest

from colosseum.sp2.core.specs import ObsSpec
from colosseum.sp2.envs.game import StepResult
from game_helpers import (
    TOY_GAMES,
    EliminationFFA,
    SoloCounterGame,
    TeamDeadTeammateGame,
    TurnTakingGame,
    UnitsGame,
)


@pytest.mark.parametrize("name", sorted(TOY_GAMES))
def test_every_toy_game_resets_every_layout(name):
    env = TOY_GAMES[name]()
    env.spec.validate()
    for layout in env.spec.layouts:
        result = env.reset(seed=0, layout=layout)
        assert isinstance(result, StepResult) and result.acting and not result.rewards
        for seat in result.acting:
            role = env.spec.roles[env.spec.role_of(layout, seat)]
            ObsSpec.from_space(role.observation_space).check(result.obs[seat], f"{name} seat {seat}")


def test_solo_counter_rewards_and_truncation():
    env = SoloCounterGame(length=8, truncate_at=3)
    env.reset(seed=None, layout="solo")
    assert env.step({0: 1}).rewards == {0: 1.0}
    env.step({0: 0})
    last = env.step({0: 1})
    assert last.episode_over and last.truncated and set(last.final_obs) == {0} and not last.acting


def test_turn_taking_pays_the_waiting_seat():
    env = TurnTakingGame(length=2)
    first = env.reset(seed=None, layout="2p")
    assert first.acting == {0} and first.action_masks[0].tolist() == [True, True, False]
    second = env.step({0: 1})
    assert second.acting == {1} and second.rewards == {1: 1.0}
    assert env.step({1: 0}).episode_over


def test_ffa_default_eliminations_and_ranks():
    env = EliminationFFA(max_players=4)
    r = env.reset(seed=None, layout="4p")
    terminated = []
    while not r.episode_over:
        r = env.step({p: 0 for p in r.acting})
        terminated.append(sorted(r.terminated))
    assert terminated == [[3], [2], [1]]
    assert r.outcome.team_rank == {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    assert r.rewards[1] == pytest.approx(-0.9)


def test_dead_teammate_keeps_getting_rewards():
    env = TeamDeadTeammateGame(length=4, dead_at=2)
    r = env.reset(seed=None, layout="2v2")
    for _ in range(2):
        r = env.step({p: 1 for p in r.acting})
    assert r.acting == {0, 2, 3} and r.rewards[1] == 1.0


def test_units_game_births_deaths_and_masks():
    env = UnitsGame(max_units=4)
    r = env.reset(seed=None, layout="solo")
    assert r.obs[0]["grid"].dtype == np.uint8
    mask = r.action_masks[0]["units"]
    assert mask["unit"].tolist() == [True, False, False, False]
    assert not mask["action"][1].any() and mask["action"][0].tolist() == [True] * 4 + [True, False, False, False]
    actions = {"base": 1, "units": {"move": np.zeros(4, np.int64), "target": np.zeros(4, np.int64)}}
    r = env.step({0: actions})
    assert r.rewards == {0: 0.5 + 0.25}
    assert r.action_masks[0]["units"]["unit"].tolist() == [True, True, False, False]
    assert UnitsGame(uint8_grid=False).reset(None, "solo").obs[0]["grid"].dtype == np.float32
