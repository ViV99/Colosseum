"""Eval engine: scheduling, seat rotation, turn-based per-seat state (T7.1)."""
import itertools
from collections import Counter

import gymnasium
import numpy as np
import pytest

from colosseum.core.errors import EnvContractError
from colosseum.envs.base_env import BaseEnv
from colosseum.eval import MatchRecord, play_matches, schedule_lineups
from helpers import AlternatingEnv, MoveCounterModel

EXPECTED_EPISODE = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_pairwise_schedule_balances_seats(num_players):
    names = ["a", "b", "c"]
    lineups = schedule_lineups(names, num_players, num_matches=6)
    assert len(lineups) == 3 * 6
    for a, b in itertools.combinations(names, 2):
        pair_lineups = [lu for lu in lineups if set(lu) == {a, b}]
        assert len(pair_lineups) == 6
        for seat in range(num_players):
            counts = Counter(lu[seat] for lu in pair_lineups)
            assert counts[a] == counts[b] == 3


def test_solo_schedule():
    assert schedule_lineups(["a", "b"], 1, 3) == [("a",)] * 3 + [("b",)] * 3
    assert schedule_lineups(["a"], 2, 2) == [("a", "a"), ("a", "a")]
    with pytest.raises(ValueError):
        schedule_lineups(["a", "a"], 2, 2)
    with pytest.raises(ValueError):
        schedule_lineups(["a", "b"], 2, 0)


def test_turn_based_state_advances_only_on_own_moves_and_resets_per_episode():
    envs = []

    def env_fn():
        env = AlternatingEnv()
        envs.append(env)
        return env

    models = {"a": MoveCounterModel(), "b": MoveCounterModel()}
    lineups = schedule_lineups(["a", "b"], 2, 4)
    records = play_matches(models, env_fn, lineups, num_envs=2, deterministic=True)

    assert len(records) == 4
    assert sorted(r.lineup for r in records) == sorted(lineups)
    assert all(isinstance(r, MatchRecord) and r.length == 6 for r in records)
    assert all(r.outcomes == (1.0, 0.0) and r.returns == (1.0, -1.0) for r in records)
    complete = [ep for env in envs for ep in env.episodes if len(ep) == 6]
    assert len(complete) >= 4
    assert all(ep == EXPECTED_EPISODE for ep in complete)


def test_all_lineups_play_even_with_more_envs_than_matches():
    lineups = schedule_lineups(["a", "b", "c"], 2, 2)
    records = play_matches({n: MoveCounterModel() for n in "abc"}, AlternatingEnv, lineups, num_envs=16)
    assert Counter(r.lineup for r in records) == Counter(lineups)


def test_unknown_agent_in_lineup_raises():
    with pytest.raises(ValueError, match="unknown"):
        play_matches({"a": MoveCounterModel()}, AlternatingEnv, [("a", "z")])


class _NoLegalActionEnv(BaseEnv):
    """1-player env whose acting seat has an all-False mask."""

    @property
    def num_players(self):
        return 1

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(3)

    def reset(self, seed=None):
        return {0: np.zeros(2, np.float32)}, {0: {"action_mask": np.zeros(3, dtype=bool)}}

    def step(self, actions):
        return {0: np.zeros(2, np.float32)}, {0: 0.0}, {0: True}, {0: False}, {0: {}}


def test_acting_seat_without_legal_action_raises():
    with pytest.raises(EnvContractError, match="seat 0"):
        play_matches({"a": MoveCounterModel(3)}, _NoLegalActionEnv, [("a",)])
