"""Eval schedules by the CLI rules of spec block 8 (T6.1)."""
from __future__ import annotations

from collections import Counter

import gymnasium
import numpy as np
import pytest

from colosseum.sp2.envs.game import GameSpec, RoleSpec, SeatSpec
from colosseum.sp2.eval import default_layouts, schedule_lineups

OBS = gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)
ACT = gymnasium.spaces.Discrete(3)
HUNTER = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (4,), np.float32), gymnasium.spaces.Discrete(5))
PREY = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (3,), np.float32), gymnasium.spaces.Discrete(4))


def agents(lineup) -> list[str]:
    return [seat.agent_id for seat in lineup.seats]


def test_pairs_rotate_teams_in_a_two_player_game():
    spec = GameSpec.symmetric(2, OBS, ACT)
    lineups = schedule_lineups(spec, "2p", {"a": ["player"], "b": ["player"]}, 4)
    assert [agents(lu) for lu in lineups] == [["a", "b"], ["b", "a"]] * 2
    assert all(not s.collect and s.network_id == "latest" for lu in lineups for s in lu.seats)


def test_pairs_alternate_over_ffa_teams_and_cover_every_pair():
    spec = GameSpec.symmetric(4, OBS, ACT)
    lineups = schedule_lineups(spec, "4p", {"a": ["player"], "b": ["player"], "c": ["player"]}, 2)
    assert len(lineups) == 3 * 2
    assert agents(lineups[0]) == ["a", "b", "a", "b"] and agents(lineups[1]) == ["b", "a", "b", "a"]
    assert {tuple(sorted(set(agents(lu)))) for lu in lineups} == {("a", "b"), ("a", "c"), ("b", "c")}


def test_teams_are_filled_homogeneously():
    spec = GameSpec.teams_of([2, 2], OBS, ACT)
    lineups = schedule_lineups(spec, "2v2", {"a": ["player"], "b": ["player"]}, 2)
    assert [agents(lu) for lu in lineups] == [["a", "a", "b", "b"], ["b", "b", "a", "a"]]


def test_asymmetric_pair_is_not_rotated():
    spec = GameSpec(roles={"hunter": HUNTER, "prey": PREY},
                    layouts={"1v2": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1))})
    lineups = schedule_lineups(spec, "1v2", {"h": ["hunter"], "p": ["prey"]}, 3)
    assert [agents(lu) for lu in lineups] == [["h", "p", "p"]] * 3
    assert schedule_lineups(spec, "1v2", {"h": ["hunter"], "h2": ["hunter"]}, 2) == []  # nobody plays prey


def test_one_agent_in_a_competitive_layout_plays_every_team():
    spec = GameSpec.symmetric(3, OBS, ACT)
    lineups = schedule_lineups(spec, "3p", {"a": ["player"]}, 5)
    assert len(lineups) == 5 and all(agents(lu) == ["a"] * 3 for lu in lineups)


def test_one_team_gives_homogeneous_teams_then_cross_play():
    spec = GameSpec.teams_of([2], OBS, ACT)
    lineups = schedule_lineups(spec, "coop2", {"a": ["player"], "b": ["player"]}, 4)
    counts = Counter(tuple(agents(lu)) for lu in lineups)
    assert counts == {("a", "a"): 4, ("b", "b"): 4, ("a", "b"): 2, ("b", "a"): 2}
    solo = GameSpec.solo(OBS, ACT)
    assert [agents(lu) for lu in schedule_lineups(solo, "solo", {"a": ["player"], "b": ["player"]}, 2)] == [
        ["a"], ["a"], ["b"], ["b"]]


def test_cross_play_respects_roles():
    spec = GameSpec(roles={"hunter": HUNTER, "prey": PREY},
                    layouts={"duo": (SeatSpec("hunter", 0), SeatSpec("prey", 0))})
    lineups = schedule_lineups(spec, "duo", {"h": ["hunter"], "p": ["prey"]}, 3)
    assert [agents(lu) for lu in lineups] == [["h", "p"]] * 3  # no homogeneous team is possible


def test_default_layouts_and_errors():
    spec = GameSpec(roles={"player": RoleSpec(OBS, ACT), "hunter": HUNTER, "prey": PREY},
                    layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1)),
                             "1v1": (SeatSpec("hunter", 0), SeatSpec("prey", 1))})
    assert default_layouts(spec, {"a": ["player"], "b": ["player"]}) == ["2p"]
    assert default_layouts(spec, {"h": ["hunter"], "p": ["prey"]}) == ["1v1"]
    with pytest.raises(ValueError, match="unknown layout"):
        schedule_lineups(spec, "9p", {"a": ["player"]}, 1)
    with pytest.raises(ValueError, match="num_matches"):
        schedule_lineups(spec, "2p", {"a": ["player"]}, 0)
