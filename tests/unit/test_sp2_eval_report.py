"""Eval reports per layout and outcome kind, with SP1's interval math (T6.1)."""
from __future__ import annotations

import json
import math

import gymnasium
import numpy as np
import pytest

from colosseum.sp2.core.types import MatchResult, SeatResult, TeamResult
from colosseum.sp2.envs.game import GameSpec, RoleSpec, SeatSpec
from colosseum.sp2.eval import normal_interval, summarize, wilson_interval

OBS = gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)
ACT = gymnasium.spaces.Discrete(3)
SPEC = GameSpec(
    roles={"player": RoleSpec(OBS, ACT), "hunter": RoleSpec(OBS, gymnasium.spaces.Discrete(5))},
    layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1)),
             "4p": tuple(SeatSpec("player", i) for i in range(4)),
             "coop2": (SeatSpec("player", 0), SeatSpec("player", 0)),
             "hunt": (SeatSpec("hunter", 0), SeatSpec("player", 1))},
)


def match(layout, names, ranks, rewards=None, scores=None, roles=None, length=6) -> MatchResult:
    teams_of = [s.team for s in SPEC.layouts[layout]]
    seats = [SeatResult(seat=i, role=(roles or [s.role for s in SPEC.layouts[layout]])[i], team=teams_of[i],
                        agent_id=name, network_id="latest", reward=float((rewards or [0.0] * len(names))[i]))
             for i, name in enumerate(names)]
    teams = [TeamResult(team=t, rank=float(r), score=float((scores or {}).get(t, 0.0))) for t, r in ranks.items()]
    kind = SPEC.outcome_kind(layout)
    return MatchResult(match_id="m", layout=layout, outcome_kind=kind, seats=seats, teams=teams,
                       episode_length=length)


def test_wilson_and_normal_intervals_match_sp1():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(8, 10)
    assert low < 0.8 < high and 0.0 <= low and high <= 1.0
    assert wilson_interval(10, 10)[1] == 1.0
    mean, low, high = normal_interval([1.0, 3.0])
    assert mean == 2.0 and high - mean == pytest.approx(1.959963984540054 * math.sqrt(2.0) / math.sqrt(2))
    assert normal_interval([5.0]) == (5.0, 5.0, 5.0) and normal_interval([]) == (0.0, 0.0, 0.0)


def test_wdl_pairs_with_sides_and_reversed_rows():
    results = [
        match("2p", ["a", "b"], {0: 1, 1: 2}, rewards=[1, -1]),
        match("2p", ["b", "a"], {0: 1, 1: 1}),
        match("2p", ["b", "a"], {0: 2, 1: 1}, rewards=[-1, 1]),
    ]
    report = summarize(SPEC, results, agents=["a", "b"], num_matches=3).to_dict()
    section = report["layouts"]["2p"]
    assert section["outcome_kind"] == "wdl" and section["n"] == 3 and section["unattributed"] == 0
    a_row, b_row = section["pairs"]
    assert (a_row["agent_a"], a_row["wins"], a_row["draws"], a_row["losses"]) == ("a", 2, 1, 0)
    assert (b_row["agent_a"], b_row["wins"], b_row["losses"]) == ("b", 0, 2)
    assert a_row["score"] == pytest.approx(2.5 / 3) and a_row["score_ci"] == list(wilson_interval(2.5, 3))
    assert a_row["per_side"] == {"0": {"n": 1, "wins": 1, "draws": 0, "losses": 0, "score": 1.0},
                                 "1": {"n": 2, "wins": 1, "draws": 1, "losses": 0, "score": 0.75}}
    assert b_row["per_side"]["0"]["n"] == 2
    assert a_row["mean_return_a"] == pytest.approx(2 / 3)
    assert section["by_role"]["player"]["a"]["n"] == 3


def test_one_agent_in_a_wdl_layout_reports_per_seat():
    results = [match("2p", ["a", "a"], {0: 1, 1: 2}, rewards=[1, -1]), match("2p", ["a", "a"], {0: 2, 1: 1})]
    section = summarize(SPEC, results).to_dict()["layouts"]["2p"]
    assert section["pairs"] == []
    (row,) = section["solo"]
    assert row["agent"] == "a" and row["n"] == 4
    assert row["per_seat"]["0"]["wins"] == 1 and row["per_seat"]["0"]["losses"] == 1


def test_rank_layout_reports_mean_rank_first_places_and_higher_table():
    results = [
        match("4p", ["a", "b", "a", "b"], {0: 1, 1: 2, 2: 3, 3: 4}),
        match("4p", ["b", "a", "b", "a"], {0: 1, 1: 1, 2: 3, 3: 4}),
    ]
    section = summarize(SPEC, results).to_dict()["layouts"]["4p"]
    assert section["outcome_kind"] == "rank"
    a = section["agents"]["a"]
    assert a["n"] == 4 and a["mean_rank"] == pytest.approx((1 + 3 + 1 + 4) / 4) and a["first_place_rate"] == 0.5
    assert section["higher"]["a"]["b"]["n"] == 8
    assert section["higher"]["a"]["b"]["rate"] + section["higher"]["b"]["a"]["rate"] == pytest.approx(1.0)


def test_score_layout_reports_compositions_and_cross_play():
    results = [
        match("coop2", ["a", "a"], {0: 1}, scores={0: 4.0}),
        match("coop2", ["a", "a"], {0: 1}, scores={0: 6.0}),
        match("coop2", ["b", "a"], {0: 1}, scores={0: 1.0}),
    ]
    section = summarize(SPEC, results).to_dict()["layouts"]["coop2"]
    assert section["compositions"]["a+a"]["mean_score"] == 5.0 and section["compositions"]["a+a"]["homogeneous"]
    assert section["compositions"]["a+b"] == {"n": 1, "mean_score": 1.0, "score_ci": [1.0, 1.0],
                                              "homogeneous": False}


def test_roles_are_reported_separately_and_text_and_json_work(tmp_path):
    results = [match("hunt", ["h", "p"], {0: 1, 1: 2}, rewards=[2.0, -2.0])]
    report = summarize(SPEC, results, agents=["h", "p"], num_matches=1)
    section = report.to_dict()["layouts"]["hunt"]
    assert section["by_role"] == {"hunter": {"h": {"n": 1, "mean_return": 2.0}},
                                  "player": {"p": {"n": 1, "mean_return": -2.0}}}
    assert section["pairs"][0]["agent_a"] == "h" and section["pairs"][0]["wins"] == 1
    text = report.text()
    assert "layout hunt" in text and "role hunter" in text
    report.write_json(tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text())["layouts"]["hunt"]["n"] == 1


def test_mixed_teams_are_unattributed():
    spec = GameSpec.teams_of([2, 2], OBS, ACT)
    result = MatchResult(
        match_id="m", layout="2v2", outcome_kind="wdl",
        seats=[SeatResult(i, "player", i // 2, name, "latest", 0.0) for i, name in enumerate("abba")],
        teams=[TeamResult(0, 1.0, 0.0), TeamResult(1, 2.0, 0.0)], episode_length=3)
    section = summarize(spec, [result]).to_dict()["layouts"]["2v2"]
    assert section["unattributed"] == 1 and section["pairs"] == []


def test_sections_follow_the_layout_order_of_the_game_and_cross_play_is_marked():
    results = [match("coop2", ["b", "a"], {0: 1}, scores={0: 1.0}), match("2p", ["a", "b"], {0: 1, 1: 2})]
    report = summarize(SPEC, results)
    assert list(report.to_dict()["layouts"]) == ["2p", "coop2"]
    assert "(cross-play)" in report.text()
    with pytest.raises(ValueError, match="does not have"):
        summarize(GameSpec.symmetric(2, OBS, ACT), [match("coop2", ["a", "a"], {0: 1})])
