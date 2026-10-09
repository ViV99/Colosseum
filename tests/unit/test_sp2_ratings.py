"""Per-layout ratings: team-pair weights, ELO, win rates, past, scores, cross-play (T5.2)."""
from __future__ import annotations

import gymnasium
import numpy as np
import pytest

from colosseum.coordinator.ratings import (
    CrossPlayTable,
    EloRating,
    MemberPair,
    PastWinRate,
    RatingBook,
    ScoreTracker,
    WinRateTracker,
    member_pairs,
)
from colosseum.core.types import MatchResult, SeatResult, TeamResult
from colosseum.envs.game import GameSpec, RoleSpec, SeatSpec

OBS = gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)
ACT = gymnasium.spaces.Discrete(3)


def seat(i, team, agent, network="latest", role="player", reward=0.0) -> SeatResult:
    return SeatResult(seat=i, role=role, team=team, agent_id=agent, network_id=network, reward=reward)


def result(layout, seats, ranks, scores=None, kind=None, length=5) -> MatchResult:
    kind = kind or ("score" if len(ranks) == 1 else "wdl" if len(ranks) == 2 else "rank")
    teams = [TeamResult(team=t, rank=float(r), score=float((scores or {}).get(t, 0.0))) for t, r in ranks.items()]
    return MatchResult(match_id="m", layout=layout, outcome_kind=kind, seats=seats, teams=teams,
                       episode_length=length)


def test_one_vs_one_is_one_pair_of_weight_one():
    pairs = member_pairs(result("2p", [seat(0, 0, "a"), seat(1, 1, "b")], {0: 1, 1: 2}))
    assert pairs == [MemberPair("cross", "a", "b", 1.0, 1.0, "player", "player")]


def test_two_v_two_splits_the_team_pair_weight_between_member_pairs():
    r = result("2v2", [seat(0, 0, "a"), seat(1, 0, "a"), seat(2, 1, "b"), seat(3, 1, "b")], {0: 2, 1: 1})
    pairs = member_pairs(r)
    assert len(pairs) == 4 and all(p.score_a == 0.0 and p.weight == pytest.approx(0.25) for p in pairs)
    assert sum(p.weight for p in pairs) == pytest.approx(1.0)  # the same total weight as a 1v1


def test_ffa_team_pairs_get_one_over_t_minus_one():
    r = result("4p", [seat(i, i, agent) for i, agent in enumerate("abcd")], {0: 1, 1: 2, 2: 3, 3: 4})
    pairs = member_pairs(r)
    assert len(pairs) == 6 and all(p.weight == pytest.approx(1 / 3) for p in pairs)
    assert {(p.a, p.b): p.score_a for p in pairs}[("a", "d")] == 1.0


def test_teammates_are_not_compared_and_skipped_pairs_leave_the_divisor():
    r = result("2v1v1", [seat(0, 0, "a"), seat(1, 0, "b"), seat(2, 1, "c"), seat(3, 2, "c")],
               {0: 1, 1: 2, 2: 2})
    pairs = member_pairs(r)
    assert ("a", "b") not in {(p.a, p.b) for p in pairs}
    by_teams = [p for p in pairs if p.a in ("a", "b") and p.b == "c"]
    assert len(by_teams) == 4 and all(p.weight == pytest.approx(0.5 / 2) for p in by_teams)
    # team 1 vs team 2: c vs c, both latest -> skipped, no pair at all
    assert not [p for p in pairs if p.a == p.b == "c"]

    same = result("2v2", [seat(0, 0, "a"), seat(1, 0, "a"), seat(2, 1, "a"), seat(3, 1, "b")], {0: 1, 1: 2})
    pairs = member_pairs(same)  # a-vs-a latest pairs are skipped: 2 counted pairs share the weight
    assert [(p.a, p.b, p.weight) for p in pairs] == [("a", "b", 0.5), ("a", "b", 0.5)]


def test_latest_vs_own_checkpoint_is_a_past_pair_from_the_latest_side():
    r = result("2p", [seat(0, 0, "a", "ckpt_v3"), seat(1, 1, "a")], {0: 1, 1: 2})
    assert member_pairs(r) == [MemberPair("past", "a", "a", 0.0, 1.0, "player", "player")]
    latest_first = result("2p", [seat(0, 0, "a"), seat(1, 1, "a", "ckpt_v3")], {0: 1, 1: 2})
    assert member_pairs(latest_first) == [MemberPair("past", "a", "a", 1.0, 1.0, "player", "player")]
    two_checkpoints = result("2p", [seat(0, 0, "a", "ckpt_v1"), seat(1, 1, "a", "ckpt_v2")], {0: 1, 1: 2})
    assert member_pairs(two_checkpoints) == []
    assert member_pairs(result("solo", [seat(0, 0, "a")], {0: 1})) == []


def test_elo_weighted_updates_match_sp1_and_ignore_pair_order():
    sp1, v2 = EloRating(), EloRating()
    sp1.update_pairs([("a", "b", 1.0), ("a", "c", 0.5)], k_scale=0.5)
    v2.update_weighted([("a", "c", 0.5, 0.5), ("a", "b", 1.0, 0.5)])
    assert sp1.all_ratings == pytest.approx(v2.all_ratings)
    assert v2.get("a") == pytest.approx(1200 + 32 * 0.5 * 0.5)
    assert v2.get("b") + v2.get("c") + v2.get("a") == pytest.approx(3600)


def test_win_rates_are_weighted_and_draws_count_half():
    w = WinRateTracker()
    w.record_pair("a", "b", 1.0, weight=0.25)
    w.record_pair("a", "b", 0.5, weight=0.75)
    assert w.get_win_rate("a", "b") == pytest.approx(0.25 + 0.375)
    assert w.get_win_rate("b", "a") == pytest.approx(1 - 0.625)
    assert w.games("a", "b") == 2 and w.get_win_rate("a", "z") == 0.5
    with pytest.raises(ValueError):
        w.record_pair("a", "b", 1.5)
    with pytest.raises(ValueError):
        w.record_pair("a", "b", 1.0, weight=0.0)


def test_past_win_rate_window_and_weights():
    past = PastWinRate(window=2)
    assert past.get("a") is None
    past.record("a", 1.0, 3.0)
    past.record("a", 0.0, 1.0)
    assert past.get("a") == pytest.approx(0.75) and past.games("a") == 2
    past.record("a", 0.0, 1.0)  # the oldest entry falls out of the window
    assert past.get("a") == 0.0
    with pytest.raises(ValueError):
        past.record("a", -0.5)
    with pytest.raises(ValueError):
        past.record("a", 1.0, weight=0.0)


def test_score_tracker_mean_ema_and_ci():
    tracker = ScoreTracker(ema_alpha=0.5)
    for score in (1.0, 3.0):
        tracker.update("a", score)
    summary = tracker.summary()["a"]
    assert summary["n"] == 2 and summary["mean"] == 2.0 and summary["ema"] == 2.0
    half = 1.959963984540054 * np.std([1.0, 3.0], ddof=1) / np.sqrt(2)
    assert summary["ci_low"] == pytest.approx(2.0 - half) and summary["ci_high"] == pytest.approx(2.0 + half)
    tracker.update("b", 5.0)
    assert tracker.summary()["b"]["ci_low"] == tracker.summary()["b"]["ci_high"] == 5.0


def test_cross_play_keys_are_sorted_multisets():
    table = CrossPlayTable()
    table.update(["b", "a"], 2.0)
    table.update(["a", "b"], 4.0)
    table.update(["a", "a"], 1.0)
    assert table.summary() == {"a+a": {"n": 1, "mean": 1.0}, "a+b": {"n": 2, "mean": 3.0}}


def test_rating_book_keeps_layouts_apart_and_routes_score_layouts():
    spec = GameSpec(
        roles={"player": RoleSpec(OBS, ACT)},
        layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1)),
                 "4p": tuple(SeatSpec("player", i) for i in range(4)),
                 "coop2": (SeatSpec("player", 0), SeatSpec("player", 0))},
    )
    book = RatingBook(spec, ["a", "b"])
    book.update(result("2p", [seat(0, 0, "a"), seat(1, 1, "b")], {0: 1, 1: 2}))
    book.update(result("2p", [seat(0, 0, "a"), seat(1, 1, "a", "ckpt_v1")], {0: 1, 1: 1}))
    book.update(result("coop2", [seat(0, 0, "a"), seat(1, 0, "b")], {0: 1}, scores={0: 6.0}))
    snap = book.snapshot()
    assert set(snap) == {"2p", "4p", "coop2"}
    assert set(snap["2p"]) == {"outcome_kind", "elo", "win_rates", "games", "wr_vs_past", "past_games",
                               "scores", "cross_play", "role_win_rates"}
    assert snap["2p"]["elo"]["a"] > 1200 > snap["2p"]["elo"]["b"]
    assert snap["4p"]["elo"] == {"a": 1200.0, "b": 1200.0} and snap["4p"]["games"]["a"]["b"] == 0
    assert snap["2p"]["wr_vs_past"] == {"a": 0.5, "b": None} and snap["2p"]["past_games"]["a"] == 1
    assert snap["coop2"]["scores"]["a"]["mean"] == 6.0 and snap["coop2"]["scores"]["b"]["n"] == 1
    assert snap["coop2"]["cross_play"] == {"a+b": {"n": 1, "mean": 6.0}}
    assert book.win_rate("2p", "a", "b") == 1.0 and book.win_rate("4p", "a", "b") == 0.5
    assert book.win_rate("2p", "b", "z") == 0.5  # an unseen pair of a known layout keeps the SP1 prior
    assert book.elo("2p", "b") < 1200
    with pytest.raises(ValueError, match=r"unknown layout 'unknown'.*\['2p', '4p', 'coop2'\]"):
        book.win_rate("unknown", "a", "b")
    with pytest.raises(ValueError, match=r"unknown layout 'unknown'"):
        book.elo("unknown", "a")


def test_rating_book_records_win_rates_by_role():
    spec = GameSpec(
        roles={"hunter": RoleSpec(OBS, ACT), "prey": RoleSpec(OBS, gymnasium.spaces.Discrete(4))},
        layouts={"1v2": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1))},
    )
    book = RatingBook(spec, ["h", "p"])
    book.update(result("1v2", [seat(0, 0, "h", role="hunter"), seat(1, 1, "p", role="prey"),
                               seat(2, 1, "p", role="prey")], {0: 1, 1: 2}))
    roles = book.snapshot()["1v2"]["role_win_rates"]
    assert roles == {"hunter": {"h": {"p": 1.0}}, "prey": {"p": {"h": 0.0}}}
    assert book.snapshot()["1v2"]["games"]["h"]["p"] == 2


def test_rating_book_rejects_an_unknown_layout_and_a_mismatched_outcome_kind():
    spec = GameSpec(roles={"player": RoleSpec(OBS, ACT)},
                    layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1))})
    book = RatingBook(spec, ["a", "b"])
    with pytest.raises(ValueError, match=r"unknown layout '3p'.*\['2p'\]"):
        book.update(result("3p", [seat(0, 0, "a"), seat(1, 1, "b"), seat(2, 2, "b")], {0: 1, 1: 2, 2: 3}))
    with pytest.raises(ValueError, match=r"outcome_kind 'rank'.*'2p'.*'wdl'"):
        book.update(result("2p", [seat(0, 0, "a"), seat(1, 1, "b")], {0: 1, 1: 2}, kind="rank"))
    assert "3p" not in book.snapshot() and book.snapshot()["2p"]["games"]["a"]["b"] == 0


def ffa_spec(*layouts) -> GameSpec:
    return GameSpec(roles={"player": RoleSpec(OBS, ACT)},
                    layouts={name: tuple(SeatSpec("player", t) for t in teams) for name, teams in layouts})


def test_rating_book_elo_values_for_team_layouts():
    book = RatingBook(ffa_spec(("2v1v1", (0, 0, 1, 2))), ["a", "b", "c", "d"])
    book.update(result("2v1v1", [seat(0, 0, "a"), seat(1, 0, "b"), seat(2, 1, "c"), seat(3, 2, "d")],
                       {0: 1, 1: 2, 2: 3}))
    assert book.snapshot()["2v1v1"]["elo"] == pytest.approx({"a": 1208.0, "b": 1208.0, "c": 1200.0, "d": 1184.0})

    same = RatingBook(ffa_spec(("2v1v1", (0, 0, 1, 2))), ["a", "b", "c"])
    same.update(result("2v1v1", [seat(0, 0, "a"), seat(1, 0, "a"), seat(2, 1, "b"), seat(3, 2, "c")],
                       {0: 1, 1: 2, 2: 3}))
    assert same.snapshot()["2v1v1"]["elo"] == pytest.approx({"a": 1216.0, "b": 1200.0, "c": 1184.0})


def test_rating_book_decisive_past_and_mixed_team_pairs():
    book = RatingBook(ffa_spec(("2p", (0, 1)), ("2v2", (0, 0, 1, 1))), ["a", "b"])
    book.update(result("2p", [seat(0, 0, "a"), seat(1, 1, "a", "ckpt_v2")], {0: 1, 1: 2}))
    assert book.snapshot()["2p"]["wr_vs_past"] == {"a": 1.0, "b": None}

    # team 0: a latest + a checkpoint; team 1: a latest + b. Team 0 wins.
    mixed = result("2v2", [seat(0, 0, "a"), seat(1, 0, "a", "ckpt_v1"), seat(2, 1, "a"), seat(3, 1, "b")],
                   {0: 1, 1: 2})
    third = pytest.approx(1 / 3)
    assert member_pairs(mixed) == [  # latest a vs latest a is skipped: 3 counted pairs share the weight
        MemberPair("cross", "a", "b", 1.0, third, "player", "player"),
        MemberPair("past", "a", "a", 0.0, third, "player", "player"),  # team 1's latest a lost to the checkpoint
        MemberPair("cross", "a", "b", 1.0, third, "player", "player"),
    ]
    book.update(mixed)
    snap = book.snapshot()["2v2"]
    assert snap["wr_vs_past"] == {"a": 0.0, "b": None} and snap["past_games"] == {"a": 1, "b": 0}
    assert snap["win_rates"]["a"]["b"] == 1.0 and snap["games"]["a"]["b"] == 2
    assert snap["elo"] == pytest.approx({"a": 1200 + 32 * (2 / 3) * 0.5, "b": 1200 - 32 * (2 / 3) * 0.5})
