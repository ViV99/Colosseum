"""Pairwise ratings from per-seat match results (T5.2)."""
from __future__ import annotations

import json

import pytest

from colosseum.coordinator.coordinator import Coordinator
from colosseum.coordinator.ratings import EloRating, PastWinRate, WinRateTracker, pairwise_score
from colosseum.core.config import ColosseumConfig
from colosseum.core.types import MatchResult, SeatResult

TTT = "examples.tic_tac_toe"


def make_coordinator(tmp_path, agents, num_players=2) -> Coordinator:
    cfg = ColosseumConfig.model_validate({
        "env": {"env_class": f"{TTT}.env.TicTacToeEnv", "num_players": num_players},
        "networks": {
            "encoder_class": f"{TTT}.networks.TicTacToeEncoder",
            "policy_class": f"{TTT}.networks.TicTacToePolicy",
            "value_class": f"{TTT}.networks.TicTacToeValue",
        },
        "training": {"phase": "league"},
        "agents": {a: {} for a in agents},
    })
    return Coordinator(cfg, checkpoint_dir=tmp_path / "ckpt")


def seat(i, agent, outcome, network="latest", reward=0.0):
    return SeatResult(seat=i, agent_id=agent, network_id=network, outcome=outcome, reward=reward)


def result(*seats, length=5):
    return MatchResult(match_id="m", seats=list(seats), episode_length=length)


def test_pairwise_score():
    assert pairwise_score(1.0, 0.0) == 1.0
    assert pairwise_score(0.0, 1.0) == 0.0
    assert pairwise_score(0.5, 0.5) == 0.5


def test_win_rate_tracker_stores_fractional_sums():
    wr = WinRateTracker()
    wr.record_pair("a", "b", 0.5)
    wr.record_pair("a", "b", 1.0)
    wr.record_pair("a", "b", 0.5)
    assert wr.get_win_rate("a", "b") == pytest.approx(2.0 / 3.0)
    assert wr.get_win_rate("b", "a") == pytest.approx(1.0 / 3.0)
    assert wr.games("a", "b") == 3
    assert wr.get_win_rate("a", "c") == 0.5  # never met
    with pytest.raises(ValueError):
        wr.record_pair("a", "b", 1.5)


def test_elo_update_pairs_is_order_independent():
    pairs = [("x", "y", 1.0), ("x", "z", 1.0), ("y", "z", 0.5)]
    e1, e2 = EloRating(), EloRating()
    e1.update_pairs(pairs, k_scale=0.5)
    e2.update_pairs(list(reversed(pairs)), k_scale=0.5)
    assert e1.all_ratings == pytest.approx(e2.all_ratings)
    assert e1.get("y") == pytest.approx(e1.get("z"))


def test_elo_update_pair_two_player_win_moves_16_points():
    elo = EloRating(k_factor=32.0)
    elo.update_pair("a", "b", 1.0)
    assert elo.get("a") == pytest.approx(1216.0)
    assert elo.get("b") == pytest.approx(1184.0)


def test_ffa_ranking_gives_correct_pairwise_results(tmp_path):
    coord = make_coordinator(tmp_path, ["x", "y", "z"], num_players=3)
    for _ in range(10):
        coord.report_match_result(result(seat(0, "x", 1.0), seat(1, "y", 0.5), seat(2, "z", 0.0)))
    wr = coord.win_rates
    assert wr.get_win_rate("x", "y") == 1.0
    assert wr.get_win_rate("x", "z") == 1.0
    assert wr.get_win_rate("y", "z") == 1.0
    assert wr.get_win_rate("z", "x") == 0.0
    assert coord.elo.get("x") > coord.elo.get("y") > coord.elo.get("z")


def test_eight_player_ffa_winner_moves_like_a_two_player_win(tmp_path):
    agents = [f"p{i}" for i in range(8)]
    coord = make_coordinator(tmp_path, agents, num_players=8)
    coord.report_match_result(result(*[seat(i, a, 1.0 if i == 0 else 0.0) for i, a in enumerate(agents)]))
    assert coord.elo.get("p0") == pytest.approx(1216.0)  # K/(N-1) scaling (R4-08)


@pytest.mark.parametrize("order", [[0, 1, 2], [2, 1, 0], [1, 2, 0]])
def test_ties_are_order_independent(tmp_path, order):
    coord = make_coordinator(tmp_path, ["w", "y", "z"], num_players=3)
    seats = [seat(0, "w", 1.0), seat(1, "y", 0.0), seat(2, "z", 0.0)]
    for _ in range(5):
        coord.report_match_result(result(*[seats[i] for i in order]))
    assert coord.win_rates.get_win_rate("y", "z") == 0.5
    assert coord.win_rates.get_win_rate("z", "y") == 0.5
    assert coord.elo.get("y") == pytest.approx(coord.elo.get("z"))
    reference = make_coordinator(tmp_path / "ref", ["w", "y", "z"], num_players=3)
    for _ in range(5):
        reference.report_match_result(result(*seats))
    assert coord.elo.all_ratings == pytest.approx(reference.elo.all_ratings)


def test_agent_holding_two_seats_counts_every_cross_pair(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b"], num_players=4)
    coord.report_match_result(result(
        seat(0, "a", 1.0), seat(1, "b", 0.5), seat(2, "a", 0.0), seat(3, "b", 0.5)))
    # cross pairs: a0>b1, a0>b3, a2<b1, a2<b3 -> 2 wins out of 4
    assert coord.win_rates.games("a", "b") == 4
    assert coord.win_rates.get_win_rate("a", "b") == 0.5
    assert coord.elo.get("a") == pytest.approx(coord.elo.get("b"))


def test_same_agent_pairs_do_not_touch_elo_but_update_wr_vs_past(tmp_path):
    coord = make_coordinator(tmp_path, ["a"])
    coord.report_match_result(result(seat(0, "a", 0.0, network="ckpt_v10"), seat(1, "a", 1.0)))
    coord.report_match_result(result(seat(0, "a", 0.5), seat(1, "a", 0.5, network="ckpt_v10")))
    coord.report_match_result(result(seat(0, "a", 1.0), seat(1, "a", 0.0)))  # latest vs latest: no signal
    assert coord.elo.all_ratings == {}
    assert coord.win_rates.games("a", "a") == 0
    assert coord.past_win_rate.get("a") == pytest.approx(0.75)
    assert coord.past_win_rate.games("a") == 2


def test_past_win_rate_window():
    past = PastWinRate(window=4)
    for s in [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]:
        past.record("a", s)
    assert past.get("a") == 1.0
    assert past.get("b") is None


def test_ratings_snapshot_is_json_serializable(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b"])
    coord.report_match_result(result(seat(0, "b", 0.0), seat(1, "a", 1.0)))
    snap = coord.ratings_snapshot()
    assert set(snap) == {"elo", "win_rates", "games", "wr_vs_past", "past_games"}
    assert snap["win_rates"]["a"]["b"] == 1.0
    assert snap["games"]["b"]["a"] == 1
    assert snap["wr_vs_past"] == {"a": None, "b": None}
    json.dumps(snap)
