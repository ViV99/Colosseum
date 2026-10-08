"""Tests for rating systems and PFSP matchmaker."""
import tempfile

from colosseum.coordinator.ratings import EloRating, WinRateTracker
from dataflow_helpers import two_seat_result
from helpers import example_config


def test_elo_win():
    elo = EloRating(k_factor=32)
    ra, rb = elo.update("a", "b")
    assert ra > 1200
    assert rb < 1200
    assert abs((ra + rb) - 2400) < 0.01  # ELO is zero-sum


def test_elo_draw():
    elo = EloRating()
    ra, rb = elo.update("a", "b", draw=True)
    assert abs(ra - 1200) < 0.01
    assert abs(rb - 1200) < 0.01


def test_elo_repeated_wins():
    elo = EloRating()
    for _ in range(10):
        elo.update("strong", "weak")
    assert elo.get("strong") > elo.get("weak")
    assert elo.get("strong") > 1300


def test_win_rate_basic():
    wrt = WinRateTracker()
    wrt.record("a", "b", 1.0)
    wrt.record("a", "b", 1.0)
    wrt.record("a", "b", 0.0)
    assert abs(wrt.get_win_rate("a", "b") - 2 / 3) < 0.01
    assert abs(wrt.get_win_rate("b", "a") - 1 / 3) < 0.01


def test_win_rate_draw():
    wrt = WinRateTracker()
    wrt.record("a", "b", 0.5)
    assert abs(wrt.get_win_rate("a", "b") - 0.5) < 0.01
    assert abs(wrt.get_win_rate("b", "a") - 0.5) < 0.01


def test_win_rate_no_matches():
    wrt = WinRateTracker()
    assert wrt.get_win_rate("x", "y") == 0.5


def test_win_rate_matrix():
    wrt = WinRateTracker()
    wrt.record("a", "b", 1.0)
    wrt.record("b", "c", 1.0)
    matrix = wrt.get_win_rate_matrix(["a", "b", "c"])
    assert matrix["a"]["b"] == 1.0
    assert matrix["b"]["a"] == 0.0
    assert matrix["b"]["c"] == 1.0


def test_coordinator_match_reporting():
    """Test coordinator properly updates ELO and win rates from match results."""
    from colosseum.coordinator.coordinator import Coordinator
    from colosseum.core.config import ColosseumConfig, load_config

    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config(example_config("tic_tac_toe.yaml"))
        cd = config.model_dump()
        cd["checkpoint"]["dir"] = tmpdir
        cd["training"]["phase"] = "league"
        config = ColosseumConfig(**cd)

        coord = Coordinator(config)
        coord.agent_pool.register_trainable("a0")
        coord.agent_pool.register_trainable("a1")

        # a0 wins
        coord.report_match_result(two_seat_result("a0", 1.0, "a1", 0.0, match_id="m1"))

        assert coord.elo.get("a0") > coord.elo.get("a1")
        assert coord.win_rates.get_win_rate("a0", "a1") == 1.0

        summary = coord.get_ratings_summary()
        assert "elo" in summary
        assert "win_rates" in summary
