"""Eval statistics, reversed rows, solo mode and the pinned JSON schema (T7.2)."""
import json

import pytest

from colosseum.eval import MatchRecord, evaluate, normal_interval, summarize, wilson_interval
from helpers import AlternatingEnv, MoveCounterModel

TOP_KEYS = {"mode", "agents", "num_players", "num_matches_per_pair", "deterministic",
            "ci_level", "score_ci_method", "pairs", "solo"}
PAIR_ROW_KEYS = {"agent_a", "agent_b", "n", "wins", "draws", "losses", "win_rate", "win_rate_ci",
                 "score", "score_ci", "mean_return_a", "mean_return_b", "mean_episode_length", "per_seat"}
PAIR_SEAT_KEYS = {"n", "wins", "draws", "losses", "score"}
SOLO_ROW_KEYS = {"agent", "n", "mean_return", "return_ci", "mean_outcome", "outcome_ci",
                 "mean_episode_length", "per_seat"}
SOLO_SEAT_KEYS = {"n", "mean_return", "mean_outcome"}


def _rec(lineup, outcomes, returns=None, length=5):
    returns = returns if returns is not None else [0.0] * len(lineup)
    return MatchRecord(tuple(lineup), tuple(outcomes), tuple(returns), length)


def _pair_records(results):
    """'W'/'D'/'L' from agent a's side; lineups alternate (a, b), (b, a)."""
    out = []
    for i, r in enumerate(results):
        lineup = ("a", "b") if i % 2 == 0 else ("b", "a")
        score_a = {"W": 1.0, "D": 0.5, "L": 0.0}[r]
        outcomes = (score_a, 1.0 - score_a) if lineup[0] == "a" else (1.0 - score_a, score_a)
        out.append(_rec(lineup, outcomes))
    return out


def test_wilson_interval_contains_point_and_handles_edges():
    for successes, n in [(0, 10), (10, 10), (5, 10), (45.0, 100), (0.5, 1), (0, 1)]:
        lo, hi = wilson_interval(successes, n)
        assert 0.0 <= lo <= successes / n <= hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60


def test_normal_interval():
    mean, lo, hi = normal_interval([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5
    assert lo == pytest.approx(2.5 - 1.959963984540054 * 1.2909944487358056 / 2)
    assert hi == pytest.approx(2.5 + 1.959963984540054 * 1.2909944487358056 / 2)
    assert normal_interval([3.0]) == (3.0, 3.0, 3.0)
    assert normal_interval([]) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("w,d,l", [(10, 80, 10), (30, 10, 60), (0, 100, 0), (7, 0, 0)])
def test_pair_rows_are_consistent_and_contain_their_estimates(w, d, l):  # noqa: E741
    results = ["W"] * w + ["D"] * d + ["L"] * l
    report = summarize(_pair_records(results), ["a", "b"], num_players=2, num_matches=len(results))
    ab, ba = report.rows()
    n = len(results)
    assert (ab["agent_a"], ab["agent_b"], ba["agent_a"], ba["agent_b"]) == ("a", "b", "b", "a")
    assert (ab["n"], ab["wins"], ab["draws"], ab["losses"]) == (n, w, d, l)
    assert (ba["n"], ba["wins"], ba["draws"], ba["losses"]) == (n, l, d, w)
    for row in (ab, ba):
        assert row["win_rate_ci"][0] <= row["win_rate"] <= row["win_rate_ci"][1]
        assert row["score_ci"][0] <= row["score"] <= row["score_ci"][1]
    assert ab["win_rate"] == pytest.approx(w / n)
    assert ab["score"] == pytest.approx((w + d / 2) / n)
    assert ba["score"] == pytest.approx(1.0 - ab["score"])
    assert ba["score_ci"] == pytest.approx([1.0 - ab["score_ci"][1], 1.0 - ab["score_ci"][0]])
    # per-seat: a in seat 0 is b in seat 1 of the reversed row, with W and L swapped
    assert sum(cell["n"] for cell in ab["per_seat"].values()) == n
    for seat_a, seat_b in (("0", "1"), ("1", "0")):
        if seat_a in ab["per_seat"]:
            assert ab["per_seat"][seat_a]["wins"] == ba["per_seat"][seat_b]["losses"]
            assert ab["per_seat"][seat_a]["draws"] == ba["per_seat"][seat_b]["draws"]


def test_three_player_pair_uses_seat_groups():
    records = [_rec(("a", "b", "a"), (1.0, 0.0, 1.0)), _rec(("b", "a", "b"), (0.0, 1.0, 0.0))]
    ab, ba = summarize(records, ["a", "b"], num_players=3, num_matches=2).rows()
    assert set(ab["per_seat"]) == {"0,2", "1"}
    assert set(ba["per_seat"]) == {"0,2", "1"}
    assert ab["wins"] == 2 and ba["losses"] == 2
    assert ba["per_seat"]["1"] == {"n": 1, "wins": 0, "draws": 0, "losses": 1, "score": 0.0}


def test_solo_report_has_normal_intervals():
    records = [_rec(("a",), (0.5,), (r,), 10) for r in [1.0, 2.0, 3.0, 4.0]]
    report = summarize(records, ["a"], num_players=1, num_matches=4)
    assert report.mode == "solo"
    row = report.to_dict()["solo"][0]
    assert set(row) == SOLO_ROW_KEYS
    assert row["n"] == 4 and row["mean_return"] == 2.5
    assert row["return_ci"][0] < 2.5 < row["return_ci"][1]
    assert row["mean_episode_length"] == 10.0


def test_evaluate_end_to_end_per_seat_breakdown():
    # AlternatingEnv: seat 0 always wins -> each agent wins exactly its seat-0 matches.
    report = evaluate({"a": MoveCounterModel(), "b": MoveCounterModel()}, AlternatingEnv, num_matches=4, num_envs=2)
    ab, ba = report.rows()
    assert (ab["n"], ab["wins"], ab["draws"], ab["losses"]) == (4, 2, 0, 2)
    assert ab["per_seat"]["0"] == {"n": 2, "wins": 2, "draws": 0, "losses": 0, "score": 1.0}
    assert ab["per_seat"]["1"] == {"n": 2, "wins": 0, "draws": 0, "losses": 2, "score": 0.0}
    assert ba["per_seat"]["0"]["wins"] == 2
    text = report.summary()
    assert "win_rate" in text and "score CI" in text


def test_json_schema_is_pinned(tmp_path):
    report = evaluate({n: MoveCounterModel() for n in "abc"}, AlternatingEnv, num_matches=2, num_envs=4)
    path = tmp_path / "result.json"
    report.write_json(path)
    data = json.loads(path.read_text())
    assert set(data) == TOP_KEYS
    assert data["mode"] == "pairwise"
    assert data["agents"] == ["a", "b", "c"]
    assert data["num_players"] == 2
    assert data["num_matches_per_pair"] == 2
    assert data["ci_level"] == 0.95
    assert data["solo"] == []
    assert [(r["agent_a"], r["agent_b"]) for r in data["pairs"]] == [
        ("a", "b"), ("b", "a"), ("a", "c"), ("c", "a"), ("b", "c"), ("c", "b"),
    ]
    for row in data["pairs"]:
        assert set(row) == PAIR_ROW_KEYS
        assert len(row["win_rate_ci"]) == 2 and len(row["score_ci"]) == 2
        assert row["n"] == 2
        for cell in row["per_seat"].values():
            assert set(cell) == PAIR_SEAT_KEYS

    solo = evaluate({"a": MoveCounterModel()}, AlternatingEnv, num_matches=2, num_envs=2).to_dict()
    assert set(solo) == TOP_KEYS
    assert solo["mode"] == "solo" and solo["pairs"] == []
    assert set(solo["solo"][0]) == SOLO_ROW_KEYS
    assert set(solo["solo"][0]["per_seat"]) == {"0", "1"}
    for cell in solo["solo"][0]["per_seat"].values():
        assert set(cell) == SOLO_SEAT_KEYS
