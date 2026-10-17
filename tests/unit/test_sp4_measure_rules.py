"""Decision rules of the SP4 measurements (plan T8.2), fixed before any number is seen: tournament default
sizes, BT prior / smoothness by cross-validation, pulse cadence and the pipeline trend threshold."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from arena_kit import spearman
from colosseum.arena import bt
from colosseum.arena.comparisons import ComparisonTable, PairStat, TeamKey
from colosseum.core.types import MatchResult, SeatResult, TeamResult

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


defaults = _script("measure_tournament_defaults")
crossval = _script("bt_crossval")
cadence = _script("measure_pulse_cadence")


def _rows(cell, game, n, fail=()):
    return [{"game": game, "blocks": cell[0], "confirm": cell[1], "seed": s, "seconds": 10.0,
             "checks": {c: not (s == 0 and c in fail) for c in defaults.CHECKS}} for s in range(n)]


def test_tournament_defaults_take_the_first_passing_cell_and_the_floors():
    rows = (_rows((8, 32), "ttt", 10, fail=("winner",)) + _rows((8, 32), "tron", 10)
            + _rows((16, 64), "ttt", 10, fail=("confirmed",)) + _rows((16, 64), "tron", 10, fail=("order",)))
    decision = defaults.decide(rows)
    assert decision["chosen"] == [16, 64]
    assert decision["defaults"] == {"halving_blocks": 16, "confirm_blocks": 100, "blocks": 50}
    small = defaults.decide(_rows((8, 32), "ttt", 3) + _rows((8, 32), "tron", 3))
    assert small["defaults"] == {"halving_blocks": 8, "confirm_blocks": 100, "blocks": 50}
    assert defaults.decide(_rows((8, 32), "ttt", 10, fail=("winner",)))["chosen"] is None
    big = defaults.decide(_rows((32, 128), "ttt", 5) + _rows((32, 128), "tron", 5))
    assert big["defaults"] == {"halving_blocks": 32, "confirm_blocks": 128, "blocks": 64}


def test_problem_strings_are_classified():
    checks = defaults.classify(["ranking ['e025', 'e000'] != true order", "not confirmed; indistinguishable"])
    assert checks == {"winner": True, "order": False, "confirmed": False, "bt": True}


def test_crossval_prefers_most_smoothing_then_weakest_prior_within_one_se():
    rows = []
    for prior in crossval.PRIOR_GRID:
        for smooth in crossval.SMOOTH_GRID:
            loss = 0.60 + (0.05 if (prior, smooth) == (0.5, 64.0) else 0.0) + (0.0 if smooth <= 16 else 0.02)
            rows.append({"game": "g", "prior": prior, "smooth": smooth, "weight": [100.0] * crossval.FOLDS,
                         "loss": [100.0 * (loss + 0.001 * k) for k in range(crossval.FOLDS)]})
    decision = crossval.choose(rows)
    assert (decision["prior"], decision["smooth"]) == (8.0, 16.0)


def test_pulse_cadence_and_trend_rules():
    rows = [{"every": 1, "pulse_skipped": 3, "evaluated": 9}, {"every": 2, "pulse_skipped": 1, "evaluated": 6},
            {"every": 5, "pulse_skipped": 0, "evaluated": 2}]
    assert cadence.decide_cadence(rows)["every"] == 2
    assert cadence.decide_cadence(rows[2:])["every"] is None
    trend = cadence.decide_trend([{"spearman": 0.83, "gain": 50.0, "vs_random_win": 1.0, "confirmed": False},
                                  {"spearman": 0.71, "gain": 40.0, "vs_random_win": 1.0, "confirmed": True}])
    assert trend["rho_min"] == pytest.approx(0.6)
    assert cadence.decide_trend([{"spearman": 0.2, "gain": 1.0, "vs_random_win": 1.0, "confirmed": True}])["rho_min"] \
        == pytest.approx(0.3)



def test_pulse_rules_fail_with_a_reason_on_missing_data():
    missing = cadence.decide_cadence([{"every": 1, "pulse_skipped": None, "evaluated": 9},
                                      {"every": 2, "pulse_skipped": 0, "evaluated": 3}])
    assert missing["every"] is None
    assert "counter" in missing["reasons"][1] and "evaluated" in missing["reasons"][2]
    assert cadence.decide_cadence([{"every": 1, "pulse_skipped": None, "evaluated": 9},
                                   {"every": 2, "pulse_skipped": 1, "evaluated": 6}])["every"] == 2
    rows = [{"seed": 0, "spearman": 0.8, "gain": 10.0, "vs_random_win": 1.0, "confirmed": True},
            {"seed": 1, "spearman": float("nan"), "gain": None, "vs_random_win": 1.0, "confirmed": True},
            {"seed": 2, "spearman": None, "gain": None, "vs_random_win": 1.0, "confirmed": False}]
    trend = cadence.decide_trend(rows)
    assert trend["rho_min"] is None and len(trend["reasons"]) == 2 and "seed 1" in trend["reasons"][0]
    assert cadence.decide_trend([])["rho_min"] is None
    assert cadence.decide_trend([{"spearman": 0.75, "gain": 1.0, "vs_random_win": 1.0, "confirmed": True}])[
        "rho_min"] == pytest.approx(0.65)

def test_spearman_handles_ties_and_degenerate_inputs():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3], [1, 1, 2]) == pytest.approx(0.8660254, abs=1e-6)
    assert math.isnan(spearman([1, 2], [1, 2])) and math.isnan(spearman([1, 2, 3], [5, 5, 5]))


def _two_cell_table(seed: int = 0):
    """Hunters (1 seat) against prey (2 seats) with known logit strengths: two rating cells (two roles, two
    lineages), 200 games per pair; ``(table, [(a, b, empirical win rate of a)])``."""
    rng = np.random.default_rng(seed)
    true = {f"hunter@v{100 * v}": 1.0 + 0.3 * v for v in range(4)} | {f"prey@v{100 * v}": -0.5 + 0.2 * v
                                                                      for v in range(4)}
    table, stats = ComparisonTable(), []
    for h in (k for k in true if k.startswith("hunter")):
        for p in (k for k in true if k.startswith("prey")):
            n, prob = 200, 1.0 / (1.0 + math.exp(-(true[h] - true[p])))
            wins = int(rng.binomial(n, prob))
            a, b = TeamKey(0, ((h, "hunter"),)), TeamKey(1, ((p, "prey"), (p, "prey")))
            table.add("1v2", "train", a, b, PairStat(n=n, sw=float(n), swy=float(wins)))
            stats.append((a, b, wins / n))
    return table, stats


def _fit(table, prior: float):
    return bt.fit(table, "1v2", sources=("train",),
                  options=bt.BTOptions(prior_sigma=prior, smoothness=1.0, agent_order=("hunter", "prey")))


@pytest.mark.parametrize("prior", [0.5, 2.0, 8.0])
def test_crossval_predict_is_the_models_own_prediction_across_rating_cells(prior):
    table, _ = _two_cell_table()
    fit_ = _fit(table, prior)
    assert len(fit_.groups) >= 2           # the pairs cross rating cells (the case the anchored ratings got wrong)
    design = fit_.design
    eta = np.zeros(len(design.keys))       # the model's linear predictor of every design row: X @ params
    np.add.at(eta, design.nz_row, design.nz_val * fit_.params[design.nz_col])
    for (a, b), z in zip(design.keys, eta):
        assert crossval.predict(fit_, a, b) == pytest.approx(1.0 / (1.0 + math.exp(-z)), abs=1e-12)
    unknown = TeamKey(0, (("hunter@v900", "hunter"),))
    assert crossval.predict(fit_, unknown, design.keys[0][1]) is None


@pytest.mark.parametrize("prior", [0.5, 2.0, 8.0])
def test_crossval_predict_has_no_mean_error_at_the_fit(prior):
    """At the MAP the bias column's score equation makes the residuals sum to the prior's pull on the bias,
    which is tiny with 200 games per pair (measured |mean error| < 5e-4 for these priors; the band is 10x)."""
    table, stats = _two_cell_table()
    fit_ = _fit(table, prior)
    errors = [crossval.predict(fit_, a, b) - emp for a, b, emp in stats]
    assert abs(float(np.mean(errors))) < 0.005


def _match(uid: str, a_team: int, a_wins: bool, *, opponent: str = "b", block: str | None = None):
    agents = ["a", opponent] if a_team == 0 else [opponent, "a"]
    seats = [SeatResult(seat=s, role="player", team=s, agent_id=agents[s], network_id="x", reward=0.0)
             for s in range(2)]
    winner = a_team if a_wins else 1 - a_team
    teams = [TeamResult(team=t, rank=1.0 if t == winner else 2.0, score=0.0) for t in range(2)]
    result = MatchResult(match_id=uid, layout="2p", outcome_kind="wdl", seats=seats, teams=teams,
                         episode_length=5, uid=uid, block=block)
    return result, [100 if agents[s] == "a" else None for s in range(2)]


def test_crossval_evaluate_scores_held_out_pairs_and_leaves_self_play_out():
    """``a@v100`` beats the fixed ``b`` with p = 0.75 (seats alternate): the held-out log-loss estimates the
    Bernoulli entropy H(0.75) = 0.5623 with SE ~0.021 over 500 matches; band 0.1 (> 4.5 SE, fixed seed)."""
    rng = np.random.default_rng(1)
    items = []
    for m in range(500):
        result, versions = _match(f"run/train/m{m}", m % 2, bool(rng.random() < 0.75))
        items.append((crossval.fold_of(crossval.fold_key(result)), result, versions))
    for m in range(40):                                   # self-play: a@v100 on both sides
        result, versions = _match(f"run/train/s{m}", m % 2, bool(m % 3), opponent="a")
        items.append((crossval.fold_of(crossval.fold_key(result)), result, versions))
    inputs = (bt.BTOptions(), ["b"], None)
    tables = crossval.fold_tables(items)
    res = crossval.evaluate(inputs, items, 4.0, 1.0, tables)
    assert res["self_play"] == 40 and res["skipped"] == 0
    assert sum(res["weight"]) == pytest.approx(500.0)
    assert sum(res["loss"]) / sum(res["weight"]) == pytest.approx(0.5623, abs=0.1)
    assert crossval.evaluate(inputs, items, 4.0, 1.0) == res          # tables built inside: the same numbers


def test_crossval_folds_keep_a_block_together():
    rotations = [_match(f"t/{k}", k % 2, True, block="P/main@v100/ref/3")[0] for k in range(6)]
    assert {crossval.fold_of(crossval.fold_key(r)) for r in rotations} == {crossval.fold_of("P/main@v100/ref/3")}
    loose = _match("run/train/m7", 0, True)[0]
    assert crossval.fold_key(loose) == "run/train/m7"


def test_crossval_decision_lists_the_points_inside_the_one_se_window():
    rows = [{"game": "g", "prior": p, "smooth": s, "weight": [100.0] * crossval.FOLDS,
             "loss": [100.0 * (0.6 + (0.0 if (p, s) in ((1.0, 4.0), (2.0, 4.0)) else 0.1) + 0.001 * k)
                      for k in range(crossval.FOLDS)]}
            for p in crossval.PRIOR_GRID for s in crossval.SMOOTH_GRID]
    decision = crossval.choose(rows)
    assert [(e["prior"], e["smooth"]) for e in decision["eligible"]] == [(1.0, 4.0), (2.0, 4.0)]
    assert (decision["prior"], decision["smooth"]) == (2.0, 4.0)
