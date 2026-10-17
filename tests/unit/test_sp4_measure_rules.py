"""Decision rules of the SP4 measurements (plan T8.2), fixed before any number is seen: tournament default
sizes, BT prior / smoothness by cross-validation, pulse cadence and the pipeline trend threshold."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from arena_kit import spearman

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


def test_spearman_handles_ties_and_degenerate_inputs():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3], [1, 1, 2]) == pytest.approx(0.8660254, abs=1e-6)
    assert math.isnan(spearman([1, 2], [1, 2])) and math.isnan(spearman([1, 2, 3], [5, 5, 5]))
