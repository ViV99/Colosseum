"""The ``unit_trace`` decision rule of scripts/units_experiment.py (T8.4): no processes, no training."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "units_experiment.py"
_spec = importlib.util.spec_from_file_location("units_experiment", _SCRIPT)
ue = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ue)


def _rows(shares: dict[tuple[int, str], float], ratio_mode: str = "per_unit") -> list[dict]:
    return [{"k": k, "ratio_mode": ratio_mode, "unit_trace": t, "share_vs_scripted": s} for (k, t), s in shares.items()]


def _grid(k128: tuple[float, float, float], k8: tuple[float, float, float]) -> list[dict]:
    traces = ("joint", "geo_mean", "none")
    return _rows({**{(128, t): s for t, s in zip(traces, k128)}, **{(8, t): s for t, s in zip(traces, k8)}})


def test_keeps_geo_mean_without_a_clear_win():
    choice, _, note = ue.recommend(_grid((0.44, 0.41, 0.43), (0.55, 0.55, 0.55)))
    assert (choice, note) == ("geo_mean", None)


def test_picks_the_best_trace_that_wins_at_k128_and_holds_at_k8():
    choice, detail, note = ue.recommend(_grid((0.236, 0.044, 0.160), (0.5, 0.5, 0.5)))
    assert (choice, note) == ("joint", None)
    assert "joint: K=128 0.236, K=8 0.500" in detail


def test_a_trace_that_loses_more_than_the_margin_at_k8_is_not_chosen():
    choice, _, _ = ue.recommend(_grid((0.30, 0.20, 0.10), (0.40, 0.50, 0.50)))
    assert choice == "geo_mean"


def test_ratio_mode_joint_rows_do_not_count():
    rows = _grid((0.20, 0.20, 0.20), (0.5, 0.5, 0.5)) + _rows({(128, "none"): 0.9, (8, "none"): 0.5}, "joint")
    assert ue.recommend(rows)[0] == "geo_mean"


def test_missing_shares_give_no_recommendation_instead_of_the_default():
    """A grid without K=8 rows (e.g. the seed-1 recheck) cannot be decided: None plus a note."""
    rows = _rows({(128, "joint"): 0.176, (128, "geo_mean"): 0.143, (128, "none"): 0.219})
    choice, detail, note = ue.recommend(rows)
    assert choice is None
    assert "K=8" in note and "joint" in note
    assert "none: K=128 0.219, K=8 nan" in detail
