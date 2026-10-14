"""The SP3 rule for the APPO Units defaults (spec block 10, T6.5): scripts/units_experiment.py's
decide_defaults on synthetic rows. No processes, no training."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "units_experiment.py"
_spec = importlib.util.spec_from_file_location("units_experiment_sp3", _SCRIPT)
ue = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ue)


def _rows(cells: dict[tuple[int, str, str], list[float]]) -> list[dict]:
    return [{"k": k, "ratio_mode": r, "unit_trace": t, "seed": s, "share_vs_scripted": v}
            for (k, r, t), values in cells.items() for s, v in enumerate(values)]


BASE8 = {(8, "per_unit", "auto"): [0.50, 0.50, 0.50], (8, "joint", "auto"): [0.50, 0.50, 0.50],
         (8, "per_unit", "none"): [0.50, 0.50, 0.50]}


def test_a_clear_win_at_k128_that_holds_at_k8_switches():
    rows = _rows({**BASE8, (128, "per_unit", "auto"): [0.20, 0.18, 0.22], (128, "joint", "auto"): [0.30, 0.29, 0.31],
                  (128, "per_unit", "none"): [0.21, 0.19, 0.20]})
    decisions = ue.decide_defaults(rows)
    assert decisions["ratio_mode"]["switch"] is True and decisions["ratio_mode"]["choice"] == "joint"
    assert decisions["unit_trace"]["switch"] is False and decisions["unit_trace"]["choice"] == "joint"
    assert decisions["ratio_mode"]["diff_k128"] == pytest.approx(0.10)


def test_a_win_within_two_standard_errors_does_not_switch():
    rows = _rows({**BASE8, (128, "per_unit", "auto"): [0.10, 0.30, 0.20], (128, "joint", "auto"): [0.15, 0.35, 0.25],
                  (128, "per_unit", "none"): [0.20, 0.20, 0.20]})
    assert ue.decide_defaults(rows)["ratio_mode"]["switch"] is False


def test_losing_more_than_003_at_k8_blocks_the_switch():
    rows = _rows({(8, "per_unit", "auto"): [0.50, 0.50, 0.50], (8, "joint", "auto"): [0.46, 0.46, 0.46],
                  (8, "per_unit", "none"): [0.48, 0.48, 0.48],
                  (128, "per_unit", "auto"): [0.20, 0.18, 0.22], (128, "joint", "auto"): [0.30, 0.29, 0.31],
                  (128, "per_unit", "none"): [0.30, 0.29, 0.31]})
    decisions = ue.decide_defaults(rows)
    assert decisions["ratio_mode"]["switch"] is False                     # 0.04 worse at K = 8
    assert decisions["unit_trace"]["switch"] is True and decisions["unit_trace"]["choice"] == "none"   # 0.02 worse


def test_missing_cells_give_no_decision():
    rows = _rows({(128, "per_unit", "auto"): [0.2, 0.2, 0.2], (128, "joint", "auto"): [0.3]})
    decisions = ue.decide_defaults(rows)
    assert decisions["ratio_mode"]["switch"] is None and "K=8" in decisions["ratio_mode"]["note"]
    assert decisions["unit_trace"]["switch"] is None


def test_cells_parse_ratio_and_trace():
    assert ue.parse_cells(["joint:auto", "per_unit:none"]) == [("joint", "auto"), ("per_unit", "none")]
    with pytest.raises(ValueError, match="ratio_mode:unit_trace"):
        ue.parse_cells(["joint"])
