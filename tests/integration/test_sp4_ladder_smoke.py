"""Fast tournament smoke on the tic-tac-toe ladder (SP4 section 6, "tournament ... with a short ladder";
T8.2): ``--mode select`` with tiny sizes finishes and its report has every key the slow ladder test and the
report page read. No thresholds. ``report.html`` is required once T7.2's renderer exists (until then the
tournament writes only ``report.json`` / ``report.txt``, contract amendment A27)."""
from __future__ import annotations

import importlib.util

import arena_kit as kit


def test_select_on_the_ladder_reports_what_the_checks_read(tmp_path):
    run = kit.run_select(kit.TIC_TAC_TOE, tmp_path, seed=1, blocks=2, confirm_blocks=2, num_envs=4)
    assert set(run.ranking) == set(kit.TRUE_ORDER)
    assert run.winner in kit.TRUE_ORDER and isinstance(run.confirmed, bool)
    for name in kit.TRUE_ORDER:
        row = run.rows[name]
        assert {"score", "score_ci", "bt", "bt_ci", "games"} <= set(row)
    assert isinstance(kit.ladder_problems(run), list)
    assert list(run.out.rglob("report.txt"))
    if importlib.util.find_spec("colosseum.arena.report.render") is not None:     # T7.2 landed
        assert list(run.out.rglob("report.html"))
