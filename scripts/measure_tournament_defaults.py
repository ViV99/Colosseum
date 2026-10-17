"""Tournament default sizes on the bot ladders (SP4 criterion 3.3; plan T8.2).

For every grid point (first halving round ``--halving-blocks``, ``--confirm-blocks``) and seed, ``tournament --mode
select`` runs on both ladders (``tests/arena_kit.py``: tic_tac_toe 2p, tron 3p; 2 worker processes). Each run
is checked by ``arena_kit.ladder_problems``.

Decision rule (fixed before the measurement, ``decide``): the first grid point (smallest first) at which, on
both games, the winner is right in every seed and the order, ``confirmed`` and the BT-neighbour check each
hold in all seeds but at most one. Defaults: ``DEFAULT_HALVING_BLOCKS = max(8, blocks)``,
``DEFAULT_CONFIRM_BLOCKS = max(100, confirm)``, ``DEFAULT_BLOCKS`` (gauntlet, round-robin) ``= max(50, 2 *
blocks)``; the floors are T5.3's provisional values. They keep real selections (differences far smaller
than the ladder's) from defaulting to tiny tournaments. No passing point: report to the controller (no
defaults are set).

Usage::

    env -u PYTHONPATH .venv/bin/python scripts/measure_tournament_defaults.py --seeds 0 1 2 3 4 5 6 7 8 9 \
        --work-dir /tmp/sp4-ladder --json docs/benchmarks/sp4-tournament-defaults.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

GRID = ((8, 32), (16, 64), (32, 128))      # (first halving round blocks, confirm blocks), smallest first
FLOORS = {"halving_blocks": 8, "confirm_blocks": 100, "blocks": 50}   # Part 2's provisional defaults
CHECKS = ("winner", "order", "confirmed", "bt")


def classify(problems: list[str]) -> dict[str, bool]:
    """Which checks of ``arena_kit.ladder_problems`` passed."""
    return {"winner": not any(p.startswith("winner") for p in problems),
            "order": not any(p.startswith("ranking") for p in problems),
            "confirmed": not any(p.startswith("not confirmed") for p in problems),
            "bt": not any(p.startswith("BT") for p in problems)}


def decide(rows: list[dict], grid=GRID, floors=FLOORS) -> dict:
    """``rows``: ``{"game", "blocks", "confirm", "seed", "checks": {check: bool}, "seconds"}``."""
    games = sorted({r["game"] for r in rows})
    for blocks, confirm in grid:
        cell = [r for r in rows if (r["blocks"], r["confirm"]) == (blocks, confirm)]
        ok, summary = True, {}
        for game in games:
            runs = [r for r in cell if r["game"] == game]
            passed = {c: sum(r["checks"][c] for r in runs) for c in CHECKS}
            n = len(runs)
            summary[game] = {"runs": n, **passed, "mean_seconds": sum(r["seconds"] for r in runs) / max(n, 1)}
            ok &= n > 0 and passed["winner"] == n and all(passed[c] >= n - 1 for c in ("order", "confirmed", "bt"))
        if ok:
            return {"chosen": [blocks, confirm], "summary": summary,
                    "defaults": {"halving_blocks": max(floors["halving_blocks"], blocks),
                                 "confirm_blocks": max(floors["confirm_blocks"], confirm),
                                 "blocks": max(floors["blocks"], 2 * blocks)}}
    return {"chosen": None, "summary": None, "defaults": None}


def main() -> int:
    import arena_kit as kit

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/sp4-ladder"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    rows = []
    for blocks, confirm in GRID:
        for game in kit.LADDER_GAMES:
            for seed in args.seeds:
                work = args.work_dir / f"{game.name}-b{blocks}-c{confirm}-s{seed}"
                work.mkdir(parents=True, exist_ok=True)
                run = kit.run_select(game, work, seed=seed, blocks=blocks, confirm_blocks=confirm)
                problems = kit.ladder_problems(run)
                rows.append({"game": game.name, "blocks": blocks, "confirm": confirm, "seed": seed,
                             "checks": classify(problems), "problems": problems, "seconds": round(run.seconds, 1),
                             "ranking": run.ranking, "winner": run.winner, "confirmed": run.confirmed})
                print(json.dumps(rows[-1]), flush=True)
    decision = decide(rows)
    per_cell = defaultdict(dict)
    for r in rows:
        key = f"b{r['blocks']}-c{r['confirm']}"
        cell = per_cell[key].setdefault(r["game"], {c: 0 for c in CHECKS} | {"runs": 0})
        cell["runs"] += 1
        for c in CHECKS:
            cell[c] += r["checks"][c]
    print(json.dumps(per_cell, indent=1))
    print(json.dumps(decision, indent=1))
    if args.json:
        args.json.write_text(json.dumps({"grid": GRID, "floors": FLOORS, "rows": rows, "decision": decision},
                                        indent=2) + "\n")
    return 0 if decision["chosen"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
