"""Pulse cadence and the pipeline-test thresholds (SP4 criterion 3.4; plan T8.2).

Phase 1 (``--every 1 2 5``, seed 0): ``unit_harvest_league`` from scratch with pulse (``tests/arena_kit.py``,
1 rollout worker + the pulse process), ``evaluation.blocks`` = ``--blocks``. Recorded: snapshots stored,
snapshots evaluated by pulse (journal), the ``system`` counters ``pulse_skipped_snapshots`` /
``pulse_evaluated_snapshots`` (T6.1), training seconds. Rule ``decide_cadence``: the smallest ``every`` with
``pulse_skipped <= 1`` (the final snapshot may be cut by the budget) and at least 6 evaluated snapshots.

Phase 2 (``--calibrate-seeds 0 1 2`` with the chosen ``every``): per seed the rating trend (Spearman rho of BT
rating with version, gain newest third - oldest third), ``tournament --mode select`` over the run's
snapshots (field: the anchors) and ``eval`` of the winner against ``random`` (60 matches, deterministic).
Rule ``decide_trend``: ``RHO_MIN = max(0.3, floor_0.05(min rho - 0.1))``; ``GAIN_MIN = 0``; the
winner-vs-random bar stays the SP3 one (0.95) and is reported, not tuned.

Usage::

    env -u PYTHONPATH .venv/bin/python scripts/measure_pulse_cadence.py --every 1 2 5 --blocks 25 \
        --work-dir /tmp/sp4-pulse --json docs/benchmarks/sp4-pulse-cadence.json
    env -u PYTHONPATH .venv/bin/python scripts/measure_pulse_cadence.py --every <chosen> --blocks 25 \
        --calibrate-seeds 0 1 2 --work-dir /tmp/sp4-pulse --json docs/benchmarks/sp4-pipeline-calibration.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MIN_EVALUATED = 6
RHO_FLOOR = 0.3


def _system_counter(run, name: str) -> int | None:
    records = [r for r in run.run.records("system") if name in r]
    return int(max(r[name] for r in records)) if records else None


def decide_cadence(rows: list[dict]) -> dict:
    for row in sorted(rows, key=lambda r: r["every"]):
        if (row["pulse_skipped"] or 0) <= 1 and row["evaluated"] >= MIN_EVALUATED:
            return {"every": row["every"], "row": row}
    return {"every": None, "row": None}


def decide_trend(rows: list[dict]) -> dict:
    rhos = [r["spearman"] for r in rows]
    rho_min = max(RHO_FLOOR, math.floor((min(rhos) - 0.1) / 0.05) * 0.05) if rhos else None
    return {"rho_min": rho_min, "gain_min": 0.0, "rhos": rhos, "gains": [r["gain"] for r in rows],
            "vs_random": [r["vs_random_win"] for r in rows], "confirmed": [r["confirmed"] for r in rows]}


def main() -> int:
    import arena_kit as kit
    import pipeline_kit as pk

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--every", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--blocks", type=int, default=25)
    parser.add_argument("--calibrate-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/sp4-pulse"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    out: dict = {"blocks": args.blocks}
    if not args.calibrate_seeds:
        rows = []
        for every in args.every:
            work = args.work_dir / f"every{every}"
            work.mkdir(parents=True, exist_ok=True)
            run = kit.train_with_pulse(work, seed=0, settings=kit.PulseSettings(every=every, blocks=args.blocks))
            rows.append({"every": every, "stored": run.stored_versions, "evaluated": len(run.pulse_versions),
                         "pulse_versions": run.pulse_versions,
                         "pulse_skipped": _system_counter(run, "pulse_skipped_snapshots"),
                         "pulse_evaluated": _system_counter(run, "pulse_evaluated_snapshots"),
                         "train_s": round(run.seconds["train"], 1), "spearman": run.trend.spearman,
                         "gain": run.trend.gain})
            print(json.dumps(rows[-1]), flush=True)
        out.update(rows=rows, decision=decide_cadence(rows))
    else:
        (every,) = args.every
        rows = []
        for seed in args.calibrate_seeds:
            work = args.work_dir / f"calib-s{seed}"
            work.mkdir(parents=True, exist_ok=True)
            run = kit.train_with_pulse(work, seed=seed, settings=kit.PulseSettings(every=every, blocks=args.blocks))
            report = kit.select_snapshots(run, seed=seed)
            group = kit.group_of(report, "2p")
            winner = group["select"]["winner"]
            version = int(winner.rsplit("@v", 1)[1])
            pairs = pk.evaluate(run.config, run.run.root / "checkpoints" / pk.MAIN / f"ckpt_v{version}", [pk.RANDOM],
                                work / "winner-eval.json", layout="2p", num_matches=60)
            rows.append({"seed": seed, "every": every, "spearman": run.trend.spearman, "gain": run.trend.gain,
                         "evaluated": len(run.pulse_versions), "winner": winner,
                         "confirmed": bool(group["select"]["confirmed"]),
                         "vs_random_win": pairs[pk.RANDOM]["win_rate"], "train_s": round(run.seconds["train"], 1)})
            print(json.dumps(rows[-1]), flush=True)
        out.update(rows=rows, decision=decide_trend(rows))
    print(json.dumps(out["decision"], indent=1))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
