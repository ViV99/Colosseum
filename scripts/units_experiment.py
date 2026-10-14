"""Units experiment (SP2 spec, section 3, criterion 4).

``unit_harvest`` at K=8 and K=128 units x ``algorithm.ratio_mode`` in {joint, per_unit} x
``algorithm.unit_trace`` in {joint, geo_mean, none}, every run on the same env-step budget.
Each run is a real ``colosseum train`` (subprocess) on ``configs/examples/unit_harvest.yaml``.
Afterwards the newest checkpoint of every run plays greedily (in-process ``play_lineups``):
- against the uniformly random legal player: win rate and mean own score;
- against the scripted reference bot (``examples.unit_harvest.game.scripted_action``): win rate and
  score share own / (own + opponent), which does not saturate when every mode beats random.
Diagnostics are the means over the last quarter of the run's ``train`` records in
``metrics.jsonl``: ``clip_fraction`` (per decider), ``clip_fraction_joint``, ``ess``,
``log_rho_abs_p95``, ``log_rho_joint_abs_mean``, ``c_clip_frac``, ``deciders_valid_mean``.

The script prints a Markdown table and the ``unit_trace`` recommendation of the decision rule
(T8.4): keep ``geo_mean`` unless another trace, with ``ratio_mode=per_unit``, has a K=128 score
share vs scripted at least 0.05 higher and a K=8 share at most 0.05 lower. ``geo_mean`` is the
rule's baseline because it was the spec's default before the experiment; the experiment chose
``joint``, which is the current default (``unit_trace: auto``, docs/benchmarks.md).

SP3 (spec block 10): ``--cells`` replaces the ratio x trace product by explicit cells. The SP3 grid
is ``--cells per_unit:auto joint:auto per_unit:none --seeds 0 1 2`` (the current default and the two
alternatives); ``decide_defaults`` applies the SP3 rule: a default changes only if at K=128 the
alternative's mean ``share_vs_scripted`` beats the default's by more than 2 standard errors of the
difference (unpaired, sample variances) and at K=8 it is at most 0.03 worse.

Usage (the evaluation imports the test kit, so it puts ``tests/`` and ``tests/learning`` on
``sys.path``)::

    .venv/bin/python scripts/units_experiment.py --steps 340000 --seeds 0 --parallel 1 \\
        --json docs/benchmarks/units-experiment.json
    .venv/bin/python scripts/units_experiment.py --cells per_unit:auto joint:auto per_unit:none --seeds 0 1 2 \\
        --steps 340000 --parallel 1 --json docs/benchmarks/units-defaults-sp3.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tests", REPO_ROOT / "tests" / "learning"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CONFIG = REPO_ROOT / "configs" / "examples" / "unit_harvest.yaml"
K_KWARGS = {
    8: {"max_units": 8, "size": 8, "num_resources": 6, "initial_workers": 2, "max_steps": 50},
    128: {"max_units": 128, "size": 16, "num_resources": 16, "initial_workers": 32, "max_steps": 80},
}
RATIO_MODES = ("joint", "per_unit")
UNIT_TRACES = ("auto", "joint", "geo_mean", "none")
SP2_TRACES = ("joint", "geo_mean", "none")
SP3_CELLS = (("per_unit", "auto"), ("joint", "auto"), ("per_unit", "none"))
SE_FACTOR = 2.0          # SP3 rule: the K=128 gain must exceed 2 standard errors of the difference
K8_TOLERANCE = 0.03      # SP3 rule: at K=8 the alternative may be at most this much worse
DIAG_KEYS = ("clip_fraction", "clip_fraction_joint", "ess", "log_rho_abs_p95", "log_rho_joint_abs_mean",
             "c_clip_frac", "deciders_valid_mean")
DEFAULT_TRACE = "geo_mean"   # the rule's baseline (the spec's pre-experiment default); the current default is joint
MARGIN = 0.05


def run_name(k: int, ratio: str, trace: str, seed: int) -> str:
    return f"k{k}-{ratio}-{trace}-s{seed}"


def train(k: int, ratio: str, trace: str, seed: int, *, steps: int, workers: int, run_dir: Path) -> dict:
    from demo_learning import train_cmd

    name = run_name(k, ratio, trace, seed)
    sets = {"training.total_timesteps": steps, "training.seed": seed, "rollout.num_workers": workers,
            "algorithm.ratio_mode": ratio, "algorithm.unit_trace": trace, "env.kwargs": json.dumps(K_KWARGS[k])}
    env = {**os.environ, "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"}
    start = time.monotonic()
    with open(run_dir / f"{name}.out", "w") as out:
        proc = subprocess.run(train_cmd(CONFIG, run_dir, name, sets), cwd=REPO_ROOT, env=env, stdout=out,
                              stderr=subprocess.STDOUT, timeout=4 * 3600)
    return {"k": k, "ratio_mode": ratio, "unit_trace": trace, "seed": seed, "name": name,
            "returncode": proc.returncode, "wall_s": time.monotonic() - start}


def diagnostics(run) -> dict:
    train_records = run.records("train")
    tail = train_records[-max(1, len(train_records) // 4):]
    if not tail:
        raise KeyError(f"no train records in {run.root}")
    out = {}
    for key in DIAG_KEYS:
        values = [r[key] for r in tail]      # KeyError for a metric the run does not log
        out[key] = sum(values) / len(values)
    out["env_steps_per_s"] = run.env_steps_per_sec()
    out["train_steps"] = train_records[-1]["train_step"]
    return out


def evaluate(run, matches: int) -> dict:
    from colosseum.core.types import MatchResult
    from demo_learning import ScriptedPolicy, greedy_agent, play, random_model, rotating_lineups, team_of, win_rate
    from examples.unit_harvest.game import scripted_action

    trained, role = greedy_agent(run, "agent_0")
    action_spec = trained.action_spec
    models = {"trained": trained, "random": random_model(role),
              "scripted": ScriptedPolicy(scripted_action, action_spec)}

    def scores(r: MatchResult) -> tuple[float, float]:
        mine = team_of(r, "trained")
        own = next(t.score for t in r.teams if t.team == mine)
        opp = next(t.score for t in r.teams if t.team != mine)
        return own, opp

    vs_random = play(run.env_fn(), models, rotating_lineups("2p", 2, "trained", "random", matches))
    vs_scripted = play(run.env_fn(), models, rotating_lineups("2p", 2, "trained", "scripted", matches))
    shares = [own / (own + opp) if own + opp > 0 else 0.5 for own, opp in map(scores, vs_scripted)]
    return {"win_vs_random": win_rate(vs_random, "trained"),
            "score_vs_random": sum(scores(r)[0] for r in vs_random) / len(vs_random),
            "win_vs_scripted": win_rate(vs_scripted, "trained"),
            "share_vs_scripted": sum(shares) / len(shares)}


def recommend(rows: list[dict]) -> tuple[str | None, str, str | None]:
    """The T8.4 decision rule for the ``unit_trace: auto`` default (with Units).

    Returns ``(choice, detail, note)``. When a share the rule needs is missing (no evaluated
    ``per_unit`` row for some K and trace, e.g. a recheck grid without K=8), the rule cannot be
    applied: ``choice`` is None and ``note`` names the missing cells (no silent default)."""
    def share(k: int, trace: str) -> float:
        vals = [r["share_vs_scripted"] for r in rows if "share_vs_scripted" in r
                and r["k"] == k and r["ratio_mode"] == "per_unit" and r["unit_trace"] == trace]
        return sum(vals) / len(vals) if vals else float("nan")

    detail = "; ".join(f"{t}: K=128 {share(128, t):.3f}, K=8 {share(8, t):.3f}" for t in SP2_TRACES)
    missing = [f"K={k} {t}" for k in (128, 8) for t in SP2_TRACES if math.isnan(share(k, t))]
    if missing:
        return None, detail, ("no recommendation: the rule needs share_vs_scripted of every per_unit trace at "
                              f"K=128 and K=8; missing: {', '.join(missing)}")
    base128, base8 = share(128, DEFAULT_TRACE), share(8, DEFAULT_TRACE)
    better = [t for t in SP2_TRACES if t != DEFAULT_TRACE
              and share(128, t) >= base128 + MARGIN and share(8, t) >= base8 - MARGIN]
    choice = max(better, key=lambda t: share(128, t)) if better else DEFAULT_TRACE
    return choice, detail, None


def parse_cells(values: list[str]) -> list[tuple[str, str]]:
    cells = []
    for value in values:
        ratio, sep, trace = value.partition(":")
        if not sep or ratio not in RATIO_MODES or trace not in UNIT_TRACES:
            raise ValueError(f"cell {value!r}: expected ratio_mode:unit_trace with ratio_mode in {RATIO_MODES} "
                             f"and unit_trace in {UNIT_TRACES}")
        cells.append((ratio, trace))
    return cells


def _shares(rows: list[dict], k: int, cell: tuple[str, str]) -> list[float]:
    return [float(r["share_vs_scripted"]) for r in rows if "share_vs_scripted" in r and r["k"] == k
            and (r["ratio_mode"], r["unit_trace"]) == cell]


def _mean_var(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    return mean, sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def compare(rows: list[dict], baseline: tuple[str, str], alternative: tuple[str, str]) -> dict:
    """The SP3 rule for one question: switch to ``alternative`` only if at K=128 its mean share beats
    ``baseline`` by more than ``SE_FACTOR`` standard errors of the difference and at K=8 it is at most
    ``K8_TOLERANCE`` worse. ``switch`` is None (with a ``note``) when a cell lacks the data."""
    a128, b128 = _shares(rows, 128, alternative), _shares(rows, 128, baseline)
    a8, b8 = _shares(rows, 8, alternative), _shares(rows, 8, baseline)
    missing = [f"K=128 {c} (need >= 2 seeds)" for c, v in ((alternative, a128), (baseline, b128)) if len(v) < 2]
    missing += [f"K=8 {c}" for c, v in ((alternative, a8), (baseline, b8)) if not v]
    out: dict = {"baseline": list(baseline), "alternative": list(alternative)}
    if missing:
        return {**out, "switch": None, "note": "no decision: missing " + ", ".join(missing)}
    mean_a, var_a = _mean_var(a128)
    mean_b, var_b = _mean_var(b128)
    se = math.sqrt(var_a / len(a128) + var_b / len(b128))
    diff128 = mean_a - mean_b
    diff8 = sum(a8) / len(a8) - sum(b8) / len(b8)
    switch = diff128 > SE_FACTOR * se and diff8 >= -K8_TOLERANCE
    return {**out, "switch": switch, "diff_k128": diff128, "se_k128": se, "diff_k8": diff8,
            "mean_k128": {"alternative": mean_a, "baseline": mean_b}}


def decide_defaults(rows: list[dict]) -> dict[str, dict]:
    """Both SP3 questions; ``choice`` is the resolved value with Units (``auto`` stays when no switch)."""
    ratio = compare(rows, ("per_unit", "auto"), ("joint", "auto"))
    trace = compare(rows, ("per_unit", "auto"), ("per_unit", "none"))
    ratio["choice"] = "joint" if ratio["switch"] else "per_unit"
    trace["choice"] = "none" if trace["switch"] else "joint"
    return {"ratio_mode": ratio, "unit_trace": trace}


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, nargs="+", default=[8, 128], choices=sorted(K_KWARGS))
    parser.add_argument("--ratio-modes", nargs="+", default=list(RATIO_MODES), choices=RATIO_MODES)
    parser.add_argument("--unit-traces", nargs="+", default=list(SP2_TRACES), choices=UNIT_TRACES)
    parser.add_argument("--cells", nargs="+", default=None,
                        help="explicit ratio_mode:unit_trace cells (SP3 grid: per_unit:auto joint:auto per_unit:none)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=400_000, help="env-step budget of every run")
    parser.add_argument("--workers", type=int, default=2, help="rollout workers per run")
    parser.add_argument("--parallel", type=int, default=2, help="runs at the same time")
    parser.add_argument("--eval-matches", type=int, default=100, help="matches per opponent")
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/colosseum-units-experiment"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from demo_learning import TrainedRun

    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.cells:
        grid = [(k, r, t, s) for k in args.k for (r, t) in parse_cells(args.cells) for s in args.seeds]
    else:
        grid = list(itertools.product(args.k, args.ratio_modes, args.unit_traces, args.seeds))
    print(f"{len(grid)} runs, {args.steps} env steps each, {args.parallel} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        rows = list(pool.map(lambda g: train(*g, steps=args.steps, workers=args.workers, run_dir=args.run_dir), grid))
    for row in rows:
        if row["returncode"] != 0:
            log = args.run_dir / f"{row['name']}.out"
            print(f"ERROR {row['name']}: exit {row['returncode']}, see {log}", file=sys.stderr)
            continue
        run = TrainedRun.open(args.run_dir / row["name"])
        row.update(diagnostics(run))
        row.update(evaluate(run, args.eval_matches))
        print(f"{row['name']}: {json.dumps({k: row[k] for k in row if k not in ('name',)}, default=str)}", flush=True)

    cols = ["k", "ratio_mode", "unit_trace", "seed", "win_vs_random", "score_vs_random", "win_vs_scripted",
            "share_vs_scripted", *DIAG_KEYS, "env_steps_per_s", "wall_s"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(_fmt(row.get(c, "-")) for c in cols) + " |")
    choice, detail, note = recommend(rows)
    print(f"\nunit_trace recommendation (per_unit, share vs scripted): {choice} ({detail})")
    if note:
        print(note)
    decisions = decide_defaults(rows) if args.cells else None
    if decisions:
        for question, d in decisions.items():
            if d["switch"] is None:
                print(f"{question}: {d['note']}")
            else:
                print(f"{question}: {'SWITCH to' if d['switch'] else 'keep'} {d['choice']} "
                      f"(K=128 diff {d['diff_k128']:+.3f}, 2*SE {SE_FACTOR * d['se_k128']:.3f}; "
                      f"K=8 diff {d['diff_k8']:+.3f})")
    if args.json:
        result = {"steps": args.steps, "workers": args.workers, "k_kwargs": K_KWARGS, "rows": rows,
                  "recommendation": choice, "cells": args.cells, "decisions": decisions}
        if note:
            result["recommendation_note"] = note
        args.json.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return 1 if any(r["returncode"] != 0 for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
