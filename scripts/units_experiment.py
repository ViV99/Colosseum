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
share vs scripted at least 0.05 higher and a K=8 share at most 0.05 lower.

Usage (the evaluation imports the test kit, so it puts ``tests/`` and ``tests/learning`` on
``sys.path``)::

    .venv/bin/python scripts/units_experiment.py --steps 400000 --seeds 0 --parallel 2 \
        --json docs/benchmarks/units-experiment.json
"""
from __future__ import annotations

import argparse
import itertools
import json
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
UNIT_TRACES = ("joint", "geo_mean", "none")
DIAG_KEYS = ("clip_fraction", "clip_fraction_joint", "ess", "log_rho_abs_p95", "log_rho_joint_abs_mean",
             "c_clip_frac", "deciders_valid_mean")
DEFAULT_TRACE = "geo_mean"   # the spec's pre-experiment default: the baseline of the decision rule
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
    out = {}
    if not tail:
        raise KeyError(f"no train records in {run.root}")
    for key in DIAG_KEYS:
        values = [r[key] for r in tail]      # KeyError for a metric the run does not log
        out[key] = sum(values) / len(values)
    out["env_steps_per_s"] = run.env_steps_per_sec()
    out["train_steps"] = train_records[-1]["train_step"] if train_records else 0
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


def recommend(rows: list[dict]) -> tuple[str, str]:
    """The T8.4 decision rule for the ``unit_trace: auto`` default (with Units)."""
    def share(k: int, trace: str) -> float:
        vals = [r["share_vs_scripted"] for r in rows if "share_vs_scripted" in r
                and r["k"] == k and r["ratio_mode"] == "per_unit" and r["unit_trace"] == trace]
        return sum(vals) / len(vals) if vals else float("nan")

    base128, base8 = share(128, DEFAULT_TRACE), share(8, DEFAULT_TRACE)
    better = [t for t in UNIT_TRACES if t != DEFAULT_TRACE
              and share(128, t) >= base128 + MARGIN and share(8, t) >= base8 - MARGIN]
    choice = max(better, key=lambda t: share(128, t)) if better else DEFAULT_TRACE
    detail = "; ".join(f"{t}: K=128 {share(128, t):.3f}, K=8 {share(8, t):.3f}" for t in UNIT_TRACES)
    return choice, detail


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, nargs="+", default=[8, 128], choices=sorted(K_KWARGS))
    parser.add_argument("--ratio-modes", nargs="+", default=list(RATIO_MODES), choices=RATIO_MODES)
    parser.add_argument("--unit-traces", nargs="+", default=list(UNIT_TRACES), choices=UNIT_TRACES)
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
    choice, detail = recommend(rows)
    print(f"\nunit_trace recommendation (per_unit, share vs scripted): {choice} ({detail})")
    if args.json:
        args.json.write_text(json.dumps({"steps": args.steps, "workers": args.workers, "k_kwargs": K_KWARGS,
                                         "rows": rows, "recommendation": choice}, indent=2, default=str) + "\n")
    return 1 if any(r["returncode"] != 0 for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
