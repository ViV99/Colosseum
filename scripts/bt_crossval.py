"""Bradley-Terry prior and smoothness by cross-validation (SP4 spec block 2; plan T8.2).

Journals: short training runs of four demo games (``--make-journals``: tic_tac_toe, tron, unit_harvest_league,
predator_prey; the journal is on by default). Every match of a journal goes to fold ``sha1(uid) % FOLDS``;
for each fold and grid point the fit on the other folds predicts the held-out team pairs
(``match_pairs``: weight ``1/(T-1)``, outcome 1 / 0.5 / 0) and scores them by the weighted log-loss.
Pairs with an entity the training folds never saw are skipped (counted). Score layouts have no pairs.

Decision rule (fixed before the measurement, ``choose``): the grid point with the smallest mean held-out
log-loss (games weighted equally); among the points within one standard error (over folds) of it, the one
with the largest ``smoothness`` (the spec wants few-game snapshots to lean on their neighbours: the most
smoothing that predicts as well), then the largest ``prior_sigma`` (the weakest pull towards the anchor).

Usage::

    env -u PYTHONPATH .venv/bin/python scripts/bt_crossval.py --make-journals --work-dir /tmp/sp4-cv
    env -u PYTHONPATH .venv/bin/python scripts/bt_crossval.py --work-dir /tmp/sp4-cv \
        --json docs/benchmarks/sp4-bt-crossval.json
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GAMES = {    # game -> (example config, extra --set)
    "tic_tac_toe": ("tic_tac_toe", {}),
    "tron": ("tron", {}),
    "unit_harvest_league": ("unit_harvest_league", {}),
    "predator_prey": ("predator_prey", {}),
}
PRIOR_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)            # prior standard deviation of a strength, logits
SMOOTH_GRID = (0.0, 1.0, 4.0, 16.0, 64.0)         # random-walk precision between consecutive versions
FOLDS = 5
ELO = 400.0 / math.log(10.0)                      # Elo units per logit (BTFit ratings are in Elo units)
SOURCES = ("train", "pulse")


def fold_of(uid: str) -> int:
    return int(hashlib.sha1(uid.encode("utf-8")).hexdigest(), 16) % FOLDS


def make_journals(work_dir: Path, seed: int) -> None:
    for game, (config, extra) in GAMES.items():
        if (work_dir / "runs" / game / "matches.db").exists():
            print(f"{game}: journal exists", flush=True)
            continue
        path = REPO_ROOT / "configs" / "examples" / f"{config}.yaml"
        cmd = [sys.executable, "-m", "colosseum", "train", "-c", str(path),
               "--set", f"run.dir={work_dir / 'runs'}", "--set", f"run.name={game}", "--set", f"training.seed={seed}",
               "--set", "rollout.num_workers=2"]
        for key, value in extra.items():
            cmd += ["--set", f"{key}={value}"]
        start = time.monotonic()
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        print(f"{game}: trained in {time.monotonic() - start:.0f} s", flush=True)


def load(journal: Path):
    """``(rating inputs, [(fold, result, rating_versions)])`` of one run's journal (detail rows only)."""
    from colosseum.arena.live import rating_inputs
    from colosseum.arena.matchlog import MatchLog
    from colosseum.core.config import load_config

    config = load_config(journal.parent / "config.resolved.yaml")
    reader = MatchLog.open_readonly(journal)
    try:
        rows = reader.matches(limit=None)
        items = []
        for row in rows:
            result, versions = row.to_result()
            if result.source in SOURCES:
                items.append((fold_of(result.uid), result, versions))
    finally:
        reader.close()
    return rating_inputs(config), items


def predict(fit_, a, b) -> float | None:
    """P(team a beats team b) under the fit; None if a member was not rated."""
    def strength(team) -> float | None:
        ratings = [fit_.rating.get(e) for e, _ in team.members]
        if any(r is None for r in ratings):
            return None
        return (sum(ratings) / len(ratings) + fit_.bias.get(team.team, 0.0)) / ELO

    sa, sb = strength(a), strength(b)
    if sa is None or sb is None:
        return None
    return 1.0 / (1.0 + math.exp(-(sa - sb)))


def evaluate(inputs, items, prior: float, smooth: float) -> dict:
    """Held-out log-loss per fold: ``{"loss": [...], "weight": [...], "skipped": n}``."""
    from colosseum.arena import bt
    from colosseum.arena.comparisons import ComparisonTable, match_pairs

    options, fixed, random_bot = inputs
    options = dataclasses.replace(options, prior_sigma=prior, smoothness=smooth)
    losses, weights, skipped = [], [], 0
    for k in range(FOLDS):
        table = ComparisonTable()
        for fold, result, versions in items:
            if fold != k:
                table.add_result(result, versions)
        loss = weight = 0.0
        fits = {}
        for fold, result, versions in items:
            if fold != k or result.outcome_kind == "score" or result.layout not in table.layouts():
                continue
            if result.layout not in fits:
                fits[result.layout] = bt.fit(table, result.layout, sources=SOURCES, options=options,
                                             fixed_agents=fixed, random_bot=random_bot)
            for a, b, w, y in match_pairs(result, versions):
                p = predict(fits[result.layout], a, b)
                if p is None:
                    skipped += 1
                    continue
                p = min(max(p, 1e-9), 1 - 1e-9)
                loss -= w * (y * math.log(p) + (1 - y) * math.log(1 - p))
                weight += w
        losses.append(loss)
        weights.append(weight)
    return {"loss": losses, "weight": weights, "skipped": skipped}


def choose(rows: list[dict]) -> dict:
    """The decision rule (module docstring). ``rows``: ``{"game", "prior", "smooth", "loss": [per fold],
    "weight": [per fold]}``; returns ``{"prior", "smooth", "mean", "se", "best": (prior, smooth)}``."""
    games = sorted({r["game"] for r in rows})
    by_point: dict[tuple[float, float], list[float]] = defaultdict(lambda: [0.0] * FOLDS)
    for r in rows:
        for k in range(FOLDS):
            per_game = r["loss"][k] / r["weight"][k] if r["weight"][k] else 0.0
            by_point[(r["prior"], r["smooth"])][k] += per_game / len(games)
    stats = {}
    for point, folds in by_point.items():
        mean = sum(folds) / FOLDS
        sd = math.sqrt(sum((f - mean) ** 2 for f in folds) / (FOLDS - 1))
        stats[point] = (mean, sd / math.sqrt(FOLDS))
    best = min(stats, key=lambda p: stats[p][0])
    limit = stats[best][0] + stats[best][1]
    eligible = [p for p in stats if stats[p][0] <= limit]
    prior, smooth = max(eligible, key=lambda p: (p[1], p[0]))
    return {"prior": prior, "smooth": smooth, "mean": stats[(prior, smooth)][0], "se": stats[(prior, smooth)][1],
            "best": list(best), "best_mean": stats[best][0], "best_se": stats[best][1]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--make-journals", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.make_journals:
        make_journals(args.work_dir, args.seed)
    rows = []
    for game in GAMES:
        journal = args.work_dir / "runs" / game / "matches.db"
        inputs, items = load(journal)
        print(f"{game}: {len(items)} matches", flush=True)
        for prior in PRIOR_GRID:
            for smooth in SMOOTH_GRID:
                start = time.monotonic()
                res = evaluate(inputs, items, prior, smooth)
                rows.append({"game": game, "prior": prior, "smooth": smooth, **res,
                             "seconds": round(time.monotonic() - start, 1)})
                mean = sum(res["loss"]) / max(sum(res["weight"]), 1e-12)
                print(f"  prior {prior:>4} smooth {smooth:>5}: log-loss {mean:.5f} (skipped {res['skipped']})",
                      flush=True)
    decision = choose(rows)
    print(json.dumps(decision))
    if args.json:
        args.json.write_text(json.dumps({"folds": FOLDS, "rows": rows, "decision": decision}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
