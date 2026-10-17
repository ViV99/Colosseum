"""Bradley-Terry prior and smoothness by cross-validation (SP4 spec block 2; plan T8.2).

Journals: short training runs of four demo games (``--make-journals``: tic_tac_toe, tron, unit_harvest_league,
predator_prey; the journal is on by default). Every match of a journal goes to fold ``sha1(key) % FOLDS``, the
key being the match's block id when it has one (the seat rotations of one pulse block stay in one fold), else
its uid; for each fold and grid point the fit on the other folds predicts the held-out team pairs
(``match_pairs``: weight ``1/(T-1)``, outcome 1 / 0.5 / 0) and scores them by the weighted log-loss. The
prediction is the model's own linear predictor in logits (raw parameters, not the per-cell anchored ratings,
so pairs across rating cells - e.g. predator_prey's hunter vs prey - are predicted as the fit models them).
Pairs with an entity the training folds never saw are skipped (counted); self-play pairs (every entity
cancels, only the seat bias is left) are left out of the score (counted). Score layouts have no pairs.

Decision rule (``RULE`` = "1-SE, paired by fold"; fixed before the measurement, ``choose``): the grid point
with the smallest mean held-out log-loss (games weighted equally) is the best; a point is eligible when its
mean exceeds the best's by at most one PAIRED standard error, the standard error of its per-fold differences
from the best (``sd_k(L_p[k] - L_best[k]) / sqrt(FOLDS)``; the fold-to-fold variation all points share
cancels, so the window measures only how consistently a point is worse). Among the eligible points the one
with the largest ``smoothness`` (the spec wants few-game snapshots to lean on their neighbours: the most
smoothing that predicts as well), then the largest ``prior_sigma`` (the weakest pull towards the anchor).
(Controller ruling in the T8.2a review, before any T8.2b data: paired instead of each point's own SE.)

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
RULE = "1-SE, paired by fold"


def fold_of(key: str) -> int:
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % FOLDS


def fold_key(result) -> str:
    """The match's block id when it has one (a pulse block's rotations share a fold), else its uid."""
    return result.block if result.block else result.uid


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
                items.append((fold_of(fold_key(result)), result, versions))
    finally:
        reader.close()
    return rating_inputs(config), items


def predict(fit_, a, b) -> float | None:
    """P(team a beats team b) under the fit: the model's linear predictor in logits (``comparisons.coefficients``
    over the raw parameters ``fit_.params`` plus the seat-bias difference), so pairs across rating cells keep the
    cells' offsets; None if a member or a team is not in the fit's design."""
    from colosseum.arena.comparisons import coefficients

    design = fit_.design
    if design is None:
        return None
    if any(e not in design.index for team in (a, b) for e, _ in team.members):
        return None
    if a.team not in fit_.bias or b.team not in fit_.bias:
        return None
    z = sum(c * float(fit_.params[design.index[e]]) for e, c in coefficients(a, b).items())
    z += (fit_.bias[a.team] - fit_.bias[b.team]) / ELO
    return 1.0 / (1.0 + math.exp(-z))


def is_self_play(a, b) -> bool:
    """Every entity cancels in ``S_a - S_b`` (the same entities on both sides): the pair tests only the bias."""
    from colosseum.arena.comparisons import coefficients

    return not coefficients(a, b)


def fold_tables(items) -> list:
    """The training table of every fold (all matches of the other folds); built once, reused for every grid point."""
    from colosseum.arena.comparisons import ComparisonTable

    tables = []
    for k in range(FOLDS):
        table = ComparisonTable()
        for fold, result, versions in items:
            if fold != k:
                table.add_result(result, versions)
        tables.append(table)
    return tables


def evaluate(inputs, items, prior: float, smooth: float, tables: list | None = None) -> dict:
    """Held-out log-loss per fold: ``{"loss": [...], "weight": [...], "skipped": n, "self_play": n}``
    (``tables``: ``fold_tables(items)``, built here when not given)."""
    from colosseum.arena import bt
    from colosseum.arena.comparisons import match_pairs

    options, fixed, random_bot = inputs
    options = dataclasses.replace(options, prior_sigma=prior, smoothness=smooth)
    tables = fold_tables(items) if tables is None else tables
    losses, weights, skipped, self_play = [], [], 0, 0
    for k in range(FOLDS):
        table = tables[k]
        layouts = set(table.layouts())
        loss = weight = 0.0
        fits = {}
        for fold, result, versions in items:
            if fold != k or result.outcome_kind == "score" or result.layout not in layouts:
                continue
            if result.layout not in fits:
                fits[result.layout] = bt.fit(table, result.layout, sources=SOURCES, options=options,
                                             fixed_agents=fixed, random_bot=random_bot)
            for a, b, w, y in match_pairs(result, versions):
                if is_self_play(a, b):
                    self_play += 1
                    continue
                p = predict(fits[result.layout], a, b)
                if p is None:
                    skipped += 1
                    continue
                p = min(max(p, 1e-9), 1 - 1e-9)
                loss -= w * (y * math.log(p) + (1 - y) * math.log(1 - p))
                weight += w
        losses.append(loss)
        weights.append(weight)
    return {"loss": losses, "weight": weights, "skipped": skipped, "self_play": self_play}


def choose(rows: list[dict]) -> dict:
    """The decision rule (module docstring). ``rows``: ``{"game", "prior", "smooth", "loss": [per fold],
    "weight": [per fold]}``; returns ``{"rule", "prior", "smooth", "mean", "paired_se", "best": [prior, smooth],
    "best_mean", "points": [{prior, smooth, mean, diff, paired_se}], "eligible": [same]}`` (``diff``: mean minus
    the best's mean; ``eligible``: the points with ``diff <= paired_se``)."""
    games = sorted({r["game"] for r in rows})
    by_point: dict[tuple[float, float], list[float]] = defaultdict(lambda: [0.0] * FOLDS)
    for r in rows:
        for k in range(FOLDS):
            per_game = r["loss"][k] / r["weight"][k] if r["weight"][k] else 0.0
            by_point[(r["prior"], r["smooth"])][k] += per_game / len(games)
    means = {point: sum(folds) / FOLDS for point, folds in by_point.items()}
    best = min(means, key=means.get)
    points = {}
    for point, folds in by_point.items():
        diffs = [f - b for f, b in zip(folds, by_point[best])]
        mean_diff = sum(diffs) / FOLDS
        sd = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / (FOLDS - 1))
        points[point] = {"prior": point[0], "smooth": point[1], "mean": means[point], "diff": mean_diff,
                         "paired_se": sd / math.sqrt(FOLDS)}
    eligible = [p for p in points if points[p]["diff"] <= points[p]["paired_se"]]
    prior, smooth = max(eligible, key=lambda p: (p[1], p[0]))
    return {"rule": RULE, "prior": prior, "smooth": smooth, "mean": means[(prior, smooth)],
            "paired_se": points[(prior, smooth)]["paired_se"], "best": list(best), "best_mean": means[best],
            "points": [points[p] for p in sorted(points)], "eligible": [points[p] for p in sorted(eligible)]}


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
        tables = fold_tables(items)
        print(f"{game}: {len(items)} matches", flush=True)
        for prior in PRIOR_GRID:
            for smooth in SMOOTH_GRID:
                start = time.monotonic()
                res = evaluate(inputs, items, prior, smooth, tables)
                rows.append({"game": game, "prior": prior, "smooth": smooth, **res,
                             "seconds": round(time.monotonic() - start, 1)})
                mean = sum(res["loss"]) / max(sum(res["weight"]), 1e-12)
                print(f"  prior {prior:>4} smooth {smooth:>5}: log-loss {mean:.5f} (skipped {res['skipped']}, "
                      f"self-play {res['self_play']})",
                      flush=True)
    decision = choose(rows)
    print(json.dumps(decision))
    if args.json:
        args.json.write_text(json.dumps({"folds": FOLDS, "rows": rows, "decision": decision}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
