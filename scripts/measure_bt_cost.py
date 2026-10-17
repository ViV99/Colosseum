"""Cost of the Bradley-Terry fit at about 2 000 entities (SP4 spec §8 risk "fit slow on thousands of entities";
plan T8.2). One BLAS thread, as in the training main process (spec block 2).

Synthetic 1v1 layout: 20 lineages x 100 versions + 3 fixed agents (2 003 entities); every entity meets 10
random others (20 matches each pair, outcomes from known strengths); timings: cold ``fit``, warm ``fit``
after 1 % more data, ``laplace`` (1 000 samples, top-k 5) and ``bootstrap`` (``--bootstrap`` replicas).

Rule (fixed before the measurement): live refits are acceptable if cold fit <= 30 s, warm fit <= 5 s and
laplace <= 5 s on the reference machine (refit every 60 s in a background thread); report bootstrap is
informational. Otherwise the controller decides (iterative solver task, or a larger default refit interval).

Usage::

    env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
        scripts/measure_bt_cost.py --json docs/benchmarks/sp4-bt-cost.json
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LINEAGES, VERSIONS, FIXED, OPPONENTS, PER_PAIR = 20, 100, ("random", "greedy", "old"), 10, 20


def synthetic_table(seed: int, extra: float = 0.0):
    from colosseum.arena.comparisons import ComparisonTable, PairStat, TeamKey

    rng = np.random.default_rng(seed)
    entities = [f"a{i}@v{100 * v}" for i in range(LINEAGES) for v in range(VERSIONS)] + list(FIXED)
    strength = {e: (0.0 if e == "random" else rng.normal(0.0, 1.0)) for e in entities}
    table = ComparisonTable()
    n_pairs = int(len(entities) * OPPONENTS * (1.0 + extra))
    for _ in range(n_pairs):
        a, b = rng.choice(len(entities), size=2, replace=False)
        ea, eb = entities[a], entities[b]
        p = 1.0 / (1.0 + np.exp(-(strength[ea] - strength[eb])))
        wins = float(rng.binomial(PER_PAIR, p))
        table.add("1v1", "train", TeamKey(0, ((ea, "player"),)), TeamKey(1, ((eb, "player"),)),
                  PairStat(n=PER_PAIR, sw=float(PER_PAIR), swy=wins))
    return table, len(entities)


def main() -> int:
    from colosseum.arena import bt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    options = bt.BTOptions(prior_sigma=bt.PRIOR_SIGMA, smoothness=bt.SMOOTHNESS)
    out = {"blas_threads": os.environ.get("OMP_NUM_THREADS")}
    table, n = synthetic_table(0)
    out["entities"] = n
    t = time.monotonic()
    cold = bt.fit(table, "1v1", sources=("train",), options=options, fixed_agents=FIXED, random_bot="random")
    out["fit_cold_s"] = time.monotonic() - t
    out["iterations_cold"] = cold.iterations
    more, _ = synthetic_table(0, extra=0.01)
    t = time.monotonic()
    warm = bt.fit(more, "1v1", sources=("train",), options=options, fixed_agents=FIXED, random_bot="random", warm=cold)
    out["fit_warm_s"] = time.monotonic() - t
    out["iterations_warm"] = warm.iterations
    t = time.monotonic()
    bt.laplace(cold, samples=1000, rng=np.random.default_rng(0), top_k=5)
    out["laplace_s"] = time.monotonic() - t
    if args.bootstrap:
        from colosseum.arena.bt import BootstrapData

        t = time.monotonic()
        bt.bootstrap(cold, BootstrapData.from_table(table, "1v1", sources=("train",)), samples=args.bootstrap,
                     rng=np.random.default_rng(0))
        out["bootstrap_s"] = time.monotonic() - t
        out["bootstrap_samples"] = args.bootstrap
    out["pass"] = out["fit_cold_s"] <= 30 and out["fit_warm_s"] <= 5 and out["laplace_s"] <= 5
    print(json.dumps(out, indent=1))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
