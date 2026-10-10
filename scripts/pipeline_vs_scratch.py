"""Pipeline vs training from scratch on an equal env-step budget (SP3 spec section 3, criterion 3;
reported, not asserted).

For every seed: the full pipeline of the slow test (``tests/pipeline_kit.py``: record the
unit_harvest bot -> bc -> train with init + critic warm-up + the bot as DAgger teacher + anchors ->
eval), then the same training budget and league from scratch (no ``init``, no kickstart; the frozen
BC net stays in the config so both leagues are identical). Both trained agents are evaluated
against the bot, RandomBot and the BC net (``colosseum eval --deterministic``).

Usage::

    .venv/bin/python scripts/pipeline_vs_scratch.py --seeds 0 1 2 --json docs/benchmarks/pipeline-vs-scratch.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tests", REPO_ROOT / "tests" / "learning"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _row(variant: str, seed: int, run, game) -> dict:
    import pipeline_kit as kit

    row = {"variant": variant, "seed": seed, **{f"{k}_s": round(v, 1) for k, v in run.seconds.items()}}
    for name, key in ((game.bot, "bot"), (kit.RANDOM, "random"), (kit.BC_NET, "bc")):
        cell = run.pairs[name]
        row.update({f"{key}_score": cell["score"], f"{key}_win": cell["win_rate"],
                    f"{key}_draw": cell["draws"] / cell["n"]})
    return row


def main() -> int:
    import pipeline_kit as kit

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/colosseum-pipeline-vs-scratch"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    game, settings = kit.UNIT_HARVEST, kit.UNIT_HARVEST_SETTINGS
    rows = []
    for seed in args.seeds:
        workdir = args.work_dir / f"s{seed}"
        workdir.mkdir(parents=True, exist_ok=True)
        pipeline = kit.run_pipeline(game, workdir, seed=seed, settings=settings, name="pipeline")
        rows.append(_row("pipeline", seed, pipeline, game))
        print(json.dumps(rows[-1]), flush=True)
        scratch = kit.run_pipeline(game, workdir, seed=seed, settings=settings, warm_start=False,
                                   bc_path=pipeline.bc_path, name="scratch")
        rows.append(_row("scratch", seed, scratch, game))
        print(json.dumps(rows[-1]), flush=True)
    cols = ["variant", "seed", "bot_score", "bot_win", "bot_draw", "random_win", "bc_score", "record_s", "bc_s",
            "train_s", "eval_s"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(f"{row[c]:.3f}" if isinstance(row.get(c), float) else str(row.get(c, "-"))
                                for c in cols) + " |")
    for variant in ("pipeline", "scratch"):
        scores = [r["bot_score"] for r in rows if r["variant"] == variant]
        print(f"{variant}: mean score vs the bot {sum(scores) / len(scores):.3f} over {len(scores)} seeds")
    if args.json:
        result = {"seeds": args.seeds, "settings": repr(settings), "rows": rows}
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
