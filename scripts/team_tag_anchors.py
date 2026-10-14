"""team_tag with RandomBot anchors (SP3 spec block 9; T6.3 calibration, T6.7 acceptance).

For every (anchors share, draw reward, seed) the script trains ``configs/examples/team_tag.yaml``
with ``colosseum train`` (2 workers, the config's budget), then plays the newest checkpoint greedily
as a whole team against a team of uniformly random legal players (200 matches, the trained team is
team ``m % 2``), exactly like the slow test. The shares are set with ``--set``:
``matchmaking.opponents.anchors = a`` and ``matchmaking.opponents.latest = 0.8 - a`` (``snapshots``
stays 0.2, ``rivals`` 0). ``--draw-reward`` (the spec's fallback) sets the env's ``draw_reward``.

One JSON line per run is appended to ``--jsonl`` (win / draw / loss rates, mean episode length,
training seconds, env steps per second); finished (anchors, draw_reward, seed) rows are skipped on a
rerun, so an interrupted measurement continues where it stopped.

Decision rules (T6.3, fixed before the measurement):
- ``choose_anchor_share``: a share passes if its minimum win rate over ``min_seeds`` seeds is
  >= ``threshold`` (0.80); the passing share with the highest minimum is chosen, and a smaller
  share within ``tie`` (0.02) of that minimum wins the tie;
- ``choose_draw_reward`` (fallback): the same rule over draw rewards at one anchors share; the
  mildest penalty (closest to 0) wins the tie.

Usage (the evaluation imports the test kit, so ``tests/`` and ``tests/learning`` go on ``sys.path``)::

    .venv/bin/python scripts/team_tag_anchors.py --anchors 0.1 0.2 0.3 --seeds 0 1 2 3 4 5 \\
        --jsonl docs/benchmarks/team-tag-anchors.jsonl --run-dir /tmp/colosseum-team-tag
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tests", REPO_ROOT / "tests" / "learning"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CONFIG = REPO_ROOT / "configs" / "examples" / "team_tag.yaml"
THRESHOLD = 0.80
TIE = 0.02
MIN_SEEDS = 6
LATEST_PLUS_ANCHORS = 0.8
EVAL_MATCHES = 200


def wdl(results, agent: str) -> tuple[float, float, float]:
    """Win / draw / loss rates of ``agent``'s team (two-team matches)."""
    from demo_learning import team_of

    w = d = losses = 0
    for r in results:
        ranks = {t.team: t.rank for t in r.teams}
        mine = team_of(r, agent)
        (other,) = set(ranks) - {mine}
        if ranks[mine] < ranks[other]:
            w += 1
        elif ranks[mine] == ranks[other]:
            d += 1
        else:
            losses += 1
    n = len(results)
    return w / n, d / n, losses / n


def _choose(groups: dict[float, list[float]], *, threshold: float, tie: float, min_seeds: int,
            prefer_small: bool) -> float | None:
    passing = {key: min(wins) for key, wins in groups.items() if len(wins) >= min_seeds and min(wins) >= threshold}
    if not passing:
        return None
    best = max(passing.values())
    near = [key for key, low in passing.items() if low >= best - tie]
    return min(near) if prefer_small else max(near)


def _detail(name: str, groups: dict[float, list[float]]) -> list[str]:
    return [f"{name} {key}: min {min(v):.3f}, mean {sum(v) / len(v):.3f}, n {len(v)}"
            for key, v in sorted(groups.items())]


def choose_anchor_share(rows: Sequence[dict], *, draw_reward: float | None = None, threshold: float = THRESHOLD,
                        tie: float = TIE, min_seeds: int = MIN_SEEDS) -> tuple[float | None, list[str]]:
    """The T6.3 rule over anchors shares (module docstring); rows of other draw rewards are ignored."""
    groups: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("draw_reward") == draw_reward and "win" in r:
            groups[float(r["anchors"])].append(float(r["win"]))
    return (_choose(groups, threshold=threshold, tie=tie, min_seeds=min_seeds, prefer_small=True),
            _detail("anchors", groups))


def choose_draw_reward(rows: Sequence[dict], *, anchors: float, threshold: float = THRESHOLD, tie: float = TIE,
                       min_seeds: int = MIN_SEEDS) -> tuple[float | None, list[str]]:
    """The fallback rule over draw rewards at one anchors share; the mildest penalty wins the tie."""
    groups: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("draw_reward") is not None and float(r["anchors"]) == anchors and "win" in r:
            groups[float(r["draw_reward"])].append(float(r["win"]))
    return (_choose(groups, threshold=threshold, tie=tie, min_seeds=min_seeds, prefer_small=False),
            _detail("draw_reward", groups))


def train_and_eval(anchors: float, draw_reward: float | None, seed: int, run_dir: Path) -> dict:
    import yaml

    from colosseum.core.registry import env_spec
    from demo_learning import greedy_agent, play, random_model, team_lineups, train_example

    sets: dict = {"rollout.num_workers": 2, "training.seed": seed, "matchmaking.opponents.anchors": anchors,
                  "matchmaking.opponents.latest": round(LATEST_PLUS_ANCHORS - anchors, 6)}
    if draw_reward is not None:
        kwargs = dict(yaml.safe_load(CONFIG.read_text())["env"]["kwargs"])
        kwargs["draw_reward"] = draw_reward
        sets["env.kwargs"] = json.dumps(kwargs)
    parent = run_dir / f"a{anchors}-d{draw_reward}-s{seed}"
    parent.mkdir(parents=True, exist_ok=True)
    run = train_example("team_tag", parent, sets, timeout=1800)
    trained, role = greedy_agent(run, "agent_0")
    spec = env_spec(run.config)
    results = play(run.env_fn(), {"trained": trained, "random": random_model(role)},
                   team_lineups(spec.teams("2v2"), "2v2", "trained", "random", EVAL_MATCHES))
    w, d, losses = wdl(results, "trained")
    return {"anchors": anchors, "draw_reward": draw_reward, "seed": seed, "win": w, "draw": d, "loss": losses,
            "mean_len": sum(r.episode_length for r in results) / len(results), "train_sec": round(run.elapsed, 1),
            "env_steps_per_s": round(run.env_steps_per_sec())}


def _done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(r["anchors"], r["draw_reward"], r["seed"]) for r in rows if "win" in r}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anchors", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--draw-reward", type=float, nargs="+", default=None)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/colosseum-team-tag"))
    args = parser.parse_args()
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = _done(args.jsonl)
    failures = 0
    for draw_reward in args.draw_reward or [None]:
        for anchors in args.anchors:
            for seed in args.seeds:
                if (anchors, draw_reward, seed) in done:
                    continue
                start = time.monotonic()
                try:
                    row = train_and_eval(anchors, draw_reward, seed, args.run_dir)
                except Exception as e:  # noqa: BLE001 - keep measuring; the row records the failure
                    row = {"anchors": anchors, "draw_reward": draw_reward, "seed": seed,
                           "error": f"{type(e).__name__}: {str(e)[-1500:]}"}
                    failures += 1
                row["wall_sec"] = round(time.monotonic() - start, 1)
                with args.jsonl.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    for draw_reward in args.draw_reward or [None]:
        choice, detail = choose_anchor_share(rows, draw_reward=draw_reward)
        print(f"draw_reward {draw_reward}: anchors share by the rule: {choice}")
        for line in detail:
            print("  " + line)
    if args.draw_reward:
        for anchors in args.anchors:
            choice, detail = choose_draw_reward(rows, anchors=anchors)
            print(f"anchors {anchors}: draw reward by the rule: {choice}")
            for line in detail:
                print("  " + line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
