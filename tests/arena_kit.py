"""SP4 selection helpers for measurements and slow tests (T8.2, T8.3), driving the real CLI.

Ladder (criterion 3.3): scripted bots of known strength - one bot with a share ``epsilon`` of random moves
(``EPSILONS``; larger is weaker against any fixed field, measured in docs/benchmarks.md) - on two demo games:
tic_tac_toe (1v1, layout ``2p``) and tron (FFA, layout ``3p``). The field is the config's ``random``
(``RandomBot``) and ``ref`` (the same bot with ``FIELD_EPSILON``), never a candidate.

Pulse pipeline (criterion 3.4): ``unit_harvest_league`` trained from scratch with pulse evaluation, the
rating trend of its snapshots, ``tournament --mode select`` over the run's snapshots with the anchors as
the field, and ``eval`` of the winner against ``random``.

Report shape: ``arena.tournament.modes.report_view`` of the tournament's ``report.json`` (contract amendment
A26): ``layouts.<layout>.groups.<group>`` with ``candidates`` (rows: ``name``, ``score``, ``score_ci``, ``bt``,
``bt_ci``, ``games``, ...), ``ranking`` (final order, best first) and ``select`` (``winner``, ``confirmed``,
``indistinguishable``).
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import pipeline_kit as kit
from cli_runner import REPO_ROOT, TrainRun

EPSILONS = (0.0, 0.25, 0.5, 1.0)
FIELD_EPSILON = 0.1
FIELD = ("random", "ref")
RANDOM_BOT_CLASS = "colosseum.players.RandomBot"


def candidate_name(epsilon: float) -> str:
    """0.25 -> ``e025``."""
    return f"e{round(epsilon * 100):03d}"


TRUE_ORDER = tuple(candidate_name(e) for e in EPSILONS)     # strongest first


@dataclass(frozen=True)
class LadderGame:
    name: str
    config: str            # configs/examples/<config>.yaml
    bot_class: str         # ScriptedBot with an ``epsilon`` kwarg
    layout: str


TIC_TAC_TOE = LadderGame("tic_tac_toe", "tic_tac_toe", "examples.tic_tac_toe.ladder.EpsilonTicTacToeBot", "2p")
TRON = LadderGame("tron", "tron", "examples.tron.bots.FloodFillTronBot", "3p")
LADDER_GAMES = (TIC_TAC_TOE, TRON)


def ladder_config(game: LadderGame) -> dict:
    """The example config with the ladder agents only (candidates ``e000`` ... ``e100``, field ``random``, ``ref``)."""
    raw = yaml.safe_load((REPO_ROOT / "configs" / "examples" / f"{game.config}.yaml").read_text())
    agents: dict[str, Any] = {candidate_name(e): {"kind": "scripted", "class": game.bot_class, "kwargs": {"epsilon": e}}
                              for e in EPSILONS}
    agents["random"] = {"kind": "scripted", "class": RANDOM_BOT_CLASS}
    agents["ref"] = {"kind": "scripted", "class": game.bot_class, "kwargs": {"epsilon": FIELD_EPSILON}}
    raw["agents"] = agents
    return raw


def find_report(out: Path) -> dict:
    """The tournament's ``report.json`` under ``out`` (``--output`` dir or ``<run>/tournaments/<name>``), as
    ``report_view`` (contract amendment A26); exactly one must exist (a second tournament in the same dir would
    make the choice arbitrary)."""
    from colosseum.arena.tournament.modes import report_view

    found = sorted(Path(out).rglob("report.json"))
    assert len(found) == 1, f"expected exactly one report.json under {out}, found {[str(p) for p in found]}"
    return report_view(json.loads(found[0].read_text()))


def group_of(report: Mapping, layout: str) -> dict:
    """The only role group of ``layout`` (the ladders are symmetric games)."""
    groups = report["layouts"][layout]["groups"]
    assert len(groups) == 1, f"expected one role group in {layout}, got {list(groups)}"
    return next(iter(groups.values()))


@dataclass
class SelectRun:
    report: dict
    group: dict
    seconds: float
    out: Path

    @property
    def ranking(self) -> list[str]:
        return list(self.group["ranking"])

    @property
    def winner(self) -> str:
        return self.group["select"]["winner"]

    @property
    def confirmed(self) -> bool:
        return bool(self.group["select"]["confirmed"])

    @property
    def rows(self) -> dict[str, dict]:
        return {row["name"]: row for row in self.group["candidates"]}


def run_select(game: LadderGame, workdir: Path, *, seed: int, blocks: int | None = None,
               confirm_blocks: int | None = None, num_workers: int = 2, num_envs: int = 8,
               timeout: float = 900.0, in_process: bool = False) -> SelectRun:
    """``colosseum tournament --mode select`` of the ladder candidates against the field (defaults unless
    ``blocks`` / ``confirm_blocks`` are given; ``blocks`` is the first halving round's ``--halving-blocks``,
    pre-flight ruling P11), into a fresh ``--output`` dir under ``workdir``."""
    workdir = Path(workdir)
    config = kit.write_config(workdir / f"ladder-{game.name}.yaml", ladder_config(game))
    out = workdir / f"select-{game.name}-s{seed}-b{blocks}-c{confirm_blocks}"
    assert not out.exists(), f"{out} exists: run_select needs a fresh --output dir (use another work dir)"
    args: list[Any] = ["tournament", "-c", config, "--mode", "select", "--layout", game.layout,
                       "--field", ",".join(FIELD), "--seed", seed, "--num-workers", num_workers,
                       "--num-envs", num_envs, "--output", out]
    for name in TRUE_ORDER:
        args += ["-a", name]
    if blocks is not None:
        args += ["--halving-blocks", blocks]          # select: the first halving round's blocks per composition
    if confirm_blocks is not None:
        args += ["--confirm-blocks", confirm_blocks]
    start = time.monotonic()
    result = kit.cli(args, timeout=timeout, in_process=in_process)
    seconds = time.monotonic() - start
    assert result.code == 0, f"tournament failed ({result.code}):\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    report = find_report(out)
    return SelectRun(report, group_of(report, game.layout), seconds, out)


def ladder_problems(run: SelectRun) -> list[str]:
    """Criterion 3.3 checks (empty = pass): true order, the strongest selected and confirmed, BT intervals
    of neighbours not contradicting the order (the stronger's interval is not entirely below the weaker's)."""
    problems = []
    if run.ranking != list(TRUE_ORDER):
        problems.append(f"ranking {run.ranking} != true order {list(TRUE_ORDER)}")
    if run.winner != TRUE_ORDER[0]:
        problems.append(f"winner {run.winner} != {TRUE_ORDER[0]}")
    if not run.confirmed:
        problems.append(f"not confirmed; indistinguishable {run.group['select'].get('indistinguishable')}")
    rows = run.rows
    for stronger, weaker in zip(TRUE_ORDER, TRUE_ORDER[1:]):
        hi_s, lo_w = rows[stronger]["bt_ci"][1], rows[weaker]["bt_ci"][0]
        if hi_s < lo_w:
            problems.append(f"BT intervals reverse the order: {stronger} {rows[stronger]['bt_ci']} is below "
                            f"{weaker} {rows[weaker]['bt_ci']}")
    return problems


# ---------------- pulse pipeline (criterion 3.4) ----------------

@dataclass(frozen=True)
class PulseSettings:
    every: int                       # evaluation.every (T8.2 ruling)
    blocks: int                      # evaluation.blocks (T8.2 ruling)
    num_workers: int = 1             # rollout workers: + the pulse process = 2 processes (test rule)
    total_timesteps: int | None = None   # None: the config's budget


@dataclass
class Trend:
    spearman: float                  # rank correlation of BT rating with version
    gain: float                      # mean rating of the newest third minus the oldest third (Elo units)
    points: list[tuple[int, float]] = field(default_factory=list)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rho (average ranks for ties); nan with fewer than 3 points or a constant input."""
    if len(x) < 3:
        return float("nan")

    def ranks(v):
        v = np.asarray(v, dtype=np.float64)
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=np.float64)
        for value in np.unique(v):
            tie = v == value
            r[tie] = r[tie].mean()
        return r

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def rating_trend(report_data: Mapping, *, layout: str = "2p", agent: str = kit.MAIN) -> Trend:
    """BT rating by version of ``agent``'s snapshots in ``layout`` (report data, ratings section; series points
    are ``[version, rating, ci_low, ci_high, games]``, unrated points skipped)."""
    groups = report_data["sections"]["ratings"]["layouts"][layout]["groups"]
    points = sorted({(int(p[0]), float(p[1])) for g in groups for s in g["series"] if s["lineage"] == agent
                     for p in s["points"] if p[1] is not None})
    versions, ratings = [p[0] for p in points], [p[1] for p in points]
    third = max(1, len(points) // 3)
    gain = float(np.mean(ratings[-third:]) - np.mean(ratings[:third])) if points else float("nan")
    return Trend(spearman(versions, ratings), gain, points)


@dataclass
class PulseRun:
    run: TrainRun
    config: Path
    data: dict                       # build_report_data of the run
    trend: Trend
    pulse_versions: list[int]        # snapshots with pulse matches in the journal
    stored_versions: list[int]       # snapshot dirs at the end
    seconds: dict[str, float]


def train_with_pulse(workdir: Path, *, seed: int, settings: PulseSettings, name: str = "pulse") -> PulseRun:
    """``unit_harvest_league`` from scratch (no init, no kickstart) with pulse evaluation against the anchors."""
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
    from colosseum.arena.report.data import build_report_data

    workdir = Path(workdir)
    config = kit.write_config(workdir / f"{name}.yaml", kit.pipeline_config(kit.UNIT_HARVEST))
    sets: dict[str, Any] = {"training.seed": seed, "rollout.num_workers": settings.num_workers,
                            "evaluation.enabled": True, "evaluation.every": settings.every,
                            "evaluation.blocks": settings.blocks}
    if settings.total_timesteps is not None:
        sets["training.total_timesteps"] = settings.total_timesteps
    start = time.monotonic()
    run = kit.train(config, workdir, name, sets, timeout=1500.0)
    seconds = {"train": time.monotonic() - start}
    start = time.monotonic()
    data = build_report_data(run.root)
    seconds["report_data"] = time.monotonic() - start
    reader = MatchLog.open_readonly(run.root / MATCH_LOG_FILE)
    try:
        pulse = sorted({v for row in reader.matches(source="pulse", limit=None)
                        for seat in row.to_dict()["seats"] if seat["agent_id"] == kit.MAIN
                        and (v := seat["rating_version"]) is not None})
    finally:
        reader.close()
    stored = sorted(int(m.group(1)) for d in (run.root / "checkpoints" / kit.MAIN).iterdir()
                    if (m := re.fullmatch(r"ckpt_v(\d+)", d.name)))
    return PulseRun(run, config, data, rating_trend(data), pulse, stored, seconds)


def select_snapshots(run: PulseRun, *, seed: int, name: str = "sel", num_workers: int = 2,
                     timeout: float = 1200.0) -> dict:
    """``tournament --run <run> --agent main --snapshots all --field greedy,random --mode select``; the report
    as ``report_view`` (contract amendment A26)."""
    from colosseum.arena.tournament.modes import report_view

    args = ["tournament", "-c", run.config, "--run", run.run.root, "--agent", kit.MAIN, "--snapshots", "all",
            "--field", f"{kit.UNIT_HARVEST.bot},{kit.RANDOM}", "--mode", "select", "--seed", seed,
            "--num-workers", num_workers, "--name", name]
    result = kit.cli(args, timeout=timeout)
    assert result.code == 0, f"tournament failed ({result.code}):\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    return report_view(json.loads((run.run.root / "tournaments" / name / "report.json").read_text()))
