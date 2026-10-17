"""Synthetic run dirs and report data for the SP4 report and dashboard tests (T7.1-T7.3).

``write_run_dir`` builds a run dir the way a training run leaves it - ``config.resolved.yaml``,
``metrics.jsonl``, ``ratings.json``, ``matches.db`` (a real ``MatchLog``), ``run_state.json``,
``checkpoints/<agent>/ckpt_v<N>/`` - without training: matches between ``main`` (versions every
``INTERVAL``, strength rising with the version), ``greedy`` (strong scripted) and ``random`` are drawn from
known strengths with a fixed seed. Three outcome kinds: ``wdl`` (toy game ``turns``, layout ``2p``),
``rank`` (``ffa``, ``3p``), ``score`` (``solo``, ``solo``).

``sample_report_data`` is a hand-made report data dict covering every section (render tests need no journal).
``write_multi_run_dir`` adds a second layout (``3p``) and a second trainable agent (``second``) to a ``wdl`` run dir
(the live dashboard's filters); ``page_data`` / ``select_groups`` read the data embedded in a report page.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from colosseum.core.types import MatchResult, SeatResult, TeamResult
from game_helpers import make_test_config, scripted_agent

INTERVAL = 20
VERSIONS = tuple(range(0, 201, INTERVAL))
KINDS = {"wdl": ("turns", "2p", 2), "rank": ("ffa", "3p", 3), "score": ("solo", "solo", 1)}
FIXED_STRENGTH = {"greedy": 1.0, "random": 0.0}       # logits


def main_strength(version: int) -> float:
    return -1.0 + 2.5 * version / VERSIONS[-1]


@dataclass
class SyntheticRun:
    root: Path
    kind: str
    layout: str
    results: list[MatchResult]
    run_id: str


def _seat(seat: int, agent: str, version: int | None, *, latest: bool, offset: int = 3) -> SeatResult:
    if agent != "main":
        return SeatResult(seat, "player", seat, agent, "fixed", 0.0)
    if latest:     # latest weights: versions inside the bucket of ``version`` (rating_version = version)
        return SeatResult(seat, "player", seat, agent, "latest", 0.0, version_start=version + offset,
                          version_end=version + offset + 2)
    return SeatResult(seat, "player", seat, agent, f"ckpt_v{version}", 0.0, version_start=version,
                      version_end=version)


def make_results(kind: str, *, run_id: str, per_pair: int = 30, seed: int = 0, source: str = "train",
                 versions: Sequence[int] = VERSIONS) -> list[MatchResult]:
    """Matches of ``main@v`` against ``greedy`` and ``random`` (and, for ``rank``, all three together;
    for ``score``, homogeneous one-seat teams), outcomes drawn from the strengths; uids unique per call."""
    _, layout, teams = KINDS[kind]
    rng = np.random.default_rng(seed)
    results: list[MatchResult] = []

    def add(seats: list[SeatResult], ranks: list[float], scores: list[float]) -> None:
        i = len(results)
        block = f"P/{seats[0].version_start}/{i}" if source == "pulse" else None
        results.append(MatchResult(
            match_id=f"m{i}", layout=layout, outcome_kind="score" if teams == 1 else ("wdl" if teams == 2 else "rank"),
            seats=seats, teams=[TeamResult(t, ranks[t], scores[t]) for t in range(teams)], episode_length=5 + i % 4,
            uid=f"{run_id}/{source}/w0/e{i % 4}/ep{i}", source=source, seed=1000 + i, env_steps=10 * i, block=block))

    for v in versions:
        for k in range(per_pair):
            latest = k % 2 == 1
            if teams == 1:
                for agent in ("main", "greedy", "random"):
                    s = main_strength(v) if agent == "main" else FIXED_STRENGTH[agent]
                    score = float(2.0 + s + rng.normal(0.0, 0.5))
                    add([_seat(0, agent, v, latest=latest)], [1.0], [score])
                continue
            if teams == 2:
                for opp in ("greedy", "random"):
                    order = ["main", opp] if k % 4 < 2 else [opp, "main"]         # both sides
                    seats = [_seat(s, a, v, latest=latest) for s, a in enumerate(order)]
                    sa = main_strength(v) - FIXED_STRENGTH[opp]
                    p_main = 1.0 / (1.0 + math.exp(-sa))
                    draw = rng.random() < 0.1
                    main_wins = rng.random() < p_main
                    main_seat = order.index("main")
                    ranks = [1.0, 1.0] if draw else [0.0, 0.0]
                    if not draw:
                        ranks[main_seat] = 1.0 if main_wins else 2.0
                        ranks[1 - main_seat] = 2.0 if main_wins else 1.0
                    add(seats, ranks, [-r for r in ranks])
                continue
            order = ["main", "greedy", "random"]
            shift = k % 3
            order = order[shift:] + order[:shift]
            seats = [_seat(s, a, v, latest=latest) for s, a in enumerate(order)]
            noisy = [(main_strength(v) if a == "main" else FIXED_STRENGTH[a]) + rng.gumbel() for a in order]
            ranking = np.argsort(-np.asarray(noisy))
            ranks = [0.0] * 3
            for place, seat in enumerate(ranking):
                ranks[int(seat)] = float(place + 1)
            add(seats, ranks, [-r for r in ranks])
        if teams >= 2:          # latest against its own snapshot (self lineage: no lineage-matrix cell)
            for k in range(per_pair // 3):
                prev = max(0, v - INTERVAL)
                seats = [_seat(0, "main", v, latest=True), _seat(1, "main", prev, latest=False)]
                seats += [_seat(s, "random", None, latest=False) for s in range(2, teams)]
                ranks = [1.0, 2.0] + [3.0] * (teams - 2) if k % 2 else [2.0, 1.0] + [3.0] * (teams - 2)
                add(seats, ranks, [-r for r in ranks])
    return results


def _metrics_lines(kind: str, layout: str, *, train_records: int, decreasing_ts: bool) -> list[str]:
    lines = []
    for i in range(train_records):
        ts = 2.0e9 - i if decreasing_ts else 1.0e9 + i      # ts must not order anything (WSL2 clock jumps)
        lines.append({"ts": ts, "kind": "train", "agent": "main", "train_step": 10 * i, "total_loss": 1.0 / (1 + i),
                      "entropy": 1.0 - i / (2 * train_records), "rho_trunc_mass": 0.0, "lr": 1.0e-3})
    for i in range(40):
        lines.append({"ts": 1.0e9, "kind": "system", "env_steps": 1000 * i, "env_steps_per_sec": 900.0 + i,
                      "train_steps_per_sec": {"main": 5.0}, "queue_depths": {"main": 3}, "parked_buffers": 0,
                      "dropped_reward_episodes": 0, "dropped_match_results": 0, "workers_reporting": 1})
        if kind != "score":
            lines.append({"ts": 1.0e9, "kind": "episodes", "agent": "main", "env_steps": 1000 * i, "episodes": 10,
                          "return_mean": 0.1, "length_mean": 6.0, "wdl": {}, "seat_counts": [5, 5], "by_layout": {},
                          "opponents": {layout: {"teams": 10, "categories": {"latest": 0.5, "snapshots": 0.2,
                                                                              "rivals": 0.0, "anchors": 0.3,
                                                                              "fallback": 0.0},
                                                 "anchors": {"greedy": 0.2, "random": 0.1}}}})
    return [json.dumps(line) for line in lines]


def write_run_dir(tmp_path: Path, kind: str = "wdl", *, name: str = "synthetic", journal: bool = True,
                  status: str | None = "finished", pid: int | None = None, live_bt: bool = False,
                  evaluation: bool = False, pulse_versions: Sequence[int] = (), per_pair: int = 30,
                  train_records: int = 201, decreasing_ts: bool = True, partial_last_line: bool = False,
                  stored_versions: Sequence[int] = (160, 180, 200), bootstrap_samples: int = 20,
                  trainable: Sequence[str] = ("main",)) -> SyntheticRun:
    """A finished (or ``status``) run dir of the given outcome ``kind`` under ``tmp_path / name``; ``trainable``: the
    config's trainable agents (only ``main`` plays the journal's matches)."""
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog, RunInfo
    from colosseum.core.run_state import write_run_state

    game, layout, _ = KINDS[kind]
    root = Path(tmp_path) / name
    root.mkdir(parents=True)
    # greedy is not a RandomBot, so the default anchor (the first RandomBot of the config) is ``random``
    agents = {**{aid: {} for aid in trainable}, "greedy": scripted_agent("game_helpers.ConstantBot"),
              "random": scripted_agent()}
    cfg = make_test_config(game, agents=agents, run={"name": name, "dir": str(tmp_path)},
                           checkpoint={"interval": INTERVAL}, ratings={"bootstrap_samples": bootstrap_samples},
                           evaluation={"enabled": evaluation})
    (root / "config.resolved.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json", by_alias=True),
                                                              sort_keys=False))
    run_id = uuid.uuid4().hex
    results = make_results(kind, run_id=run_id, per_pair=per_pair)
    if pulse_versions:
        results += make_results(kind, run_id=run_id, per_pair=4, seed=1, source="pulse", versions=pulse_versions)
    if journal:
        log = MatchLog(root / MATCH_LOG_FILE, RunInfo(run_id=run_id, name=name, parent_run_id=None, game=game,
                                                      config_hash="synthetic", checkpoint_interval=INTERVAL))
        try:
            for i in range(0, len(results), 500):
                log.append(results[i:i + 500])
        finally:
            log.close()
    text = "\n".join(_metrics_lines(kind, layout, train_records=train_records, decreasing_ts=decreasing_ts)) + "\n"
    if partial_last_line:
        text += '{"ts": 1.0, "kind": "train", "agent": "ma'
    (root / "metrics.jsonl").write_text(text)
    table = {"outcome_kind": "score" if kind == "score" else kind,
             "elo": {"main": 1250.0, "greedy": 1230.0, "random": 1010.0},
             "games": {"main": {"greedy": 100, "random": 100}}, "wr_vs_past": {"main": 0.6},
             "past_games": {"main": 50},
             "pfsp": {"main": {"greedy@fixed": {"score": 0.4, "games": 300.0},
                               "main@ckpt_v180": {"score": 0.6, "games": 40.0}}}}
    ratings = {"env_steps": 39000, "layouts": {layout: table}}
    if live_bt:
        # the shape of T2.2's live.bt_section (contract amendment A14): rows per role-group label, '@' extras
        def bt_row(entity, version, rating, lo, hi, games, lcb, p_lin, p_grp, anchor=False):
            return {"entity": entity, "lineage": entity.split("@v")[0], "version": version, "rating": rating,
                    "se": (hi - lo) / 3.92, "ci_low": lo, "ci_high": hi, "lcb": lcb, "games": games, "p_best": p_grp,
                    "p_best_lineage": p_lin, "component": 0, "anchor": anchor}

        ratings["bt"] = {layout: {
            "player": [bt_row("main@v200", 200, 123.0, 100.0, 146.0, 70, 105.0, 0.9, 0.6),
                       bt_row("greedy", None, 80.0, 60.0, 100.0, 700, 64.0, 1.0, 0.4),
                       bt_row("random", None, 0.0, 0.0, 0.0, 700, 0.0, 1.0, 0.0, anchor=True)],
            "@bias": {"0": 12.0, "1": -12.0},
            "@anchors": {"player/0": {"anchor": "random", "members": ["random"], "rule": "random_bot"}},
            "@transitivity": {"flag": False, "g2": 1.2, "dof": 1, "p_value": 0.27, "pairs_tested": 2,
                              "worst_pairs": []},
            "@fit": {"converged": True, "iterations": 5, "approximate": False}}}
    (root / "ratings.json").write_text(json.dumps(ratings))
    for v in stored_versions:
        (root / "checkpoints" / "main" / f"ckpt_v{v}").mkdir(parents=True)
    if status is not None:
        write_run_state(root, status, pid if pid is not None else os.getpid(),
                        exit_code=None if status == "running" else (0 if status == "finished" else 1))
    return SyntheticRun(root, kind, layout, results, run_id)


def as_agent(results: Sequence[MatchResult], agent: str, tag: str) -> list[MatchResult]:
    """Copies of ``results`` with ``main``'s seats played by ``agent``; uids get ``/<tag>`` (still unique)."""
    out = []
    for result in results:
        result = copy.deepcopy(result)
        result.uid = f"{result.uid}/{tag}"
        for seat in result.seats:
            if seat.agent_id == "main":
                seat.agent_id = agent
        out.append(result)
    return out


def append_results(run: SyntheticRun, results: Sequence[MatchResult]) -> None:
    """Append ``results`` to the run dir's journal (under the run's id)."""
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog, RunInfo

    log = MatchLog(run.root / MATCH_LOG_FILE, RunInfo(run_id=run.run_id, name=run.root.name, parent_run_id=None,
                                                      game=KINDS[run.kind][0], config_hash="synthetic",
                                                      checkpoint_interval=INTERVAL))
    try:
        log.append(list(results))
    finally:
        log.close()


def write_multi_run_dir(tmp_path: Path, *, per_pair: int = 8) -> SyntheticRun:
    """A finished ``wdl`` run dir with two layouts (``2p``; ``3p``, a rank layout of main, greedy and random) and
    two trainable agents (``main`` plays both layouts, ``second`` the ``2p`` layout)."""
    run = write_run_dir(tmp_path, "wdl", per_pair=per_pair, trainable=("main", "second"))
    rank = make_results("rank", run_id=run.run_id, per_pair=per_pair, seed=1)
    second = as_agent(make_results("wdl", run_id=run.run_id, per_pair=per_pair, seed=2), "second", "second")
    append_results(run, [*as_agent(rank, "main", "3p"), *second])
    return run


def page_data(page: str) -> dict:
    """The report data embedded in a report page (``render.render_html``)."""
    m = re.search(r'<script type="application/json" id="colosseum-data">(.*?)</script>', page, re.S)
    assert m, "the page embeds no report data"
    return json.loads(m.group(1))


def select_groups(page: str, tournament: str) -> list[dict]:
    """The role groups with a selection (``select`` not null) of the tournament item named ``tournament`` in a
    report page's tournaments section; AssertionError when the section has no item of that name."""
    items = [item for item in page_data(page)["sections"]["tournaments"]["items"] if item["name"] == tournament]
    assert len(items) == 1, f"tournament {tournament!r}: {len(items)} items in the report"
    layouts = items[0]["report"].get("layouts") or {}
    return [g for lay in layouts.values() for g in (lay.get("groups") or {}).values() if g.get("select")]


def sample_report_data() -> dict:
    """A report data dict with every section (wdl, rank and score layouts), for render tests."""
    versions = list(range(0, 2100, 100))
    main_rows = [{"entity": f"main@v{v}", "lineage": "main", "version": v, "kind": "trainable",
                  "rating": -50.0 + 8 * i, "ci": [-90.0 + 8 * i, -10.0 + 8 * i], "games": 50 + 10 * i,
                  "p_best_lineage": 0.05, "p_best_group": 0.04, "lcb": -80.0 + 8 * i} for i, v in enumerate(versions)]
    fixed = [{"entity": "greedy", "lineage": "greedy", "version": None, "kind": "scripted", "rating": 60.0,
              "ci": [40.0, 80.0], "games": 4000, "p_best_lineage": 1.0, "p_best_group": 0.2, "lcb": 45.0},
             {"entity": "random", "lineage": "random", "version": None, "kind": "scripted", "rating": 0.0,
              "ci": [0.0, 0.0], "games": 4000, "p_best_lineage": 1.0, "p_best_group": 0.0, "lcb": 0.0}]

    def cell(s: float, n: int) -> dict:
        return {"score": s, "ci": [max(0.0, s - 0.05), min(1.0, s + 0.05)], "n": n}

    snap_axis = [f"main@v{v}" for v in versions[::4]] + ["greedy", "random"]
    k = len(snap_axis)
    snap_cells = [[None if i == j else cell(0.5 + 0.03 * (j - i), 10 if (i + j) % 5 == 0 else 60) for j in range(k)]
                  for i in range(k)]
    steps = list(range(0, 200001, 4000))
    match_rows = [{"uid": f"r1/train/w0/e{i % 8}/ep{i}", "source": "pulse" if i % 3 == 0 else "train", "layout": "2p",
                   "outcome_kind": "wdl", "episode_length": 50, "seed": 1000 + i, "env_steps": 1000 * i, "block": None,
                   "replay": None,
                   "seats": [{"seat": 0, "team": 0, "role": "player", "agent_id": "main", "network_id": "latest",
                              "version_start": 100 * (i // 10), "version_end": 100 * (i // 10),
                              "rating_version": 100 * (i // 10), "reward": 1.0, "eliminated_step": None},
                             {"seat": 1, "team": 1, "role": "player", "agent_id": "greedy", "network_id": "fixed",
                              "version_start": None, "version_end": None, "rating_version": None, "reward": -1.0,
                              "eliminated_step": None}],
                   "teams": [{"team": 0, "rank": 1 + i % 2, "score": 1.0},
                             {"team": 1, "rank": 2 - i % 2, "score": 0.0}]}
                  for i in range(60)]
    tournament = {
        "name": "select-1", "mode": "select", "seed": 7, "winner": "main@v2000", "confirmed": True,
        "layouts": {"2p": {"outcome_kind": "wdl", "groups": {"default": {
            "candidates": [
                {"name": "main@v2000", "display": "main@v2000", "score": 0.71, "score_ci": [0.66, 0.76],
                 "worst": {"opponent": "greedy", "score": 0.49}, "bt": 120.0, "bt_ci": [90.0, 150.0], "p_best": 0.9,
                 "games": 400, "eliminated_round": None, "repeated": []},
                {"name": "main@v1500", "display": "main@v1500", "score": 0.62, "score_ci": [0.57, 0.67],
                 "worst": {"opponent": "greedy", "score": 0.41}, "bt": 70.0, "bt_ci": [40.0, 100.0], "p_best": 0.1,
                 "games": 400, "eliminated_round": 1, "repeated": []}],
            "ranking": ["main@v2000", "main@v1500"],
            "matrix": {"main@v2000": {"greedy": {"score": 0.49, "ci": [0.42, 0.56], "games": 200},
                                      "random": {"score": 0.93, "ci": [0.90, 0.96], "games": 200}}},
            "rounds": [{"round": 1, "blocks": 16, "survivors": ["main@v2000"], "eliminated": ["main@v1500"]}],
            "select": {"winner": "main@v2000", "confirmed": True, "indistinguishable": [], "diff_ci": [0.03, 0.15],
                       "repeated_columns": [], "diff_repeated": []},
            "transitivity": {"flag": False, "p_value": 0.4}, "repeated_pairs": [], "candidate_weights": {},
            "combined": None}}, "dropped": [], "self_excluded": []}}}
    return {
        "format": 1, "title": "synthetic-run", "generated": "colosseum report",
        "sections": {
            "summary": {"run": "synthetic-run", "path": "runs/synthetic-run", "status": "finished",
                        "status_detail": "exit code 0", "pid": 1234, "exit_code": 0, "env_steps": 160384,
                        "retention": "snapshots: newest 5 (keep_last, with trainer state); the final one of this run; "
                                     "pinned; top-k off; no cap",
                        "total_timesteps": 160000, "budget_frac": 1.0024, "env_steps_per_sec": 1620.5,
                        "agents": [{"agent": "main", "kind": "trainable", "elo": 1250.0, "wr_vs_past": 0.61,
                                    "bt": {"layout": "2p", "entity": "main@v2000", "rating": 110.0,
                                           "ci": [70.0, 150.0]}},
                                   {"agent": "greedy", "kind": "scripted", "elo": 1230.0, "wr_vs_past": None,
                                    "bt": {"layout": "2p", "entity": "greedy", "rating": 60.0, "ci": [40.0, 80.0]}},
                                   {"agent": "random", "kind": "scripted", "bt": None, "elo": 1010.0,
                                    "wr_vs_past": None}],
                        "parents": ["synthetic-parent"], "notes": []},
            "ratings": {"source": "journal", "uncertainty": "bootstrap", "env_steps": 160384, "notes": [], "layouts": {
                "2p": {"outcome_kind": "wdl", "kind": "bt", "bias": {"0": 12.0, "1": -12.0},
                       "transitivity": {"flag": True, "g2": 40.2, "dof": 12, "p_value": 0.0001,
                                        "worst_pairs": [{"a": "main@v300", "b": "greedy", "residual": 3.1,
                                                         "games": 120}]},
                       "groups": [{"label": "default", "component": 0, "anchor": "random",
                                   "entities": main_rows + fixed, "truncated": 0,
                                   "series": [{"lineage": "main", "points": [[r["version"], r["rating"], r["ci"][0],
                                                                              r["ci"][1], r["games"]]
                                                                             for r in main_rows]}],
                                   "anchors": [{"entity": f["entity"], "rating": f["rating"], "ci": f["ci"]}
                                               for f in fixed]}]},
                "4p": {"outcome_kind": "rank", "kind": "bt", "bias": {}, "transitivity": None,
                       "groups": [{"label": "default", "component": 0, "anchor": "random", "entities": fixed,
                                   "truncated": 3, "series": [], "anchors": []}]},
                "solo": {"outcome_kind": "score", "kind": "score",
                         "entities": [{"entity": "main@v100", "lineage": "main", "version": 100, "kind": "trainable",
                                       "n": 40, "mean": 3.2, "ci": [2.9, 3.5]}],
                         "truncated": 0, "compositions_truncated": 0,
                         "compositions": [{"composition": "main@v100+random", "n": 20, "mean": 2.1, "ci": [1.8, 2.4],
                                           "homogeneous": False}]}}},
            "matrix": {"min_games": 20, "notes": [], "layouts": {
                "2p": {"kind": "bt",
                       "lineages": {"axis": ["main", "greedy", "random"],
                                    "cells": [[None, cell(0.48, 900), cell(0.97, 900)],
                                              [cell(0.52, 900), None, cell(0.99, 12)],
                                              [cell(0.03, 900), cell(0.01, 12), None]]},
                       "snapshots": {"agent": "main", "axis": snap_axis, "cells": snap_cells, "thinned_from": 23}},
                "solo": {"kind": "score", "truncated": 0,
                         "cross_play": [{"composition": "main@v100+random", "n": 20, "mean": 2.1, "ci": [1.8, 2.4],
                                         "homogeneous": False}]}}},
            "anchors": {"source": "train", "biased": True, "notes": [], "unevaluated": {}, "layouts": {"2p": {"main": {
                "metric": "pairwise", "unevaluated": [],
                "anchors": {"greedy": [[v, 0.3 + 0.01 * i, 0.25 + 0.01 * i, 0.35 + 0.01 * i, 40]
                                       for i, v in enumerate(versions)],
                            "random": [[v, 0.9, 0.85, 0.95, 40] for v in versions]}}}}},
            "opponents": {"notes": [], "agents": {"main": {"2p": {
                "env_steps": steps,
                "categories": {"latest": [0.5] * len(steps), "snapshots": [0.2] * len(steps),
                               "rivals": [0.0] * len(steps), "anchors": [0.3] * len(steps),
                               "fallback": [0.0] * len(steps)},
                "anchors": {"greedy": [0.15] * len(steps), "random": [0.15] * len(steps)}}}},
                "pfsp": {"2p": {"main": [{"player": "greedy@fixed", "score": 0.48, "games": 900.0},
                                         {"player": "main@ckpt_v300", "score": 0.7, "games": 120.0}]}}},
            "training": {"notes": [], "default": ["total_loss", "entropy", "rho_trunc_mass"], "agents": {"main": {
                "train_step": list(range(0, 1000, 10)),
                "metrics": {"total_loss": [1.0 / (1 + i) for i in range(100)],
                            "entropy": [1.5 - 0.005 * i for i in range(100)],
                            "rho_trunc_mass": [0.001 * (i % 7) for i in range(100)], "lr": [1.0e-3] * 100}}}},
            "system": {"notes": [], "env_steps": steps, "default": ["env_steps_per_sec", "dropped_match_results"],
                       "series": {"env_steps_per_sec": [1600.0 + (i % 5) * 10 for i in range(len(steps))],
                                  "dropped_match_results": [0.0] * len(steps),
                                  "queue_depths.main": [4.0] * len(steps)},
                       "latest": {"env_steps_per_sec": 1620.0, "dropped_match_results": 0.0,
                                  "dropped_reward_episodes": 0.0},
                       "free_disk_mb": 120000.0,
                       "journal": {"file_mb": 12.5, "wal_mb": 0.4, "counts_rows": 1200, "counts_only": False}},
            "matches": {"limit": 1000, "filters": {}, "notes": [], "rows": match_rows},
            "tournaments": {"notes": [], "items": [{"name": "select-1", "path": "tournaments/select-1",
                                                   "report": tournament}]},
        },
    }
