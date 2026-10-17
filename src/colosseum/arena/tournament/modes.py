"""Tournament modes (SP4 spec block 3): gauntlet, round-robin, halving, select; the report.

Every mode builds blocks (``schedule``), asks ``play(blocks)`` for their results (the tournament runner: the
journal for blocks already written whole, worker processes for the rest; tests pass a fake) and turns them
into tables (``stats``):
- ``gauntlet``: every candidate plays ``blocks`` blocks per composition on stage A; tables per layout and role
  group, plus a layout-weighted combination per group when there are several layouts of one kind;
- ``round-robin``: every pair of participants (candidates and field) plays ``blocks`` blocks per pair on stage
  A (two-team layouts only, ruling C4; participants of one role group: mixed groups are a ConfigError); a score
  matrix with intervals (a repeated pair counts as one block), ``score: null`` without a playable pair;
- ``halving``: successive halving on stage A, per role group. Round ``r`` plays ``halving_blocks * 2**r`` new
  blocks per composition for the active candidates (indices continue, so every round has fresh seeds and
  every active candidate the same ones); ranking by the combined score over all blocks played so far; the
  lower ``floor(n / 2)`` leave, but never below ``finalists``; rounds end when ``finalists`` are left (one round
  when there are no more than ``finalists`` to begin with; ruling D6);
- ``select``: ``halving`` (stage A) to ``finalists``, then a gauntlet of the finalists with ``confirm_blocks``
  per composition on fresh seeds (stage B). The winner per group is the best stage B score; ``confirmed`` by
  the paired stratified bootstrap of stage B only (``stats.select_winner``). Stage A games never enter stage B
  numbers (winner's curse); the winner's own stage B score is still biased up by the selection among the
  finalists (the report says so), the interval of the difference is not.

Several layouts: per-layout tables, and a combination with ``layout_weights`` (``matchmaking.layouts``; the CLI
fills in defaults) computed on the same bootstrap replicates as the per-layout tables. ``score`` layouts (one
team) use the mean team score; ``halving`` and ``select`` refuse to mix them with pairwise layouts (ConfigError:
choose ``--layout``), ``gauntlet`` keeps both kinds per layout and never averages them. The BT rating of every
pairwise layout is fitted on the tournament's own matches (``bt_section``; ruling D9).

What the schedule could not seat (``schedule.DroppedComposition``) is listed per layout in the report
(``layouts[i]["dropped"]``); a candidate that is also a field member plays the field without itself
(``flags["self_excluded"]``; the weights its score uses are in the table's ``candidate_weights``). Repeated
deterministic games (``stats``) are marked next to every interval they enter: a row's ``repeated`` columns, a
combined row's ``repeated`` ``[layout, column]`` pairs, a selection's ``diff_repeated``. The builders' INFO
lines about dropped compositions and self-excluded candidates are logged once per tournament, not once per
halving round. ``select`` with ``finalists=1`` logs one WARNING: nothing can be confirmed.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from colosseum.arena.tournament import schedule as _schedule
from colosseum.arena.tournament.schedule import (
    Block,
    DroppedComposition,
    check_role_groups,
    gauntlet_blocks,
    round_robin_blocks,
    score_blocks,
)
from colosseum.arena.tournament.stats import (
    PAIRED_BOOTSTRAP_SAMPLES,
    SELF,
    CandidateTable,
    block_values,
    candidate_scores,
    cell_interval,
    combine,
    quantiles,
    repeated_columns,
    select_winner,
)
from colosseum.core.errors import ConfigError
from colosseum.core.types import MatchResult
from colosseum.envs.game import GameSpec
from colosseum.eval import wilson_interval

logger = logging.getLogger(__name__)

MODES = ("gauntlet", "round-robin", "halving", "select")
REPORT_FORMAT = 1
# T8.2 ruling (ladder measurement, docs/benchmarks.md «Лесенка ботов и размеры турнира»): on both bot ladders
# (tic_tac_toe 2p, tron 3p; 10 seeds per grid point) the smallest grid point, 8 halving blocks / 32 confirm
# blocks, passed (winner right 10/10, order 9/10, confirmed 10/10, BT neighbours 10/10 on each game), so the
# pre-fixed rule kept the floors: max(8, 8), max(100, 32), max(50, 2 * 8). The floors keep real selections
# (differences far smaller than the ladder's) from defaulting to tiny tournaments.
DEFAULT_BLOCKS = 50            # gauntlet / round-robin: blocks per composition (pair)
DEFAULT_HALVING_BLOCKS = 8     # halving round 0: blocks per composition
DEFAULT_CONFIRM_BLOCKS = 100   # select stage B: blocks per composition
DEFAULT_FINALISTS = 3
WINNER_BIAS_NOTE = ("the winner's stage B score is biased up by the selection among the finalists; the interval "
                    "of its difference to the runner-up is not")
LONE_FINALIST_NOTE = "a single finalist: there is no difference to confirm"

PlayFn = Callable[[Sequence[Block]], Mapping[str, Sequence[MatchResult]]]
BTSectionFn = Callable[[str, Sequence[MatchResult], Sequence[Block]], "dict | None"]


@dataclass(frozen=True)
class TournamentSettings:
    name: str                              # the tournament id: block-id prefix and journal key
    layouts: list[str]
    layout_weights: dict[str, float]
    blocks: int = DEFAULT_BLOCKS
    halving_blocks: int = DEFAULT_HALVING_BLOCKS
    confirm_blocks: int = DEFAULT_CONFIRM_BLOCKS
    finalists: int = DEFAULT_FINALISTS
    seed: int = 0
    deterministic: bool = False
    bootstrap_samples: int = PAIRED_BOOTSTRAP_SAMPLES
    level: float = 0.95


@dataclass
class TournamentReport:
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.data

    def text(self) -> str:
        return report_text(self.data)


class _OnceFilter(logging.Filter):
    """Lets every distinct message of the schedule logger through once (halving calls the builders per round)."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if message in self.seen:
            return False
        self.seen.add(message)
        return True


def _player_row(players: Mapping[str, Any], name: str) -> dict[str, Any]:
    p = players[name]
    return {"name": name, "display": p.display, "kind": p.kind, "lineage": p.lineage, "roles": list(p.roles),
            "source": p.source.origin}


class _Context:
    def __init__(self, spec: GameSpec, players: Mapping[str, Any], candidates: Sequence[str], field_: Sequence[str],
                 settings: TournamentSettings, play: PlayFn) -> None:
        self.spec, self.players, self.settings, self.play = spec, players, settings, play
        self.candidates, self.field = list(candidates), list(field_)
        self.names = {(p.agent_id, p.network_id): n for n, p in players.items()}
        self.lineage = {n: p.lineage for n, p in players.items()}
        self.metric_of = {L: ("score" if spec.num_teams(L) == 1 else "pairwise") for L in settings.layouts}
        self.blocks: list[Block] = []
        self.results: dict[str, list[MatchResult]] = {}
        self.dropped: dict[tuple[str, str, tuple[str, ...]], DroppedComposition] = {}
        self.rng = np.random.default_rng(np.random.SeedSequence([int(settings.seed) % 2**63, 0x7EB0]))

    def run(self, blocks: Sequence[Block]) -> None:
        blocks = [b for b in blocks if b.id not in self.results]
        if not blocks:
            return
        got = self.play(blocks)
        for b in blocks:
            if b.id not in got:
                raise ValueError(f"tournament: the results source returned nothing for block {b.id}")
            self.results[b.id] = list(got[b.id])
        self.blocks.extend(blocks)

    def _note_dropped(self, dropped: Sequence[DroppedComposition]) -> None:
        for d in dropped:
            self.dropped.setdefault((d.layout, d.candidate, d.composition), d)

    def schedule(self, layout: str, candidates: Sequence[str], *, blocks: int, stage: str,
                 first_index: int = 0) -> list[Block]:
        dropped: list[DroppedComposition] = []
        common = dict(blocks=blocks, seed=self.settings.seed, stage=stage, first_index=first_index,
                      prefix=self.settings.name, dropped=dropped)
        if self.metric_of[layout] == "score":
            out = score_blocks(self.spec, layout, candidates, self.field, self.players, **common)
        else:
            out = gauntlet_blocks(self.spec, layout, candidates, self.field, self.players, **common)
        self._note_dropped(dropped)
        return out

    def round_robin(self, layout: str, participants: Sequence[str]) -> list[Block]:
        dropped: list[DroppedComposition] = []
        out = round_robin_blocks(self.spec, layout, participants, self.players, blocks=self.settings.blocks,
                                 seed=self.settings.seed, stage="A", prefix=self.settings.name, dropped=dropped)
        self._note_dropped(dropped)
        return out

    def tables(self, blocks: Sequence[Block], groups: Mapping[tuple[str, str], str]) -> dict:
        return candidate_scores(blocks, self.results, names=self.names, lineage=self.lineage, group_of=groups,
                                metric_of=self.metric_of, deterministic=self.settings.deterministic)


def _groups(ctx: _Context) -> tuple[dict[str, list[str]], dict[tuple[str, str], str]]:
    """``{group: candidates}`` over all layouts and ``{(layout, candidate): group}``; a candidate must be in one
    group label in every layout (ConfigError otherwise)."""
    group_of: dict[tuple[str, str], str] = {}
    members: dict[str, list[str]] = {}
    for layout in ctx.settings.layouts:
        for label, names in check_role_groups(ctx.spec, layout, ctx.candidates, ctx.players).items():
            for name in names:
                group_of[(layout, name)] = label
    for name in ctx.candidates:
        labels = {group_of[(L, name)] for L in ctx.settings.layouts}
        if len(labels) > 1:
            raise ConfigError(f"candidate {name!r} plays different role groups in the layouts {ctx.settings.layouts} "
                              f"({sorted(labels)}); choose layouts with --layout")
        members.setdefault(labels.pop(), []).append(name)
    return members, group_of


def _table_dict(ctx: _Context, table: CandidateTable, reps: np.ndarray) -> dict[str, Any]:
    """The per-layout table of the report schema; ``reps`` are the table's cell replicates ``[R, C, L]``."""
    score_reps = table.scores_of(reps)
    cells, scores = table.cells(), table.scores()
    rows = []
    for i, c in enumerate(table.candidates):
        cols = {col: float(cells[i, j]) for j, col in enumerate(table.columns) if table.has_cell(c, col)}
        opponents = {col: v for col, v in cols.items() if col != SELF}
        worst = min(opponents, key=lambda col: (opponents[col], col)) if opponents else None
        rows.append({
            "name": c, "score": float(scores[i]), "ci95": list(quantiles(score_reps[:, i], ctx.settings.level)),
            "worst": None if worst is None else {"opponent": worst, "score": opponents[worst]},
            "games": int(table.matches.get(c, 0)),
            "blocks": table.blocks_played(c),
            "repeated": sorted(col for (cand, col) in table.repeated if cand == c),
        })
    rows.sort(key=lambda r: (-r["score"], table.candidates.index(r["name"])))
    matrix = {c: {col: {"score": float(cells[i, j]),
                        "ci95": _interval(cell_interval(table, c, col, reps, ctx.settings.level)),
                        "games": int(table.games[c][col]), "blocks": table.blocks_of(col, c),
                        "repeated": (c, col) in table.repeated}
                  for j, col in enumerate(table.columns) if table.has_cell(c, col)}
              for i, c in enumerate(table.candidates)}
    weights, used = table.weights, table.candidate_weights
    return {"stage": table.stage, "layout": table.layout, "group": table.group, "metric": table.metric,
            "columns": list(table.columns), "column_weights": [float(w) for w in weights],
            # the weights a candidate's score actually uses, where they differ (a candidate in the field)
            "candidate_weights": {c: [float(w) for w in used[i]] for i, c in enumerate(table.candidates)
                                  if not np.allclose(used[i], weights)},
            "rows": rows, "matrix": matrix}


def _interval(ci: tuple[float, float] | None) -> list[float] | None:
    return None if ci is None else [float(ci[0]), float(ci[1])]


def _combined_dict(stage: str, group: str, combined, level: float,
                   tables: Sequence[CandidateTable]) -> dict[str, Any]:
    rows = [{"name": c, "score": float(combined.point[i]), "ci95": list(quantiles(combined.replicates[:, i], level)),
             "repeated": [list(x) for x in repeated_columns(tables, c)]}
            for i, c in enumerate(combined.candidates)]
    rows.sort(key=lambda r: (-r["score"], combined.candidates.index(r["name"])))
    return {"stage": stage, "layout": "*", "group": group, "rows": rows}


def _stage_tables(ctx: _Context, stage: str, names: Sequence[str], group: str, group_of) -> list[CandidateTable]:
    blocks = [b for b in ctx.blocks if b.stage == stage and b.candidate in names]
    tables = ctx.tables(blocks, group_of)
    return [tables[(stage, L, group)] for L in ctx.settings.layouts if (stage, L, group) in tables]


def _halving(ctx: _Context, group: str, names: list[str], group_of, rounds: list[dict]) -> list[str]:
    s = ctx.settings
    active, offset, size, r = list(names), 0, s.halving_blocks, 0
    while True:
        for layout in s.layouts:
            ctx.run(ctx.schedule(layout, active, blocks=size, stage="A", first_index=offset))
        tables = _stage_tables(ctx, "A", active, group, group_of)
        comb = combine(tables, s.layout_weights, samples=0, rng=ctx.rng)
        # by name: comb.candidates follow the blocks (round 0's input order), ``active`` is ranked
        point = {name: float(v) for name, v in zip(comb.candidates, comb.point)}
        missing = [n for n in active if n not in point]
        if missing:
            raise ValueError(f"halving round {r}: no stage A score for {missing}")
        ranked = sorted(active, key=lambda n: (-point[n], names.index(n)))
        keep = len(active) if len(active) <= s.finalists else max(s.finalists, len(active) - len(active) // 2)
        rounds.append({"group": group, "round": r, "stage": "A", "blocks_per_composition": size,
                       "first_index": offset, "active": list(ranked),
                       "scores": {n: point[n] for n in ranked},
                       "eliminated": ranked[keep:]})
        active = ranked[:keep]
        offset, size, r = offset + size, size * 2, r + 1
        if len(active) <= s.finalists:
            return active


def _check_settings(mode: str, spec: GameSpec, settings: TournamentSettings) -> None:
    if mode not in MODES:
        raise ConfigError(f"--mode {mode!r}: expected one of {list(MODES)}")
    if not settings.layouts:
        raise ConfigError("the tournament has no layout; choose layouts with --layout")
    unknown = [L for L in settings.layouts if L not in spec.layouts]
    if unknown:
        raise ConfigError(f"layouts {unknown} are not layouts of the game {sorted(spec.layouts)}")
    missing = [L for L in settings.layouts if not float(settings.layout_weights.get(L, 0.0)) > 0]
    if missing:
        raise ConfigError(f"layouts {missing} have no positive weight in {dict(settings.layout_weights)}; give every "
                          f"tournament layout a weight (matchmaking.layouts) or choose layouts with --layout")
    for name, value in (("blocks", settings.blocks), ("halving_blocks", settings.halving_blocks),
                        ("confirm_blocks", settings.confirm_blocks), ("finalists", settings.finalists),
                        ("bootstrap_samples", settings.bootstrap_samples)):
        if int(value) < 1:
            raise ConfigError(f"tournament setting {name} must be >= 1, got {value}")
    kinds = {spec.num_teams(L) == 1 for L in settings.layouts}
    if mode in ("halving", "select") and len(kinds) > 1:
        raise ConfigError(f"--mode {mode}: the layouts {settings.layouts} mix score (one team) and win/rank layouts, "
                          f"whose results cannot be averaged; choose layouts of one kind with --layout")


def run(mode: str, *, spec: GameSpec, players: Mapping[str, Any], candidates: Sequence[str], field: Sequence[str],
        settings: TournamentSettings, play: PlayFn, bt_fn: BTSectionFn | None = None) -> TournamentReport:
    """Play ``mode`` (module docstring) and build the report (``TournamentReport.to_dict()``: the T5.3 schema)."""
    started = time.monotonic()
    _check_settings(mode, spec, settings)
    candidates, field = list(dict.fromkeys(candidates)), list(dict.fromkeys(field))
    if mode == "select" and settings.finalists == 1:
        logger.warning("--mode select with --finalists 1: stage B has no runner-up, so the winner can never be "
                       "confirmed; pin it by hand (colosseum pin) if you accept it")
    ctx = _Context(spec, players, candidates, field, settings, play)
    once = _OnceFilter()
    _schedule.logger.addFilter(once)
    try:
        tables, rounds, selection = _play_mode(ctx, mode)
    finally:
        _schedule.logger.removeFilter(once)
    bt: dict[str, dict | None] = {}
    for layout in settings.layouts:
        if bt_fn is not None and ctx.metric_of[layout] == "pairwise":
            layout_blocks = [b for b in ctx.blocks if b.layout == layout]
            results = [r for b in layout_blocks for r in ctx.results[b.id]]
            bt[layout] = bt_fn(layout, results, layout_blocks)
        else:
            bt[layout] = None
    for t in tables:                      # the BT columns of every per-layout table row (spec block 3)
        section = bt.get(t["layout"])
        by_name = {e["name"]: e for e in section["entities"]} if section else {}
        for row in t["rows"]:
            e = by_name.get(row["name"])
            row["bt"] = None if e is None else {"rating": e["rating"], "ci95": e["ci95"], "p_best": e["p_best"]}
    repeated_pairs = sorted({(t["layout"], r["name"], col) for t in tables if "matrix" in t
                             for r in t["rows"] for col in r.get("repeated", [])})
    if mode == "round-robin":         # both rows of a pair carry its mark: list the pair once, as scheduled
        order = {n: i for i, n in enumerate(dict.fromkeys([*candidates, *field]))}
        repeated_pairs = [x for x in repeated_pairs if order[x[1]] < order[x[2]]]
    self_excluded = ([] if mode == "round-robin" else
                     [[L, c] for L in settings.layouts for c in candidates
                      if c in field and (ctx.metric_of[L] == "pairwise" or spec.layout_size(L) >= 2)])
    data = {
        "format": REPORT_FORMAT, "kind": "tournament", "name": settings.name, "mode": mode,
        "seed": settings.seed, "deterministic": settings.deterministic, "ci_level": settings.level,
        "bootstrap_samples": settings.bootstrap_samples,
        "settings": {"blocks": settings.blocks, "halving_blocks": settings.halving_blocks,
                     "confirm_blocks": settings.confirm_blocks, "finalists": settings.finalists},
        "candidates": [_player_row(players, n) for n in candidates],
        "field": [_player_row(players, n) for n in field],
        "layouts": [{"layout": L, "outcome_kind": spec.outcome_kind(L), "metric": ctx.metric_of[L],
                     "weight": float(settings.layout_weights[L]),
                     "dropped": [{"candidate": d.candidate, "composition": list(d.composition), "reason": d.reason}
                                 for (layout, _c, _k), d in ctx.dropped.items() if layout == L]}
                    for L in settings.layouts],
        "tables": tables, "rounds": rounds, "selection": selection or None, "bt": bt,
        "games": {"matches": sum(len(v) for v in ctx.results.values()), "blocks": len(ctx.results),
                  "by_stage": {st: {"blocks": sum(1 for b in ctx.blocks if b.stage == st),
                                    "matches": sum(len(ctx.results[b.id]) for b in ctx.blocks if b.stage == st)}
                               for st in sorted({b.stage for b in ctx.blocks})}},
        "flags": {"repeated_pairs": [list(x) for x in repeated_pairs],
                  "non_transitive": sorted(L for L, sec in bt.items()
                                           if sec and (sec.get("transitivity") or {}).get("flag")),
                  "self_excluded": self_excluded},
        "elapsed_sec": time.monotonic() - started,
    }
    return TournamentReport(_json_safe(data))


def _play_mode(ctx: _Context, mode: str) -> tuple[list[dict], list[dict], dict[str, dict]]:
    settings = ctx.settings
    tables: list[dict] = []
    rounds: list[dict] = []
    selection: dict[str, dict] = {}
    if mode == "round-robin":
        participants = list(dict.fromkeys([*ctx.candidates, *ctx.field]))
        for layout in settings.layouts:
            groups = check_role_groups(ctx.spec, layout, participants, ctx.players)
            if len(groups) > 1:
                raise ConfigError(f"--mode round-robin ranks every participant in one table, but in layout {layout!r} "
                                  f"they play different role groups: {groups}; compare the groups with gauntlet or "
                                  f"select (they rank per role group), or give round-robin participants of one group")
            ctx.run(ctx.round_robin(layout, participants))
        tables.extend(_round_robin_tables(ctx, participants))
        return tables, rounds, selection
    one_kind = len(set(ctx.metric_of.values())) == 1
    groups, group_of = _groups(ctx)
    for group, names in groups.items():
        if mode == "gauntlet":
            for layout in settings.layouts:
                ctx.run(ctx.schedule(layout, names, blocks=settings.blocks, stage="A"))
            stages = [("A", names)]
        else:
            finalists = _halving(ctx, group, names, group_of, rounds)
            stages = [("A", finalists)]        # stage A table: the blocks every finalist played (ruling D6)
            if mode == "select":
                for layout in settings.layouts:
                    ctx.run(ctx.schedule(layout, finalists, blocks=settings.confirm_blocks, stage="B"))
                stages.append(("B", finalists))
        for stage, members in stages:
            stage_tables = _stage_tables(ctx, stage, members, group, group_of)
            reps = [t.bootstrap_cells(settings.bootstrap_samples, ctx.rng) for t in stage_tables]
            tables.extend(_table_dict(ctx, t, r) for t, r in zip(stage_tables, reps))
            selecting = mode == "select" and stage == "B"
            if not selecting and (len(stage_tables) == 1 or not one_kind):
                continue
            comb = combine(stage_tables, settings.layout_weights, samples=settings.bootstrap_samples, rng=ctx.rng,
                           replicates=[t.scores_of(r) for t, r in zip(stage_tables, reps)])
            if len(stage_tables) > 1:
                tables.append(_combined_dict(stage, group, comb, settings.level, stage_tables))
            if selecting:
                sel = select_winner(comb, stage_tables, level=settings.level)
                selection[group] = {
                    "winner": sel.winner, "runner_up": sel.runner_up, "confirmed": sel.confirmed,
                    "diff": sel.diff, "diff_ci95": None if sel.diff_ci95 is None else list(sel.diff_ci95),
                    "indistinguishable": sel.indistinguishable,
                    "repeated_columns": [list(x) for x in sel.repeated_columns],
                    # every repeated column of the winner or the runner-up: diff_ci95 rests on it in part
                    "diff_repeated": [list(x) for x in repeated_columns(
                        stage_tables, *[n for n in (sel.winner, sel.runner_up) if n is not None])],
                    "note": WINNER_BIAS_NOTE if sel.runner_up is not None else LONE_FINALIST_NOTE}
    return tables, rounds, selection


def _json_safe(obj: Any) -> Any:
    """Tuples to lists, non-finite floats to None (``report.json`` is strict JSON)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _round_robin_tables(ctx: _Context, participants: Sequence[str]) -> list[dict]:
    """Pairwise matrix per layout: ``cells[a][b]`` = a's mean pairwise score against b (1 - b's against a), with a
    bootstrap interval over the pair's blocks. Under ``deterministic`` a pair whose blocks (two or more) all gave
    the same games is ``repeated``: it counts as one block (Wilson interval of one observation, as the gauntlet's
    cells), and both rows list it in ``repeated``."""
    out = []
    for layout in ctx.settings.layouts:
        per_pair: dict[tuple[str, str], list[tuple[float, tuple]]] = {}
        for b in ctx.blocks:
            if b.layout != layout:
                continue
            mean, _n, signature = block_values(b, ctx.results[b.id], ctx.names, "pairwise")[b.composition[0]]
            per_pair.setdefault((b.candidate, b.composition[0]), []).append((mean, signature))
        matrix: dict[str, dict[str, dict]] = {a: {} for a in participants}
        for (a, b), entries in per_pair.items():
            x = np.asarray([v for v, _sig in entries])
            repeated = ctx.settings.deterministic and len(entries) >= 2 and len({sig for _v, sig in entries}) == 1
            if repeated:
                lo, hi = wilson_interval(float(x.mean()), 1)
            else:
                idx = ctx.rng.integers(len(x), size=(ctx.settings.bootstrap_samples, len(x)))
                lo, hi = quantiles(x[idx].mean(axis=1), ctx.settings.level)
            matrix[a][b] = {"score": float(x.mean()), "ci95": [lo, hi], "blocks": len(x), "repeated": repeated}
            matrix[b][a] = {"score": float(1 - x.mean()), "ci95": [1 - hi, 1 - lo], "blocks": len(x),
                            "repeated": repeated}
        rows = []
        for a in participants:
            row = matrix[a]
            rows.append({"name": a, "score": float(np.mean([c["score"] for c in row.values()])) if row else None,
                         "opponents": len(row), "repeated": sorted(col for col, c in row.items() if c["repeated"])})
        # a participant without a playable pair has no score (None) and comes last
        rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0.0), participants.index(r["name"])))
        out.append({"stage": "A", "layout": layout, "group": "*", "metric": "pairwise",
                    "columns": list(participants), "rows": rows, "matrix": matrix})
    return out


def report_text(data: Mapping[str, Any]) -> str:
    """Plain-text report (``report.txt`` and the console)."""
    lines = [f"Tournament {data['name']} ({data['mode']}, seed {data['seed']}"
             f"{', deterministic' if data['deterministic'] else ''}): {data['games']['matches']} matches in "
             f"{data['games']['blocks']} blocks"]
    for t in data["tables"]:
        title = "all layouts (weighted)" if t["layout"] == "*" else f"layout {t['layout']}"
        lines.append(f"== stage {t['stage']}, {title}, group {t['group']}")
        for r in t["rows"]:
            ci = r.get("ci95")
            text = f"  {r['name']:<28} {'-' if r['score'] is None else format(r['score'], '.3f')}"
            if ci and None not in ci:
                text += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
            if r.get("worst"):
                text += f"  worst {r['worst']['opponent']} {r['worst']['score']:.3f}"
            if "games" in r:
                text += f"  games {r['games']}"
            if r.get("repeated"):
                text += "  REPEATED vs " + ", ".join(x if isinstance(x, str) else "/".join(x) for x in r["repeated"])
            lines.append(text)
    for r in data["rounds"]:
        lines.append(f"-- halving {r['group']} round {r['round']}: {r['blocks_per_composition']} blocks/composition, "
                     f"out: {', '.join(r['eliminated']) or '-'}")
    for group, sel in (data["selection"] or {}).items():
        state = "confirmed" if sel["confirmed"] else "NOT confirmed"
        line = f"** {group}: winner {sel['winner']} ({state}"
        if sel["diff_ci95"] is not None and sel["diff"] is not None and None not in sel["diff_ci95"]:
            lo, hi = sel["diff_ci95"]
            line += f"; vs {sel['runner_up']}: {sel['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        if sel.get("diff_repeated"):
            line += "; REPEATED games in " + ", ".join(f"{L}: {c}" for L, c in sel["diff_repeated"])
        lines.append(line + ")")
        if sel["indistinguishable"]:
            lines.append(f"   indistinguishable from the winner: {', '.join(sel['indistinguishable'])}")
        if sel.get("repeated_columns"):
            lines.append("   the lead rests on repeated games: " +
                         ", ".join(f"{L}: {c}" for L, c in sel["repeated_columns"]))
        lines.append(f"   note: {sel['note']}")
    for layout, sec in data["bt"].items():
        if sec:
            anchors = ", ".join(f"{a['group']}: {a['anchor']}" for a in sec["anchor"])
            lines.append(f"== BT ratings, layout {layout} (anchor {anchors})")
            for e in sec["entities"]:
                ci = e["ci95"] if e["ci95"] and None not in e["ci95"] else None
                lines.append(f"  {e['name']:<28} {e['rating']:+8.1f} "
                             f"{f'[{ci[0]:+.1f}, {ci[1]:+.1f}]' if ci else '[-]'}  "
                             f"P(best) {e['p_best']:.2f}  games {e['games']}")
            if sec["transitivity"]["flag"]:
                lines.append(f"  possible cycles (G2 p={sec['transitivity']['p_value']:.3g}): "
                             f"{sec['transitivity']['worst_pairs']}")
    for entry in data.get("layouts") or []:
        for d in entry.get("dropped") or []:
            lines.append(f"Not scheduled in {entry['layout']}: {d['candidate']} vs "
                         f"{'+'.join(d['composition']) or 'its homogeneous team'} ({d['reason']})")
    flags = data["flags"]
    for layout, candidate in flags.get("self_excluded") or []:
        lines.append(f"{layout}: {candidate} is also in the field and plays the field without itself")
    if flags["repeated_pairs"]:
        lines.append("Games repeat (not enough randomness): " +
                     "; ".join(f"{L}: {c} vs {f}" for L, c, f in flags["repeated_pairs"]))
    return "\n".join(lines)


def report_view(data: Mapping[str, Any]) -> dict[str, Any]:
    """The report by layout and role group (contract amendment A26; Part 3's page, kits and slow tests read it):
    ``{name, mode, seed, [winner, confirmed], layouts: {layout: {outcome_kind, dropped, self_excluded, groups:
    {group: {candidates, ranking, matrix, rounds, select, transitivity, repeated_pairs, candidate_weights,
    combined}}}}}``.

    - ``candidates``: the rows of the group's last-stage table (B, else A), then the candidates halving eliminated
      (their last round's score, no interval); each ``{name, display, lineage, score, score_ci, worst, bt, bt_ci,
      p_best, games, eliminated_round, repeated}`` (``bt*`` from the layout's BT section, None without one;
      ``repeated``: the columns where the candidate's games repeated, ``[]`` for the eliminated);
    - ``ranking``: the last stage's order, then the eliminated by later round and higher score;
    - ``matrix``: ``{candidate: {column: {score, ci, games, repeated}}}`` of that table; ``rounds``: ``{round,
      blocks, survivors, eliminated}``; ``select``: the group's selection (``winner``, ``runner_up``,
      ``confirmed``, ``diff``, ``diff_ci``, ``indistinguishable``, ``repeated_columns``, ``diff_repeated``,
      ``note``) or None;
    - ``candidate_weights``: the table's per-candidate column weights where they differ (a candidate in the field);
      ``combined``: the group's layout combination of the last stage (``[{name, score, score_ci, repeated}]``,
      ``repeated`` as ``[layout, column]`` pairs) or None (one layout);
    - ``dropped``: the layout's compositions the schedule could not seat (``{candidate, composition, reason}``);
      ``self_excluded``: the layout's candidates that are also field members (they play the field without
      themselves);
    - ``winner`` / ``confirmed`` at the top only when one group was selected.
    A partial report (e.g. ``{"name", "mode"}``) gives the view of what it has; a view (``layouts`` is already a
    mapping) is returned as it is."""
    if isinstance(data.get("layouts"), Mapping):
        return dict(data)
    players = {p["name"]: p for p in [*(data.get("candidates") or []), *(data.get("field") or [])]}
    selection = data.get("selection") or {}
    flags = (data.get("flags") or {}).get("repeated_pairs") or []
    self_excluded = (data.get("flags") or {}).get("self_excluded") or []
    entries = {e["layout"]: e for e in data.get("layouts") or []}
    last: dict[tuple[str, str], dict] = {}
    combined: dict[str, dict] = {}
    for t in data.get("tables") or []:
        if t["layout"] == "*":
            if t["group"] not in combined or t["stage"] > combined[t["group"]]["stage"]:
                combined[t["group"]] = t
            continue
        key = (t["layout"], t["group"])
        if key not in last or t["stage"] > last[key]["stage"]:
            last[key] = t
    layouts: dict[str, dict] = {}
    for (layout, group), t in last.items():
        section = (data.get("bt") or {}).get(layout) or {}
        rated = {e["name"]: e for e in section.get("entities") or []}

        def row(name: str, score, ci, worst, games, out_round, repeated=(), rated=rated) -> dict:
            e = rated.get(name)
            p = players.get(name, {})
            return {"name": name, "display": p.get("display", name), "lineage": p.get("lineage", name),
                    "score": score, "score_ci": ci, "worst": worst, "bt": None if e is None else e["rating"],
                    "bt_ci": None if e is None else e["ci95"], "p_best": None if e is None else e["p_best"],
                    "games": games if games is not None else (None if e is None else e["games"]),
                    "eliminated_round": out_round, "repeated": list(repeated)}

        candidates = [row(r["name"], r["score"], r.get("ci95"), r.get("worst"), r.get("games"), None,
                          r.get("repeated") or ()) for r in t["rows"]]
        rounds = [r for r in data.get("rounds") or [] if r["group"] == group]
        eliminated = sorted(((r["round"], r["scores"][n], n) for r in rounds for n in r["eliminated"]),
                            key=lambda x: (-x[0], -x[1]))
        candidates += [row(n, score, None, None, None, rnd) for rnd, score, n in eliminated]
        sel = selection.get(group)
        entry = entries.get(layout, {})
        comb = combined.get(group)
        layouts.setdefault(layout, {"outcome_kind": entry.get("outcome_kind"),
                                    "dropped": list(entry.get("dropped") or []),
                                    "self_excluded": [c for L, c in self_excluded if L == layout],
                                    "groups": {}})["groups"][group] = {
            "candidates": candidates,
            "ranking": [r["name"] for r in t["rows"]] + [n for _, _, n in eliminated],
            "matrix": {c: {col: {"score": cell["score"], "ci": cell.get("ci95"), "games": cell.get("games"),
                                 "repeated": bool(cell.get("repeated", False))}
                           for col, cell in cols.items()} for c, cols in (t.get("matrix") or {}).items()},
            "rounds": [{"round": r["round"], "blocks": r["blocks_per_composition"],
                        "survivors": [n for n in r["active"] if n not in r["eliminated"]],
                        "eliminated": list(r["eliminated"])} for r in rounds],
            "select": None if sel is None else {
                "winner": sel["winner"], "runner_up": sel.get("runner_up"), "confirmed": sel["confirmed"],
                "diff": sel.get("diff"), "diff_ci": sel.get("diff_ci95"),
                "indistinguishable": list(sel.get("indistinguishable") or []),
                "repeated_columns": [list(x) for x in sel.get("repeated_columns") or []],
                "diff_repeated": [list(x) for x in sel.get("diff_repeated") or []], "note": sel.get("note")},
            "transitivity": section.get("transitivity"),
            "repeated_pairs": [[c, col] for L, c, col in flags if L == layout],
            "candidate_weights": dict(t.get("candidate_weights") or {}),
            "combined": None if comb is None else [
                {"name": r["name"], "score": r["score"], "score_ci": r.get("ci95"),
                 "repeated": [list(x) for x in r.get("repeated") or []]} for r in comb["rows"]],
        }
    out = {"name": data.get("name"), "mode": data.get("mode"), "seed": data.get("seed"), "layouts": layouts}
    if len(selection) == 1:
        (sel,) = selection.values()
        out["winner"], out["confirmed"] = sel["winner"], sel["confirmed"]
    return out


def _is_random_bot(class_path: str) -> bool:
    from colosseum.core.registry import import_class
    from colosseum.players import RandomBot

    try:
        return import_class(class_path) is RandomBot
    except Exception:  # noqa: BLE001 - not importable here: not the RandomBot
        return False


def bt_section(layout: str, results: Sequence[MatchResult], blocks: Sequence[Block], *, players: Mapping[str, Any],
               anchor: str | None = None, samples: int = 200, seed: int = 0) -> dict[str, Any]:
    """BT ratings of one pairwise layout from the tournament's own matches (spec block 3): ``arena.bt.fit`` on a
    ``ComparisonTable`` of ``results`` (``source`` tournament), intervals by ``bt.bootstrap`` over whole blocks
    (strata: stage and composition; keys: block indices, shared by every candidate), ``P(best)`` within the role
    group, the G2 non-transitivity test (``TransitivityResult.to_dict()``; flagged at p < 0.01). The anchor rule is
    the run's (``ratings.anchor``, else the first ``RandomBot``, else the mean of the fixed players); every
    non-snapshot participant counts as fixed (ruling D9)."""
    from colosseum.arena import bt
    from colosseum.arena.comparisons import ComparisonTable
    from colosseum.arena.entities import snapshot_version

    table = ComparisonTable()
    for result in results:
        table.add_result(result, [snapshot_version(seat.network_id) for seat in result.seats])
    fixed = [name for name, p in players.items() if p.kind != "snapshot"]
    random_bot = next((name for name, p in players.items() if p.kind == "scripted"
                       and getattr(p.source, "bot", None) is not None and _is_random_bot(p.source.bot.class_path)),
                      None)
    fit = bt.fit(table, layout, sources=("tournament",), options=bt.default_options(anchor=anchor),
                 fixed_agents=fixed, random_bot=random_bot)
    by_block: dict[str, list[MatchResult]] = {}
    for result in results:
        by_block.setdefault(result.block, []).append(result)
    strata = {b.id: f"{b.stage}/{b.stratum}" for b in blocks}
    unc = bt.bootstrap(fit, bt.BootstrapData.from_blocks(by_block, strata=strata), samples=samples,
                       rng=np.random.default_rng(np.random.SeedSequence([int(seed) % 2**63, 0xB7])))
    tt = bt.transitivity_test(table, fit, sources=("tournament",))
    entities = sorted((e for e in fit.rating if e in unc.ci95), key=lambda e: (-fit.rating[e], e))
    return _json_safe({
        "anchor": [{"group": g.group, "component": int(g.component), "anchor": g.anchor} for g in fit.groups],
        "entities": [{"name": e, "rating": float(fit.rating[e]), "ci95": [float(x) for x in unc.ci95[e]],
                      "p_best": float(unc.p_best_group.get(e, 0.0)), "games": int(fit.games.get(e, 0))}
                     for e in entities],
        "transitivity": tt.to_dict(),
        "converged": bool(fit.converged),
    })
