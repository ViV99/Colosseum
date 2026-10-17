"""Tournament statistics (SP4 spec block 3): candidate scores, the paired stratified bootstrap, ``confirmed``,
repeated deterministic games.

Tables. ``candidate_scores`` turns played blocks into one ``CandidateTable`` per (stage, layout, role group):
- **columns** are the opponents the score is averaged over: the field members (pairwise layouts) or
  ``SELF`` (the candidate's homogeneous team) plus the field members it was paired with (one-team ``score``
  layouts);
- **strata** are the compositions (``Block.stratum``) and **indices** the block indices of each stratum;
  every candidate of a table that holds a stratum has played the same block indices in it (the same seeds).
  Candidates hold the same strata, except a candidate that is also a field member: the schedule leaves it out
  of its own compositions (T5.2), so it lacks those strata and its own column (``self_excluded``);
- ``values[c, s, k, col]`` is the block's mean for that column: pairwise layouts average the candidate's
  pairwise score against the column's team (1 / 0.5 / 0 by team ranks, over the block's rotations and over
  every team that member holds); ``score`` layouts average the team score over the block's matches.

Candidate score: ``S_c = sum_col w_col * cell(c, col)`` with ``cell`` the mean of the column's block values
over every block that holds the column and ``w_col`` equal per lineage (``1 / lineages``), split equally
between the columns of a lineage (snapshots of one run weigh as one opponent); a candidate without a column
(``self_excluded``) has the weights of its own columns. The worst opponent is the column with the smallest cell.

Paired stratified bootstrap: a replicate redraws, with replacement, the block indices within every stratum,
the SAME draw for every candidate (common seeds), and recomputes every cell and score; several layouts are
redrawn independently and combined per replicate with the layout weights. Intervals are the 2.5 % / 97.5 %
quantiles of the replicates. ``confirmed``: the 95 % interval of ``S_winner - S_runner_up`` lies above 0, and
no column where the two differ is a repeated pair of either (below). A lone finalist is never ``confirmed``
(there is no difference to test).

Repeated games (``--deterministic`` only): a (candidate, column) pair whose every block gave the same
outcomes and lengths (two or more blocks) is flagged. It counts as one block: its per-cell interval is the
Wilson interval of one observation (pairwise) or the point itself (score), and a lead that rests on such a
column is never ``confirmed`` (spec block 3: «Такие пары в ДИ и confirmed считаются одним блоком»; ruling D5).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from colosseum.core.outcomes import pairwise_rank_score
from colosseum.core.types import MatchResult
from colosseum.eval import wilson_interval

SELF = "(self)"
# Provisional: T8.2 measures the bootstrap size on the bot ladder and the controller fixes it as a ruling.
PAIRED_BOOTSTRAP_SAMPLES = 2000
_CHUNK = 250                      # replicates per vectorized chunk (memory bound)


@dataclass
class CandidateTable:
    stage: str
    layout: str
    metric: str                          # "pairwise" | "score"
    group: str
    candidates: list[str]
    columns: list[str]
    lineage: dict[str, str]              # column -> lineage
    strata: list[str]
    indices: dict[str, list[int]]        # stratum -> block indices (sorted)
    values: np.ndarray                   # [C, S, Kmax, L] block means; 0 where absent
    present: np.ndarray                  # [C, S, L] bool: the candidate's blocks of the stratum hold the column
    games: dict[str, dict[str, int]]     # candidate -> column -> comparisons (pairwise) or matches (score)
    matches: dict[str, int] = field(default_factory=dict)        # candidate -> matches played in the table
    repeated: set[tuple[str, str]] = field(default_factory=set)
    self_excluded: list[str] = field(default_factory=list)       # candidates that are also columns (field members)

    @property
    def weights(self) -> np.ndarray:
        """Column weights: equal per lineage, split within a lineage (module docstring); sums to 1."""
        return _lineage_weights(self.columns, self.lineage, np.ones(len(self.columns), dtype=bool))

    @property
    def candidate_weights(self) -> np.ndarray:
        """``[C, L]``: every candidate's weights over the columns it holds (``weights`` when it holds all)."""
        held = self.present.any(axis=1)
        return np.stack([_lineage_weights(self.columns, self.lineage, held[i]) for i in range(len(self.candidates))])

    def has_cell(self, candidate: str, column: str) -> bool:
        return bool(self.present[self.candidates.index(candidate), :, self.columns.index(column)].any())

    def blocks_of(self, column: str, candidate: str | None = None) -> int:
        """Blocks that hold ``column`` for ``candidate`` (default: the most any candidate has)."""
        j = self.columns.index(column)
        rows = range(len(self.candidates)) if candidate is None else [self.candidates.index(candidate)]
        return max(int(sum(len(self.indices[s]) for i, s in enumerate(self.strata) if self.present[c, i, j]))
                   for c in rows)

    def blocks_played(self, candidate: str) -> int:
        """Blocks of the table the candidate played."""
        c = self.candidates.index(candidate)
        return int(sum(len(self.indices[s]) for i, s in enumerate(self.strata) if self.present[c, i].any()))

    def _den(self) -> np.ndarray:
        sizes = np.array([len(self.indices[s]) for s in self.strata], dtype=np.float64)
        return np.einsum("s,csl->cl", sizes, self.present.astype(np.float64))       # [C, L]

    def cells(self) -> np.ndarray:
        """Point estimates ``[C, L]`` (0 where a candidate lacks the column)."""
        num = self.values.sum(axis=2).sum(axis=1)                     # [C, L] (absent entries are 0)
        return num / np.maximum(self._den(), 1)

    def scores(self) -> np.ndarray:
        """Candidate scores ``[C]``."""
        return (self.cells() * self.candidate_weights).sum(axis=1)

    def bootstrap_cells(self, samples: int, rng: np.random.Generator) -> np.ndarray:
        """Paired stratified bootstrap replicates of every cell, ``[samples, C, L]`` (module docstring)."""
        kmax = self.values.shape[2]
        sizes = [len(self.indices[s]) for s in self.strata]
        den = np.maximum(self._den(), 1)                              # [C, L]: constant over replicates
        out = np.empty((samples, len(self.candidates), len(self.columns)))
        values = self.values.astype(np.float64)
        for start in range(0, samples, _CHUNK):
            r = min(_CHUNK, samples - start)
            counts = np.zeros((r, len(self.strata), kmax))
            for i, size in enumerate(sizes):
                counts[:, i, :size] = rng.multinomial(size, np.full(size, 1.0 / size), size=r)
            out[start:start + r] = np.einsum("rsk,cskl->rcl", counts, values) / den
        return out

    def bootstrap_scores(self, samples: int, rng: np.random.Generator) -> np.ndarray:
        """``[samples, C]`` candidate-score replicates."""
        return self.scores_of(self.bootstrap_cells(samples, rng))

    def scores_of(self, cell_replicates: np.ndarray) -> np.ndarray:
        """``[R, C]`` scores of cell replicates ``[R, C, L]``."""
        return np.einsum("rcl,cl->rc", cell_replicates, self.candidate_weights)


def _lineage_weights(columns: Sequence[str], lineage: Mapping[str, str], held: np.ndarray) -> np.ndarray:
    kept = [c for c, h in zip(columns, held) if h]
    lineages = sorted({lineage[c] for c in kept})
    per = {lin: sum(1 for c in kept if lineage[c] == lin) for lin in lineages}
    return np.array([1.0 / (len(lineages) * per[lineage[c]]) if h else 0.0 for c, h in zip(columns, held)])


def _rotation(uid: str) -> int:
    return int(uid.rsplit("/r", 1)[1])


def _team_names(result: MatchResult, names: Mapping[tuple[str, str], str]) -> dict[int, str]:
    teams: dict[int, set[str]] = defaultdict(set)
    for seat in result.seats:
        teams[seat.team].add(names[(seat.agent_id, seat.network_id)])
    out = {}
    for team, members in teams.items():
        if len(members) != 1:
            raise ValueError(f"match {result.uid}: team {team} mixes players {sorted(members)}")
        out[team] = next(iter(members))
    return out


def block_values(block, results: Sequence[MatchResult], names: Mapping[tuple[str, str], str],
                 metric: str) -> dict[str, tuple[float, int, tuple]]:
    """``{column: (block mean, comparisons or matches, signature)}`` of one played block (module docstring);
    the signature (sorted ``(rotation, value, length)``) detects repeated games."""
    if len(results) != len(block.lineups):
        raise ValueError(f"block {block.id}: {len(results)} results for {len(block.lineups)} matches")
    acc: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    for result in results:
        rotation = _rotation(result.uid)
        if metric == "score":
            column = block.composition[0] if block.composition else SELF
            acc[column].append((rotation, float(result.teams[0].score), int(result.episode_length)))
            continue
        teams = _team_names(result, names)
        ranks = {t.team: t.rank for t in result.teams}
        own = [t for t, name in teams.items() if name == block.candidate]
        if len(own) != 1:
            raise ValueError(f"match {result.uid}: candidate {block.candidate!r} holds {len(own)} teams")
        for team, name in teams.items():
            if team != own[0]:
                acc[name].append((rotation, pairwise_rank_score(ranks[own[0]], ranks[team]),
                                  int(result.episode_length)))
    return {col: (float(np.mean([v for _r, v, _l in rows])), len(rows), tuple(sorted(rows)))
            for col, rows in acc.items()}


def candidate_scores(blocks: Sequence, results_by_block: Mapping[str, Sequence[MatchResult]], *,
                     names: Mapping[tuple[str, str], str], lineage: Mapping[str, str],
                     group_of: Mapping[tuple[str, str], str], metric_of: Mapping[str, str],
                     deterministic: bool = False) -> dict[tuple[str, str, str], CandidateTable]:
    """One ``CandidateTable`` per (stage, layout, group) of ``blocks`` (all played: ``results_by_block`` holds
    every block id). ``names``: (agent_id, network_id) -> player name; ``lineage``: player name -> lineage;
    ``group_of``: (layout, candidate) -> group label; ``metric_of``: layout -> "pairwise" | "score".
    ValueError when the candidates of a table did not play the same blocks (module docstring)."""
    raw: dict[tuple[str, str, str], dict[str, dict[tuple[str, int], dict[str, tuple]]]] = \
        defaultdict(lambda: defaultdict(dict))
    played: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for block in blocks:
        key = (block.stage, block.layout, group_of[(block.layout, block.candidate)])
        results = results_by_block[block.id]
        raw[key][block.candidate][(block.stratum, block.index)] = block_values(
            block, results, names, metric_of[block.layout])
        played[key][block.candidate] += len(results)
    tables = {}
    for (stage, layout, group), per_candidate in raw.items():
        candidates = list(per_candidate)
        columns = sorted({col for c in candidates for cell in per_candidate[c].values() for col in cell},
                         key=lambda col: (col != SELF, col))
        strata, indices = _common_blocks(stage, layout, group, per_candidate, set(columns))
        kmax = max(len(v) for v in indices.values())
        values = np.zeros((len(candidates), len(strata), kmax, len(columns)))
        present = np.zeros((len(candidates), len(strata), len(columns)), dtype=bool)
        games = {c: dict.fromkeys(columns, 0) for c in candidates}
        sigs: dict[tuple[str, str], set] = defaultdict(set)
        nblocks: dict[tuple[str, str], int] = defaultdict(int)
        for ci, c in enumerate(candidates):
            for si, s in enumerate(strata):
                for ki, k in enumerate(indices[s]):
                    for col, (mean, n, sig) in per_candidate[c].get((s, k), {}).items():
                        j = columns.index(col)
                        values[ci, si, ki, j] = mean
                        present[ci, si, j] = True
                        games[c][col] += n
                        sigs[(c, col)].add(sig)
                        nblocks[(c, col)] += 1
        repeated = {pair for pair, found in sigs.items() if deterministic and nblocks[pair] >= 2 and len(found) == 1}
        tables[(stage, layout, group)] = CandidateTable(
            stage=stage, layout=layout, metric=metric_of[layout], group=group, candidates=candidates,
            columns=columns, lineage={col: (SELF if col == SELF else lineage[col]) for col in columns},
            strata=strata, indices=indices, values=values, present=present, games=games,
            matches={c: int(played[(stage, layout, group)][c]) for c in candidates}, repeated=repeated,
            self_excluded=[c for c in candidates if c in columns])
    return tables


def _common_blocks(stage: str, layout: str, group: str, per_candidate: Mapping[str, Mapping[tuple[str, int], dict]],
                   columns: set[str]) -> tuple[list[str], dict[str, list[int]]]:
    """The strata and their block indices; every holder of a stratum has its indices, and only a candidate that
    is also a column (a field member left out of its own compositions) may lack strata (ValueError otherwise)."""
    held: dict[str, dict[str, set[int]]] = {c: defaultdict(set) for c in per_candidate}
    for c, cells in per_candidate.items():
        for s, k in cells:
            held[c][s].add(k)
    strata = sorted({s for c in held for s in held[c]})
    indices: dict[str, list[int]] = {}
    for s in strata:
        sets = {c: frozenset(held[c][s]) for c in held if s in held[c]}
        if len(set(sets.values())) > 1:
            raise ValueError(f"stage {stage}, layout {layout}, group {group}: the candidates did not play the same "
                             f"blocks (composition {s}: {', '.join(f'{c}: {len(v)}' for c, v in sets.items())})")
        indices[s] = sorted(next(iter(sets.values())))
    for c in held:
        missing = [s for s in strata if s not in held[c]]
        if missing and c not in columns:
            raise ValueError(f"stage {stage}, layout {layout}, group {group}: the candidates did not play the same "
                             f"blocks ({c} lacks the compositions {missing})")
    return strata, indices


def quantiles(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.quantile(x, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def cell_interval(table: CandidateTable, candidate: str, column: str, replicates: np.ndarray,
                  level: float = 0.95) -> tuple[float, float]:
    """95 % interval of one cell from its bootstrap replicates ``[R, C, L]``; a repeated pair counts as one
    block (Wilson of one observation for pairwise, the point for score)."""
    ci, j = table.candidates.index(candidate), table.columns.index(column)
    value = float(table.cells()[ci, j])
    if (candidate, column) in table.repeated:
        return wilson_interval(value, 1) if table.metric == "pairwise" else (value, value)
    return quantiles(replicates[:, ci, j], level)


def repeated_conflict(tables: Sequence[CandidateTable], a: str, b: str) -> list[tuple[str, str]]:
    """``(layout, column)`` pairs where ``a`` and ``b`` differ (in the cell, or one lacks it) and either one's
    games repeat."""
    out = []
    for t in tables:
        if a not in t.candidates or b not in t.candidates:
            continue
        cells = t.cells()
        ia, ib = t.candidates.index(a), t.candidates.index(b)
        for j, col in enumerate(t.columns):
            if (a, col) not in t.repeated and (b, col) not in t.repeated:
                continue
            same = t.has_cell(a, col) == t.has_cell(b, col) and np.isclose(cells[ia, j], cells[ib, j])
            if not same:
                out.append((t.layout, col))
    return out


@dataclass
class Combined:
    """Candidate scores of one (stage, group) over its layouts, with per-replicate combination."""

    candidates: list[str]
    point: np.ndarray        # [C]
    replicates: np.ndarray   # [R, C]


def combine(tables: Sequence[CandidateTable], weights: Mapping[str, float], *, samples: int,
            rng: np.random.Generator, replicates: Sequence[np.ndarray] | None = None) -> Combined:
    """Layout-weighted scores of the candidates common to every table, per replicate (module docstring).
    ``replicates``: the tables' score replicates ``[samples, C_t]`` when already drawn (the report shows the same
    replicates per layout and combined); ``samples=0`` gives the point only."""
    candidates = [c for c in tables[0].candidates if all(c in t.candidates for t in tables)]
    total = sum(weights[t.layout] for t in tables)
    point = np.zeros(len(candidates))
    reps = np.zeros((samples, len(candidates)))
    for i, t in enumerate(tables):
        idx = [t.candidates.index(c) for c in candidates]
        w = weights[t.layout] / total
        point += w * t.scores()[idx]
        if samples:
            table_reps = replicates[i] if replicates is not None else t.bootstrap_scores(samples, rng)
            reps += w * table_reps[:, idx]
    return Combined(candidates=candidates, point=point, replicates=reps)


def confirmed(diff_replicates: np.ndarray, *, level: float = 0.95, conflict: bool = False) -> bool:
    """The ``level`` interval of ``S_winner - S_runner_up`` lies above 0 and no repeated column decides it."""
    return (not conflict) and quantiles(diff_replicates, level)[0] > 0


@dataclass
class Selection:
    winner: str
    runner_up: str | None
    confirmed: bool
    diff: float | None
    diff_ci95: tuple[float, float] | None
    indistinguishable: list[str]
    repeated_columns: list[tuple[str, str]]


def select_winner(combined: Combined, tables: Sequence[CandidateTable], *, level: float = 0.95) -> Selection:
    """The winner by point score (ties: input order), ``confirmed`` against the runner-up and every finalist
    whose difference interval contains 0 or whose difference rests on a repeated column (``indistinguishable``).
    A lone finalist is the winner, not confirmed."""
    order = sorted(range(len(combined.candidates)), key=lambda i: (-combined.point[i], i))
    w = order[0]
    winner = combined.candidates[w]
    if len(order) == 1:
        return Selection(winner, None, False, None, None, [], [])
    u = order[1]
    runner = combined.candidates[u]
    diff = combined.replicates[:, w] - combined.replicates[:, u]
    conflict = repeated_conflict(tables, winner, runner)
    indistinct = []
    for i in order[1:]:
        d = combined.replicates[:, w] - combined.replicates[:, i]
        name = combined.candidates[i]
        if quantiles(d, level)[0] <= 0 or repeated_conflict(tables, winner, name):
            indistinct.append(name)
    return Selection(winner=winner, runner_up=runner, confirmed=confirmed(diff, level=level, conflict=bool(conflict)),
                     diff=float(combined.point[w] - combined.point[u]), diff_ci95=quantiles(diff, level),
                     indistinguishable=indistinct, repeated_columns=conflict)
