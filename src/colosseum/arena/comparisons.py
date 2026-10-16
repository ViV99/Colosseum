"""Team-pair sufficient statistics of match results (SP4 spec blocks 1-2).

The unit of the Bradley-Terry ratings is a pair of teams of one match with ``T >= 2`` teams:
weight ``1 / (T - 1)``, outcome 1 / 0.5 / 0 from the team ranks (``pairwise_rank_score``, the rule of SP2's
``member_pairs`` at team level). A team is a :class:`TeamKey`: its team index in the layout (the index also
fixes its roles, so the side bias ``b[layout, team]`` of ``colosseum.arena.bt`` is keyed by it) and its
members as sorted ``(entity, role)`` pairs. For every ``(layout, source)`` a :class:`ComparisonTable` keeps,
per ordered team pair ``(a, b)`` with ``a.team < b.team``, the sums ``n`` (matches), ``sw`` (weights) and
``swy`` (weight * outcome of ``a``): exactly what the fit needs (point estimate and Hessian), so the journal's
compacted ``counts`` rows and its detailed matches give the same table.

One-team layouts (``score``) have no pairs: their team composition keeps ``n``, the sum and the sum of squares
of the team score (:class:`ScoreStat`).

A pair whose two teams have the same members (an entity against itself, e.g. latest against latest of one
version bucket) carries information about the side bias only: no entity's net coefficient is non-zero, so it
counts for no entity's games (``entity_games``).

The table is thread-safe: the journal writer thread adds to it, the live-rating thread takes ``copy()``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from colosseum.arena.entities import EntityKey, entity_key
from colosseum.core.outcomes import pairwise_rank_score
from colosseum.core.types import MatchResult

Pair = tuple["TeamKey", "TeamKey", float, float]   # (a, b, weight, outcome of a)

_COEF_EPS = 1e-12


@dataclass(frozen=True, order=True)
class TeamKey:
    """A team of a match as a rating unit: team index in the layout and sorted ``(entity, role)`` members."""

    team: int
    members: tuple[tuple[EntityKey, str], ...]

    def entities(self) -> list[EntityKey]:
        """Member entities (with repeats: two seats of one entity count twice)."""
        return [entity for entity, _ in self.members]

    def encode(self) -> str:
        """Canonical text form (reports, tests): ``[team, [[entity, role], ...]]`` as compact JSON."""
        return json.dumps([self.team, [list(m) for m in self.members]], separators=(",", ":"))

    @classmethod
    def decode(cls, text: str) -> TeamKey:
        team, members = json.loads(text)
        return cls(int(team), tuple((str(e), str(r)) for e, r in members))

    @classmethod
    def of(cls, team: int, members: Iterable[tuple[EntityKey, str]]) -> TeamKey:
        return cls(int(team), tuple(sorted((str(e), str(r)) for e, r in members)))


@dataclass
class PairStat:
    """Sufficient statistics of one ordered team pair: matches, sum of weights, sum of weight * outcome."""

    n: int = 0
    sw: float = 0.0
    swy: float = 0.0

    def add(self, other: PairStat) -> None:
        self.n += int(other.n)
        self.sw += float(other.sw)
        self.swy += float(other.swy)


@dataclass
class ScoreStat:
    """Team-score statistics of one composition of a one-team layout: matches, sum, sum of squares."""

    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, other: ScoreStat) -> None:
        self.n += int(other.n)
        self.total += float(other.total)
        self.total_sq += float(other.total_sq)


def team_keys(result: MatchResult, rating_versions: Sequence[int | None]) -> dict[int, TeamKey]:
    """``{team index: TeamKey}`` of a result; ``rating_versions`` has one entry per ``result.seats`` entry."""
    if len(rating_versions) != len(result.seats):
        raise ValueError(f"match {result.uid or result.match_id!r}: {len(rating_versions)} rating versions for "
                         f"{len(result.seats)} seats")
    members: dict[int, list[tuple[EntityKey, str]]] = {}
    for seat, version in zip(result.seats, rating_versions, strict=True):
        members.setdefault(int(seat.team), []).append((entity_key(seat.agent_id, version), seat.role))
    return {team: TeamKey.of(team, m) for team, m in sorted(members.items())}


def pairs_of(keys: Mapping[int, TeamKey], ranks: Mapping[int, float]) -> list[Pair]:
    """``(a, b, weight, outcome_a)`` of every pair of teams (``a.team < b.team``) that has members and a rank;
    weight ``1 / (T - 1)`` with ``T`` such teams; ``[]`` for fewer than two. The one rule for results and for
    journal rows (``colosseum.arena.matchlog`` folds detailed matches with it)."""
    teams = [t for t in sorted(keys) if t in ranks]
    if len(teams) < 2:
        return []
    weight = 1.0 / (len(teams) - 1)
    return [(keys[ta], keys[tb], weight, pairwise_rank_score(float(ranks[ta]), float(ranks[tb])))
            for i, ta in enumerate(teams) for tb in teams[i + 1:]]


def score_of(keys: Mapping[int, TeamKey], scores: Mapping[int, float]) -> tuple[TeamKey, float] | None:
    """``(team, team score)`` when exactly one team has members and a score (a one-team layout), else None."""
    if len(keys) != 1 or len(scores) != 1:
        return None
    ((team, key),) = keys.items()
    return (key, float(scores[team])) if team in scores else None


def match_pairs(result: MatchResult, rating_versions: Sequence[int | None]) -> list[Pair]:
    """``(a, b, weight, outcome_a)`` for every pair of teams of ``result`` (``a.team < b.team``); ``[]`` for a
    one-team layout (``pairs_of``)."""
    return pairs_of(team_keys(result, rating_versions), {int(t.team): float(t.rank) for t in result.teams})


def match_score(result: MatchResult, rating_versions: Sequence[int | None]) -> tuple[TeamKey, float] | None:
    """``(team, team score)`` of a one-team result, else None (``score_of``)."""
    return score_of(team_keys(result, rating_versions), {int(t.team): float(t.score) for t in result.teams})


def coefficients(a: TeamKey, b: TeamKey) -> dict[EntityKey, float]:
    """Net coefficient of every entity in ``S_a - S_b`` (team strength = mean of its members): non-zero
    entries only. An entity that sits on both teams in the same share cancels."""
    out: dict[EntityKey, float] = {}
    for key, sign in ((a, 1.0), (b, -1.0)):
        share = sign / len(key.members)
        for entity, _ in key.members:
            out[entity] = out.get(entity, 0.0) + share
    return {e: c for e, c in out.items() if abs(c) > _COEF_EPS}


def _pair_order(item) -> tuple:
    (a, b), _ = item
    return a.team, a.members, b.team, b.members


@dataclass
class _Cell:
    pairs: dict[tuple[TeamKey, TeamKey], PairStat] = field(default_factory=dict)
    scores: dict[TeamKey, ScoreStat] = field(default_factory=dict)


class ComparisonTable:
    """Sufficient statistics of team pairs per ``(layout, source)``, plus score sums for one-team layouts
    (module docstring). Every method holds the table's lock; ``version`` counts mutations."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cells: dict[tuple[str, str], _Cell] = {}
        self._version = 0

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def _cell(self, layout: str, source: str) -> _Cell:
        return self._cells.setdefault((str(layout), str(source)), _Cell())

    def add_result(self, result: MatchResult, rating_versions: Sequence[int | None]) -> None:
        """Add one match (pairs, or the team score of a one-team layout) under ``(result.layout, result.source)``."""
        pairs = match_pairs(result, rating_versions)
        with self._lock:
            if pairs:
                for a, b, w, y in pairs:
                    self._add_pair(result.layout, result.source, a, b, PairStat(1, w, w * y))
            else:
                score = match_score(result, rating_versions)
                if score is None:
                    return
                key, value = score
                self._add_score(result.layout, result.source, key, ScoreStat(1, value, value * value))
            self._version += 1

    def _add_pair(self, layout: str, source: str, a: TeamKey, b: TeamKey, stat: PairStat) -> None:
        if a.team > b.team:      # canonical order: the lower team index first
            a, b, stat = b, a, PairStat(stat.n, stat.sw, stat.sw - stat.swy)
        if a.team == b.team:
            raise ValueError(f"a team pair needs two different team indices, got {a.team} twice")
        cell = self._cell(layout, source)
        cell.pairs.setdefault((a, b), PairStat()).add(stat)

    def _add_score(self, layout: str, source: str, key: TeamKey, stat: ScoreStat) -> None:
        self._cell(layout, source).scores.setdefault(key, ScoreStat()).add(stat)

    def add(self, layout: str, source: str, a: TeamKey, b: TeamKey, stat: PairStat) -> None:
        """Add aggregated pair statistics (a journal ``counts`` row); the pair is stored in canonical order."""
        with self._lock:
            self._add_pair(layout, source, a, b, stat)
            self._version += 1

    def add_score(self, layout: str, source: str, key: TeamKey, stat: ScoreStat) -> None:
        """Add aggregated score statistics of a one-team composition."""
        with self._lock:
            self._add_score(layout, source, key, stat)
            self._version += 1

    def merge(self, other: ComparisonTable) -> None:
        """Add every statistic of ``other``."""
        snapshot = other.copy()
        with self._lock:
            for (layout, source), cell in snapshot._cells.items():
                for (a, b), stat in cell.pairs.items():
                    self._add_pair(layout, source, a, b, stat)
                for key, stat in cell.scores.items():
                    self._add_score(layout, source, key, stat)
            self._version += 1

    def copy(self) -> ComparisonTable:
        """An independent copy (taken under the lock)."""
        out = ComparisonTable()
        with self._lock:
            for key, cell in self._cells.items():
                out._cells[key] = _Cell(
                    pairs={k: PairStat(s.n, s.sw, s.swy) for k, s in cell.pairs.items()},
                    scores={k: ScoreStat(s.n, s.total, s.total_sq) for k, s in cell.scores.items()},
                )
            out._version = self._version
        return out

    def layouts(self) -> list[str]:
        with self._lock:
            return sorted({layout for layout, _ in self._cells})

    def sources(self, layout: str) -> list[str]:
        with self._lock:
            return sorted(source for lay, source in self._cells if lay == layout)

    def pairs(self, layout: str, sources: Collection[str]) -> list[tuple[TeamKey, TeamKey, PairStat]]:
        """Pair statistics of ``layout`` summed over ``sources`` (sorted by key)."""
        merged: dict[tuple[TeamKey, TeamKey], PairStat] = {}
        with self._lock:
            for source in sources:
                cell = self._cells.get((layout, source))
                if cell is None:
                    continue
                for key, stat in cell.pairs.items():
                    merged.setdefault(key, PairStat()).add(stat)
        return [(a, b, stat) for (a, b), stat in sorted(merged.items(), key=_pair_order)]

    def scores(self, layout: str, sources: Collection[str]) -> list[tuple[TeamKey, ScoreStat]]:
        """Score statistics of ``layout`` summed over ``sources`` (sorted by key)."""
        merged: dict[TeamKey, ScoreStat] = {}
        with self._lock:
            for source in sources:
                cell = self._cells.get((layout, source))
                if cell is None:
                    continue
                for key, stat in cell.scores.items():
                    merged.setdefault(key, ScoreStat()).add(stat)
        return sorted(merged.items(), key=lambda kv: (kv[0].team, kv[0].members))

    def entity_games(self, layout: str, sources: Collection[str]) -> dict[EntityKey, int]:
        """Matches per entity with a non-zero net coefficient (pairs) or a seat (scores); self-play excluded."""
        games: dict[EntityKey, int] = {}
        for a, b, stat in self.pairs(layout, sources):
            for entity in coefficients(a, b):
                games[entity] = games.get(entity, 0) + stat.n
        for key, stat in self.scores(layout, sources):
            for entity in set(key.entities()):
                games[entity] = games.get(entity, 0) + stat.n
        return games

    def as_dict(self) -> dict:
        """Plain nested form for tests and debugging: ``{(layout, source): {"pairs": {(a, b): (n, sw, swy)},
        "scores": {key: (n, total, total_sq)}}}``."""
        with self._lock:
            return {key: {"pairs": {(a.encode(), b.encode()): (s.n, s.sw, s.swy) for (a, b), s in cell.pairs.items()},
                          "scores": {k.encode(): (s.n, s.total, s.total_sq) for k, s in cell.scores.items()}}
                    for key, cell in sorted(self._cells.items())}
