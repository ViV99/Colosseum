"""Ratings per layout (spec block 7).

Every layout of the game has its own tables; a result of a layout the game does not have is
rejected (ValueError), never given new tables. Rating entities follow SP1's code (overview,
"Deviations"): ELO and the win-rate matrix are keyed by the base ``agent_id``, so a checkpoint of
X playing Y counts as X vs Y; "latest of X vs a checkpoint of X" feeds ``wr_vs_past``; two seats of the same
agent that are both latest (or both checkpoints) carry no signal and are skipped.

Layouts with two or more teams (``wdl`` / ``rank``): every pair of teams (A, B) is one comparison
by team rank (``pairwise_rank_score``). Its weight ``1 / (T - 1)`` (times ``k_factor`` for ELO) is
split equally between the counted member pairs ``(a in A, b in B)`` (``member_pairs``);
teammates are never compared, and skipped member pairs are not in the divisor. ELO deltas of one
match are computed from the ratings before the match. Win rates and ``wr_vs_past`` are weighted
means of the same member-pair scores (fractional: a draw counts 0.5).

One-team layouts (solo, cooperative, ``score``): ``ScoreTracker`` keeps every agent's team score
(mean, EMA, 95% CI) and ``CrossPlayTable`` the mean score per team composition.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from colosseum.sp2.core.outcomes import pairwise_rank_score
from colosseum.sp2.core.types import LATEST_NETWORK_ID, MatchResult, SeatResult
from colosseum.sp2.envs.game import GameSpec

Z_95 = 1.959963984540054
SCORE_EMA_ALPHA = 0.05


@dataclass(frozen=True)
class MemberPair:
    """One counted comparison between seats of two different teams.

    ``cross``: different agents ``a`` and ``b``. ``past``: ``a`` is an agent's latest weights and
    ``b`` (== ``a``) one of its checkpoints. ``score_a`` is from ``a``'s side.
    """

    kind: Literal["cross", "past"]
    a: str
    b: str
    score_a: float
    weight: float
    role_a: str = ""
    role_b: str = ""


def _classify(sa: SeatResult, sb: SeatResult, score: float) -> tuple | None:
    if sa.agent_id != sb.agent_id:
        return "cross", sa.agent_id, sb.agent_id, score, sa.role, sb.role
    a_latest = sa.network_id == LATEST_NETWORK_ID
    b_latest = sb.network_id == LATEST_NETWORK_ID
    if a_latest and not b_latest:
        return "past", sa.agent_id, sa.agent_id, score, sa.role, sb.role
    if b_latest and not a_latest:
        return "past", sb.agent_id, sb.agent_id, 1.0 - score, sb.role, sa.role
    return None


def member_pairs(result: MatchResult) -> list[MemberPair]:
    """Counted member pairs of ``result`` with their weights (module docstring)."""
    by_team: dict[int, list[SeatResult]] = {}
    for seat in result.seats:
        by_team.setdefault(seat.team, []).append(seat)
    ranks = {team.team: team.rank for team in result.teams}
    teams = sorted(by_team)
    if len(teams) < 2:
        return []
    share = 1.0 / (len(teams) - 1)
    pairs: list[MemberPair] = []
    for i, team_a in enumerate(teams):
        for team_b in teams[i + 1:]:
            score = pairwise_rank_score(ranks[team_a], ranks[team_b])
            counted = [c for sa in by_team[team_a] for sb in by_team[team_b]
                       if (c := _classify(sa, sb, score)) is not None]
            for kind, a, b, s, role_a, role_b in counted:
                pairs.append(MemberPair(kind, a, b, s, share / len(counted), role_a, role_b))
    return pairs


class EloRating:
    """Pairwise ELO; all deltas of one update are computed from the ratings before it."""

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1200.0) -> None:
        self.k_factor = float(k_factor)
        self.initial_rating = float(initial_rating)
        self._ratings: dict[str, float] = {}

    def get(self, agent_id: str) -> float:
        return self._ratings.get(agent_id, self.initial_rating)

    def register(self, agent_id: str) -> None:
        self._ratings.setdefault(agent_id, self.initial_rating)

    def expected(self, a: str, b: str) -> float:
        """Expected score of ``a`` against ``b``."""
        return 1.0 / (1.0 + math.pow(10.0, (self.get(b) - self.get(a)) / 400.0))

    def update_weighted(self, pairs: Iterable[tuple[str, str, float, float]]) -> None:
        """Apply ``(a, b, score_a, weight)`` results together; each moves by ``k_factor * weight``."""
        deltas: dict[str, float] = {}
        for a, b, score_a, weight in pairs:
            k = self.k_factor * float(weight)
            e_a = self.expected(a, b)
            deltas[a] = deltas.get(a, 0.0) + k * (score_a - e_a)
            deltas[b] = deltas.get(b, 0.0) + k * ((1.0 - score_a) - (1.0 - e_a))
        for agent_id, delta in deltas.items():
            self._ratings[agent_id] = self.get(agent_id) + delta

    def update_pairs(self, pairs: Iterable[tuple[str, str, float]], k_scale: float = 1.0) -> None:
        """SP1 form: every ``(a, b, score_a)`` with the same weight ``k_scale``."""
        self.update_weighted((a, b, s, k_scale) for a, b, s in pairs)

    @property
    def all_ratings(self) -> dict[str, float]:
        return dict(self._ratings)


class WinRateTracker:
    """Pairwise score rates as weighted means (a draw is 0.5); ``games`` counts member pairs."""

    def __init__(self) -> None:
        self._scores: dict[str, dict[str, float]] = {}
        self._weights: dict[str, dict[str, float]] = {}
        self._games: dict[str, dict[str, int]] = {}

    def record_pair(self, a: str, b: str, score: float, weight: float = 1.0) -> None:
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score}")
        if not weight > 0.0:
            raise ValueError(f"weight must be > 0, got {weight}")
        for x, y, s in ((a, b, score), (b, a, 1.0 - score)):
            self._scores.setdefault(x, {}).setdefault(y, 0.0)
            self._weights.setdefault(x, {}).setdefault(y, 0.0)
            self._games.setdefault(x, {}).setdefault(y, 0)
            self._scores[x][y] += weight * s
            self._weights[x][y] += weight
            self._games[x][y] += 1

    def games(self, a: str, b: str) -> int:
        return self._games.get(a, {}).get(b, 0)

    def get_win_rate(self, a: str, b: str) -> float:
        """a's score rate against b; 0.5 if they never met."""
        weight = self._weights.get(a, {}).get(b, 0.0)
        return self._scores[a][b] / weight if weight > 0.0 else 0.5

    def get_overall_win_rate(self, agent_id: str) -> float:
        weight = sum(self._weights.get(agent_id, {}).values())
        return sum(self._scores.get(agent_id, {}).values()) / weight if weight > 0.0 else 0.5

    def get_win_rate_matrix(self, agent_ids: Sequence[str]) -> dict[str, dict[str, float]]:
        return {a: {b: self.get_win_rate(a, b) for b in agent_ids if b != a} for a in agent_ids}

    def get_games_matrix(self, agent_ids: Sequence[str]) -> dict[str, dict[str, int]]:
        return {a: {b: self.games(a, b) for b in agent_ids if b != a} for a in agent_ids}


class PastWinRate:
    """Weighted score rate of an agent's latest weights against its own checkpoints, over the
    last ``window`` member pairs (the self-play progress signal)."""

    def __init__(self, window: int = 500) -> None:
        self._window = window
        self._scores: dict[str, deque[tuple[float, float]]] = {}

    def record(self, agent_id: str, score: float, weight: float = 1.0) -> None:
        self._scores.setdefault(agent_id, deque(maxlen=self._window)).append((float(score), float(weight)))

    def get(self, agent_id: str) -> float | None:
        entries = self._scores.get(agent_id)
        if not entries:
            return None
        total = sum(w for _, w in entries)
        return sum(s * w for s, w in entries) / total if total > 0.0 else None

    def games(self, agent_id: str) -> int:
        return len(self._scores.get(agent_id, ()))


class ScoreTracker:
    """Per agent: count, mean, EMA and a 95% normal CI of the team score (one-team layouts)."""

    def __init__(self, ema_alpha: float = SCORE_EMA_ALPHA) -> None:
        self._alpha = float(ema_alpha)
        self._stats: dict[str, tuple[int, float, float, float]] = {}  # n, mean, M2, ema

    def update(self, agent_id: str, score: float) -> None:
        score = float(score)
        n, mean, m2, ema = self._stats.get(agent_id, (0, 0.0, 0.0, score))
        n += 1
        delta = score - mean
        mean += delta / n
        m2 += delta * (score - mean)
        ema = score if n == 1 else (1.0 - self._alpha) * ema + self._alpha * score
        self._stats[agent_id] = (n, mean, m2, ema)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for agent_id, (n, mean, m2, ema) in self._stats.items():
            half = Z_95 * math.sqrt(m2 / (n - 1)) / math.sqrt(n) if n >= 2 else 0.0
            out[agent_id] = {"n": n, "mean": mean, "ema": ema, "ci_low": mean - half, "ci_high": mean + half}
        return out


def composition_key(agent_ids: Iterable[str]) -> str:
    """Team composition as a multiset: sorted agent ids joined by ``+`` (``"a+a+b"``)."""
    return "+".join(sorted(agent_ids))


class CrossPlayTable:
    """Mean team score per team composition (one-team layouts)."""

    def __init__(self) -> None:
        self._cells: dict[str, list[float]] = {}  # key -> [n, sum]

    def update(self, composition: Sequence[str], score: float) -> None:
        cell = self._cells.setdefault(composition_key(composition), [0, 0.0])
        cell[0] += 1
        cell[1] += float(score)

    def summary(self) -> dict[str, dict[str, float]]:
        return {key: {"n": n, "mean": total / n} for key, (n, total) in sorted(self._cells.items())}


class _LayoutRatings:
    def __init__(self, outcome_kind: str, k_factor: float, initial_rating: float, past_window: int) -> None:
        self.outcome_kind = outcome_kind
        self.elo = EloRating(k_factor, initial_rating)
        self.win_rates = WinRateTracker()
        self.past = PastWinRate(past_window)
        self.scores = ScoreTracker()
        self.cross_play = CrossPlayTable()
        self.role_scores: dict[str, dict[str, dict[str, list[float]]]] = {}  # role -> a -> b -> [sum, weight]

    def record_role(self, pair: MemberPair) -> None:
        for role, x, y, s in ((pair.role_a, pair.a, pair.b, pair.score_a),
                              (pair.role_b, pair.b, pair.a, 1.0 - pair.score_a)):
            cell = self.role_scores.setdefault(role, {}).setdefault(x, {}).setdefault(y, [0.0, 0.0])
            cell[0] += pair.weight * s
            cell[1] += pair.weight

    def role_win_rates(self) -> dict[str, dict[str, dict[str, float]]]:
        return {role: {a: {b: s / w for b, (s, w) in row.items()} for a, row in table.items()}
                for role, table in sorted(self.role_scores.items())}


class RatingBook:
    """Rating tables of every layout of a game (module docstring)."""

    def __init__(self, spec: GameSpec, agent_ids: Sequence[str], k_factor: float = 32.0,
                 initial_rating: float = 1200.0, past_window: int = 500) -> None:
        self._agent_ids = list(agent_ids)
        self._args = (k_factor, initial_rating, past_window)
        self._books = {name: _LayoutRatings(spec.outcome_kind(name), *self._args) for name in spec.layouts}

    def update(self, result: MatchResult) -> None:
        """Record one finished match in the tables of its layout.

        Raises ValueError for a layout the game does not have, or an ``outcome_kind`` that
        differs from the layout's (both mean the result comes from another game).
        """
        book = self._books.get(result.layout)
        if book is None:
            raise ValueError(f"match {result.match_id!r}: unknown layout {result.layout!r}; "
                             f"the game has {list(self._books)}")
        if result.outcome_kind != book.outcome_kind:
            raise ValueError(f"match {result.match_id!r}: outcome_kind {result.outcome_kind!r} does not match "
                             f"layout {result.layout!r}, which is {book.outcome_kind!r}")
        if book.outcome_kind == "score":
            score = float(result.teams[0].score)
            for agent_id in sorted({seat.agent_id for seat in result.seats}):
                book.scores.update(agent_id, score)
            if len(result.seats) >= 2:
                book.cross_play.update([seat.agent_id for seat in result.seats], score)
            return
        pairs = member_pairs(result)
        cross = [p for p in pairs if p.kind == "cross"]
        book.elo.update_weighted((p.a, p.b, p.score_a, p.weight) for p in cross)
        for p in cross:
            book.win_rates.record_pair(p.a, p.b, p.score_a, p.weight)
            book.record_role(p)
        for p in pairs:
            if p.kind == "past":
                book.past.record(p.a, p.score_a, p.weight)

    def win_rate(self, layout: str, a: str, b: str) -> float:
        """a's score rate against b in ``layout``; 0.5 for pairs that never met (SP1 prior)."""
        book = self._books.get(layout)
        return 0.5 if book is None else book.win_rates.get_win_rate(a, b)

    def elo(self, layout: str, agent_id: str) -> float:
        book = self._books.get(layout)
        return float(self._args[1]) if book is None else book.elo.get(agent_id)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        ids = self._agent_ids
        return {
            layout: {
                "outcome_kind": book.outcome_kind,
                "elo": {a: book.elo.get(a) for a in ids},
                "win_rates": book.win_rates.get_win_rate_matrix(ids),
                "games": book.win_rates.get_games_matrix(ids),
                "wr_vs_past": {a: book.past.get(a) for a in ids},
                "past_games": {a: book.past.games(a) for a in ids},
                "scores": book.scores.summary(),
                "cross_play": book.cross_play.summary(),
                "role_win_rates": book.role_win_rates(),
            }
            for layout, book in self._books.items()
        }
