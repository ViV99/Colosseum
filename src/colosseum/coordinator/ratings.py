"""Rating systems: pairwise ELO, pairwise win rates, latest-vs-past win rate.

Every tracker consumes *pairwise* scores (1.0 win, 0.5 draw, 0.0 loss). The
coordinator extracts these scores from the per-seat outcomes of each match.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def pairwise_score(outcome_a: float, outcome_b: float) -> float:
    """1.0 if a's outcome is higher, 0.0 if lower, 0.5 if equal."""
    if outcome_a > outcome_b:
        return 1.0
    if outcome_a < outcome_b:
        return 0.0
    return 0.5


@dataclass
class EloRating:
    """Pairwise ELO.

    ``update_pairs`` computes every delta from the ratings *before* the match and
    applies them together, so the result does not depend on the order of the pairs.
    """

    k_factor: float = 32.0
    initial_rating: float = 1200.0
    _ratings: dict[str, float] = field(default_factory=dict)

    def get(self, agent_id: str) -> float:
        return self._ratings.get(agent_id, self.initial_rating)

    def register(self, agent_id: str) -> None:
        self._ratings.setdefault(agent_id, self.initial_rating)

    def expected(self, a: str, b: str) -> float:
        """Expected score of ``a`` against ``b``."""
        return 1.0 / (1.0 + math.pow(10.0, (self.get(b) - self.get(a)) / 400.0))

    def update_pairs(self, pairs: Iterable[tuple[str, str, float]], k_scale: float = 1.0) -> None:
        """Apply several pairwise results ``(a, b, score_a)`` simultaneously."""
        k = self.k_factor * k_scale
        deltas: dict[str, float] = {}
        for a, b, score_a in pairs:
            e_a = self.expected(a, b)
            deltas[a] = deltas.get(a, 0.0) + k * (score_a - e_a)
            deltas[b] = deltas.get(b, 0.0) + k * ((1.0 - score_a) - (1.0 - e_a))
        for agent_id, delta in deltas.items():
            self._ratings[agent_id] = self.get(agent_id) + delta

    def update_pair(self, a: str, b: str, score_a: float, k_scale: float = 1.0) -> None:
        """Update both ratings from one pairwise result; ``score_a`` in {0, 0.5, 1}."""
        self.update_pairs([(a, b, score_a)], k_scale=k_scale)

    @property
    def all_ratings(self) -> dict[str, float]:
        return dict(self._ratings)


@dataclass
class WinRateTracker:
    """Pairwise win rates stored as float score sums (a draw adds 0.5 to both sides)."""

    _scores: dict[str, dict[str, float]] = field(default_factory=dict)
    _games: dict[str, dict[str, int]] = field(default_factory=dict)

    def record_pair(self, a: str, b: str, score_a: float) -> None:
        """Record one pairwise result: 1.0 a won, 0.5 draw, 0.0 b won."""
        if not 0.0 <= score_a <= 1.0:
            raise ValueError(f"score_a must be in [0, 1], got {score_a}")
        self._scores.setdefault(a, {}).setdefault(b, 0.0)
        self._scores.setdefault(b, {}).setdefault(a, 0.0)
        self._games.setdefault(a, {}).setdefault(b, 0)
        self._games.setdefault(b, {}).setdefault(a, 0)
        self._scores[a][b] += score_a
        self._scores[b][a] += 1.0 - score_a
        self._games[a][b] += 1
        self._games[b][a] += 1

    def games(self, a: str, b: str) -> int:
        return self._games.get(a, {}).get(b, 0)

    def get_win_rate(self, a: str, b: str) -> float:
        """a's score rate against b; 0.5 if they never met."""
        n = self.games(a, b)
        if n == 0:
            return 0.5
        return self._scores[a][b] / n

    def get_overall_win_rate(self, agent_id: str) -> float:
        n = sum(self._games.get(agent_id, {}).values())
        if n == 0:
            return 0.5
        return sum(self._scores.get(agent_id, {}).values()) / n

    def get_win_rate_matrix(self, agent_ids: list[str]) -> dict[str, dict[str, float]]:
        return {a: {b: self.get_win_rate(a, b) for b in agent_ids if b != a} for a in agent_ids}

    def get_games_matrix(self, agent_ids: list[str]) -> dict[str, dict[str, int]]:
        return {a: {b: self.games(a, b) for b in agent_ids if b != a} for a in agent_ids}


class PastWinRate:
    """Score rate of an agent's latest weights against its own checkpoints.

    Only the last ``window`` pairwise scores per agent are kept, so the value tracks
    current progress (the self-play progress signal, R4-07) rather than an all-time mean.
    """

    def __init__(self, window: int = 500) -> None:
        self._window = window
        self._scores: dict[str, deque[float]] = {}

    def record(self, agent_id: str, score_latest: float) -> None:
        self._scores.setdefault(agent_id, deque(maxlen=self._window)).append(float(score_latest))

    def get(self, agent_id: str) -> float | None:
        scores = self._scores.get(agent_id)
        if not scores:
            return None
        return sum(scores) / len(scores)

    def games(self, agent_id: str) -> int:
        return len(self._scores.get(agent_id, ()))
