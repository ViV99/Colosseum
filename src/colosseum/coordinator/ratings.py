"""Rating systems for tracking agent skill: ELO and win-rate tracking.

Used by the coordinator to track relative agent strength and by PFSP
matchmakers to prioritize opponents.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EloRating:
    """ELO rating tracker for pairwise matches."""

    k_factor: float = 32.0
    initial_rating: float = 1200.0
    _ratings: dict[str, float] = field(default_factory=dict)

    def get(self, agent_id: str) -> float:
        return self._ratings.get(agent_id, self.initial_rating)

    def register(self, agent_id: str) -> None:
        if agent_id not in self._ratings:
            self._ratings[agent_id] = self.initial_rating

    def update(self, winner_id: str, loser_id: str, draw: bool = False) -> tuple[float, float]:
        """Update ratings after a match.

        Args:
            winner_id: The winning agent (or either agent if draw).
            loser_id: The losing agent (or the other agent if draw).
            draw: If True, the match was a draw.

        Returns:
            (new_winner_rating, new_loser_rating)
        """
        self.register(winner_id)
        self.register(loser_id)

        ra = self._ratings[winner_id]
        rb = self._ratings[loser_id]

        expected_a = 1.0 / (1.0 + math.pow(10.0, (rb - ra) / 400.0))
        expected_b = 1.0 - expected_a

        if draw:
            score_a, score_b = 0.5, 0.5
        else:
            score_a, score_b = 1.0, 0.0

        self._ratings[winner_id] = ra + self.k_factor * (score_a - expected_a)
        self._ratings[loser_id] = rb + self.k_factor * (score_b - expected_b)

        return self._ratings[winner_id], self._ratings[loser_id]

    @property
    def all_ratings(self) -> dict[str, float]:
        return dict(self._ratings)


@dataclass
class WinRateTracker:
    """Tracks pairwise win rates between agents."""

    _wins: dict[str, dict[str, int]] = field(default_factory=dict)
    _matches: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, agent_a: str, agent_b: str, outcome_a: float) -> None:
        """Record a match outcome.

        Args:
            agent_a: First agent.
            agent_b: Second agent.
            outcome_a: 1.0 = a wins, 0.0 = b wins, 0.5 = draw.
        """
        for agent in (agent_a, agent_b):
            if agent not in self._wins:
                self._wins[agent] = {}
                self._matches[agent] = {}

        self._matches[agent_a].setdefault(agent_b, 0)
        self._matches[agent_b].setdefault(agent_a, 0)
        self._wins[agent_a].setdefault(agent_b, 0)
        self._wins[agent_b].setdefault(agent_a, 0)

        self._matches[agent_a][agent_b] += 1
        self._matches[agent_b][agent_a] += 1

        # Record using integer counts scaled by 2 to handle draws cleanly
        # win=2, draw=1, loss=0 (then divide by 2*matches for win rate)
        self._wins[agent_a][agent_b] += int(outcome_a * 2)
        self._wins[agent_b][agent_a] += int((1.0 - outcome_a) * 2)

    def get_win_rate(self, agent_a: str, agent_b: str) -> float:
        """Get agent_a's win rate against agent_b. Returns 0.5 if no matches."""
        matches = self._matches.get(agent_a, {}).get(agent_b, 0)
        if matches == 0:
            return 0.5
        wins = self._wins.get(agent_a, {}).get(agent_b, 0)
        return wins / (2.0 * matches)

    def get_overall_win_rate(self, agent_id: str) -> float:
        """Get agent's overall win rate across all opponents."""
        total_wins = 0
        total_matches = 0
        for opp, count in self._matches.get(agent_id, {}).items():
            total_matches += count
            total_wins += self._wins.get(agent_id, {}).get(opp, 0)
        if total_matches == 0:
            return 0.5
        return total_wins / (2.0 * total_matches)

    def get_win_rate_matrix(self, agent_ids: list[str]) -> dict[str, dict[str, float]]:
        """Get full pairwise win rate matrix."""
        return {
            a: {b: self.get_win_rate(a, b) for b in agent_ids}
            for a in agent_ids
        }
