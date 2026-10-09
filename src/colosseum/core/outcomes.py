"""Team ranks and scores of a finished episode (SP2 spec block 1, "Конец эпизода").

- No ``Outcome`` (or neither field set): team score = mean of its seats' episode returns
  (the mean, so teams of different sizes compare fairly), ranks from the scores.
- Only ``team_score``: ranks from the scores (higher is better).
- Only ``team_rank``: score = the same mean of returns.
Ranks from scores: ``rank = 1 + number of strictly better teams`` (ties share a rank).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from colosseum.core.errors import EnvContractError
from colosseum.envs.game import Outcome


def _ranks_from_scores(score: Mapping[int, float]) -> dict[int, float]:
    return {t: float(1 + sum(1 for u in score if score[u] > s)) for t, s in score.items()}


def _keys(values: Mapping) -> list:
    """The keys sorted when they are comparable, else in the env's order (mixed key types)."""
    try:
        return sorted(values)
    except TypeError:
        return list(values)


def _checked(values: Mapping[int, float], field: str, num_teams: int, where: str) -> dict[int, float]:
    prefix = f"{where}: " if where else ""
    if not isinstance(values, Mapping) or set(values) != set(range(num_teams)):
        got = _keys(values) if isinstance(values, Mapping) else type(values).__name__
        raise EnvContractError(f"{prefix}outcome.{field} keys {got} must be exactly the layout's teams "
                               f"{list(range(num_teams))}")
    out: dict[int, float] = {}
    for team in range(num_teams):
        try:
            value = float(values[team])
        except (TypeError, ValueError):
            raise EnvContractError(f"{prefix}outcome.{field}[{team}] must be a number, "
                                   f"got {values[team]!r}") from None
        if not math.isfinite(value):
            raise EnvContractError(f"{prefix}outcome.{field}[{team}] must be finite, got {value}")
        out[team] = value
    return out


def resolve_outcome(outcome: Outcome | None, teams: list[list[int]], seat_returns: Sequence[float],
                    where: str = "") -> tuple[dict[int, float], dict[int, float]]:
    """``(team_rank, team_score)`` for the layout's ``teams`` (team index -> seats)."""
    mean_returns = {t: sum(float(seat_returns[s]) for s in seats) / len(seats) for t, seats in enumerate(teams)}
    rank = score = None
    if outcome is not None:
        if outcome.team_score is not None:
            score = _checked(outcome.team_score, "team_score", len(teams), where)
        if outcome.team_rank is not None:
            rank = _checked(outcome.team_rank, "team_rank", len(teams), where)
    if score is None:
        score = mean_returns
    if rank is None:
        rank = _ranks_from_scores(score)
    return rank, score


def pairwise_rank_score(rank_a: float, rank_b: float) -> float:
    """Score of A against B from team ranks: 1 (A better), 0.5 (tie), 0 (B better)."""
    if rank_a < rank_b:
        return 1.0
    if rank_a == rank_b:
        return 0.5
    return 0.0
