"""PFSP statistics per player (SP3 spec block 5).

For every layout and every trainable agent O (the owner) an EMA of O@latest's score against each player X
it met: the latest weights of another agent, a snapshot of any agent (O's own included) or a scripted /
frozen agent (network ``"fixed"``). Source: the member pairs of a result (``coordinator.ratings.member_pairs``:
seats of different teams, score by team ranks, weight ``1 / (T - 1)`` split between the counted pairs, a
draw = 0.5). Only pairs with O@latest on one side and anything but O@latest on the other count. The EMA step
of a pair of weight ``w`` is ``1 - 2 ** (-w / halflife_games)``; before the first game the score is the prior
(0.5). Statistics of an evicted snapshot are dropped (``forget``); a resumed run starts empty (as SP2 ratings).

``pfsp_weight`` turns a score into a candidate weight for the matchmaker: ``hard`` ``(1 - x) ** p`` (focus on
the players O loses to), ``balanced`` ``x * (1 - x)`` (focus on even players), ``uniform`` 1; floor 1e-6.
"""

from __future__ import annotations

from collections.abc import Mapping

from colosseum.coordinator.ratings import member_pairs
from colosseum.core.types import LATEST_NETWORK_ID, MatchResult

PlayerKey = tuple[str, str]  # (agent_id, network_id)

DEFAULT_HALFLIFE_GAMES = 200.0
PFSP_MIN_WEIGHT = 1e-6
PFSP_WEIGHTINGS = ("hard", "balanced", "uniform")


def player_name(player: PlayerKey) -> str:
    """``"agent@network"``: the key of a player in snapshots (``ratings.json``)."""
    return f"{player[0]}@{player[1]}"


def pfsp_weight(score: float, weighting: str, exponent: float) -> float:
    """Candidate weight of a player against whom the owner's score is ``score`` (module docstring)."""
    x = min(1.0, max(0.0, float(score)))
    if weighting == "hard":
        weight = (1.0 - x) ** float(exponent)
    elif weighting == "balanced":
        weight = x * (1.0 - x)
    elif weighting == "uniform":
        weight = 1.0
    else:
        raise ValueError(f"unknown PFSP weighting {weighting!r}; expected one of {PFSP_WEIGHTINGS}")
    return max(weight, PFSP_MIN_WEIGHT)


class PfspStats:
    """Per layout, per owner, per player: ``[score EMA, games]`` (module docstring).

    ``halflife_by_agent`` maps every owner to track (the trainable agents) to its ``halflife_games``; pairs of
    other owners are ignored.
    """

    def __init__(self, halflife_by_agent: Mapping[str, float], prior: float = 0.5) -> None:
        for agent_id, halflife in halflife_by_agent.items():
            if not float(halflife) > 0.0:
                raise ValueError(f"PFSP halflife_games of {agent_id!r} must be > 0, got {halflife}")
        self._halflife = {agent_id: float(h) for agent_id, h in halflife_by_agent.items()}
        self._prior = float(prior)
        self._table: dict[str, dict[str, dict[PlayerKey, list[float]]]] = {}

    def update(self, result: MatchResult) -> None:
        """Fold one finished match into the statistics of its layout."""
        for pair in member_pairs(result):
            self._record(result.layout, pair.a, pair.net_a, (pair.b, pair.net_b), pair.score_a, pair.weight)
            self._record(result.layout, pair.b, pair.net_b, (pair.a, pair.net_a), 1.0 - pair.score_a, pair.weight)

    def _record(self, layout: str, owner: str, network: str, player: PlayerKey, score: float, weight: float) -> None:
        if network != LATEST_NETWORK_ID or player == (owner, LATEST_NETWORK_ID):
            return
        halflife = self._halflife.get(owner)
        if halflife is None:
            return
        cell = self._table.setdefault(layout, {}).setdefault(owner, {}).setdefault(player, [self._prior, 0.0])
        cell[0] += (1.0 - 2.0 ** (-float(weight) / halflife)) * (float(score) - cell[0])
        cell[1] += float(weight)

    def _cell(self, layout: str, owner: str, player: PlayerKey) -> list[float] | None:
        return self._table.get(layout, {}).get(owner, {}).get(tuple(player))

    def score(self, layout: str, owner: str, player: PlayerKey) -> float:
        """The owner's score EMA against ``player`` in ``layout`` (the prior before the first game)."""
        cell = self._cell(layout, owner, player)
        return self._prior if cell is None else cell[0]

    def games(self, layout: str, owner: str, player: PlayerKey) -> float:
        """Summed pair weights behind ``score`` (fractional for team and FFA layouts)."""
        cell = self._cell(layout, owner, player)
        return 0.0 if cell is None else cell[1]

    def forget(self, player: PlayerKey) -> None:
        """Drop ``player`` everywhere (an evicted snapshot)."""
        key = tuple(player)
        for owners in self._table.values():
            for players in owners.values():
                players.pop(key, None)

    def snapshot(self) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
        """``{layout: {owner: {"agent@net": {"score", "games"}}}}`` (JSON-ready; players sorted)."""
        return {
            layout: {
                owner: {player_name(p): {"score": cell[0], "games": cell[1]} for p, cell in sorted(players.items())}
                for owner, players in owners.items()
            }
            for layout, owners in self._table.items()
        }
