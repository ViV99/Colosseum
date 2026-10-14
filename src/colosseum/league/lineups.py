"""Lineup helpers of the league (spec block 5): enabled layouts, seat permutation, lineup checks.

``enabled_layouts`` and ``permute_seats`` moved here from SP2's ``coordinator/matchmaker.py``.
``check_lineup`` is the framework's check of every lineup a matchmaker (built-in or custom)
returns.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from colosseum.core.config import MatchmakingConfig
from colosseum.core.types import (
    FIXED_NETWORK_ID,
    LATEST_NETWORK_ID,
    OPPONENT_CATEGORIES,
    SOURCE_OWNER,
    Lineup,
    SeatAssignment,
)
from colosseum.envs.game import GameSpec

if TYPE_CHECKING:
    from colosseum.league.base import MatchmakerContext

_SOURCES = ("", SOURCE_OWNER, *OPPONENT_CATEGORIES)


def enabled_layouts(spec: GameSpec, config: MatchmakingConfig) -> dict[str, float]:
    """``{layout: weight}`` matchmaking draws from: ``config.layouts``, or every layout with weight 1."""
    if not config.layouts:
        return {name: 1.0 for name in spec.layouts}
    return {name: float(weight) for name, weight in config.layouts.items()}


def playable_layouts(spec: GameSpec, config: MatchmakingConfig, roles: Iterable[str]) -> list[str]:
    """Enabled layouts (of the game) with a seat of one of ``roles``, in config order."""
    roles = set(roles)
    return [name for name in enabled_layouts(spec, config)
            if name in spec.layouts and any(seat.role in roles for seat in spec.layouts[name])]


def permute_seats(spec: GameSpec, layout: str, seats: Sequence[SeatAssignment],
                  rng: random.Random) -> list[SeatAssignment]:
    """Randomly permute ``seats`` while keeping the layout's structure.

    Whole teams move only onto teams with the same multiset of roles; inside a team, an
    assignment moves only onto a seat of the same role. Assignments move as objects, so each
    keeps its ``source``.
    """
    seat_specs = spec.layouts[layout]
    if len(seats) != len(seat_specs):
        raise ValueError(f"permute_seats: layout {layout!r} has {len(seat_specs)} seats, got {len(seats)}")
    teams = spec.teams(layout)
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for team, members in enumerate(teams):
        groups[tuple(sorted(seat_specs[s].role for s in members))].append(team)
    out: list[SeatAssignment | None] = [None] * len(seats)
    for team_ids in groups.values():
        targets = list(team_ids)
        rng.shuffle(targets)
        for source, target in zip(team_ids, targets, strict=True):
            target_by_role: dict[str, list[int]] = defaultdict(list)
            for s in teams[target]:
                target_by_role[seat_specs[s].role].append(s)
            for role_seats in target_by_role.values():
                rng.shuffle(role_seats)
            for s in teams[source]:
                out[target_by_role[seat_specs[s].role].pop()] = seats[s]
    return out  # type: ignore[return-value]


def check_lineup(context: MatchmakerContext, lineup: Lineup, who: str) -> None:
    """ValueError naming ``who`` and the lineup unless ``lineup`` is valid (spec block 5).

    The layout exists; the seat count is the layout's; every seat's agent exists and plays the
    seat's role; a trainable agent plays ``latest`` or one of its stored snapshots, a scripted or
    frozen agent plays ``fixed``; only the latest weights of trainable agents collect; ``source``
    is empty, ``owner`` or an opponent category.
    """
    if not isinstance(lineup, Lineup):
        raise ValueError(f"{who}: lineup_for must return a Lineup, got {type(lineup).__name__}")

    def fail(problem: str) -> ValueError:
        return ValueError(f"{who}: invalid lineup {lineup}: {problem}")

    spec = context.spec
    if lineup.layout not in spec.layouts:
        raise fail(f"unknown layout {lineup.layout!r} (the game has {sorted(spec.layouts)})")
    seat_specs = spec.layouts[lineup.layout]
    if len(lineup.seats) != len(seat_specs):
        raise fail(f"{len(lineup.seats)} seats for layout {lineup.layout!r}, which has {len(seat_specs)}")
    snapshots: dict[str, set[str]] = {}
    for seat, (assignment, seat_spec) in enumerate(zip(lineup.seats, seat_specs, strict=True)):
        agent_id, network_id = assignment.agent_id, assignment.network_id
        view = context.agents.get(agent_id)
        if view is None:
            raise fail(f"seat {seat}: unknown agent {agent_id!r} (agents: {list(context.agents)})")
        if seat_spec.role not in view.roles:
            raise fail(f"seat {seat}: agent {agent_id!r} does not play role {seat_spec.role!r} "
                       f"(it plays {sorted(view.roles)})")
        trainable = view.kind == "trainable"
        if trainable and network_id == FIXED_NETWORK_ID:
            raise fail(f"seat {seat}: trainable agent {agent_id!r} plays {LATEST_NETWORK_ID!r} or a stored "
                       f"snapshot, not network {FIXED_NETWORK_ID!r}")
        if trainable and network_id != LATEST_NETWORK_ID:
            stored = snapshots.setdefault(agent_id, set(context.snapshots(agent_id)))
            if network_id not in stored:
                raise fail(f"seat {seat}: agent {agent_id!r} has no snapshot {network_id!r} (stored: {sorted(stored)})")
        if not trainable and network_id != FIXED_NETWORK_ID:
            raise fail(f"seat {seat}: {view.kind} agent {agent_id!r} must play network {FIXED_NETWORK_ID!r}, "
                       f"not {network_id!r}")
        if assignment.collect and not (trainable and network_id == LATEST_NETWORK_ID):
            raise fail(f"seat {seat}: only the latest weights of trainable agents collect; "
                       f"{agent_id!r}@{network_id} has collect=True")
        if assignment.source not in _SOURCES:
            raise fail(f"seat {seat}: unknown source {assignment.source!r} (expected one of {list(_SOURCES)})")
