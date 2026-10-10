"""Lineup matchmaking (spec block 7): one ``Lineup`` per env, built for an owner agent.

A lineup is a layout (teams of seats with roles) plus one ``SeatAssignment`` per seat. One
algorithm covers solo, FFA, teams, cooperative and asymmetric games:

1. Layout: drawn by the ``matchmaking.layouts`` weights (every layout, equal weights, when
   empty), only among layouts with a seat of one of the owner's roles.
2. Match type, once per match: self-play with probability ``self_play_ratio`` (always under
   ``mode: self_play``), else arena.
3. Owner's team: uniform among the teams with a seat of one of the owner's roles.
4. A core per team. The owner's team: the owner's latest weights. Every other team, drawn
   independently: under self-play, when the owner plays a role of the team, the owner's latest
   weights (probability ``latest_prob``) or one of its checkpoints; otherwise (arena, or the
   owner plays none of the team's roles) an agent drawn by PFSP among the OTHER trainable agents
   that play a role of the team, ``f(wr) = (1 - wr) ** pfsp_exponent`` with the win rate of the
   layout; without such agents, the self-play choice.
5. The core takes one seat of a role it plays. The team's other seats of roles the core plays
   follow ``teammates``: ``self`` gives them to the core; ``mixed`` gives each to the core with
   probability ``teammate_self_prob``, else uniformly to the latest weights of another trainable
   agent with that role or to a checkpoint of the core's agent. A seat of a role the core does
   not play gets the latest weights of a uniformly drawn trainable agent that plays it (the
   owner included).
6. Seats with latest weights collect; checkpoint seats do not.
7. ``shuffle_seats``: ``permute_seats`` permutes teams of equal role composition and, inside a
   team, seats of the same role.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

from colosseum.core.config import MatchmakingConfig
from colosseum.core.errors import ConfigError
from colosseum.core.types import LATEST_NETWORK_ID, Lineup, SeatAssignment
from colosseum.envs.game import GameSpec
from colosseum.league.schedule import schedule_value

logger = logging.getLogger(__name__)

PFSP_MIN_WEIGHT = 1e-6


def enabled_layouts(spec: GameSpec, config: MatchmakingConfig) -> dict[str, float]:
    """``{layout: weight}`` matchmaking draws from: ``config.layouts``, or every layout with weight 1."""
    if not config.layouts:
        return {name: 1.0 for name in spec.layouts}
    return {name: float(weight) for name, weight in config.layouts.items()}


def validate_matchmaking(spec: GameSpec, agent_roles: Mapping[str, Sequence[str]],
                         config: MatchmakingConfig) -> None:
    """Spec block 7 checks; ConfigError with a fix hint.

    - ``matchmaking.layouts`` names only layouts of the game, with weights > 0;
    - every role of every enabled layout is played by at least one agent;
    - every agent has a seat in at least one enabled layout.
    """
    unknown = sorted(set(config.layouts) - set(spec.layouts))
    if unknown:
        raise ConfigError(
            f"matchmaking.layouts names unknown layouts {unknown}; the game's layouts are {sorted(spec.layouts)}"
        )
    not_positive = sorted(name for name, weight in config.layouts.items() if not weight > 0)
    if not_positive:
        raise ConfigError(f"matchmaking.layouts weights must be > 0; fix {not_positive}")
    layouts = enabled_layouts(spec, config)
    played = {role for roles in agent_roles.values() for role in roles}
    for name in layouts:
        missing = sorted({seat.role for seat in spec.layouts[name]} - played)
        if missing:
            raise ConfigError(
                f"layout {name!r}: no trainable agent plays role(s) {missing}; add them to some "
                f"agents.<id>.roles or leave {name!r} out of matchmaking.layouts"
            )
    for agent_id, roles in agent_roles.items():
        if not any(seat.role in roles for name in layouts for seat in spec.layouts[name]):
            raise ConfigError(
                f"agent {agent_id!r} (roles {list(roles)}) has no seat in the enabled layouts "
                f"{sorted(layouts)}; check agents.{agent_id}.roles and matchmaking.layouts"
            )


def permute_seats(spec: GameSpec, layout: str, seats: Sequence[SeatAssignment],
                  rng: random.Random) -> list[SeatAssignment]:
    """Randomly permute ``seats`` while keeping the layout's structure.

    Whole teams move only onto teams with the same multiset of roles; inside a team, an
    assignment moves only onto a seat of the same role.
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


def _seat(agent_id: str, network_id: str) -> SeatAssignment:
    """Latest weights collect, checkpoints never do."""
    return SeatAssignment(agent_id=agent_id, network_id=network_id, collect=network_id == LATEST_NETWORK_ID)


class LineupMatchmaker:
    """Builds the lineup of one match whose training data belongs to ``owner`` (module docstring)."""

    def __init__(
        self,
        *,
        spec: GameSpec,
        agent_roles: Mapping[str, Sequence[str]],
        config: MatchmakingConfig,
        checkpoints: Callable[[str], list[str]],
        win_rate: Callable[[str, str, str], float],
        rng: random.Random,
    ) -> None:
        validate_matchmaking(spec, agent_roles, config)
        self._spec = spec
        self._roles = {agent_id: frozenset(roles) for agent_id, roles in agent_roles.items()}
        self._agents = list(agent_roles)
        self._config = config
        self._checkpoints = checkpoints
        self._win_rate = win_rate
        self._rng = rng
        self._layouts = enabled_layouts(spec, config)
        # SP3 T3.1 shim: SP2's numbers read back from the v3 shares at env step 0 (exact for translated
        # SP2 configs; anchors are ignored). The built-in MixtureMatchmaker replaces this class in T3.2.
        latest = schedule_value(config.opponents.latest, 0)
        snapshots = schedule_value(config.opponents.snapshots, 0)
        self._self_play_ratio = 1.0 - schedule_value(config.opponents.rivals, 0)
        self._latest_prob = latest / (latest + snapshots) if latest + snapshots > 0 else 1.0
        self._pfsp_exponent = float(config.pfsp.exponent)

    @property
    def self_play_ratio(self) -> float:
        return self._self_play_ratio

    def playable_layouts(self, agent_id: str) -> list[str]:
        """Enabled layouts with a seat of one of ``agent_id``'s roles, in config order."""
        roles = self._roles[agent_id]
        return [name for name in self._layouts if any(seat.role in roles for seat in self._spec.layouts[name])]

    def pfsp_weight(self, layout: str, owner: str, candidate: str) -> float:
        p = self._pfsp_exponent
        return max(PFSP_MIN_WEIGHT, (1.0 - self._win_rate(layout, owner, candidate)) ** p)

    def lineup_for(self, owner: str) -> Lineup:
        if owner not in self._roles:
            raise KeyError(f"unknown agent {owner!r}; known agents: {self._agents}")
        layouts = self.playable_layouts(owner)
        layout = self._rng.choices(layouts, weights=[self._layouts[name] for name in layouts], k=1)[0]
        seat_specs = self._spec.layouts[layout]
        teams = self._spec.teams(layout)
        self_play = self._rng.random() < self._self_play_ratio
        owner_teams = [t for t, members in enumerate(teams)
                       if any(seat_specs[s].role in self._roles[owner] for s in members)]
        owner_team = self._rng.choice(owner_teams)
        seats: list[SeatAssignment | None] = [None] * len(seat_specs)
        for team, members in enumerate(teams):
            if team == owner_team:
                core = (owner, LATEST_NETWORK_ID)
            else:
                core = self._opponent_core(owner, layout, members, self_play)
            self._fill_team(core, layout, members, seats)
        lineup_seats: list[SeatAssignment] = seats  # type: ignore[assignment]
        if self._config.shuffle_seats:
            lineup_seats = permute_seats(self._spec, layout, lineup_seats, self._rng)
        return Lineup(layout=layout, seats=lineup_seats)

    def _opponent_core(self, owner: str, layout: str, members: Sequence[int], self_play: bool) -> tuple[str, str]:
        team_roles = {self._spec.layouts[layout][s].role for s in members}
        owner_plays = bool(team_roles & self._roles[owner])
        if not (self_play and owner_plays):
            candidates = [a for a in self._agents if a != owner and team_roles & self._roles[a]]
            if candidates:
                weights = [self.pfsp_weight(layout, owner, c) for c in candidates]
                return self._rng.choices(candidates, weights=weights, k=1)[0], LATEST_NETWORK_ID
        return self._self_play_core(owner)

    def _self_play_core(self, owner: str) -> tuple[str, str]:
        checkpoints = self._checkpoints(owner)
        if not checkpoints or self._rng.random() < self._latest_prob:
            return owner, LATEST_NETWORK_ID
        return owner, self._rng.choice(checkpoints)

    def _fill_team(self, core: tuple[str, str], layout: str, members: Sequence[int],
                   seats: list[SeatAssignment | None]) -> None:
        agent_id, network_id = core
        seat_specs = self._spec.layouts[layout]
        own = [s for s in members if seat_specs[s].role in self._roles[agent_id]]
        core_seat = self._rng.choice(own) if own else None
        for s in members:
            role = seat_specs[s].role
            if role not in self._roles[agent_id]:
                players = [a for a in self._agents if role in self._roles[a]]
                seats[s] = _seat(self._rng.choice(players), LATEST_NETWORK_ID)
            elif s == core_seat or self._config.teammates == "self":
                seats[s] = _seat(agent_id, network_id)
            else:
                seats[s] = self._mixed_teammate(agent_id, network_id, role)

    def _mixed_teammate(self, agent_id: str, network_id: str, role: str) -> SeatAssignment:
        if self._rng.random() < self._config.teammate_self_prob:
            return _seat(agent_id, network_id)
        candidates = [(a, LATEST_NETWORK_ID) for a in self._agents if a != agent_id and role in self._roles[a]]
        candidates += [(agent_id, c) for c in self._checkpoints(agent_id)]
        if not candidates:
            return _seat(agent_id, network_id)
        return _seat(*self._rng.choice(candidates))
