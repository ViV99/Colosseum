"""The SP2 env contract: ``GameSpec`` + ``MultiAgentEnv`` + ``StepResult`` (spec block 1).

A game has *roles* (observation, action and optional global-state spaces) and *layouts*
(match variants): a layout is a tuple of seats, each with a role and a team ``0..T-1``.
Seats ``0..n-1`` of a layout are used; ``n..max_seats-1`` stay empty. The outcome kind
follows from the number of teams: 1 -> ``score``, 2 -> ``wdl``, 3 or more -> ``rank``.

``reset(seed, layout)`` and ``step(actions)`` return a :class:`StepResult`. ``acting`` lists
the seats that act on the NEXT step, and ``step`` receives exactly one action per acting
seat. :class:`colosseum.sp2.envs.contract.EpisodeTracker` checks every rule.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import gymnasium

from colosseum.core.errors import EnvContractError

_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]*")
DEFAULT_ROLE = "player"


@dataclass(frozen=True)
class RoleSpec:
    observation_space: gymnasium.Space
    action_space: gymnasium.Space
    global_state_space: gymnasium.Space | None = None


@dataclass(frozen=True)
class SeatSpec:
    role: str
    team: int


@dataclass(frozen=True)
class GameSpec:
    roles: dict[str, RoleSpec]
    layouts: dict[str, tuple[SeatSpec, ...]]

    def __post_init__(self) -> None:
        # Accept lists of seats; store tuples (the spec is immutable and comparable).
        object.__setattr__(self, "roles", dict(self.roles))
        object.__setattr__(self, "layouts", {name: tuple(seats) for name, seats in self.layouts.items()})

    # ---- structure --------------------------------------------------------

    def _seats(self, layout: str) -> tuple[SeatSpec, ...]:
        try:
            return self.layouts[layout]
        except KeyError:
            raise EnvContractError(f"unknown layout {layout!r}; the game has {list(self.layouts)}") from None

    @property
    def max_seats(self) -> int:
        return max(len(seats) for seats in self.layouts.values())

    def layout_size(self, layout: str) -> int:
        return len(self._seats(layout))

    def teams(self, layout: str) -> list[list[int]]:
        seats = self._seats(layout)
        out: list[list[int]] = [[] for _ in range(1 + max(s.team for s in seats))]
        for i, seat in enumerate(seats):
            out[seat.team].append(i)
        return out

    def num_teams(self, layout: str) -> int:
        return len(self.teams(layout))

    def outcome_kind(self, layout: str) -> Literal["score", "wdl", "rank"]:
        n = self.num_teams(layout)
        return "score" if n == 1 else "wdl" if n == 2 else "rank"

    def role_of(self, layout: str, seat: int) -> str:
        seats = self._seats(layout)
        if not 0 <= seat < len(seats):
            raise EnvContractError(f"layout {layout!r} has seats 0..{len(seats) - 1}, not {seat}")
        return seats[seat].role

    def validate(self) -> None:
        """Raise EnvContractError naming the first problem of the spec."""
        from colosseum.sp2.core.specs import ActionSpec, ObsSpec

        if not self.roles:
            raise EnvContractError("GameSpec: no roles")
        if not self.layouts:
            raise EnvContractError("GameSpec: no layouts (at least one is needed)")
        for name, role in self.roles.items():
            if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
                raise EnvContractError(f"GameSpec: role name {name!r} is not a safe identifier "
                                       f"(letters, digits, '_' and '-', not starting with '-')")
            if not isinstance(role, RoleSpec):
                raise EnvContractError(f"GameSpec: role {name!r} must be a RoleSpec, got {type(role).__name__}")
            try:
                ObsSpec.from_space(role.observation_space)
                ActionSpec.from_space(role.action_space)
                if role.global_state_space is not None:
                    ObsSpec.from_space(role.global_state_space)
            except (TypeError, ValueError) as e:
                raise EnvContractError(f"GameSpec: role {name!r}: {e}") from e
        for layout, seats in self.layouts.items():
            if not isinstance(layout, str) or _NAME_RE.fullmatch(layout) is None:
                raise EnvContractError(f"GameSpec: layout name {layout!r} is not a safe identifier "
                                       f"(letters, digits, '_' and '-', not starting with '-')")
            if not seats:
                raise EnvContractError(f"GameSpec: layout {layout!r} has no seats")
            for i, seat in enumerate(seats):
                if not isinstance(seat, SeatSpec):
                    raise EnvContractError(f"GameSpec: layout {layout!r} seat {i} must be a SeatSpec, "
                                           f"got {type(seat).__name__}")
                if seat.role not in self.roles:
                    raise EnvContractError(f"GameSpec: layout {layout!r} seat {i} has unknown role {seat.role!r} "
                                           f"(roles: {list(self.roles)})")
            teams = sorted({seat.team for seat in seats})
            if teams != list(range(len(teams))):
                raise EnvContractError(f"GameSpec: layout {layout!r} uses teams {teams}; team numbers must be "
                                       f"exactly 0..T-1")

    # ---- helpers -----------------------------------------------------------

    @classmethod
    def solo(cls, obs: gymnasium.Space, act: gymnasium.Space,
             global_state: gymnasium.Space | None = None) -> GameSpec:
        """One seat, role ``"player"``, layout ``"solo"``."""
        return cls(roles={DEFAULT_ROLE: RoleSpec(obs, act, global_state)},
                   layouts={"solo": (SeatSpec(DEFAULT_ROLE, 0),)})

    @classmethod
    def symmetric(cls, num_players: int | Iterable[int], obs: gymnasium.Space, act: gymnasium.Space,
                  global_state: gymnasium.Space | None = None) -> GameSpec:
        """Free-for-all: team = seat; one layout ``"<n>p"`` per player count."""
        counts = [num_players] if isinstance(num_players, int) else list(num_players)
        if not counts or any(int(n) < 1 for n in counts):
            raise ValueError(f"GameSpec.symmetric: player counts must be >= 1, got {counts}")
        layouts = {f"{n}p": tuple(SeatSpec(DEFAULT_ROLE, i) for i in range(int(n))) for n in counts}
        if len(layouts) != len(counts):
            raise ValueError(f"GameSpec.symmetric: duplicate player counts in {counts}")
        return cls(roles={DEFAULT_ROLE: RoleSpec(obs, act, global_state)}, layouts=layouts)

    @classmethod
    def teams_of(cls, sizes: Sequence[int] | Iterable[Sequence[int]], obs: gymnasium.Space, act: gymnasium.Space,
                 global_state: gymnasium.Space | None = None) -> GameSpec:
        """Teams of the given sizes: ``[2, 2]`` -> ``"2v2"``, ``[2, 1, 1]`` -> ``"2v1v1"``, ``[n]`` ->
        ``"coop<n>"``; a list of such lists gives several layouts. Seats are numbered team by team."""
        items = list(sizes)
        groups = [items] if items and all(isinstance(x, int) for x in items) else [list(g) for g in items]
        layouts: dict[str, tuple[SeatSpec, ...]] = {}
        for group in groups:
            if not group or any(int(n) < 1 for n in group):
                raise ValueError(f"GameSpec.teams_of: team sizes must be >= 1, got {group}")
            name = f"coop{group[0]}" if len(group) == 1 else "v".join(str(int(n)) for n in group)
            if name in layouts:
                raise ValueError(f"GameSpec.teams_of: duplicate layout {name!r}")
            layouts[name] = tuple(SeatSpec(DEFAULT_ROLE, t) for t, n in enumerate(group) for _ in range(int(n)))
        return cls(roles={DEFAULT_ROLE: RoleSpec(obs, act, global_state)}, layouts=layouts)


@dataclass
class Outcome:
    """Team-level result of an episode (keys: the layout's teams)."""

    team_rank: dict[int, float] | None = None    # 1 = best; ties share a rank; fractions allowed
    team_score: dict[int, float] | None = None   # game score of the team


@dataclass
class StepResult:
    acting: set[int]                                     # seats that act on the NEXT step
    obs: dict[int, Any]                                  # required for every acting seat
    action_masks: dict[int, Any] = field(default_factory=dict)   # acting seats only; missing = all allowed
    rewards: dict[int, float] = field(default_factory=dict)      # any live seat; missing = 0
    terminated: set[int] = field(default_factory=set)            # seats eliminated in this step
    episode_over: bool = False
    truncated: bool = False                              # ended by an artificial limit
    final_obs: dict[int, Any] | None = None              # with truncated: every live seat
    global_state: dict[int, Any] | None = None           # per seat, from its perspective
    outcome: Outcome | None = None                       # with episode_over
    infos: dict[int, dict] = field(default_factory=dict)


class MultiAgentEnv(ABC):
    """A game with seats; ``spec`` may be a class or an instance attribute."""

    spec: GameSpec

    @abstractmethod
    def reset(self, seed: int | None, layout: str) -> StepResult:
        """Start an episode of ``layout``: acting, obs, action_masks, global_state; no rewards."""

    @abstractmethod
    def step(self, actions: dict[int, Any]) -> StepResult:
        """One step; ``actions`` has exactly the seats of the previous result's ``acting``."""

    def close(self) -> None:
        """Release resources (optional)."""
