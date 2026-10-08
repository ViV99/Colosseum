"""``EpisodeTracker``: checks one env's results against the contract and tracks seat phases.

Every rule of SP2 spec block 1 lives here (with the mask rules of
:mod:`colosseum.sp2.core.specs`); ``MatchRunner`` (training and eval) and ``validate`` use it.

Seat lifecycle (seats ``0..n-1`` of the layout; ``n..max_seats-1`` are EMPTY):
- EMPTY never acts and never gets an observation, a mask, a reward or ``terminated``;
- LIVE acts (in ``acting``) or waits; it may get rewards at any step; observations,
  masks and global states of waiting seats are ignored;
- ELIMINATED (``terminated`` in some step): a reward in that same step is allowed (it is
  applied first); afterwards the seat may not act, get rewards or be terminated again.

A step with an empty ``acting`` and no ``episode_over`` is an idle tick; more than
``max_idle_steps`` of them in a row is an error (an env that forgot ``episode_over``).
"""

from __future__ import annotations

import enum
import math
import numbers
from collections.abc import Mapping, Set
from typing import Any

from colosseum.core.errors import EnvContractError
from colosseum.sp2.core.outcomes import resolve_outcome
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.tree import Tree
from colosseum.sp2.envs.game import GameSpec, StepResult


class SeatPhase(enum.Enum):
    EMPTY = "empty"
    LIVE = "live"
    ELIMINATED = "eliminated"


class _RoleRules:
    """The specs of one role, built once."""

    def __init__(self, spec: GameSpec, role: str) -> None:
        r = spec.roles[role]
        self.obs = ObsSpec.from_space(r.observation_space)
        self.action = ActionSpec.from_space(r.action_space)
        self.global_state = ObsSpec.from_space(r.global_state_space) if r.global_state_space is not None else None


class EpisodeTracker:
    """Validates one env's StepResults against its GameSpec and tracks seat phases."""

    def __init__(self, spec: GameSpec, *, max_idle_steps: int = 1000, context: str = "") -> None:
        self.spec = spec
        self.max_idle_steps = int(max_idle_steps)
        self.context = context
        self._rules = {role: _RoleRules(spec, role) for role in spec.roles}
        self.layout: str | None = None
        self.episode_step = 0
        self.episode_over = False
        self._phase: list[SeatPhase] = [SeatPhase.EMPTY] * spec.max_seats
        self._acting: set[int] = set()
        self._returns: list[float] = []
        self._eliminated: dict[int, int] = {}
        self._idle = 0
        self._team: tuple[dict[int, float], dict[int, float]] | None = None

    # ---- queries ----------------------------------------------------------------

    def phase(self, seat: int) -> SeatPhase:
        return self._phase[seat] if 0 <= seat < len(self._phase) else SeatPhase.EMPTY

    def live_seats(self) -> list[int]:
        return [s for s, p in enumerate(self._phase) if p is SeatPhase.LIVE]

    def acting(self) -> set[int]:
        return set(self._acting)

    def seat_returns(self) -> list[float]:
        """Undiscounted return of every seat of the layout in this episode so far."""
        return list(self._returns)

    def eliminated_step(self, seat: int) -> int | None:
        """The episode step in which ``seat`` was terminated, or None."""
        return self._eliminated.get(seat)

    def team_result(self) -> tuple[dict[int, float], dict[int, float]]:
        """``(team_rank, team_score)`` of the finished episode (resolved in its final step)."""
        if self._team is None:
            raise RuntimeError("team_result() is only available after episode_over")
        return self._team

    # ---- context ----------------------------------------------------------------

    def _where(self, seat: int | None = None) -> str:
        parts = [self.context] if self.context else []
        if seat is not None:
            parts.append(f"seat {seat}")
        parts.append(f"episode step {self.episode_step}")
        if self.layout is not None:
            parts.append(f"layout {self.layout}")
        return ", ".join(parts)

    def _fail(self, message: str, seat: int | None = None) -> EnvContractError:
        return EnvContractError(f"{self._where(seat)}: {message}")

    def _rules_of(self, seat: int) -> _RoleRules:
        return self._rules[self.spec.role_of(self.layout, seat)]

    # ---- lifecycle -----------------------------------------------------------------

    def on_reset(self, layout: str, result: StepResult) -> dict[int, Tree | None]:
        """Start an episode of ``layout`` with the env's reset result; returns the acting seats' masks."""
        self.layout = None
        self.episode_step = 0
        if layout not in self.spec.layouts:
            raise self._fail(f"layout {layout!r} is not one of the game's layouts {list(self.spec.layouts)}")
        self.layout = layout
        if not isinstance(result, StepResult):
            raise self._fail(f"reset must return a StepResult, got {type(result).__name__}")
        self._check_fields(result)
        n = self.spec.layout_size(layout)
        self._phase = [SeatPhase.LIVE if s < n else SeatPhase.EMPTY for s in range(self.spec.max_seats)]
        self._returns = [0.0] * n
        self._eliminated = {}
        self._idle = 0
        self.episode_over = False
        self._acting = set()
        self._team = None
        if result.rewards:
            raise self._fail(f"reset must not give rewards, got {result.rewards}")
        if result.terminated:
            raise self._fail(f"reset must not terminate seats, got {sorted(result.terminated)}")
        if result.episode_over or result.truncated:
            raise self._fail("reset must not end the episode (episode_over/truncated set)")
        if result.outcome is not None:
            raise self._fail("reset must not report an outcome")
        return self._accept_next(result)

    def on_step(self, actions: dict[int, Any], result: StepResult) -> dict[int, Tree | None]:
        """Check the actions sent and the env's step result; returns the acting seats' masks."""
        if self.layout is None:
            raise RuntimeError("on_step before on_reset")
        if self.episode_over:
            raise RuntimeError("on_step after episode_over; reset the env first")
        self.episode_step += 1
        if set(actions) != self._acting:
            raise self._fail(f"the actions sent are for seats {sorted(actions)}, but the acting seats were "
                             f"{sorted(self._acting)}")
        if not isinstance(result, StepResult):
            raise self._fail(f"step must return a StepResult, got {type(result).__name__}")
        self._check_fields(result)
        self._check_seat_keys(result.rewards, "a reward")
        for seat, value in result.rewards.items():
            try:
                reward = float(value)
            except (TypeError, ValueError):
                raise self._fail(f"reward must be a number, got {value!r}", seat) from None
            if not math.isfinite(reward):
                raise self._fail(f"reward must be finite, got {reward}", seat)
        self._check_seat_keys(result.terminated, "terminated")
        if result.outcome is not None and not result.episode_over:
            raise self._fail("outcome is only allowed with episode_over")
        if result.truncated and not result.episode_over:
            raise self._fail("truncated requires episode_over")
        if result.episode_over:
            if result.acting:
                raise self._fail(f"episode_over with a non-empty acting set {sorted(result.acting)}")
            self._check_empty_seat_data(result)
            self._check_global_state_roles(result)
            if result.truncated:
                self._check_truncation(result)
        overlap = set(result.acting) & set(result.terminated)
        if overlap:
            raise self._fail(f"seats {sorted(overlap)} are terminated and acting in the same step")
        masks = self._accept_next(result) if not result.episode_over else {}
        for seat, value in result.rewards.items():
            self._returns[seat] += float(value)
        for seat in result.terminated:
            self._phase[seat] = SeatPhase.ELIMINATED
            self._eliminated[seat] = self.episode_step
        if result.episode_over:
            self._team = resolve_outcome(result.outcome, self.spec.teams(self.layout), self._returns, self._where())
            self.episode_over = True
            self._acting = set()
        return masks

    # ---- checks ---------------------------------------------------------------------

    def _check_seat_keys(self, seats: Any, what: str) -> None:
        for seat in seats:
            phase = self.phase(seat)
            if phase is SeatPhase.EMPTY:
                raise self._fail(f"{what} for an empty seat (layout {self.layout} has "
                                 f"{self.spec.layout_size(self.layout)} seats)", seat)
            if phase is SeatPhase.ELIMINATED:
                raise self._fail(f"{what} for a seat eliminated at episode step {self._eliminated[seat]}", seat)

    def _check_truncation(self, result: StepResult) -> None:
        live = [s for s in self.live_seats() if s not in result.terminated]
        final_obs = result.final_obs or {}
        for seat in live:
            if seat not in final_obs:
                raise self._fail("truncated episode without final_obs for this live seat", seat)
            rules = self._rules_of(seat)
            rules.obs.check(final_obs[seat], f"{self._where(seat)}: final_obs")
            if rules.global_state is not None:
                gs = result.global_state or {}
                if seat not in gs:
                    raise self._fail("truncated episode without the final global_state the role declares", seat)
                rules.global_state.check(gs[seat], f"{self._where(seat)}: final global_state")

    def _check_global_state_roles(self, result: StepResult) -> None:
        for seat in result.global_state or {}:
            if self._rules_of(seat).global_state is None:
                raise self._fail(f"global_state for a seat whose role {self.spec.role_of(self.layout, seat)!r} "
                                 f"declares no global_state_space", seat)

    def _check_extra_seats(self, per_seat: dict[int, Any] | None, what: str) -> None:
        for seat in per_seat or {}:
            if self.phase(seat) is SeatPhase.EMPTY:
                raise self._fail(f"{what} for an empty seat", seat)

    def _check_empty_seat_data(self, result: StepResult) -> None:
        """Observations, masks, global states and final observations for EMPTY seats are errors."""
        self._check_extra_seats(result.obs, "an observation")
        self._check_extra_seats(result.action_masks, "an action mask")
        self._check_extra_seats(result.global_state, "a global_state")
        self._check_extra_seats(result.final_obs, "final_obs")

    def _check_fields(self, result: StepResult) -> None:
        """Container types and integer seat keys of a StepResult, before any rule reads them."""
        fields = {"acting": result.acting, "terminated": result.terminated, "obs": result.obs,
                  "action_masks": result.action_masks, "rewards": result.rewards}
        if result.final_obs is not None:
            fields["final_obs"] = result.final_obs
        if result.global_state is not None:
            fields["global_state"] = result.global_state
        for name, value in fields.items():
            if name in ("acting", "terminated"):
                if not isinstance(value, (Set, list, tuple)):
                    raise self._fail(f"StepResult.{name} must be a set of seats, got {type(value).__name__}")
            elif not isinstance(value, Mapping):
                raise self._fail(f"StepResult.{name} must be a dict of seat -> value, got {type(value).__name__}")
            for seat in value:
                if not isinstance(seat, numbers.Integral) or isinstance(seat, bool):
                    raise self._fail(f"StepResult.{name}: seats must be ints, got {seat!r} "
                                     f"({type(seat).__name__})")

    def _accept_next(self, result: StepResult) -> dict[int, Tree | None]:
        """Check the next acting set with its observations, masks and global states."""
        acting = set(result.acting)
        for seat in acting:
            phase = self.phase(seat)
            if phase is SeatPhase.EMPTY:
                raise self._fail("an empty seat is in acting", seat)
            if phase is SeatPhase.ELIMINATED:
                raise self._fail(f"a seat eliminated at episode step {self._eliminated[seat]} is in acting", seat)
        self._check_empty_seat_data(result)
        masks: dict[int, Tree | None] = {}
        for seat in sorted(acting):
            rules = self._rules_of(seat)
            where = self._where(seat)
            if seat not in result.obs:
                raise self._fail("an acting seat has no observation", seat)
            rules.obs.check(result.obs[seat], f"{where}: observation")
            gs = result.global_state or {}
            if rules.global_state is not None:
                if seat not in gs:
                    raise self._fail("the role declares a global_state_space but the acting seat got no "
                                     "global_state", seat)
                rules.global_state.check(gs[seat], f"{where}: global_state")
            mask = rules.action.normalize_mask(result.action_masks.get(seat), f"{where}: action mask")
            rules.action.check_acting_mask(mask, where)
            masks[seat] = mask
        self._check_global_state_roles(result)
        if not acting and not result.episode_over:
            self._idle += 1
            if self._idle > self.max_idle_steps:
                raise self._fail(f"more than env.max_idle_steps={self.max_idle_steps} steps in a row without "
                                 f"acting seats and without episode_over")
        elif acting:
            self._idle = 0
        self._acting = acting
        return masks
