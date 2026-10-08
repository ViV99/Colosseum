"""MatchRunner: the match core shared by training (RolloutLoop) and eval (spec block 5).

It owns a vector env, one :class:`Lineup` per env, one ``EpisodeTracker`` per env and the
model state of every occupied seat. Each :meth:`MatchRunner.step`:

1. runs inference for every acting seat of every env, grouped by ``(agent_id, network_id)``
   (policy only, ``networks.model.act``), then calls ``observer.on_act`` per acting seat in
   ``(env, seat)`` order;
2. steps every env once (``vec_env.step``; an env without acting seats gets ``{}``);
3. per env in index order: ``EpisodeTracker.on_step`` (contract checks, normalized masks)
   -> ``on_rewards`` (every reward of the step, including the elimination step) ->
   ``on_terminated`` (if any seat was eliminated) -> if the episode is over:
   ``on_episode_end`` (with the :class:`MatchResult`) -> apply the next lineup
   (``on_lineup_applied``) -> reset the env's model states;
4. resets all finished envs in one ``vec_env.reset`` with per-episode seeds.

Seat returns, elimination steps, the episode length and the team ranks/scores of the
:class:`MatchResult` come from the env's ``EpisodeTracker``, which accumulates them while it
checks the results (one source of truth; the outcome is resolved with ``resolve_outcome``:
default team score = mean of the team's seat returns).

A lineup naming a network the model pool cannot provide is seated as the agent's latest
weights with ``collect=True`` (SP1 rule; one warning per (agent, network)). Only ``latest``
seats may collect: a checkpoint seat with ``collect=True`` is a ``ValueError``.

``context`` is a prefix ending with ``", "`` (e.g. ``"worker 3, "``); env contract errors
read ``"worker 3, env 1, seat 2, episode step 7, layout 4p: ..."``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from colosseum.networks.state import State, cat_batch, slice_batch
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.tree import Tree, tree_map, tree_to_numpy, tree_to_torch
from colosseum.sp2.core.types import (
    LATEST_NETWORK_ID,
    Lineup,
    MatchResult,
    SeatAssignment,
    SeatResult,
    TeamResult,
)
from colosseum.sp2.envs.contract import EpisodeTracker
from colosseum.sp2.envs.game import GameSpec, StepResult
from colosseum.sp2.envs.vector import SubprocessVectorEnv, VectorEnv
from colosseum.sp2.networks.model import PolicyModel, act
from colosseum.sp2.worker.buffers import put_row

logger = logging.getLogger(__name__)

__all__ = ["ActRecord", "EpisodeEnd", "MatchObserver", "MatchRunner", "ModelPool"]


class ModelPool(Protocol):
    def get(self, agent_id: str, network_id: str) -> PolicyModel | None: ...


@dataclass
class ActRecord:
    """One decision of one seat, as the observer sees it (numpy only, plus the model state)."""

    agent_id: str
    network_id: str
    obs: Tree                        # numpy, cast to the role's observation dtypes
    global_state: Tree | None        # only when the seat's role declares global_state_space
    mask: Tree | None                # normalized mask (EpisodeTracker)
    action: Tree                     # numpy, as sent to the env
    log_prob: float
    unit_log_probs: np.ndarray | None   # [K] float32 when the action has K > 1 deciders
    pre_state: State                 # model state before this act, leaves [1, ...]


@dataclass
class EpisodeEnd:
    """How an env's episode ended. ``live_seats`` excludes every eliminated seat (also this step's);
    ``final_obs`` / ``final_global_state`` (truncation only) hold the live seats only."""

    truncated: bool
    live_seats: list[int]
    final_obs: dict[int, Tree] | None
    final_global_state: dict[int, Tree] | None
    result: MatchResult


class MatchObserver(Protocol):
    def on_act(self, env: int, seat: int, record: ActRecord) -> None: ...
    def on_rewards(self, env: int, rewards: dict[int, float]) -> None: ...
    def on_terminated(self, env: int, seats: list[int]) -> None: ...
    def on_episode_end(self, env: int, end: EpisodeEnd) -> None: ...
    def on_lineup_applied(self, env: int, old: Lineup, new: Lineup) -> None: ...


@dataclass
class _RoleInfo:
    obs: ObsSpec
    action: ActionSpec
    has_global_state: bool


class _EnvState:
    """Per-env bookkeeping of the runner."""

    def __init__(self, tracker: EpisodeTracker, lineup: Lineup) -> None:
        self.tracker = tracker
        self.lineup = lineup
        self.next_lineup: Lineup | None = None
        self.result: StepResult | None = None      # the latest result (its acting seats act next)
        self.masks: dict[int, Tree | None] = {}     # normalized masks of those acting seats
        self.states: dict[int, State] = {}
        self.episode_index = 0


class MatchRunner:
    """Runs matches on a vector env for given lineups and a model pool (module docstring)."""

    def __init__(
        self,
        *,
        vec_env: VectorEnv | SubprocessVectorEnv,
        lineups: Sequence[Lineup],
        models: ModelPool,
        observer: MatchObserver | None = None,
        seed: int | None = None,
        max_idle_steps: int = 1000,
        deterministic: bool = False,
        context: str = "",
        match_id_prefix: str = "m",
    ) -> None:
        self._vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.spec: GameSpec = vec_env.spec
        if len(lineups) != self.num_envs:
            raise ValueError(f"need one lineup per env: {len(lineups)} lineups for {self.num_envs} envs")
        self._models = models
        self._observer = observer
        self._seed = seed
        self._deterministic = deterministic
        self._context = context
        self._prefix = match_id_prefix
        self._episodes_finished = 0
        self._warned_missing: set[tuple[str, str]] = set()
        self._roles = {
            name: _RoleInfo(
                obs=ObsSpec.from_space(role.observation_space),
                action=ActionSpec.from_space(role.action_space),
                has_global_state=role.global_state_space is not None,
            )
            for name, role in self.spec.roles.items()
        }
        self._envs: list[_EnvState] = []
        for e, lineup in enumerate(lineups):
            tracker = EpisodeTracker(self.spec, max_idle_steps=max_idle_steps, context=f"{context}env {e}")
            env_state = _EnvState(tracker, self._resolve(lineup, e))
            self._init_model_states(env_state)
            self._envs.append(env_state)
        self._reset_envs(list(range(self.num_envs)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lineup(self, env: int) -> Lineup:
        """Env ``env``'s current lineup (after the missing-network fallback)."""
        return self._envs[env].lineup

    def set_next_lineup(self, env: int, lineup: Lineup) -> None:
        """Stage ``lineup`` for env ``env``; it is applied at the env's next episode end.

        Layout, seat count and every agent's latest model are checked now; the missing-network
        fallback is applied at the episode end (a checkpoint may be loaded in between).
        """
        self._check_lineup(lineup, env)
        self._envs[env].next_lineup = lineup

    @property
    def episodes_finished(self) -> int:
        return self._episodes_finished

    def step(self) -> int:
        """One step of every env (see the module docstring). Returns the env steps taken."""
        actions = self._infer()
        results = self._vec_env.step(actions)
        finished = [e for e in range(self.num_envs) if self._after_step(e, actions[e], results[e])]
        if finished:
            self._reset_envs(finished)
        return self.num_envs

    def close(self) -> None:
        self._vec_env.close()

    # ------------------------------------------------------------------
    # Lineups
    # ------------------------------------------------------------------

    def _check_lineup(self, lineup: Lineup, env: int) -> None:
        if lineup.layout not in self.spec.layouts:
            raise ValueError(
                f"{self._context}env {env}: lineup layout {lineup.layout!r} is not one of "
                f"{sorted(self.spec.layouts)}"
            )
        size = self.spec.layout_size(lineup.layout)
        if len(lineup.seats) != size:
            raise ValueError(
                f"{self._context}env {env}: lineup for layout {lineup.layout!r} has "
                f"{len(lineup.seats)} seats, the layout has {size}"
            )
        for seat, assignment in enumerate(lineup.seats):
            if self._models.get(assignment.agent_id, LATEST_NETWORK_ID) is None:
                raise ValueError(
                    f"{self._context}env {env}: the model pool has no model for agent {assignment.agent_id!r}"
                )
            if assignment.collect and assignment.network_id != LATEST_NETWORK_ID:
                raise ValueError(
                    f"{self._context}env {env}, seat {seat}: agent {assignment.agent_id!r} plays network "
                    f"{assignment.network_id!r} with collect=True; only {LATEST_NETWORK_ID!r} seats collect"
                )

    def _resolve(self, lineup: Lineup, env: int) -> Lineup:
        """Validate ``lineup`` and replace networks the pool cannot provide by latest + collect."""
        self._check_lineup(lineup, env)
        seats = []
        for assignment in lineup.seats:
            aid, net = assignment.agent_id, assignment.network_id
            if self._models.get(aid, net) is not None:
                seats.append(SeatAssignment(aid, net, assignment.collect))
                continue
            if (aid, net) not in self._warned_missing:
                self._warned_missing.add((aid, net))
                logger.warning(
                    f"{self._context}network {net!r} of {aid!r} is not loaded; "
                    f"seating {LATEST_NETWORK_ID!r} (collecting) instead"
                )
            seats.append(SeatAssignment(aid, LATEST_NETWORK_ID, True))
        return Lineup(layout=lineup.layout, seats=seats)

    def _model(self, agent_id: str, network_id: str) -> PolicyModel:
        model = self._models.get(agent_id, network_id)
        if model is None:
            raise RuntimeError(f"{self._context}model ({agent_id!r}, {network_id!r}) disappeared from the pool")
        return model

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _infer(self) -> dict[int, dict[int, Tree]]:
        """Batched inference for all acting seats; ``on_act`` per seat in (env, seat) order."""
        groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        for e, env_state in enumerate(self._envs):
            for seat in sorted(env_state.result.acting):
                a = env_state.lineup.seats[seat]
                groups[(a.agent_id, a.network_id)].append((e, seat))

        actions: dict[int, dict[int, Tree]] = {e: {} for e in range(self.num_envs)}
        records: dict[tuple[int, int], ActRecord] = {}
        for (aid, net), seats in groups.items():
            model = self._model(aid, net)
            # every seat of a group has the same spaces: an agent's roles share them (core.roles)
            first_env, first_seat = seats[0]
            info = self._roles[self.spec.role_of(self._envs[first_env].tracker.layout, first_seat)]
            n = len(seats)
            obs_batch = info.obs.allocate((n,))
            mask_batch = info.action.full_mask((n,)) if info.action.has_masks else None
            for j, (e, seat) in enumerate(seats):
                env_state = self._envs[e]
                put_row(obs_batch, j, env_state.result.obs[seat])
                if mask_batch is not None:
                    put_row(mask_batch, j, env_state.masks[seat])
            # the records' copies are taken before inference: a model may transform its input in place
            obs_rows = [tree_map(lambda leaf, j=j: leaf[j].copy(), obs_batch) for j in range(n)]
            state_batch = cat_batch([self._envs[e].states[seat] for e, seat in seats])
            out = act(model, tree_to_torch(obs_batch), state_batch,
                      None if mask_batch is None else tree_to_torch(mask_batch),
                      deterministic=self._deterministic)
            act_np = tree_to_numpy(out.actions)
            log_probs = out.log_probs.float().cpu().numpy()
            unit_lps = out.unit_log_probs.float().cpu().numpy() if info.action.num_deciders > 1 else None
            for j, (e, seat) in enumerate(seats):
                env_state = self._envs[e]
                action = tree_map(lambda leaf, j=j: leaf[j].copy(), act_np)
                actions[e][seat] = action
                gs = env_state.result.global_state
                records[(e, seat)] = ActRecord(
                    agent_id=aid,
                    network_id=net,
                    obs=obs_rows[j],
                    global_state=gs[seat] if info.has_global_state and gs is not None else None,
                    mask=env_state.masks.get(seat),
                    action=action,
                    log_prob=float(log_probs[j]),
                    unit_log_probs=None if unit_lps is None else unit_lps[j].copy(),
                    pre_state=env_state.states[seat],
                )
                env_state.states[seat] = None if out.state is None else slice_batch(out.state, j)
        if self._observer is not None:
            for key in sorted(records):
                self._observer.on_act(key[0], key[1], records[key])
        return actions

    def _after_step(self, e: int, actions: dict[int, Tree], result: StepResult) -> bool:
        """Process env ``e``'s step result; True if its episode ended."""
        env_state = self._envs[e]
        tracker = env_state.tracker
        env_state.masks = tracker.on_step(actions, result)
        obs = self._observer
        if obs is not None:
            obs.on_rewards(e, {int(s): float(r) for s, r in result.rewards.items()})
            if result.terminated:
                obs.on_terminated(e, sorted(int(s) for s in result.terminated))
        if not result.episode_over:
            env_state.result = result
            return False

        match_result = self._match_result(e)
        self._episodes_finished += 1
        if obs is not None:
            live = tracker.live_seats()
            truncated = bool(result.truncated)
            gs = result.global_state
            obs.on_episode_end(e, EpisodeEnd(
                truncated=truncated,
                live_seats=live,
                final_obs={s: result.final_obs[s] for s in live} if truncated else None,
                final_global_state=(
                    {s: gs[s] for s in live if s in gs} if truncated and gs is not None else None
                ),
                result=match_result,
            ))
        if env_state.next_lineup is not None:
            old, new = env_state.lineup, self._resolve(env_state.next_lineup, e)
            env_state.lineup, env_state.next_lineup = new, None
            if obs is not None:
                obs.on_lineup_applied(e, old, new)
        env_state.episode_index += 1
        self._init_model_states(env_state)
        return True

    def _match_result(self, e: int) -> MatchResult:
        """The finished episode's result from the env's tracker (returns, eliminations, teams)."""
        env_state = self._envs[e]
        tracker = env_state.tracker
        layout = tracker.layout
        ranks, scores = tracker.team_result()
        returns = tracker.seat_returns()
        seat_specs = self.spec.layouts[layout]
        seats = [
            SeatResult(
                seat=s,
                role=seat_specs[s].role,
                team=seat_specs[s].team,
                agent_id=a.agent_id,
                network_id=a.network_id,
                reward=float(returns[s]),
                eliminated_step=tracker.eliminated_step(s),
            )
            for s, a in enumerate(env_state.lineup.seats)
        ]
        return MatchResult(
            match_id=f"{self._prefix}{e}_ep{env_state.episode_index}",
            layout=layout,
            outcome_kind=self.spec.outcome_kind(layout),
            seats=seats,
            teams=[TeamResult(team=t, rank=float(ranks[t]), score=float(scores[t])) for t in sorted(ranks)],
            episode_length=tracker.episode_step,
        )

    def _init_model_states(self, env_state: _EnvState) -> None:
        """Initial model states of every seat of the env's lineup (episode start)."""
        env_state.states = {
            s: self._model(a.agent_id, a.network_id).initial_state(1) for s, a in enumerate(env_state.lineup.seats)
        }

    def _episode_seed(self, e: int, k: int) -> int | None:
        if self._seed is None:
            return None
        return int(np.random.SeedSequence([int(self._seed), e, k]).generate_state(1)[0])

    def _reset_envs(self, envs: list[int]) -> None:
        requests = {
            e: (self._episode_seed(e, self._envs[e].episode_index), self._envs[e].lineup.layout) for e in envs
        }
        results = self._vec_env.reset(requests)
        for e in envs:
            env_state = self._envs[e]
            env_state.masks = env_state.tracker.on_reset(env_state.lineup.layout, results[e])
            env_state.result = results[e]
