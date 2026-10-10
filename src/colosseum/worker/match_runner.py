"""MatchRunner: the match core shared by training (RolloutLoop), eval and record (SP2 spec block 5,
SP3 spec block 3).

It owns a vector env, one :class:`Lineup` per env, one ``EpisodeTracker`` per env, the model state of
every neural seat and the scripted bots. Each :meth:`MatchRunner.step`:

1. acts for every acting seat of every env. Neural seats are inferred in batches grouped by
   ``(agent_id, network_id)`` (policy only, ``networks.model.act``). Scripted seats (the pool gives a
   :class:`ScriptedPlayer`) call their bot's ``act(obs, mask, info)`` one by one; the action passes
   ``check_bot_action`` (the single legality gate). Then ``observer.on_act`` per acting seat in
   ``(env, seat)`` order (a scripted seat's record has ``log_prob`` 0, no unit log-probs, no state);
2. steps every env once (``vec_env.step``; an env without acting seats gets ``{}``);
3. per env in index order: ``EpisodeTracker.on_step`` (contract checks, normalized masks)
   -> ``on_rewards`` (every reward of the step, including the elimination step) ->
   ``on_terminated`` (if any seat was eliminated) -> if the episode is over:
   ``on_episode_end`` (with the :class:`MatchResult`) -> apply the next lineup
   (``on_lineup_applied``) -> reset the env's model states;
4. resets all finished envs in one ``vec_env.reset`` with per-episode seeds; after each reset the
   bots of the env's scripted seats are reset (``bot_rng(episode seed, seat, agent)``), then
   ``observer.on_episode_start(env, layout, episode_seed)`` is called if the observer defines it.

Players. ``models.get(agent_id, network_id)`` gives a ``PolicyModel`` or a ``ScriptedPlayer``: an
agent's latest weights under ``"latest"``, its snapshots under ``"ckpt_v<N>"``, scripted and frozen
agents under ``FIXED_NETWORK_ID``. A bot instance exists per ``(agent, env, seat)``: created by the
player's factory at the first reset with the agent at that seat and kept between episodes (also across
lineup changes). Bots get the observation cast to the role's dtypes, the normalized mask and
``StepResult.infos.get(seat)`` of the latest result (``None`` without one; MatchRunner already keeps
that result, so infos cost nothing beyond one ``infos.get(seat)`` per decision). Observation and mask are
copies: a bot editing them in place changes neither the legality gate nor the record. Every ``ActRecord`` (neural
seats included) carries that ``infos`` entry as ``ActRecord.info``. A bot's exception or illegal action
is a ``PlayerError`` with the context "worker W, env E, seat P, episode step K, layout L: agent 'X'".

Seat returns, elimination steps, the episode length and the team ranks/scores of the
:class:`MatchResult` come from the env's ``EpisodeTracker`` (one source of truth; the outcome is
resolved with ``resolve_outcome``: default team score = mean of the team's seat returns).
``SeatAssignment.source`` is copied into ``SeatResult.source``.

A lineup naming a snapshot the pool cannot provide is seated as the agent's latest weights with
``collect=True`` (SP1 rule; one warning per (agent, network)), unless the pool serves that agent's
``"latest"`` as a ``ScriptedPlayer`` (a ``ValueError``: a bot never collects); latest and fixed seats must
be in the pool. Only ``latest`` seats of neural players may collect: any other seat with ``collect=True`` (also a
``ScriptedPlayer`` a pool serves under ``"latest"``) is a ``ValueError``.

``context`` is a prefix ending with ``", "`` (e.g. ``"worker 3, "``); env contract errors
read ``"worker 3, env 1, seat 2, episode step 7, layout 4p: ..."``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from colosseum.core.errors import PlayerError
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.tree import Tree, tree_map, tree_to_numpy, tree_to_torch
from colosseum.core.types import (
    FIXED_NETWORK_ID,
    LATEST_NETWORK_ID,
    Lineup,
    MatchResult,
    SeatAssignment,
    SeatResult,
    TeamResult,
)
from colosseum.envs.contract import EpisodeTracker
from colosseum.envs.game import GameSpec, StepResult
from colosseum.envs.vector import SubprocessVectorEnv, VectorEnv
from colosseum.networks.model import PolicyModel, act
from colosseum.networks.state import State, cat_batch, slice_batch
from colosseum.players.scripted import ScriptedBot, bot_rng, check_bot_action
from colosseum.worker.buffers import put_row

logger = logging.getLogger(__name__)

__all__ = ["ActRecord", "EpisodeEnd", "MatchObserver", "MatchRunner", "ModelPool", "PlayerPool", "ScriptedPlayer"]


@dataclass(frozen=True)
class ScriptedPlayer:
    """A scripted player in a pool: ``factory()`` builds one bot instance (per agent, env and seat)."""

    factory: Callable[[], ScriptedBot]


class PlayerPool(Protocol):
    def get(self, agent_id: str, network_id: str) -> PolicyModel | ScriptedPlayer | None: ...


ModelPool = PlayerPool  # SP2 name


@dataclass
class ActRecord:
    """One decision of one seat, as the observer sees it (numpy only, plus the model state).

    ``info`` is ``StepResult.infos.get(seat)`` of the result the seat acted on, for EVERY acting seat
    (neural and scripted; None without an entry): a scripted seat's record holds what its bot saw, and a
    neural seat's record gives it to consumers such as a DAgger teacher (SP3 amendment A15).
    Scripted seats: ``log_prob`` 0.0, ``unit_log_probs`` None, ``pre_state`` None."""

    agent_id: str
    network_id: str
    obs: Tree                        # numpy, cast to the role's observation dtypes
    global_state: Tree | None        # only when the seat's role declares global_state_space
    mask: Tree | None                # normalized mask (EpisodeTracker)
    action: Tree                     # numpy, as sent to the env
    log_prob: float
    unit_log_probs: np.ndarray | None   # [K] float32 when the action has K > 1 deciders
    pre_state: State                 # model state before this act, leaves [1, ...]
    info: Any = None


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
    """Observer of a MatchRunner. Optional extra method: ``on_episode_start(env, layout, episode_seed)``,
    called after every env reset (after the scripted bots were reset)."""

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
        self.scripted: dict[int, ScriptedPlayer] = {}   # seats of the current lineup played by bots
        self.episode_index = 0


class MatchRunner:
    """Runs matches on a vector env for given lineups and a player pool (module docstring)."""

    def __init__(
        self,
        *,
        vec_env: VectorEnv | SubprocessVectorEnv,
        lineups: Sequence[Lineup],
        models: PlayerPool,
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
        self._players = models
        self._observer = observer
        self._on_episode_start = getattr(observer, "on_episode_start", None)
        self._seed = seed
        self._deterministic = deterministic
        self._context = context
        self._prefix = match_id_prefix
        self._episodes_finished = 0
        self._warned_missing: set[tuple[str, str]] = set()
        self._bots: dict[tuple[str, int, int], ScriptedBot] = {}
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

    def next_lineup(self, env: int) -> Lineup | None:
        """The lineup staged for env ``env`` (applied at its next episode end), if any."""
        return self._envs[env].next_lineup

    def set_next_lineup(self, env: int, lineup: Lineup) -> None:
        """Stage ``lineup`` for env ``env``; it is applied at the env's next episode end.

        Layout, seat count, the players of latest and fixed seats (and every snapshot seat's agent's latest
        model) are checked now; the missing-snapshot fallback is applied at the episode end (a snapshot may
        be loaded in between).
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
    # Lineups and players
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
            net = assignment.network_id
            needed = net if net in (LATEST_NETWORK_ID, FIXED_NETWORK_ID) else LATEST_NETWORK_ID
            player = self._players.get(assignment.agent_id, needed)
            if player is None:
                raise ValueError(
                    f"{self._context}env {env}: the player pool has no model for agent {assignment.agent_id!r} "
                    f"(network {needed!r})"
                )
            if assignment.collect and net != LATEST_NETWORK_ID:
                raise ValueError(
                    f"{self._context}env {env}, seat {seat}: agent {assignment.agent_id!r} plays network "
                    f"{net!r} with collect=True; only {LATEST_NETWORK_ID!r} seats collect"
                )
            if assignment.collect and isinstance(player, ScriptedPlayer):
                raise ValueError(
                    f"{self._context}env {env}, seat {seat}: agent {assignment.agent_id!r} is a scripted player "
                    f"with collect=True; only a trainable agent's {LATEST_NETWORK_ID!r} model collects"
                )

    def _resolve(self, lineup: Lineup, env: int) -> Lineup:
        """Validate ``lineup`` and replace snapshots the pool cannot provide by latest + collect (never for a
        scripted player: ValueError)."""
        self._check_lineup(lineup, env)
        seats = []
        for seat, assignment in enumerate(lineup.seats):
            aid, net = assignment.agent_id, assignment.network_id
            if self._players.get(aid, net) is not None:
                seats.append(SeatAssignment(aid, net, assignment.collect, assignment.source))
                continue
            if isinstance(self._players.get(aid, LATEST_NETWORK_ID), ScriptedPlayer):
                raise ValueError(
                    f"{self._context}env {env}, seat {seat}: network {net!r} of {aid!r} is not loaded and {aid!r} "
                    f"is a scripted player; the missing-network fallback seats {LATEST_NETWORK_ID!r} with "
                    f"collect=True, which only a trainable agent's model may do"
                )
            if (aid, net) not in self._warned_missing:
                self._warned_missing.add((aid, net))
                logger.warning(
                    f"{self._context}network {net!r} of {aid!r} is not loaded; "
                    f"seating {LATEST_NETWORK_ID!r} (collecting) instead"
                )
            seats.append(SeatAssignment(aid, LATEST_NETWORK_ID, True, assignment.source))
        return Lineup(layout=lineup.layout, seats=seats)

    def _player(self, agent_id: str, network_id: str) -> PolicyModel | ScriptedPlayer:
        player = self._players.get(agent_id, network_id)
        if player is None:
            raise RuntimeError(f"{self._context}player ({agent_id!r}, {network_id!r}) disappeared from the pool")
        return player

    def _where(self, e: int, seat: int, agent_id: str) -> str:
        tracker = self._envs[e].tracker
        return (f"{self._context}env {e}, seat {seat}, episode step {tracker.episode_step}, "
                f"layout {tracker.layout}: agent {agent_id!r}")

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _infer(self) -> dict[int, dict[int, Tree]]:
        """Batched inference for neural seats, ``act`` for scripted ones; ``on_act`` in (env, seat) order."""
        groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        bot_seats: list[tuple[int, int]] = []
        for e, env_state in enumerate(self._envs):
            scripted = env_state.scripted
            for seat in sorted(env_state.result.acting):
                if scripted and seat in scripted:
                    bot_seats.append((e, seat))
                    continue
                a = env_state.lineup.seats[seat]
                groups[(a.agent_id, a.network_id)].append((e, seat))

        actions: dict[int, dict[int, Tree]] = {e: {} for e in range(self.num_envs)}
        records: dict[tuple[int, int], ActRecord] = {}
        for (aid, net), seats in groups.items():
            model = self._player(aid, net)
            # every seat of a group has the same spaces: an agent's roles share them (core.roles)
            first_env, first_seat = seats[0]
            role_info = self._roles[self.spec.role_of(self._envs[first_env].tracker.layout, first_seat)]
            n = len(seats)
            obs_batch = role_info.obs.allocate((n,))
            mask_batch = role_info.action.full_mask((n,)) if role_info.action.has_masks else None
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
            unit_lps = out.unit_log_probs.float().cpu().numpy() if role_info.action.num_deciders > 1 else None
            for j, (e, seat) in enumerate(seats):
                env_state = self._envs[e]
                action = tree_map(lambda leaf, j=j: leaf[j].copy(), act_np)
                actions[e][seat] = action
                gs = env_state.result.global_state
                infos = env_state.result.infos
                records[(e, seat)] = ActRecord(
                    agent_id=aid,
                    network_id=net,
                    obs=obs_rows[j],
                    global_state=gs[seat] if role_info.has_global_state and gs is not None else None,
                    mask=env_state.masks.get(seat),
                    action=action,
                    log_prob=float(log_probs[j]),
                    unit_log_probs=None if unit_lps is None else unit_lps[j].copy(),
                    pre_state=env_state.states[seat],
                    info=infos.get(seat) if infos else None,
                )
                env_state.states[seat] = None if out.state is None else slice_batch(out.state, j)
        for e, seat in bot_seats:
            actions[e][seat], records[(e, seat)] = self._bot_act(e, seat)
        if self._observer is not None:
            for key in sorted(records):
                self._observer.on_act(key[0], key[1], records[key])
        return actions

    def _bot_act(self, e: int, seat: int) -> tuple[Tree, ActRecord]:
        """One scripted decision: the bot sees obs (role dtypes), the normalized mask and infos[seat]."""
        env_state = self._envs[e]
        a = env_state.lineup.seats[seat]
        role_name = self.spec.role_of(env_state.tracker.layout, seat)
        role_info = self._roles[role_name]
        obs_batch = role_info.obs.allocate((1,))
        put_row(obs_batch, 0, env_state.result.obs[seat])
        obs = tree_map(lambda leaf: leaf[0].copy(), obs_batch)
        mask = env_state.masks.get(seat)
        infos = env_state.result.infos
        seat_info = infos.get(seat) if infos else None
        where = self._where(e, seat, a.agent_id)
        bot = self._bots[(a.agent_id, e, seat)]
        try:
            # copies: a bot editing them in place can neither bypass the gate nor change the record
            raw = bot.act(tree_map(np.copy, obs), None if mask is None else tree_map(np.copy, mask), seat_info)
        except Exception as exc:  # noqa: BLE001 - any bot failure is reported with its context
            raise PlayerError(f"{where}: act raised {type(exc).__name__}: {exc}") from exc
        action = check_bot_action(self.spec.roles[role_name], raw, mask, where, action_spec=role_info.action)
        gs = env_state.result.global_state
        record = ActRecord(
            agent_id=a.agent_id, network_id=a.network_id, obs=obs,
            global_state=gs[seat] if role_info.has_global_state and gs is not None else None,
            mask=mask, action=action, log_prob=0.0, unit_log_probs=None, pre_state=None, info=seat_info,
        )
        return action, record

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
                source=a.source,
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
        """Initial model states of every neural seat and the scripted seats of the env's lineup (episode start)."""
        states: dict[int, State] = {}
        scripted: dict[int, ScriptedPlayer] = {}
        for s, a in enumerate(env_state.lineup.seats):
            player = self._player(a.agent_id, a.network_id)
            if isinstance(player, ScriptedPlayer):
                scripted[s] = player
                states[s] = None
            else:
                states[s] = player.initial_state(1)
        env_state.states = states
        env_state.scripted = scripted

    def _episode_seed(self, e: int, k: int) -> int | None:
        if self._seed is None:
            return None
        return int(np.random.SeedSequence([int(self._seed), e, k]).generate_state(1)[0])

    def _reset_bots(self, e: int, episode_seed: int | None) -> None:
        """Create (lazily) and reset the bot of every scripted seat of env ``e``'s lineup."""
        env_state = self._envs[e]
        layout = env_state.lineup.layout
        for seat, player in env_state.scripted.items():
            aid = env_state.lineup.seats[seat].agent_id
            key = (aid, e, seat)
            bot = self._bots.get(key)
            if bot is None:
                try:
                    bot = player.factory()
                except Exception as exc:  # noqa: BLE001 - reported with its context
                    raise PlayerError(f"{self._where(e, seat, aid)}: creating the bot failed "
                                      f"({type(exc).__name__}: {exc})") from exc
                self._bots[key] = bot
            try:
                bot.reset(role=self.spec.role_of(layout, seat), seat=seat, layout=layout,
                          rng=bot_rng(episode_seed, seat, aid))
            except Exception as exc:  # noqa: BLE001 - reported with its context
                raise PlayerError(f"{self._where(e, seat, aid)}: reset raised {type(exc).__name__}: {exc}") from exc

    def _reset_envs(self, envs: list[int]) -> None:
        requests = {
            e: (self._episode_seed(e, self._envs[e].episode_index), self._envs[e].lineup.layout) for e in envs
        }
        results = self._vec_env.reset(requests)
        for e in envs:
            env_state = self._envs[e]
            env_state.masks = env_state.tracker.on_reset(env_state.lineup.layout, results[e])
            env_state.result = results[e]
            if env_state.scripted:
                self._reset_bots(e, requests[e][0])
            if self._on_episode_start is not None:
                self._on_episode_start(e, env_state.lineup.layout, requests[e][0])
