"""In-process rollout loop: vectorized envs, batched inference, per-slot transitions.

``RolloutLoop`` owns the environments and the per-agent model pools of one
worker. All I/O (sending chunks, pulling weights, reporting match results,
receiving match re-assignments) goes through :class:`LoopIO` callbacks, so the
loop runs unchanged in a worker process (``rollout_worker_process``) and
in-process in tests.

Multi-agent: several agents can occupy different player slots of the same
environments. Inference is grouped by (agent_id, network_id) for batching, and
each chunk is routed to the agent that produced it.

Buffers are owned by agents (``worker/slots.py``): each collecting slot holds
one buffer, and on match re-assignment a slot's partial buffer is parked for
its agent instead of being discarded (spec block 2). A chunk's
``behavior_policy_version`` is the version at its FIRST transition.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from colosseum.core.action_spec import ActionSpec
from colosseum.core.outcomes import player_outcomes
from colosseum.core.types import (
    MatchResult,
    TrajectoryChunk,
    WeightPayload,
    WorkerCommand,
    state_dict_from_numpy,
)
from colosseum.envs.base_env import BaseEnv
from colosseum.envs.vec_env import VectorEnv
from colosseum.networks.model import PolicyModel, act
from colosseum.networks.state import State, cat_batch, slice_batch
from colosseum.worker.slots import BufferPool, RolloutBuffer, SlotTrack

logger = logging.getLogger(__name__)

LATEST_NETWORK_ID = "latest"


@dataclass
class LoopIO:
    """All I/O of the rollout loop goes through these callbacks."""

    send_chunk: Callable[[TrajectoryChunk], None]
    poll_weights: Callable[[str], WeightPayload | None]
    report_result: Callable[[MatchResult], None] | None = None
    poll_command: Callable[[], WorkerCommand | None] | None = None
    add_env_steps: Callable[[int], None] | None = None  # global env-step budget counter


class RolloutLoop:
    """One worker's rollout loop over ``num_envs`` vectorized envs."""

    def __init__(
        self,
        *,
        worker_id: int,
        env_fn: Callable[[], BaseEnv],
        num_envs: int,
        chunk_length: int,
        agent_ids: list[str],
        model_factories: dict[str, Callable[[], PolicyModel]],
        io: LoopIO,
        gamma: float = 0.99,
        weight_sync_interval: float = 5.0,
        slot_agent_map: list[list[str]] | None = None,
        slot_network_map: list[list[str]] | None = None,
        collect_mask: list[list[bool]] | None = None,
        checkpoint_state_dicts_by_agent: dict[str, dict[str, dict[str, np.ndarray]]] | None = None,
        seed: int | None = None,
        vec_env_kind: str = "sync",
        subproc_workers: int | None = None,
    ) -> None:
        self.worker_id = worker_id
        self._io = io
        self._agent_ids = list(agent_ids)
        self._model_factories = model_factories
        self._chunk_length = chunk_length
        self._weight_sync_interval = weight_sync_interval
        self._gamma = float(gamma)  # used for truncation bootstrapping (T3.3)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if vec_env_kind == "subprocess":
            from colosseum.envs.subproc_vec_env import SubprocessVectorEnv
            self._vec_env = SubprocessVectorEnv(env_fn, num_envs, num_workers=subproc_workers)
        elif vec_env_kind == "sync":
            self._vec_env = VectorEnv(env_fn, num_envs)
        else:
            raise ValueError(f"unknown vec_env_kind {vec_env_kind!r}")
        self._num_envs = num_envs
        self._num_players = self._vec_env.num_players
        self._action_spec = ActionSpec.from_space(self._vec_env.action_space)
        E, P = self._num_envs, self._num_players

        # Model pool: agent -> network id -> model ("latest" + frozen checkpoints).
        self._models: dict[str, dict[str, PolicyModel]] = {}
        self._policy_versions: dict[str, int] = {aid: 0 for aid in self._agent_ids}
        for aid in self._agent_ids:
            latest = model_factories[aid]()
            latest.eval()
            self._models[aid] = {LATEST_NETWORK_ID: latest}
        for aid, by_id in (checkpoint_state_dicts_by_agent or {}).items():
            for ckpt_id, sd in by_id.items():
                self._add_checkpoint(aid, ckpt_id, sd)
        self._warned_missing: set[tuple[str, str]] = set()

        # Initial weights for every agent's latest model (records their policy version, A5).
        self.sync_weights()

        # Live match assignment; WorkerCommand updates are staged in `_pending_maps`
        # and applied per env at its next episode boundary.
        if slot_agent_map is None:
            slot_agent_map = [[self._agent_ids[0]] * P for _ in range(E)]
        if collect_mask is None:
            collect_mask = [[True] * P for _ in range(E)]
        if slot_network_map is None:
            # Non-collecting slots default to the agent's first checkpoint, if any.
            ckpts = checkpoint_state_dicts_by_agent or {}
            slot_network_map = [
                [
                    LATEST_NETWORK_ID
                    if collect_mask[e][p] or not ckpts.get(slot_agent_map[e][p])
                    else next(iter(ckpts[slot_agent_map[e][p]]))
                    for p in range(P)
                ]
                for e in range(E)
            ]
        self._slot_agent_map = [list(r) for r in slot_agent_map]
        self._slot_network_map = [list(r) for r in slot_network_map]
        self._collect_mask = [[bool(c) for c in r] for r in collect_mask]
        self._pending_maps: tuple[list, list, list] | None = None

        self._obs, self._infos = self._vec_env.reset_all(seed=seed)
        self._pool = BufferPool(
            chunk_length=chunk_length,
            obs_shape=tuple(self._obs.shape[2:]),
            action_shape=tuple(self._action_spec.action_shape),
            action_dtype=self._action_spec.numpy_dtype,
            mask_size=self._action_spec.flat_mask_size,
        )
        self._tracks: list[list[SlotTrack]] = []
        for e in range(E):
            row = []
            for p in range(P):
                track = SlotTrack(buffer=None, state=self._initial_state(e, p))
                if self._collect_mask[e][p]:
                    track.buffer = self._pool.acquire(self._slot_agent_map[e][p])
                row.append(track)
            self._tracks.append(row)

        self._ep_rewards = np.zeros((E, P), dtype=np.float64)
        self._ep_lengths = np.zeros(E, dtype=np.int64)
        self._ep_counter = np.zeros(E, dtype=np.int64)
        self._env_steps = 0
        self._chunks_sent = 0
        self._recorded: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> int:
        """One vectorized env step for all envs. Returns env steps taken."""
        self._poll_command()
        obs = self._obs
        acting = self._acting_flags(self._infos)
        masks = self._extract_masks(self._infos)
        actions, log_probs, values, pre_states = self._infer(obs, masks)
        self._open_transitions(obs, actions, log_probs, values, masks, acting, pre_states)
        next_obs, rewards, terminated, truncated, infos = self._vec_env.step(actions)
        self._after_env_step(next_obs, rewards, terminated, truncated, infos)
        self._obs, self._infos = next_obs, infos
        n = self._num_envs
        self._env_steps += n
        if self._io.add_env_steps is not None:
            self._io.add_env_steps(n)
        if time.monotonic() - self._last_weight_sync >= self._weight_sync_interval:
            self.sync_weights()
        return n

    def sync_weights(self) -> None:
        """Pull the newest weights for every agent's latest model (non-blocking)."""
        for aid in self._agent_ids:
            payload = self._io.poll_weights(aid)
            if payload is not None:
                self._models[aid][LATEST_NETWORK_ID].load_state_dict(payload.to_torch_state_dict())
                self._policy_versions[aid] = int(payload.policy_version)
        self._last_weight_sync = time.monotonic()

    def run(self, should_stop: Callable[[], bool], max_env_steps: int = 0) -> None:
        """Step until ``should_stop()`` or until ``max_env_steps`` env steps (0 = no limit)."""
        while not should_stop():
            if max_env_steps > 0 and self._env_steps >= max_env_steps:
                break
            self.step()

    def close(self) -> None:
        self._vec_env.close()

    @property
    def stats(self) -> dict[str, int]:
        out = {
            "chunks_sent": self._chunks_sent,
            "env_steps": self._env_steps,
            "episodes": int(self._ep_counter.sum()),
            "parked_buffers": self._pool.parked_count(),
            "recorded_transitions": sum(self._recorded.values()),
        }
        for aid in self._agent_ids:
            out[f"recorded_transitions/{aid}"] = self._recorded[aid]
            out[f"buffered_transitions/{aid}"] = self._buffered_transitions(aid)
        return out

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def _add_checkpoint(self, aid: str, ckpt_id: str, sd: dict[str, np.ndarray]) -> None:
        if aid not in self._models or ckpt_id in self._models[aid]:
            return
        model = self._model_factories[aid]()
        model.load_state_dict(state_dict_from_numpy(sd))
        model.eval()
        self._models[aid][ckpt_id] = model
        logger.info(f"Worker {self.worker_id}: agent {aid}: loaded checkpoint {ckpt_id}")

    def _resolve_model(self, aid: str, net_id: str) -> PolicyModel:
        """The model seated for (agent, network id); falls back to latest if not loaded."""
        models = self._models[aid]
        model = models.get(net_id)
        if model is None:
            if (aid, net_id) not in self._warned_missing:
                self._warned_missing.add((aid, net_id))
                logger.warning(
                    f"Worker {self.worker_id}: network {net_id!r} of {aid!r} is not loaded; "
                    f"using {LATEST_NETWORK_ID!r}"
                )
            model = models[LATEST_NETWORK_ID]
        return model

    def _initial_state(self, e: int, p: int) -> State:
        """Episode-start state of the model actually seated in slot (e, p)."""
        model = self._resolve_model(self._slot_agent_map[e][p], self._slot_network_map[e][p])
        return model.initial_state(1)

    # ------------------------------------------------------------------
    # Commands / re-assignment
    # ------------------------------------------------------------------

    def _poll_command(self) -> None:
        """Load a command's new checkpoints now; stage its slot maps for episode ends."""
        if self._io.poll_command is None:
            return
        cmd = self._io.poll_command()
        if cmd is None:
            return
        for aid, ckpts in cmd.new_checkpoints.items():
            for ckpt_id, sd in ckpts.items():
                self._add_checkpoint(aid, ckpt_id, sd)
        if cmd.slot_agent_map:
            self._pending_maps = (
                [list(r) for r in cmd.slot_agent_map],
                [list(r) for r in cmd.slot_network_map],
                [[bool(c) for c in r] for r in cmd.collect_mask],
            )

    def _apply_pending_assignment(self, e: int) -> None:
        """Apply the staged assignment to env ``e`` (called at its episode end).

        A slot that stops collecting for its agent parks its partial buffer
        (whose last transition is done) in the pool; a slot that starts
        collecting acquires a buffer, preferring a parked one of its agent.
        """
        if self._pending_maps is None:
            return
        agents, nets, collect = self._pending_maps
        if e >= len(agents):
            return
        for p in range(self._num_players):
            track = self._tracks[e][p]
            old_aid = self._slot_agent_map[e][p]
            new_aid = agents[e][p]
            new_collect = collect[e][p]
            if track.buffer is not None and (not new_collect or new_aid != old_aid):
                self._pool.park(old_aid, track.buffer)
                track.buffer = None
            self._slot_agent_map[e][p] = new_aid
            self._slot_network_map[e][p] = nets[e][p]
            self._collect_mask[e][p] = new_collect
            if new_collect and track.buffer is None:
                track.buffer = self._pool.acquire(new_aid)

    def _buffered_transitions(self, aid: str) -> int:
        """Transitions of ``aid`` recorded but not yet sent (slot + parked buffers)."""
        n = self._pool.parked_transitions(aid)
        for e in range(self._num_envs):
            for p in range(self._num_players):
                buf = self._tracks[e][p].buffer
                if buf is not None and self._slot_agent_map[e][p] == aid:
                    n += buf.steps
        return n

    # ------------------------------------------------------------------
    # Per-step pieces
    # ------------------------------------------------------------------

    def _acting_flags(self, infos: list[dict]) -> np.ndarray:
        """[E, P] bool: ``info[p]["active"]`` per slot; slots without the key act."""
        E, P = self._num_envs, self._num_players
        acting = np.ones((E, P), dtype=bool)
        for e in range(E):
            info_e = infos[e] if e < len(infos) else {}
            for p in range(P):
                info = info_e.get(p) if isinstance(info_e, dict) else None
                if isinstance(info, dict) and "active" in info:
                    acting[e, p] = bool(info["active"])
        return acting

    def _extract_masks(self, infos: list[dict]) -> np.ndarray | None:
        """[E*P, mask_size] bool, or None if no slot provides ``action_mask``.

        Slots without a mask get an all-true row.
        """
        M = self._action_spec.flat_mask_size
        if M == 0:
            return None
        E, P = self._num_envs, self._num_players
        out: np.ndarray | None = None
        for e in range(E):
            info_e = infos[e] if e < len(infos) else {}
            for p in range(P):
                info = info_e.get(p) if isinstance(info_e, dict) else None
                if not isinstance(info, dict) or info.get("action_mask") is None:
                    continue
                if out is None:
                    out = np.ones((E * P, M), dtype=bool)
                raw = info["action_mask"]
                if isinstance(raw, dict):
                    out[e * P + p] = self._action_spec.flatten_mask(raw)
                else:
                    out[e * P + p] = np.asarray(raw, dtype=bool).reshape(M)
        return out

    def _infer(self, obs: np.ndarray, masks: np.ndarray | None):
        """Batched inference for every slot, grouped by (agent, network).

        Returns (actions [E,P,*A], log_probs [E,P], values [E,P], pre_states)
        where ``pre_states[(e, p)]`` is the slot's model state BEFORE this step.
        """
        E, P = self._num_envs, self._num_players
        spec = self._action_spec
        actions = np.zeros((E, P, *spec.action_shape), dtype=spec.numpy_dtype)
        log_probs = np.zeros((E, P), dtype=np.float32)
        values = np.zeros((E, P), dtype=np.float32)
        pre_states: dict[tuple[int, int], State] = {}
        masks3 = None if masks is None else masks.reshape(E, P, -1)

        groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        for e in range(E):
            for p in range(P):
                groups[(self._slot_agent_map[e][p], self._slot_network_map[e][p])].append((e, p))

        for (aid, net_id), slots in groups.items():
            model = self._resolve_model(aid, net_id)
            ei = [e for e, _ in slots]
            pi = [p for _, p in slots]
            obs_b = torch.from_numpy(np.ascontiguousarray(obs[ei, pi], dtype=np.float32))
            mask_b = None if masks3 is None else torch.from_numpy(np.ascontiguousarray(masks3[ei, pi]))
            state_b = cat_batch([self._tracks[e][p].state for e, p in slots])
            with torch.no_grad():
                out = act(model, obs_b, state_b, mask_b)
            actions[ei, pi] = out.actions.cpu().numpy().astype(spec.numpy_dtype, copy=False)
            log_probs[ei, pi] = out.log_probs.float().cpu().numpy()
            values[ei, pi] = out.values.float().cpu().numpy()
            for j, (e, p) in enumerate(slots):
                track = self._tracks[e][p]
                pre_states[(e, p)] = track.state
                track.state = None if out.state is None else slice_batch(out.state, j)
        return actions, log_probs, values, pre_states

    def _open_transitions(self, obs, actions, log_probs, values, masks, acting, pre_states) -> None:
        """Record a transition for every acting collecting slot.

        Slots that report ``info["active"] = False`` are not recorded. A buffer's
        first transition records the slot's pre-step state and the agent's
        current policy version.
        """
        E, P = self._num_envs, self._num_players
        masks3 = None if masks is None else masks.reshape(E, P, -1)
        for e in range(E):
            for p in range(P):
                if not (acting[e, p] and self._collect_mask[e][p]):
                    continue
                track = self._tracks[e][p]
                aid = self._slot_agent_map[e][p]
                buf = track.buffer
                if buf.steps == 0:
                    buf.begin_chunk(pre_states[(e, p)], self._policy_versions[aid])
                buf.open(
                    obs[e, p], actions[e, p], float(log_probs[e, p]), float(values[e, p]),
                    None if masks3 is None else masks3[e, p],
                )
                track.has_open = True
                self._recorded[aid] += 1

    def _after_env_step(self, next_obs, rewards, terminated, truncated, infos) -> None:
        """Give each transition recorded this step its reward and done flag;
        seal full buffers (bootstrap = V(next_obs), or 0 after a terminal step)."""
        E, P = self._num_envs, self._num_players
        self._ep_rewards += rewards
        self._ep_lengths += 1
        for e in range(E):
            done = bool(terminated[e] or truncated[e])
            for p in range(P):
                track = self._tracks[e][p]
                if not (self._collect_mask[e][p] and track.has_open):
                    continue
                buf = track.buffer
                buf.add_reward(float(rewards[e, p]))
                if done:
                    buf.mark_done()
                track.has_open = False
                if buf.is_full:
                    boot = 0.0 if done else self._bootstrap_value_forward(e, p, next_obs[e, p])
                    self._seal(self._slot_agent_map[e][p], buf, bootstrap_value=boot)
            if done:
                self._end_episode(e, infos[e])

    def _end_episode(self, e: int, info_e: dict) -> None:
        P = self._num_players
        for p in range(P):
            buf = self._tracks[e][p].buffer
            # A collecting slot that did not act on the last step still ends its
            # episode here: its last recorded transition gets done=True.
            if self._collect_mask[e][p] and buf is not None and buf.steps > 0:
                buf.mark_done()
        self._report_result(e, info_e)
        self._ep_rewards[e] = 0.0
        self._ep_lengths[e] = 0
        self._ep_counter[e] += 1
        self._apply_pending_assignment(e)
        for p in range(P):
            self._tracks[e][p].state = self._initial_state(e, p)

    def _bootstrap_value_forward(self, e: int, p: int, next_obs: np.ndarray) -> float:
        """V(next_obs) from the slot agent's latest model with the slot's state."""
        model = self._models[self._slot_agent_map[e][p]][LATEST_NETWORK_ID]
        obs_b = torch.from_numpy(np.ascontiguousarray(next_obs[None], dtype=np.float32))
        with torch.no_grad():
            out = model.step(obs_b, self._tracks[e][p].state)
        return float(out.value[0])

    def _seal(self, aid: str, buf: RolloutBuffer, bootstrap_value: float) -> None:
        chunk = buf.build_chunk(aid, bootstrap_value)
        buf.reset()
        self._io.send_chunk(chunk)
        self._chunks_sent += 1

    def _report_result(self, e: int, info_e: dict) -> None:
        """Report env ``e``'s finished episode; players keyed ``agent_id:network_id``."""
        if self._io.report_result is None:
            return
        P = self._num_players
        terminal_infos = {
            p: (info_e.get(p, {}) or {}).get("terminal_info", {}) for p in range(P)
        }
        rewards = self._ep_rewards[e]
        outcomes = player_outcomes(rewards, terminal_infos, P)
        player_outcome: dict[str, float] = {}
        total_rewards: dict[str, float] = {}
        for p in range(P):
            key = f"{self._slot_agent_map[e][p]}:{self._slot_network_map[e][p]}"
            player_outcome[key] = float(outcomes[p])
            total_rewards[key] = float(rewards[p])
        self._io.report_result(MatchResult(
            match_id=f"w{self.worker_id}_e{e}_ep{int(self._ep_counter[e])}",
            player_outcomes=player_outcome,
            total_rewards=total_rewards,
            episode_length=int(self._ep_lengths[e]),
        ))
