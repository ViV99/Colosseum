"""In-process rollout loop: vectorized envs, batched inference, trajectory chunks.

``RolloutLoop`` owns the environments and the per-agent network pools of one
worker. All I/O (sending chunks, pulling weights, reporting match results,
receiving match re-assignments) goes through the ``LoopIO`` callbacks, so the
loop can be driven step by step inside a test process. ``rollout_worker_process``
(``colosseum.worker.rollout_worker``) wraps it with multiprocessing queues.

Multi-agent: several agents can occupy different player slots of the same
environments. Inference is grouped by (agent_id, network_id) for batching, and
each chunk is routed to the agent that produced it.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from colosseum.core.action_spec import ActionSpec
from colosseum.core.types import MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand
from colosseum.envs.base_env import BaseEnv
from colosseum.envs.vec_env import VectorEnv

logger = logging.getLogger(__name__)

LATEST_NETWORK_ID = "latest"


@dataclass
class RolloutBuffer:
    """Pre-allocated trajectory buffer for a single (env, player) pair.

    Uses numpy arrays allocated once at creation with a write cursor.
    Eliminates per-step list append overhead and reduces copies at chunk build time.
    """

    chunk_length: int
    obs_shape: tuple = ()
    action_shape: tuple = ()
    action_dtype: type = np.int64
    num_actions: int = 0

    _observations: np.ndarray = field(init=False, repr=False)
    _actions: np.ndarray = field(init=False, repr=False)
    _log_probs: np.ndarray = field(init=False, repr=False)
    _rewards: np.ndarray = field(init=False, repr=False)
    _dones: np.ndarray = field(init=False, repr=False)
    _values: np.ndarray = field(init=False, repr=False)
    _action_masks: np.ndarray | None = field(init=False, default=None, repr=False)
    _cursor: int = field(init=False, default=0)
    _has_masks: bool = field(init=False, default=False)
    _lstm_h_init: torch.Tensor | None = field(init=False, default=None, repr=False)
    _lstm_c_init: torch.Tensor | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        T = self.chunk_length
        self._observations = np.zeros((T, *self.obs_shape), dtype=np.float32)
        self._actions = np.zeros((T, *self.action_shape), dtype=self.action_dtype)
        self._log_probs = np.zeros(T, dtype=np.float32)
        self._rewards = np.zeros(T, dtype=np.float32)
        self._dones = np.zeros(T, dtype=np.float32)
        self._values = np.zeros(T, dtype=np.float32)
        if self.num_actions > 0:
            self._action_masks = np.zeros((T, self.num_actions), dtype=np.bool_)

    def append(self, obs, action, log_prob, reward, done, value, action_mask=None):
        i = self._cursor
        self._observations[i] = obs
        self._actions[i] = action
        self._log_probs[i] = log_prob
        self._rewards[i] = reward
        self._dones[i] = done
        self._values[i] = value
        if action_mask is not None and self._action_masks is not None:
            self._action_masks[i] = action_mask
            self._has_masks = True
        self._cursor += 1

    @property
    def is_full(self) -> bool:
        return self._cursor >= self.chunk_length

    @property
    def steps(self) -> int:
        return self._cursor

    @property
    def has_masks(self) -> bool:
        return self._has_masks

    def set_lstm_init(self, h: torch.Tensor, c: torch.Tensor):
        """Save initial LSTM hidden state for the current chunk."""
        self._lstm_h_init = h.clone()
        self._lstm_c_init = c.clone()

    def reset(self):
        self._cursor = 0
        self._has_masks = False
        self._lstm_h_init = None
        self._lstm_c_init = None


@dataclass
class LoopIO:
    """Callbacks through which a ``RolloutLoop`` talks to the outside world."""

    send_chunk: Callable[[TrajectoryChunk], None]
    poll_weights: Callable[[str], WeightPayload | None]
    report_result: Callable[[MatchResult], None] | None = None
    poll_command: Callable[[], WorkerCommand | None] | None = None
    add_env_steps: Callable[[int], None] | None = None  # global env-step budget (wired in T2.5)


def _run_inference_group(
    net,
    indices: list[tuple],
    obs_flat: np.ndarray,
    all_masks: np.ndarray | None,
    hidden_states: dict,
    out_actions: np.ndarray,
    out_log_probs: np.ndarray,
    out_values: np.ndarray,
) -> None:
    """Batched inference for one (agent, network) group; writes into output arrays."""
    idx_list = [i[0] for i in indices]
    obs_batch = torch.from_numpy(np.ascontiguousarray(obs_flat[idx_list])).float()
    mask_batch = None
    if all_masks is not None:
        mask_batch = torch.from_numpy(np.ascontiguousarray(all_masks[idx_list])).bool()

    hidden_batch = None
    if net.is_recurrent:
        h_list, c_list = [], []
        for _, env_idx, p in indices:
            h, c = hidden_states.get((env_idx, p), net.initial_hidden(1))
            h_list.append(h)
            c_list.append(c)
        hidden_batch = (torch.cat(h_list, dim=1), torch.cat(c_list, dim=1))

    with torch.no_grad():
        actions, log_probs, values, new_hidden = net.act(
            obs_batch, action_mask=mask_batch, hidden=hidden_batch,
        )

    if net.is_recurrent and new_hidden is not None:
        for j, (_, env_idx, p) in enumerate(indices):
            hidden_states[(env_idx, p)] = (
                new_hidden[0][:, j:j + 1, :].clone(),
                new_hidden[1][:, j:j + 1, :].clone(),
            )

    idx_arr = np.asarray(idx_list, dtype=np.intp)
    out_actions[idx_arr] = actions.numpy()
    out_log_probs[idx_arr] = log_probs.numpy().astype(np.float32, copy=False)
    out_values[idx_arr] = values.numpy().astype(np.float32, copy=False)


def _apply_command(cmd, networks_by_agent, network_factories, pending) -> None:
    """Load any new checkpoints into the pool and stash the new slot maps.

    The slot maps are applied per-env at the next episode boundary (so a match
    keeps a consistent assignment for its whole episode).
    """
    for aid, ckpts in cmd.new_checkpoints.items():
        if aid not in networks_by_agent:
            continue
        for ckpt_id, sd in ckpts.items():
            if ckpt_id not in networks_by_agent[aid]:
                net = network_factories[aid]()
                net.load_state_dict(sd)
                net.eval()
                networks_by_agent[aid][ckpt_id] = net
    if cmd.slot_agent_map:
        pending["slot_agent_map"] = cmd.slot_agent_map
        pending["slot_network_map"] = cmd.slot_network_map
        pending["collect_mask"] = cmd.collect_mask


def _build_chunk(buffer: RolloutBuffer, agent_id, bootstrap_value, policy_version):
    """Build a TrajectoryChunk from a pre-allocated buffer."""
    chunk = TrajectoryChunk(
        agent_id=agent_id,
        observations=torch.from_numpy(buffer._observations.copy()),
        actions=torch.from_numpy(buffer._actions.copy()),
        action_log_probs=torch.from_numpy(buffer._log_probs.copy()),
        rewards=torch.from_numpy(buffer._rewards.copy()),
        dones=torch.from_numpy(buffer._dones.copy()),
        values=torch.from_numpy(buffer._values.copy()),
        bootstrap_value=torch.tensor(bootstrap_value, dtype=torch.float32),
        behavior_policy_version=policy_version,
    )
    if buffer._has_masks and buffer._action_masks is not None:
        chunk.action_masks = torch.from_numpy(buffer._action_masks.copy())
    if buffer._lstm_h_init is not None:
        chunk.lstm_hidden = (buffer._lstm_h_init, buffer._lstm_c_init)
    return chunk


def _make_match_result(
    worker_id: int,
    env_idx: int,
    step: int,
    ep_rewards: np.ndarray,
    ep_length: int,
    slot_nets: list[str],
    slot_agent_ids: list[str],
    terminal_infos: dict[int, dict] | None = None,
) -> MatchResult:
    """Build the episode result reported to the coordinator.

    Player outcomes are keyed by ``agent_id:network_id`` (``"agent_0:latest"``,
    ``"agent_0:ckpt_v100"``). Outcomes prefer the env's authoritative signal
    (``rank``/``outcome`` in the terminal info) and fall back to cumulative reward.
    """
    from colosseum.core.outcomes import player_outcomes as _player_outcomes

    num_players = len(slot_nets)
    outcomes = _player_outcomes(ep_rewards, terminal_infos, num_players)

    player_outcomes: dict[str, float] = {}
    total_rewards: dict[str, float] = {}
    for p in range(num_players):
        player_key = f"{slot_agent_ids[p]}:{slot_nets[p]}"
        total_rewards[player_key] = float(ep_rewards[p])
        player_outcomes[player_key] = float(outcomes[p])

    return MatchResult(
        match_id=f"w{worker_id}_e{env_idx}_{step}",
        player_outcomes=player_outcomes,
        total_rewards=total_rewards,
        episode_length=int(ep_length),
    )


def _extract_action_masks(
    infos: list[dict],
    num_envs: int,
    num_players: int,
    action_spec=None,
) -> np.ndarray | None:
    """Extract action masks from env info dicts into a flat array.

    Convention: info[env_idx][player_idx]["action_mask"] is a bool ndarray
    or dict of per-component bool ndarrays (for composite action spaces).
    Returns [num_envs * num_players, num_actions] bool array, or None if no masks.
    """
    if not infos:
        return None
    first_info = infos[0]
    if not isinstance(first_info, dict) or 0 not in first_info:
        return None
    if "action_mask" not in first_info[0]:
        return None

    masks = []
    for env_idx in range(num_envs):
        for p in range(num_players):
            raw = infos[env_idx][p]["action_mask"]
            if isinstance(raw, dict) and action_spec is not None:
                masks.append(action_spec.flatten_mask(raw))
            else:
                masks.append(raw)
    return np.array(masks, dtype=bool)


def _extract_active_flags(
    infos: list[dict],
    num_envs: int,
    num_players: int,
) -> np.ndarray | None:
    """Per-slot ``info["active"]`` flags, or None if the env doesn't provide them."""
    if not infos:
        return None
    first = infos[0]
    if not isinstance(first, dict) or 0 not in first:
        return None
    if not isinstance(first[0], dict) or "active" not in first[0]:
        return None
    flags = np.ones((num_envs, num_players), dtype=bool)
    for e in range(num_envs):
        for p in range(num_players):
            flags[e][p] = bool(infos[e][p].get("active", True))
    return flags


class RolloutLoop:
    """One worker's environments + network pools, advanced one vector step at a time."""

    def __init__(
        self,
        *,
        worker_id: int,
        env_fn: Callable[[], BaseEnv],
        num_envs: int,
        chunk_length: int,
        agent_ids: list[str],
        model_factories: dict[str, Callable[[], nn.Module]],
        io: LoopIO,
        gamma: float = 0.99,
        weight_sync_interval: float = 5.0,
        slot_agent_map: list[list[str]] | None = None,
        slot_network_map: list[list[str]] | None = None,
        collect_mask: list[list[bool]] | None = None,
        checkpoint_state_dicts_by_agent: dict[str, dict[str, dict]] | None = None,
        seed: int | None = None,
        vec_env_kind: str = "sync",
        subproc_workers: int | None = None,
    ) -> None:
        if checkpoint_state_dicts_by_agent is None:
            checkpoint_state_dicts_by_agent = {aid: {} for aid in agent_ids}

        self.worker_id = worker_id
        self.num_envs = num_envs
        self.chunk_length = chunk_length
        self.agent_ids = list(agent_ids)
        self.gamma = gamma  # used for truncation bootstrapping (T3.3)
        self._io = io
        self._model_factories = model_factories
        self._weight_sync_interval = weight_sync_interval

        logger.info(
            f"Worker {worker_id}: starting with {num_envs} envs ({vec_env_kind}), "
            f"chunk_length={chunk_length}, agents={agent_ids}"
        )

        if vec_env_kind == "subprocess":
            from colosseum.envs.subproc_vec_env import SubprocessVectorEnv
            self._vec_env = SubprocessVectorEnv(env_fn, num_envs, num_workers=subproc_workers)
        else:
            self._vec_env = VectorEnv(env_fn, num_envs)
        num_players = self._vec_env.num_players
        self.num_players = num_players

        # Per-agent network pools: "latest" receives weight updates; checkpoint
        # networks are frozen opponents.
        self._networks: dict[str, dict[str, nn.Module]] = {}
        self._policy_versions: dict[str, int] = {}
        for aid in self.agent_ids:
            nets: dict[str, nn.Module] = {}
            nets[LATEST_NETWORK_ID] = model_factories[aid]()
            nets[LATEST_NETWORK_ID].eval()
            self._policy_versions[aid] = 0
            ckpt_dicts = checkpoint_state_dicts_by_agent.get(aid, {})
            for ckpt_id, sd in ckpt_dicts.items():
                net = model_factories[aid]()
                net.load_state_dict(sd)
                net.eval()
                nets[ckpt_id] = net
            if ckpt_dicts:
                logger.info(
                    f"Worker {worker_id}: agent {aid}: loaded {len(ckpt_dicts)} "
                    f"checkpoint(s): {list(ckpt_dicts.keys())}"
                )
            self._networks[aid] = nets

        # Initial weights for every agent's latest network.
        self.sync_weights()

        self._any_recurrent = self._compute_any_recurrent()

        # Default slot assignment: every slot is the first agent's latest network.
        if slot_agent_map is None:
            slot_agent_map = [[self.agent_ids[0]] * num_players for _ in range(num_envs)]
        if slot_network_map is None:
            if collect_mask is not None:
                slot_network_map = []
                for e in range(num_envs):
                    slot_nets = []
                    for p in range(num_players):
                        aid = slot_agent_map[e][p]
                        ckpts = checkpoint_state_dicts_by_agent.get(aid, {})
                        if collect_mask[e][p] or not ckpts:
                            slot_nets.append(LATEST_NETWORK_ID)
                        else:
                            slot_nets.append(next(iter(ckpts)))
                    slot_network_map.append(slot_nets)
            else:
                slot_network_map = [[LATEST_NETWORK_ID] * num_players for _ in range(num_envs)]
        if collect_mask is None:
            collect_mask = [[True] * num_players for _ in range(num_envs)]

        # Live (mutable) match assignment; WorkerCommand updates are staged in
        # `_pending` and applied per env at its next episode boundary.
        self._slot_agent_map = [list(row) for row in slot_agent_map]
        self._slot_network_map = [list(row) for row in slot_network_map]
        self._collect_mask = [list(row) for row in collect_mask]
        self._pending: dict[str, list | None] = {
            "slot_agent_map": None, "slot_network_map": None, "collect_mask": None,
        }

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self._obs, self._infos = self._vec_env.reset_all(seed=seed)

        self._hidden_states: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        if self._any_recurrent:
            for env_idx in range(num_envs):
                for p in range(num_players):
                    aid = self._slot_agent_map[env_idx][p]
                    net = self._networks[aid][LATEST_NETWORK_ID]
                    if net.is_recurrent:
                        self._hidden_states[(env_idx, p)] = net.initial_hidden(1)

        self._action_spec = ActionSpec.from_space(self._vec_env.action_space)
        obs_shape = self._obs.shape[2:]
        # A buffer exists for every slot (not only collecting ones): runtime
        # re-assignment can flip a slot's collect flag.
        self._buffers: list[list[RolloutBuffer]] = [
            [
                RolloutBuffer(
                    chunk_length=chunk_length,
                    obs_shape=obs_shape,
                    action_shape=self._action_spec.action_shape,
                    action_dtype=self._action_spec.numpy_dtype,
                    num_actions=self._action_spec.flat_mask_size,
                )
                for _ in range(num_players)
            ]
            for _ in range(num_envs)
        ]

        self._total_steps = 0
        self._chunks_sent = 0
        self._episodes = 0
        self._last_weight_sync = time.time()
        self._ep_rewards = np.zeros((num_envs, num_players), dtype=np.float64)
        self._ep_lengths = np.zeros(num_envs, dtype=np.int64)

        num_slots = num_envs * num_players
        self._all_actions = np.zeros(
            (num_slots, *self._action_spec.action_shape), dtype=self._action_spec.numpy_dtype,
        )
        self._all_log_probs = np.zeros(num_slots, dtype=np.float32)
        self._all_values = np.zeros(num_slots, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        return {
            "chunks_sent": self._chunks_sent,
            "env_steps": self._total_steps,
            "episodes": self._episodes,
            "parked_buffers": 0,
        }

    def sync_weights(self) -> None:
        """Load the newest available weights into every agent's latest network."""
        for aid in self.agent_ids:
            payload = self._io.poll_weights(aid)
            if payload is not None:
                self._networks[aid][LATEST_NETWORK_ID].load_state_dict(payload.state_dict)
                self._policy_versions[aid] = payload.policy_version

    def run(self, should_stop: Callable[[], bool], max_env_steps: int = 0) -> None:
        """Step until ``should_stop()`` or until ``max_env_steps`` env steps (0 = no limit)."""
        while not should_stop():
            if max_env_steps > 0 and self._total_steps >= max_env_steps:
                break
            self.step()

    def close(self) -> None:
        self._vec_env.close()

    def step(self) -> int:
        """Advance every env by one step. Returns the number of env steps taken."""
        num_envs, num_players = self.num_envs, self.num_players
        obs = self._obs
        infos = self._infos

        cmd = self._io.poll_command() if self._io.poll_command is not None else None
        if cmd is not None:
            _apply_command(cmd, self._networks, self._model_factories, self._pending)
            self._any_recurrent = self._compute_any_recurrent()

        # Save the recurrent state at chunk start (before inference).
        if self._any_recurrent:
            for env_idx in range(num_envs):
                for p in range(num_players):
                    buf = self._buffers[env_idx][p]
                    if (self._collect_mask[env_idx][p] and buf.steps == 0
                            and (env_idx, p) in self._hidden_states):
                        h, c = self._hidden_states[(env_idx, p)]
                        buf.set_lstm_init(h, c)

        # Group slots by (agent_id, network_id) for batched inference.
        net_groups: dict[tuple[str, str], list[tuple]] = defaultdict(list)
        for env_idx in range(num_envs):
            for p in range(num_players):
                flat_idx = env_idx * num_players + p
                aid = self._slot_agent_map[env_idx][p]
                net_id = self._slot_network_map[env_idx][p]
                net_groups[(aid, net_id)].append((flat_idx, env_idx, p))

        obs_flat = obs.reshape(-1, *obs.shape[2:])
        self._all_actions.fill(0)
        self._all_log_probs.fill(0)
        self._all_values.fill(0)

        # Masks and turn flags describe the state the agents are about to act on.
        all_masks = _extract_action_masks(infos, num_envs, num_players, action_spec=self._action_spec)
        active_flags = _extract_active_flags(infos, num_envs, num_players)

        for (aid, net_id), indices in net_groups.items():
            agent_nets = self._networks[aid]
            net = agent_nets.get(net_id, agent_nets[LATEST_NETWORK_ID])
            _run_inference_group(
                net, indices, obs_flat, all_masks, self._hidden_states,
                self._all_actions, self._all_log_probs, self._all_values,
            )

        actions_np = self._all_actions.reshape(num_envs, num_players, *self._action_spec.action_shape)
        log_probs_np = self._all_log_probs.reshape(num_envs, num_players)
        values_np = self._all_values.reshape(num_envs, num_players)

        next_obs, rewards, terminated, truncated, infos = self._vec_env.step(actions_np)

        for env_idx in range(num_envs):
            done = terminated[env_idx] or truncated[env_idx]
            self._ep_lengths[env_idx] += 1

            for player_idx in range(num_players):
                self._ep_rewards[env_idx, player_idx] += rewards[env_idx, player_idx]

                slot_active = active_flags is None or active_flags[env_idx][player_idx]
                if self._collect_mask[env_idx][player_idx] and slot_active:
                    buf = self._buffers[env_idx][player_idx]
                    slot_aid = self._slot_agent_map[env_idx][player_idx]

                    action = actions_np[env_idx, player_idx]
                    if isinstance(action, np.ndarray) and action.ndim == 0:
                        action = action.item()

                    mask_for_step = None
                    if all_masks is not None:
                        mask_for_step = all_masks[env_idx * num_players + player_idx]

                    buf.append(
                        obs=obs[env_idx, player_idx],
                        action=action,
                        log_prob=log_probs_np[env_idx, player_idx],
                        reward=rewards[env_idx, player_idx],
                        done=done,
                        value=values_np[env_idx, player_idx],
                        action_mask=mask_for_step,
                    )

                    if buf.is_full:
                        bootstrap_net = self._networks[slot_aid][LATEST_NETWORK_ID]
                        next_obs_t = torch.tensor(
                            next_obs[env_idx, player_idx], dtype=torch.float32,
                        ).unsqueeze(0)
                        bootstrap_hidden = self._hidden_states.get((env_idx, player_idx))
                        with torch.no_grad():
                            _, _, bootstrap_val, _ = bootstrap_net.act(
                                next_obs_t, hidden=bootstrap_hidden,
                            )
                        bootstrap_val = 0.0 if done else bootstrap_val.item()

                        chunk = _build_chunk(
                            buf, slot_aid, bootstrap_val, self._policy_versions[slot_aid],
                        )
                        self._io.send_chunk(chunk)
                        self._chunks_sent += 1
                        buf.reset()

            if done and self._io.report_result is not None:
                term_infos = {
                    p: infos[env_idx][p].get("terminal_info", {})
                    for p in range(num_players)
                }
                self._io.report_result(_make_match_result(
                    self.worker_id, env_idx, self._total_steps,
                    self._ep_rewards[env_idx], self._ep_lengths[env_idx],
                    self._slot_network_map[env_idx], self._slot_agent_map[env_idx],
                    term_infos,
                ))

            if done:
                self._episodes += 1
                self._ep_rewards[env_idx] = 0.0
                self._ep_lengths[env_idx] = 0
                self._apply_pending_assignment(env_idx)

                if self._any_recurrent:
                    for p in range(num_players):
                        if (env_idx, p) in self._hidden_states:
                            aid = self._slot_agent_map[env_idx][p]
                            net = self._networks[aid][LATEST_NETWORK_ID]
                            self._hidden_states[(env_idx, p)] = net.initial_hidden(1)

        self._obs = next_obs
        self._infos = infos
        self._total_steps += num_envs

        now = time.time()
        if now - self._last_weight_sync >= self._weight_sync_interval:
            self.sync_weights()
            self._last_weight_sync = now

        return num_envs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_any_recurrent(self) -> bool:
        return any(
            nets[LATEST_NETWORK_ID].is_recurrent for nets in self._networks.values()
        )

    def _apply_pending_assignment(self, env_idx: int) -> None:
        """Apply a staged match re-assignment to one env at its episode boundary."""
        pending = self._pending
        if pending["slot_agent_map"] is None or env_idx >= len(pending["slot_agent_map"]):
            return
        new_agents = pending["slot_agent_map"][env_idx]
        new_nets = pending["slot_network_map"][env_idx]
        new_collect = pending["collect_mask"][env_idx]
        for p in range(self.num_players):
            # Discard the partial buffer when a slot's agent or collect flag changes.
            if (self._slot_agent_map[env_idx][p] != new_agents[p]
                    or self._collect_mask[env_idx][p] != new_collect[p]):
                self._buffers[env_idx][p].reset()
            self._slot_agent_map[env_idx][p] = new_agents[p]
            self._slot_network_map[env_idx][p] = new_nets[p]
            self._collect_mask[env_idx][p] = new_collect[p]
