"""Rollout worker: runs environments, collects trajectories.

Each RolloutWorker runs in a separate process. It owns a VectorEnv with
`num_envs` environments, each having `num_players` player slots.

The worker:
1. Steps all envs in parallel (sequential within VectorEnv)
2. Runs batched inference for all player slots
3. Collects trajectory data into per-(env, player) buffers (only for collecting slots)
4. When a buffer reaches chunk_length, seals it into a TrajectoryChunk
5. Sends chunks to the learner via the trajectory queue
6. Periodically pulls fresh weights from the weight store

Multi-agent: multiple agents can occupy different player slots in the same
environments. Each agent has its own network pool, trajectory queue, and
weight queue. Inference is grouped by (agent_id, network_id) for batching
efficiency. Chunks are routed to the correct agent's learner.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from colosseum.core.action_spec import ActionSpec
from colosseum.core.types import TrajectoryChunk, WeightPayload
from colosseum.envs.base_env import BaseEnv
from colosseum.envs.vec_env import VectorEnv
from colosseum.networks.actor_critic import ActorCriticNetwork

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
    _action_masks: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _cursor: int = field(init=False, default=0)
    _has_masks: bool = field(init=False, default=False)
    _lstm_h_init: Optional[torch.Tensor] = field(init=False, default=None, repr=False)
    _lstm_c_init: Optional[torch.Tensor] = field(init=False, default=None, repr=False)

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


def _run_inference_group(
    net: ActorCriticNetwork,
    indices: list[tuple],
    obs_flat: np.ndarray,
    all_masks: Optional[np.ndarray],
    hidden_states: dict,
    out_actions: np.ndarray,
    out_log_probs: np.ndarray,
    out_values: np.ndarray,
) -> None:
    """Batched inference for one (agent, network) group; writes into output arrays.

    Scatter is vectorised (fancy-indexed numpy assignment) instead of a per-slot
    Python loop with ``.item()`` calls.
    """
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


def _drain_commands(command_queue: Optional[mp.Queue]):
    """Return the most recent WorkerCommand (draining stale ones), or None."""
    if command_queue is None:
        return None
    latest = None
    while True:
        try:
            latest = command_queue.get_nowait()
        except queue.Empty:
            break
    return latest


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
    """Build TrajectoryChunk from pre-allocated buffer.

    Uses ndarray.copy() + torch.from_numpy() — one contiguous memcpy per field
    instead of the previous list→ndarray→tensor double conversion.
    """
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


def rollout_worker_process(
    worker_id: int,
    env_fn: Callable[[], BaseEnv],
    num_envs: int,
    chunk_length: int,
    agent_ids: list[str],
    network_factories: dict[str, Callable[[], ActorCriticNetwork]],
    trajectory_queues: dict[str, mp.Queue],
    weight_queues: dict[str, mp.Queue],
    stop_event: mp.Event,
    weight_sync_interval: float = 5.0,
    total_timesteps: int = 0,
    checkpoint_state_dicts_by_agent: Optional[dict[str, dict[str, dict]]] = None,
    slot_network_map: Optional[list[list[str]]] = None,
    collect_mask: Optional[list[list[bool]]] = None,
    slot_agent_map: Optional[list[list[str]]] = None,
    results_queue: Optional[mp.Queue] = None,
    seed: Optional[int] = None,
    command_queue: Optional[mp.Queue] = None,
    vec_env_kind: str = "sync",
    subproc_workers: Optional[int] = None,
) -> None:
    """Main worker process function.

    Args:
        worker_id: unique worker identifier.
        env_fn: factory to create a BaseEnv instance.
        num_envs: number of vectorized envs.
        chunk_length: steps per trajectory chunk.
        agent_ids: list of trainable agent IDs.
        network_factories: agent_id -> factory to create ActorCriticNetwork.
        trajectory_queues: agent_id -> queue to send chunks to the learner.
        weight_queues: agent_id -> queue to receive weight updates.
        stop_event: set to signal the worker to stop.
        weight_sync_interval: seconds between weight sync checks.
        total_timesteps: stop after this many env steps (0 = run until stop_event).
        checkpoint_state_dicts_by_agent: agent_id -> {ckpt_id -> state_dict}.
        slot_network_map: [num_envs][num_players] -> network_id string.
        collect_mask: [num_envs][num_players] bool mask.
        slot_agent_map: [num_envs][num_players] -> agent_id.
        results_queue: optional queue for episode results.
        seed: random seed for reproducibility.
        command_queue: optional queue of WorkerCommand updates (runtime match
            re-assignment + new checkpoints). Applied per-env at episode
            boundaries so matchmaking evolves without restarting the worker.
        vec_env_kind: "sync" (in-process) or "subprocess" (parallel env stepping).
        subproc_workers: child-process count for the subprocess vec_env backend.
    """
    if checkpoint_state_dicts_by_agent is None:
        checkpoint_state_dicts_by_agent = {aid: {} for aid in agent_ids}

    logger.info(
        f"Worker {worker_id}: starting with {num_envs} envs ({vec_env_kind}), "
        f"chunk_length={chunk_length}, agents={agent_ids}"
    )

    if vec_env_kind == "subprocess":
        from colosseum.envs.subproc_vec_env import SubprocessVectorEnv
        vec_env = SubprocessVectorEnv(env_fn, num_envs, num_workers=subproc_workers)
    else:
        vec_env = VectorEnv(env_fn, num_envs)
    num_players = vec_env.num_players

    # ------------------------------------------------------------------
    # Build per-agent network pools
    # ------------------------------------------------------------------
    networks_by_agent: dict[str, dict[str, ActorCriticNetwork]] = {}
    policy_versions: dict[str, int] = {}

    for aid in agent_ids:
        nets: dict[str, ActorCriticNetwork] = {}

        # "latest" network — receives weight updates from this agent's learner
        nets[LATEST_NETWORK_ID] = network_factories[aid]()
        nets[LATEST_NETWORK_ID].eval()
        policy_versions[aid] = 0

        # Checkpoint networks for this agent
        ckpt_dicts = checkpoint_state_dicts_by_agent.get(aid, {})
        for ckpt_id, sd in ckpt_dicts.items():
            net = network_factories[aid]()
            net.load_state_dict(sd)
            net.eval()
            nets[ckpt_id] = net

        if ckpt_dicts:
            logger.info(
                f"Worker {worker_id}: agent {aid}: loaded {len(ckpt_dicts)} "
                f"checkpoint(s): {list(ckpt_dicts.keys())}"
            )

        networks_by_agent[aid] = nets

    # Try to get initial weights for all agents' latest networks
    for aid in agent_ids:
        _sync_weights(networks_by_agent[aid][LATEST_NETWORK_ID], weight_queues[aid])

    # Check if any agent's network is recurrent
    _any_recurrent = any(
        nets[LATEST_NETWORK_ID].is_recurrent
        for nets in networks_by_agent.values()
    )

    # ------------------------------------------------------------------
    # Default slot_agent_map: all slots use the first (or single) agent
    # ------------------------------------------------------------------
    if slot_agent_map is None:
        slot_agent_map = [
            [agent_ids[0]] * num_players for _ in range(num_envs)
        ]

    # Default slot_network_map: all slots use "latest"
    if slot_network_map is None:
        # If collect_mask exists, non-collecting slots use first checkpoint
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
            slot_network_map = [
                [LATEST_NETWORK_ID] * num_players for _ in range(num_envs)
            ]

    # Collect mask: which (env, player) slots collect trajectories
    if collect_mask is None:
        collect_mask = [[True] * num_players for _ in range(num_envs)]

    # These three maps are the *live* (mutable) match assignment. Runtime
    # WorkerCommand updates re-assign them per-env at episode boundaries so
    # self-play-vs-checkpoints and PFSP opponent selection evolve during
    # training without restarting the worker.
    slot_agent_map = [list(row) for row in slot_agent_map]
    slot_network_map = [list(row) for row in slot_network_map]
    collect_mask = [list(row) for row in collect_mask]
    # Latest received-but-not-yet-applied assignment (applied per-env on done).
    pending: dict[str, Optional[list]] = {
        "slot_agent_map": None, "slot_network_map": None, "collect_mask": None,
    }

    # Seed worker RNGs for reproducibility
    if seed is not None:
        import random
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Initialize env
    obs, infos = vec_env.reset_all(seed=seed)

    # Initialize hidden states for recurrent networks
    hidden_states: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    if _any_recurrent:
        for env_idx in range(num_envs):
            for p in range(num_players):
                aid = slot_agent_map[env_idx][p]
                net = networks_by_agent[aid][LATEST_NETWORK_ID]
                if net.is_recurrent:
                    hidden_states[(env_idx, p)] = net.initial_hidden(1)

    # Derive shapes for pre-allocated buffers
    action_spec = ActionSpec.from_space(vec_env.action_space)

    obs_shape = obs.shape[2:]  # obs is [num_envs, num_players, *obs_shape]
    action_shape_tup = action_spec.action_shape
    buf_action_dtype = action_spec.numpy_dtype
    buf_num_actions = action_spec.flat_mask_size

    # Allocate a rollout buffer for EVERY slot (not just currently-collecting
    # ones): runtime re-assignment can flip a slot's collect status, so buffers
    # must always exist. Whether a slot actually appends/seals is gated on the
    # live collect_mask below.
    buffers: list[list[RolloutBuffer]] = []
    for env_idx in range(num_envs):
        env_buffers = []
        for p in range(num_players):
            env_buffers.append(RolloutBuffer(
                chunk_length=chunk_length,
                obs_shape=obs_shape,
                action_shape=action_shape_tup,
                action_dtype=buf_action_dtype,
                num_actions=buf_num_actions,
            ))
        buffers.append(env_buffers)

    total_steps = 0
    last_weight_sync = time.time()
    chunks_sent = 0

    # Episode reward tracking for results reporting
    ep_rewards = np.zeros((num_envs, num_players), dtype=np.float64)
    ep_lengths = np.zeros(num_envs, dtype=np.int64)

    # Pre-allocated inference output arrays (reused every step)
    _num_slots = num_envs * num_players
    all_actions = np.zeros((_num_slots, *action_spec.action_shape), dtype=action_spec.numpy_dtype)
    all_log_probs = np.zeros(_num_slots, dtype=np.float32)
    all_values = np.zeros(_num_slots, dtype=np.float32)

    while not stop_event.is_set():
        if total_timesteps > 0 and total_steps >= total_timesteps:
            break

        # Apply any runtime match-assignment update (new checkpoints loaded into
        # the pool now; slot maps staged in `pending`, applied per-env on done).
        cmd = _drain_commands(command_queue)
        if cmd is not None:
            _apply_command(cmd, networks_by_agent, network_factories, pending)
            _any_recurrent = any(
                nets[LATEST_NETWORK_ID].is_recurrent
                for nets in networks_by_agent.values()
            )

        # Save LSTM hidden state at chunk start (before inference)
        if _any_recurrent:
            for env_idx in range(num_envs):
                for p in range(num_players):
                    buf = buffers[env_idx][p]
                    if collect_mask[env_idx][p] and buf.steps == 0 and (env_idx, p) in hidden_states:
                        h, c = hidden_states[(env_idx, p)]
                        buf.set_lstm_init(h, c)

        # Group slot indices by (agent_id, network_id) for batched inference
        net_groups: dict[tuple[str, str], list[tuple]] = defaultdict(list)
        for env_idx in range(num_envs):
            for p in range(num_players):
                flat_idx = env_idx * num_players + p
                aid = slot_agent_map[env_idx][p]
                net_id = slot_network_map[env_idx][p]
                net_groups[(aid, net_id)].append((flat_idx, env_idx, p))

        # Batch inference (reuse pre-allocated arrays)
        obs_flat = obs.reshape(-1, *obs.shape[2:])
        all_actions.fill(0)
        all_log_probs.fill(0)
        all_values.fill(0)

        # Extract action masks + turn-based active flags from the PRE-step info
        # (these describe the state the agents are about to act on).
        all_masks = _extract_action_masks(infos, num_envs, num_players, action_spec=action_spec)
        active_flags = _extract_active_flags(infos, num_envs, num_players)

        for (aid, net_id), indices in net_groups.items():
            agent_nets = networks_by_agent[aid]
            net = agent_nets.get(net_id, agent_nets[LATEST_NETWORK_ID])
            _run_inference_group(
                net, indices, obs_flat, all_masks, hidden_states,
                all_actions, all_log_probs, all_values,
            )

        actions_np = all_actions.reshape(num_envs, num_players, *action_spec.action_shape)
        log_probs_np = all_log_probs.reshape(num_envs, num_players)
        values_np = all_values.reshape(num_envs, num_players)

        # Step all envs
        next_obs, rewards, terminated, truncated, infos = vec_env.step(actions_np)

        # Accumulate rewards and store transitions
        for env_idx in range(num_envs):
            done = terminated[env_idx] or truncated[env_idx]
            ep_lengths[env_idx] += 1

            for player_idx in range(num_players):
                ep_rewards[env_idx, player_idx] += rewards[env_idx, player_idx]

                slot_active = active_flags is None or active_flags[env_idx][player_idx]
                if collect_mask[env_idx][player_idx] and slot_active:
                    buf = buffers[env_idx][player_idx]
                    slot_aid = slot_agent_map[env_idx][player_idx]

                    action = actions_np[env_idx, player_idx]
                    if isinstance(action, np.ndarray) and action.ndim == 0:
                        action = action.item()

                    mask_for_step = None
                    if all_masks is not None:
                        flat_idx = env_idx * num_players + player_idx
                        mask_for_step = all_masks[flat_idx]

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
                        # Bootstrap value from the same agent's latest network
                        bootstrap_net = networks_by_agent[slot_aid][LATEST_NETWORK_ID]
                        next_obs_t = torch.tensor(
                            next_obs[env_idx, player_idx], dtype=torch.float32
                        ).unsqueeze(0)
                        bootstrap_hidden = hidden_states.get((env_idx, player_idx))
                        with torch.no_grad():
                            _, _, bootstrap_val, _ = bootstrap_net.act(
                                next_obs_t, hidden=bootstrap_hidden,
                            )
                        bootstrap_val = 0.0 if done else bootstrap_val.item()

                        chunk = _build_chunk(
                            buf, slot_aid, bootstrap_val,
                            policy_versions[slot_aid],
                        )
                        # Route chunk to the correct agent's learner queue
                        tq = trajectory_queues[slot_aid]
                        while not stop_event.is_set():
                            try:
                                tq.put(chunk, timeout=1.0)
                                break
                            except queue.Full:
                                continue
                        chunks_sent += 1
                        buf.reset()

            # Report episode result when done (terminal infos give the env's
            # authoritative outcome when available; else cumulative reward).
            if done and results_queue is not None:
                term_infos = {
                    p: infos[env_idx][p].get("terminal_info", {})
                    for p in range(num_players)
                }
                _report_episode_result(
                    results_queue, worker_id, env_idx, total_steps,
                    ep_rewards[env_idx], ep_lengths[env_idx],
                    slot_network_map[env_idx], slot_agent_map[env_idx],
                    term_infos,
                )

            # Reset episode trackers, apply any pending re-assignment, reset hidden.
            if done:
                ep_rewards[env_idx] = 0.0
                ep_lengths[env_idx] = 0

                # Apply staged match re-assignment for this env's next episode.
                if (pending["slot_agent_map"] is not None
                        and env_idx < len(pending["slot_agent_map"])):
                    new_agents = pending["slot_agent_map"][env_idx]
                    new_nets = pending["slot_network_map"][env_idx]
                    new_collect = pending["collect_mask"][env_idx]
                    for p in range(num_players):
                        # Discard partial buffer when a slot's agent/collect changes.
                        if (slot_agent_map[env_idx][p] != new_agents[p]
                                or collect_mask[env_idx][p] != new_collect[p]):
                            buffers[env_idx][p].reset()
                        slot_agent_map[env_idx][p] = new_agents[p]
                        slot_network_map[env_idx][p] = new_nets[p]
                        collect_mask[env_idx][p] = new_collect[p]

                if _any_recurrent:
                    for p in range(num_players):
                        if (env_idx, p) in hidden_states:
                            aid = slot_agent_map[env_idx][p]
                            net = networks_by_agent[aid][LATEST_NETWORK_ID]
                            hidden_states[(env_idx, p)] = net.initial_hidden(1)

        obs = next_obs
        total_steps += num_envs

        # Periodic weight sync for ALL agents' latest networks
        now = time.time()
        if now - last_weight_sync >= weight_sync_interval:
            for aid in agent_ids:
                new_version = _sync_weights(
                    networks_by_agent[aid][LATEST_NETWORK_ID],
                    weight_queues[aid],
                )
                if new_version is not None:
                    policy_versions[aid] = new_version
            last_weight_sync = now

    vec_env.close()
    # Detach feeder threads for queues this worker produced to, so undrained
    # data (e.g. chunks a stopped learner never consumed) can't block exit.
    for tq in trajectory_queues.values():
        try:
            tq.cancel_join_thread()
        except Exception:
            pass
    if results_queue is not None:
        try:
            results_queue.cancel_join_thread()
        except Exception:
            pass
    logger.info(f"Worker {worker_id}: finished. Steps={total_steps}, chunks_sent={chunks_sent}")


def _report_episode_result(
    results_queue: mp.Queue,
    worker_id: int,
    env_idx: int,
    step: int,
    ep_rewards: np.ndarray,
    ep_length: int,
    slot_nets: list[str],
    slot_agent_ids: list[str],
    terminal_infos: Optional[dict[int, dict]] = None,
) -> None:
    """Report episode result to the coordinator via results_queue.

    For tracking, player outcomes are keyed by ``agent_id:network_id``:
    - ``"agent_0:latest"`` for the actively-training network
    - ``"agent_0:ckpt_v100"`` for historical checkpoints
    The coordinator aggregates these composite keys to the base ``agent_id``
    for ELO / win-rate / PFSP.

    Outcomes prefer the env's authoritative signal (``rank``/``outcome`` in the
    terminal info) and fall back to cumulative reward otherwise.
    """
    from colosseum.core.outcomes import player_outcomes as _player_outcomes
    from colosseum.core.types import MatchResult

    num_players = len(slot_nets)
    outcomes = _player_outcomes(ep_rewards, terminal_infos, num_players)

    player_outcomes: dict[str, float] = {}
    total_rewards: dict[str, float] = {}
    for p in range(num_players):
        player_key = f"{slot_agent_ids[p]}:{slot_nets[p]}"
        total_rewards[player_key] = float(ep_rewards[p])
        player_outcomes[player_key] = float(outcomes[p])

    result = MatchResult(
        match_id=f"w{worker_id}_e{env_idx}_{step}",
        player_outcomes=player_outcomes,
        total_rewards=total_rewards,
        episode_length=ep_length,
    )
    try:
        results_queue.put_nowait(result)
    except queue.Full:
        pass  # Non-critical, don't block


def _extract_action_masks(
    infos: list[dict],
    num_envs: int,
    num_players: int,
    action_spec=None,
) -> Optional[np.ndarray]:
    """Extract action masks from env info dicts into a flat array.

    Convention: info[env_idx][player_idx]["action_mask"] is a bool ndarray
    or dict of per-component bool ndarrays (for composite action spaces).
    Returns [num_envs * num_players, num_actions] bool array, or None if no masks.
    """
    if not infos:
        return None
    # Check first env's first player for action_mask key
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
) -> Optional[np.ndarray]:
    """Per-slot ``info["active"]`` flags, or None if the env doesn't provide them.

    For turn-based games, an env can mark which player's action actually matters
    this step via ``info[player]["active"]``. When present, the worker only
    records transitions for active slots, so the policy is not trained on the
    ignored no-op moves of inactive players. Envs that omit the key keep the
    default simultaneous-move behavior (every slot recorded).
    """
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


def _sync_weights(network: ActorCriticNetwork, weight_queue: mp.Queue) -> Optional[int]:
    """Pull latest weights from the weight queue (non-blocking)."""
    latest_payload: Optional[WeightPayload] = None
    while True:
        try:
            payload = weight_queue.get_nowait()
            latest_payload = payload
        except queue.Empty:
            break

    if latest_payload is not None:
        network.load_state_dict(latest_payload.state_dict)
        return latest_payload.policy_version
    return None
