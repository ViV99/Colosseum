"""Core data types that flow through the Colosseum distributed RL system.

These dataclasses represent the fundamental units of data exchange between
workers, learners, the weight store, and the coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class TrajectoryChunk:
    """Fixed-length rollout chunk from a single agent slot in a single env.

    Workers collect observations, actions, rewards, etc. over T consecutive
    timesteps and package them into chunks that are sent to learners for
    training.  Episode boundaries are handled within chunks: ``dones[t]``
    marks the end of an episode, and the next timestep starts a new one.

    Attributes:
        agent_id: Identifier of the agent that generated this chunk.
        observations: Stacked observations of shape ``[T, *obs_shape]``.
        actions: Actions taken at each step, shape ``[T, *act_shape]``.
        action_log_probs: Log-probabilities of the chosen actions under the
            behavior policy, shape ``[T]``.
        rewards: Scalar rewards received after each action, shape ``[T]``.
        dones: Boolean episode-termination flags, shape ``[T]``.
        values: Value estimates from the behavior policy, shape ``[T]``.
        bootstrap_value: Value estimate at step ``T`` used for GAE / V-trace
            bootstrapping (scalar tensor).
        behavior_policy_version: Version counter of the policy that was used
            to collect this chunk.  The learner uses this, together with its
            current policy version, to compute importance-sampling ratios for
            V-trace correction.
        lstm_hidden: Optional LSTM hidden state ``(h, c)`` at the *start* of
            the chunk, each of shape ``[num_layers, hidden_size]``.  Required
            when training recurrent policies so that the learner can reproduce
            the correct hidden-state trajectory.
    """

    agent_id: str
    observations: torch.Tensor  # [T, *obs_shape]
    actions: torch.Tensor  # [T, *act_shape]
    action_log_probs: torch.Tensor  # [T]
    rewards: torch.Tensor  # [T]
    dones: torch.Tensor  # [T] bool
    values: torch.Tensor  # [T]
    bootstrap_value: torch.Tensor  # scalar
    behavior_policy_version: int
    lstm_hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    action_masks: Optional[torch.Tensor] = None  # [T, num_actions]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def chunk_length(self) -> int:
        """Number of timesteps ``T`` in this chunk."""
        return self.observations.shape[0]

    def _apply_to_tensors(self, fn) -> TrajectoryChunk:
        """Return a copy with *fn* applied to every tensor field."""
        return TrajectoryChunk(
            agent_id=self.agent_id,
            observations=fn(self.observations),
            actions=fn(self.actions),
            action_log_probs=fn(self.action_log_probs),
            rewards=fn(self.rewards),
            dones=fn(self.dones),
            values=fn(self.values),
            bootstrap_value=fn(self.bootstrap_value),
            behavior_policy_version=self.behavior_policy_version,
            lstm_hidden=(
                (fn(self.lstm_hidden[0]), fn(self.lstm_hidden[1]))
                if self.lstm_hidden is not None
                else None
            ),
            action_masks=(
                fn(self.action_masks)
                if self.action_masks is not None
                else None
            ),
        )

    def to_device(self, device: str | torch.device) -> TrajectoryChunk:
        """Return a shallow copy with all tensors moved to *device*."""
        return self._apply_to_tensors(lambda t: t.to(device))

    def pin_memory(self) -> TrajectoryChunk:
        """Pin all tensors to page-locked memory for faster host-to-device copies."""
        return self._apply_to_tensors(lambda t: t.pin_memory())


@dataclass
class PlayerSlot:
    """Assignment for one player slot in a match.

    The coordinator fills each player slot in a match with either the latest
    weights of an agent or a specific frozen checkpoint.

    Attributes:
        agent_id: Which agent occupies this slot.
        checkpoint_id: Specific checkpoint to load.  ``None`` means "use the
            latest weights from the weight store".
        collect_trajectories: Whether to collect trajectories from this slot
            and send them to the agent's learner.  Frozen sparring partners
            typically have this set to ``False``.
    """

    agent_id: str
    checkpoint_id: Optional[str] = None
    collect_trajectories: bool = True


@dataclass
class MatchConfig:
    """Full specification for a single match to be played by a worker.

    Attributes:
        match_id: Globally unique identifier for this match.
        env_config: Extra keyword arguments forwarded to the environment
            constructor (e.g. board size, time limit).
        player_slots: Ordered list of player-slot assignments.  The length
            must equal the number of players expected by the environment.
    """

    match_id: str
    env_config: dict[str, Any] = field(default_factory=dict)
    player_slots: list[PlayerSlot] = field(default_factory=list)

    @property
    def num_players(self) -> int:
        return len(self.player_slots)


@dataclass
class MatchResult:
    """Outcome of a completed match, reported back to the coordinator.

    Attributes:
        match_id: Matches the ``match_id`` from the originating
            :class:`MatchConfig`.
        player_outcomes: Mapping from ``agent_id`` to outcome value:
            ``1.0`` for win, ``0.0`` for loss, ``0.5`` for draw.
        total_rewards: Mapping from ``agent_id`` to the cumulative undiscounted
            reward earned during the match.
        episode_length: Total number of environment steps in the match.
    """

    match_id: str
    player_outcomes: dict[str, float] = field(default_factory=dict)
    total_rewards: dict[str, float] = field(default_factory=dict)
    episode_length: int = 0


@dataclass
class WorkerCommand:
    """Runtime match-assignment update pushed from the coordinator to a worker.

    Lets matchmaking evolve *during* training: the coordinator periodically
    re-generates slot assignments (e.g. PFSP opponent picks, fresh self-play
    checkpoints) and ships them to each worker, which applies them at the next
    episode boundary. ``new_checkpoints`` carries any checkpoint state_dicts the
    worker does not yet have, so historical opponents can enter its network pool
    without restarting the process.

    Attributes:
        slot_agent_map: ``[num_envs][num_players]`` -> agent_id.
        slot_network_map: ``[num_envs][num_players]`` -> network_id
            (``"latest"`` or a checkpoint id).
        collect_mask: ``[num_envs][num_players]`` -> whether the slot collects
            trajectories.
        new_checkpoints: ``{agent_id: {checkpoint_id: state_dict}}`` — checkpoints
            to load into the worker's network pool (deltas only).
    """

    slot_agent_map: list[list[str]] = field(default_factory=list)
    slot_network_map: list[list[str]] = field(default_factory=list)
    collect_mask: list[list[bool]] = field(default_factory=list)
    new_checkpoints: dict[str, dict[str, dict]] = field(default_factory=dict)


@dataclass
class WeightPayload:
    """Serialisable container for model weights flowing through the weight store.

    Attributes:
        agent_id: The agent these weights belong to.
        policy_version: Monotonically increasing version counter.  Workers
            compare this against the version used to collect a trajectory chunk
            so the learner can compute V-trace importance weights.
        state_dict: The ``state_dict()`` of the model, mapping parameter names
            to tensors.
    """

    agent_id: str
    policy_version: int
    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
