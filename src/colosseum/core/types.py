"""Core data types that flow through the Colosseum distributed RL system.

These dataclasses represent the fundamental units of data exchange between
workers, learners, the weight store, and the coordinator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from colosseum.core.ipc import numpy_to_tensor, tensor_to_numpy
from colosseum.networks.state import State, state_from_numpy, state_to_numpy, tree_map


def state_dict_to_numpy(state_dict: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Detached CPU numpy copies of a torch ``state_dict`` (bfloat16 -> float32)."""
    return {key: tensor_to_numpy(value) for key, value in state_dict.items()}


def state_dict_from_numpy(state_dict: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """CPU torch tensors from a numpy ``state_dict``; inverse of :func:`state_dict_to_numpy`."""
    return {key: numpy_to_tensor(value) for key, value in state_dict.items()}


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
        dones: Episode-termination flags, shape ``[T]`` (transition t is the
            last of its episode).
        values: Value estimates from the behavior policy, shape ``[T]``.
        bootstrap_value: Value estimate after the last transition (scalar),
            0 when the last transition is terminal.
        behavior_policy_version: Version counter of the policy that was used
            to collect this chunk.
        initial_state: Model state (a ``State`` pytree, see
            ``colosseum.networks.state``) before the chunk's first transition;
            every tensor leaf has batch dim 1 (``[1, ...]``). ``None`` for
            stateless models. The learner concatenates these along dim 0 and
            unrolls the model from them.
        action_masks: Optional ``[T, mask_size]`` bool masks the behavior
            policy acted under.
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
    initial_state: State = None  # leaves [1, ...]; None for stateless models
    action_masks: torch.Tensor | None = None  # [T, mask_size]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def chunk_length(self) -> int:
        """Number of timesteps ``T`` in this chunk."""
        return self.observations.shape[0]

    def _apply_to_tensors(self, fn) -> TrajectoryChunk:
        """Return a copy with *fn* applied to every tensor (including state leaves)."""
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
            initial_state=tree_map(fn, self.initial_state),
            action_masks=fn(self.action_masks) if self.action_masks is not None else None,
        )

    def to(self, device: str | torch.device) -> TrajectoryChunk:
        """Return a copy with all tensors moved to *device*."""
        return self._apply_to_tensors(lambda t: t.to(device))

    def pin_memory(self) -> TrajectoryChunk:
        """Pin all tensors to page-locked memory for faster host-to-device copies."""
        return self._apply_to_tensors(lambda t: t.pin_memory())

    def to_payload(self) -> dict[str, Any]:
        """Numpy + primitives form for crossing a process boundary (spec block 2)."""
        return {
            "agent_id": str(self.agent_id),
            "observations": tensor_to_numpy(self.observations),
            "actions": tensor_to_numpy(self.actions),
            "action_log_probs": tensor_to_numpy(self.action_log_probs),
            "rewards": tensor_to_numpy(self.rewards),
            "dones": tensor_to_numpy(self.dones),
            "values": tensor_to_numpy(self.values),
            "bootstrap_value": float(self.bootstrap_value),
            "behavior_policy_version": int(self.behavior_policy_version),
            "initial_state": state_to_numpy(self.initial_state),
            "action_masks": None if self.action_masks is None else tensor_to_numpy(self.action_masks),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TrajectoryChunk:
        """Rebuild a chunk (CPU tensors) from :meth:`to_payload` output."""
        masks = payload.get("action_masks")
        return cls(
            agent_id=str(payload["agent_id"]),
            observations=numpy_to_tensor(payload["observations"]),
            actions=numpy_to_tensor(payload["actions"]),
            action_log_probs=numpy_to_tensor(payload["action_log_probs"]),
            rewards=numpy_to_tensor(payload["rewards"]),
            dones=numpy_to_tensor(payload["dones"]),
            values=numpy_to_tensor(payload["values"]),
            bootstrap_value=torch.tensor(float(payload["bootstrap_value"]), dtype=torch.float32),
            behavior_policy_version=int(payload["behavior_policy_version"]),
            initial_state=state_from_numpy(payload.get("initial_state")),
            action_masks=None if masks is None else numpy_to_tensor(masks),
        )


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
    checkpoint_id: str | None = None
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
class SeatResult:
    """Outcome of one seat of a finished match.

    Attributes:
        seat: Seat (player slot) index in the env.
        agent_id: Base agent that played the seat.
        network_id: ``"latest"`` or a checkpoint id (``"ckpt_v<N>"``).
        outcome: In [0, 1] (1 = best), from :mod:`colosseum.core.outcomes`: the
            env's terminal ``outcome``/``rank`` if every seat has one, else
            derived from episode rewards.
        reward: Undiscounted episode return of the seat.
        rank: The env's terminal ``rank`` for the seat (1 = best), if provided.
    """

    seat: int
    agent_id: str
    network_id: str
    outcome: float
    reward: float
    rank: int | None = None


@dataclass
class MatchResult:
    """A finished match, one :class:`SeatResult` per seat (no key collisions)."""

    match_id: str
    seats: list[SeatResult] = field(default_factory=list)
    episode_length: int = 0


@dataclass
class WorkerCommand:
    """Runtime match-assignment update pushed from the coordinator to a worker.

    Lets matchmaking evolve *during* training: the coordinator periodically
    re-generates slot assignments (e.g. PFSP opponent picks, fresh self-play
    checkpoints) and ships them to each worker, which applies them per env at
    its next episode boundary. ``new_checkpoints`` carries the checkpoint
    weights (numpy, see :func:`state_dict_to_numpy`) the worker does not have
    yet (deltas only), so historical opponents can enter its model pool
    without restarting the process.

    Attributes:
        slot_agent_map: ``[num_envs][num_players]`` -> agent_id.
        slot_network_map: ``[num_envs][num_players]`` -> ``"latest"`` or a checkpoint id.
        collect_mask: ``[num_envs][num_players]`` -> whether the slot collects trajectories.
        new_checkpoints: ``{agent_id: {checkpoint_id: numpy state_dict}}``.
    """

    slot_agent_map: list[list[str]] = field(default_factory=list)
    slot_network_map: list[list[str]] = field(default_factory=list)
    collect_mask: list[list[bool]] = field(default_factory=list)
    new_checkpoints: dict[str, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict)


@dataclass
class WeightPayload:
    """Model weights flowing from a learner to workers (numpy, never torch).

    Attributes:
        agent_id: The agent these weights belong to.
        policy_version: Monotonically increasing version counter (number of
            train steps). Workers stamp it on the chunks they collect so the
            learner can compute V-trace importance weights.
        state_dict: ``model.state_dict()`` as numpy arrays (see
            :func:`state_dict_to_numpy`).
    """

    agent_id: str
    policy_version: int
    state_dict: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_model(cls, agent_id: str, policy_version: int, model: nn.Module) -> WeightPayload:
        """Snapshot ``model``'s current weights as a numpy payload."""
        return cls(
            agent_id=agent_id,
            policy_version=int(policy_version),
            state_dict=state_dict_to_numpy(model.state_dict()),
        )

    def to_torch_state_dict(self) -> dict[str, torch.Tensor]:
        """CPU torch ``state_dict`` for ``model.load_state_dict``."""
        return state_dict_from_numpy(self.state_dict)
