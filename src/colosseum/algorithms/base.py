from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from colosseum.core.types import TrajectoryChunk
    from colosseum.networks.actor_critic import ActorCriticNetwork


class BaseAlgorithm(ABC):
    """Base class for RL algorithms.

    On-policy algorithms (APPO, PPO) train directly on incoming trajectory
    chunks. Off-policy algorithms (R2D2, DQN) add chunks to a replay buffer
    and sample from it for training.

    Subclasses must implement: compute_loss, train_step, network, policy_version.
    Off-policy subclasses should also override is_off_policy and create_replay_buffer.
    """

    @abstractmethod
    def compute_loss(self, chunks: list[TrajectoryChunk]) -> dict[str, torch.Tensor]:
        """Compute loss from a batch of trajectory chunks.

        Returns dict with at least 'total_loss' key, plus any
        additional loss components for logging.
        """
        ...

    @abstractmethod
    def train_step(self, chunks: list[TrajectoryChunk]) -> dict[str, float]:
        """Full training step: compute loss, backward, clip grad, optimizer step.

        Returns metrics dict for logging (all values are Python floats).
        """
        ...

    @property
    @abstractmethod
    def network(self) -> ActorCriticNetwork:
        """The neural network being trained."""
        ...

    @property
    @abstractmethod
    def policy_version(self) -> int:
        """Number of training steps completed."""
        ...

    @property
    def optimizer_state_dict(self) -> dict:
        """Return the optimizer state dict for checkpointing.

        Subclasses should override if they use a different optimizer setup.
        """
        return {}

    @property
    def is_off_policy(self) -> bool:
        """If True, learner adds chunks to a replay buffer instead of training directly."""
        return False

    def create_replay_buffer(self, capacity: int) -> Any | None:
        """Create a replay buffer for off-policy training.

        Returns None for on-policy algorithms (default).
        Off-policy subclasses should return a buffer with add() and sample() methods.
        """
        return None
