from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from colosseum.core.types import TrajectoryChunk
    from colosseum.networks.model import PolicyModel


def deep_cpu_copy(obj: Any) -> Any:
    """Recursively copy ``obj`` so it shares no storage with live training state.

    Tensors -> ``.detach().to("cpu", copy=True)`` (exactly one copy from any
    device); numpy arrays -> ``.copy()``; dicts, lists and tuples are rebuilt;
    anything else is ``copy.deepcopy``'d.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: deep_cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_cpu_copy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(deep_cpu_copy(v) for v in obj)
    return copy.deepcopy(obj)


class BaseAlgorithm(ABC):
    """Base class for RL algorithms.

    On-policy algorithms (APPO, PPO) train directly on incoming trajectory
    chunks (chunk v2: act / boot / pad slots). Off-policy algorithms (R2D2, DQN) add chunks to a replay buffer
    and sample from it for training.

    Subclasses must implement: compute_loss, train_step, model, policy_version.
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
    def model(self) -> PolicyModel:
        """The PolicyModel being trained."""
        ...

    @property
    @abstractmethod
    def policy_version(self) -> int:
        """Number of training steps completed."""
        ...

    def set_progress(self, progress: float) -> None:
        """Share (0..1) of the global env-step budget consumed so far.

        The learner calls this before every train step. The default ignores it;
        algorithms with schedules (APPO's learning rate) override it.
        """
        return None

    def state_dict(self) -> dict[str, Any]:
        """Full training state except model weights, as a deep CPU copy.

        APPO keys: optimizer, progress, scaler, kickstart, policy_version,
        consumed_samples. Checkpoints store it as ``trainer_state.pt`` (T5.3).
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement state_dict()")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore what :meth:`state_dict` returned (model weights are loaded separately)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement load_state_dict()")

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
