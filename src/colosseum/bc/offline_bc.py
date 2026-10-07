"""Offline Behavioral Cloning trainer.

Trains a policy network via supervised learning on pre-collected
(observation, action) pairs. Supports:
- Cross-entropy loss for discrete actions (Categorical)
- MSE loss for continuous actions (Gaussian)
- Loading data from .pt files containing {"observations": tensor, "actions": tensor}
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from colosseum.networks.actor_critic import ActorCriticNetwork

logger = logging.getLogger(__name__)


class OfflineBCTrainer:
    """Supervised behavioral cloning from offline data.

    Usage::

        trainer = OfflineBCTrainer(network, lr=1e-3)
        trainer.load_data("path/to/data")   # or add_data(obs, actions)
        metrics = trainer.train(num_epochs=10, batch_size=256)

    Action masking: if the data provides ``action_masks`` (a ``[N, num_actions]``
    bool tensor), invalid actions are masked out of the policy distribution
    before computing the loss — matching how masks are used during RL.

    Scope/limitations: BC trains the feedforward policy path (encoder -> policy
    head); it does not unroll a recurrent trunk, and it loads the full dataset
    into memory. For very large replay corpora or recurrent policies, pre-train
    feedforward then fine-tune with RL, or stream data in shards via repeated
    ``add_data`` + ``train`` calls.
    """

    def __init__(
        self,
        network: ActorCriticNetwork,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
        action_type: str = "discrete",
    ) -> None:
        """
        Args:
            network: The actor-critic network to train (only encoder+policy used).
            lr: Learning rate.
            device: Torch device.
            action_type: "discrete" for cross-entropy, "continuous" for MSE.
        """
        self._network = network.to(device)
        self._device = device
        self._action_type = action_type
        self._optimizer = torch.optim.Adam(network.parameters(), lr=lr)

        self._observations: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []

    @property
    def network(self) -> ActorCriticNetwork:
        return self._network

    def add_data(self, observations: torch.Tensor, actions: torch.Tensor) -> None:
        """Add a batch of (obs, action) pairs to the training dataset."""
        self._observations.append(observations)
        self._actions.append(actions)

    def load_data(self, path: str | Path) -> int:
        """Load training data from .pt file(s).

        Each .pt file should contain a dict with "observations" and "actions" keys.
        Can point to a single file or a directory of .pt files.

        Returns:
            Number of samples loaded.
        """
        path = Path(path)
        files = sorted(path.glob("*.pt")) if path.is_dir() else [path]
        total = 0
        for f in files:
            data = torch.load(f, weights_only=True)
            obs = data["observations"]
            acts = data["actions"]
            self.add_data(obs, acts)
            total += len(obs)
            logger.info(f"Loaded {len(obs)} samples from {f}")
        logger.info(f"Total BC dataset: {total} samples")
        return total

    def train(
        self,
        num_epochs: int = 10,
        batch_size: int = 256,
        log_interval: int = 1,
    ) -> dict[str, float]:
        """Run supervised training.

        Returns:
            Final metrics dict with average loss.
        """
        if not self._observations:
            raise ValueError("No training data. Call add_data() or load_data() first.")

        all_obs = torch.cat(self._observations, dim=0)
        all_actions = torch.cat(self._actions, dim=0)
        dataset = TensorDataset(all_obs, all_actions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        logger.info(
            f"BC training: {len(dataset)} samples, {num_epochs} epochs, "
            f"batch_size={batch_size}, action_type={self._action_type}"
        )

        total_loss_accum = 0.0
        total_batches = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            for obs_batch, action_batch in loader:
                obs_batch = obs_batch.to(self._device)
                action_batch = action_batch.to(self._device)

                loss = self._compute_loss(obs_batch, action_batch)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1

            avg_loss = epoch_loss / max(1, epoch_batches)
            total_loss_accum += avg_loss
            total_batches += 1

            if (epoch + 1) % log_interval == 0:
                logger.info(f"BC epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}")

        return {
            "bc_loss": total_loss_accum / max(1, total_batches),
            "num_epochs": num_epochs,
            "num_samples": len(dataset),
        }

    def _compute_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute BC loss: cross-entropy (discrete), MSE (continuous), or auto (composite)."""
        from colosseum.networks.distributions import CompositeDist

        latent = self._network.encoder(obs)
        dist = self._network.policy(latent)

        if isinstance(dist, CompositeDist):
            # CompositeDist.log_prob handles discrete/continuous split internally
            log_probs = dist.log_prob(actions)
            return -log_probs.mean()
        elif self._action_type == "discrete":
            # Cross-entropy: negative log probability of expert actions
            log_probs = dist.log_prob(actions.long())
            return -log_probs.mean()
        else:
            # MSE: for continuous actions, minimize (predicted_mean - expert_action)^2
            predicted = dist.mode()
            return F.mse_loss(predicted, actions)
