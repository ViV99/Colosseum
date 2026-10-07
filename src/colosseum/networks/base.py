from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Transforms raw observation into a latent vector."""

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [B, *obs_shape] -> latent: [B, latent_dim]"""
        ...

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimension of the latent vector output."""
        ...


class BasePolicy(nn.Module, ABC):
    """Maps latent vector to action distribution."""

    @abstractmethod
    def forward(self, latent: torch.Tensor) -> "Distribution":
        """latent: [B, latent_dim] -> distribution over actions.

        Must return an object with sample(), log_prob(actions), entropy() methods.
        """
        ...


class BaseValue(nn.Module, ABC):
    """Maps latent vector to scalar value estimate."""

    @abstractmethod
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [B, latent_dim] -> values: [B]"""
        ...
