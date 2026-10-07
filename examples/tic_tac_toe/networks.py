"""Neural networks for TicTacToe example."""

from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import CategoricalDist


class TicTacToeEncoder(BaseEncoder):
    """Simple MLP encoder for 3x3x3 board observation."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(27, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    @property
    def latent_dim(self) -> int:
        return 64

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TicTacToePolicy(BasePolicy):
    """Discrete policy head for 9 board positions."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.net = nn.Linear(64, 9)

    def forward(self, latent: torch.Tensor) -> CategoricalDist:
        logits = self.net(latent)
        return CategoricalDist(logits=logits)


class TicTacToeValue(BaseValue):
    """Scalar value head."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)
