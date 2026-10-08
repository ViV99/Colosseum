"""Networks for the Tic-Tac-Toe example (parts of a ComposedModel)."""

from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import CategoricalDist


class TicTacToeEncoder(BaseEncoder):
    """MLP over the 3x3x3 board."""

    def __init__(self, hidden: int = 128, latent: int = 64, **kwargs) -> None:
        super().__init__()
        self._latent = latent
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(27, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TicTacToePolicy(BasePolicy):
    """Categorical over 9 cells. ``in_dim`` is the core's output size (set by build_model)."""

    def __init__(self, in_dim: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, 9)

    def forward(self, features: torch.Tensor) -> CategoricalDist:
        return CategoricalDist(logits=self.net(features))


class TicTacToeValue(BaseValue):
    def __init__(self, in_dim: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
