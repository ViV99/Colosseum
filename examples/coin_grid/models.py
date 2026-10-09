"""Model parts for coin_grid: a small CNN over the uint8 grid plus the vector leaf."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.heads import make_distribution


class CoinGridEncoder(BaseEncoder):
    def __init__(self, observation_space, channels: int = 16, latent: int = 128, **kwargs) -> None:
        super().__init__()
        c, h, w = observation_space["grid"].shape
        v = observation_space["vec"].shape[0]
        self._latent = latent
        self.conv = nn.Sequential(nn.Conv2d(c, channels, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(channels * h * w + v, latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: dict) -> torch.Tensor:
        grid = obs["grid"].float()           # the env sends uint8; the model casts
        return self.mlp(torch.cat([self.conv(grid), obs["vec"]], dim=-1))


class CoinGridPolicy(BasePolicy):
    def __init__(self, in_dim: int, action_spec, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.logits = nn.Linear(in_dim, action_spec.groups[0].nvec[0])

    def forward(self, features: torch.Tensor, aux: dict):
        return make_distribution(self.action_spec, self.logits(features))


class CoinGridValue(BaseValue):
    def __init__(self, in_dim: int, value_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
