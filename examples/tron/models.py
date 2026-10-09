"""Model parts for tron: a CNN over the rotated uint8 window around the head."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.heads import make_distribution


class TronEncoder(BaseEncoder):
    def __init__(self, observation_space, channels: int = 16, latent: int = 128, **kwargs) -> None:
        super().__init__()
        c, h, w = observation_space.shape
        self._latent = latent
        self.net = nn.Sequential(nn.Conv2d(c, channels, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(channels * h * w, latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float())


class TronPolicy(BasePolicy):
    def __init__(self, in_dim: int, action_spec, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.logits = nn.Linear(in_dim, action_spec.groups[0].nvec[0])

    def forward(self, features: torch.Tensor, aux: dict):
        return make_distribution(self.action_spec, self.logits(features))


class TronValue(BaseValue):
    def __init__(self, in_dim: int, value_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
