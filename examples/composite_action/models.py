"""Model parts for chase: a tree action, ``direction`` (categorical) and ``speed`` (Gaussian)."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.heads import make_distribution


class ChaseEncoder(BaseEncoder):
    def __init__(self, observation_space, latent: int = 32, **kwargs) -> None:
        super().__init__()
        self._latent = latent
        self.net = nn.Sequential(nn.Linear(observation_space.shape[0], latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ChasePolicy(BasePolicy):
    """``make_distribution`` params mirror the action tree: logits for ``direction``,
    ``{"mean", "log_std"}`` for ``speed``."""

    def __init__(self, in_dim: int, action_spec, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.direction = nn.Linear(in_dim, 4)
        self.speed_mean = nn.Linear(in_dim, 1)
        self.speed_log_std = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor, aux: dict):
        params = {"direction": self.direction(features),
                  "speed": {"mean": self.speed_mean(features), "log_std": self.speed_log_std}}
        return make_distribution(self.action_spec, params)


class ChaseValue(BaseValue):
    def __init__(self, in_dim: int, value_hidden: int = 16, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
