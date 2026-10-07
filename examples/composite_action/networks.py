"""Networks for ChaseEnv with composite actions (Dict space).

Demonstrates CompositeDist usage: policy returns a multi-head distribution
with CategoricalDist for direction and DiagGaussianDist for speed.
"""

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import (
    CategoricalDist,
    CompositeDist,
    DiagGaussianDist,
)

_OBS_DIM = 5
_LATENT = 32


class ChaseEncoder(BaseEncoder):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_OBS_DIM, _LATENT),
            nn.ReLU(),
        )

    @property
    def latent_dim(self) -> int:
        return _LATENT

    def forward(self, obs):
        return self.net(obs)


class ChasePolicy(BasePolicy):
    """Composite policy: direction (Discrete(4)) + speed (Box(1))."""

    def __init__(self):
        super().__init__()
        self.dir_head = nn.Linear(_LATENT, 4)
        self.speed_mean = nn.Linear(_LATENT, 1)
        self.speed_logstd = nn.Parameter(torch.zeros(1))

    def forward(self, latent):
        return CompositeDist({
            "direction": CategoricalDist(self.dir_head(latent)),
            "speed": DiagGaussianDist(
                self.speed_mean(latent),
                self.speed_logstd.expand(latent.shape[0], -1),
            ),
        })


class ChaseValue(BaseValue):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_LATENT, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, latent):
        return self.net(latent).squeeze(-1)
