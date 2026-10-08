"""Neural networks for Space Miners example.

Encoder: MLP 218 → 256 → 128
Policy: CompositeDist with continuous acceleration + discrete push per ship
Value: MLP 128 → 64 → 1
"""

from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import (
    CategoricalDist,
    CompositeDist,
    DiagGaussianDist,
)

_OBS_DIM = 218  # from env._obs_size with default max_asteroids=20
_LATENT = 128


class SpaceMinersEncoder(BaseEncoder):
    """MLP encoder: obs → latent vector."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_OBS_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, _LATENT),
            nn.ReLU(),
        )

    @property
    def latent_dim(self) -> int:
        return _LATENT

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class SpaceMinersPolicy(BasePolicy):
    """Composite policy: 6D continuous acceleration + 3 binary push heads."""

    def __init__(self, in_dim: int = _LATENT, **kwargs) -> None:
        super().__init__()
        # Acceleration: 3 ships × 2D = 6
        self.accel_mean = nn.Linear(in_dim, 6)
        self.accel_logstd = nn.Parameter(torch.zeros(6))

        # Push: binary per ship
        self.push_head_0 = nn.Linear(in_dim, 2)
        self.push_head_1 = nn.Linear(in_dim, 2)
        self.push_head_2 = nn.Linear(in_dim, 2)

    def forward(self, latent: torch.Tensor) -> CompositeDist:
        batch_size = latent.shape[0]
        return CompositeDist(
            {
                "accel": DiagGaussianDist(
                    self.accel_mean(latent),
                    self.accel_logstd.expand(batch_size, -1),
                ),
                "push_0": CategoricalDist(self.push_head_0(latent)),
                "push_1": CategoricalDist(self.push_head_1(latent)),
                "push_2": CategoricalDist(self.push_head_2(latent)),
            }
        )


class SpaceMinersValue(BaseValue):
    """MLP value head: latent → scalar."""

    def __init__(self, in_dim: int = _LATENT, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)
