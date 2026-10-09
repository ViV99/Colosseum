"""Model parts for predator_prey: an MLP over a flat float observation and a categorical policy.
The same classes serve every role: sizes come from the role's spaces (injected by ``build_model``)."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.heads import make_distribution


class PPEncoder(BaseEncoder):
    def __init__(self, observation_space, latent: int = 64, **kwargs) -> None:
        super().__init__()
        self._latent = latent
        self.net = nn.Sequential(nn.Linear(observation_space.shape[0], latent), nn.Tanh(),
                                 nn.Linear(latent, latent), nn.Tanh())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class PPPolicy(BasePolicy):
    def __init__(self, in_dim: int, action_spec, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.logits = nn.Linear(in_dim, action_spec.groups[0].nvec[0])

    def forward(self, features: torch.Tensor, aux: dict):
        return make_distribution(self.action_spec, self.logits(features))


class PPValue(BaseValue):
    def __init__(self, in_dim: int, value_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.Tanh(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
