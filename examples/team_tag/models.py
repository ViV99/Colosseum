"""Model parts for team_tag: the policy sees the local window, the value additionally sees the
whole map through a critic encoder over ``global_state`` (centralized critic)."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseCriticEncoder, BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.heads import make_distribution


class TagEncoder(BaseEncoder):
    def __init__(self, observation_space, channels: int = 16, latent: int = 128, **kwargs) -> None:
        super().__init__()
        c, h, w = observation_space["window"].shape
        v = observation_space["vec"].shape[0]
        self._latent = latent
        self.conv = nn.Sequential(nn.Conv2d(c, channels, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(channels * h * w + v, latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: dict) -> torch.Tensor:
        return self.mlp(torch.cat([self.conv(obs["window"].float()), obs["vec"]], dim=-1))


class TagCriticEncoder(BaseCriticEncoder):
    def __init__(self, global_state_space, channels: int = 16, critic_dim: int = 64, **kwargs) -> None:
        super().__init__()
        c, h, w = global_state_space.shape
        self._dim = critic_dim
        self.net = nn.Sequential(nn.Conv2d(c, channels, 3, padding=1), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(channels * h * w, critic_dim), nn.ReLU())

    @property
    def output_dim(self) -> int:
        return self._dim

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state.float())


class TagPolicy(BasePolicy):
    def __init__(self, in_dim: int, action_spec, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.logits = nn.Linear(in_dim, action_spec.groups[0].nvec[0])

    def forward(self, features: torch.Tensor, aux: dict):
        return make_distribution(self.action_spec, self.logits(features))


class TagValue(BaseValue):
    """``in_dim`` = core features + critic features (``build_model`` adds them up)."""

    def __init__(self, in_dim: int, value_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
