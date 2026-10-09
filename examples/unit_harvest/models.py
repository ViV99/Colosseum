"""Model parts for unit_harvest: an entity encoder with per-unit embeddings in ``aux`` and a
policy with a ``Discrete`` base head and a ``UnitsHead`` for the workers."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue, EncoderOutput
from colosseum.networks.heads import UnitsHead, make_distribution


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x[B, N, F]`` over the entities where ``mask[B, N]`` is set (0 when there are none)."""
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class HarvestEncoder(BaseEncoder):
    def __init__(self, observation_space, unit_dim: int = 64, latent: int = 128, **kwargs) -> None:
        super().__init__()
        f = observation_space["units"].shape[1]
        g = observation_space["base"].shape[0]
        self._latent = latent
        self.unit_net = nn.Sequential(nn.Linear(f, unit_dim), nn.ReLU(), nn.Linear(unit_dim, unit_dim), nn.ReLU())
        self.enemy_net = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.resource_net = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(unit_dim + 32 + 32 + g, latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: dict) -> EncoderOutput:
        units = self.unit_net(obs["units"])                                        # [B, U, E]
        own = masked_mean(units, obs["unit_mask"])
        enemies = masked_mean(self.enemy_net(obs["enemies"]), obs["enemy_mask"])
        resources = masked_mean(self.resource_net(obs["resources"]), obs["resource_mask"])
        latent = self.mlp(torch.cat([own, enemies, resources, obs["base"]], dim=-1))
        return EncoderOutput(latent=latent, aux={"units": units})


class HarvestPolicy(BasePolicy):
    def __init__(self, in_dim: int, action_spec, unit_dim: int = 64, unit_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        groups = {g.path: g for g in action_spec.groups}
        self.base_head = nn.Linear(in_dim, groups[("base",)].nvec[0])
        self.workers_head = UnitsHead(groups[("workers",)], unit_dim + in_dim, unit_hidden)

    def forward(self, features: torch.Tensor, aux: dict):
        units = aux["units"]
        context = features.unsqueeze(1).expand(-1, units.shape[1], -1)
        params = {"base": self.base_head(features),
                  "workers": self.workers_head(torch.cat([units, context], dim=-1))}
        return make_distribution(self.action_spec, params)


class HarvestValue(BaseValue):
    def __init__(self, in_dim: int, value_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
