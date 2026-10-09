"""Model parts for space_miners: ship embeddings go to a ``UnitsHead`` (``accel`` Gaussian and
``push`` categorical per ship); asteroids are pooled with their mask."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue, EncoderOutput
from colosseum.networks.heads import UnitsHead, make_distribution


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x[B, N, F]`` over the entities where ``mask[B, N]`` is set (0 when there are none)."""
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class MinersEncoder(BaseEncoder):
    def __init__(self, observation_space, ship_dim: int = 64, latent: int = 128, **kwargs) -> None:
        super().__init__()
        ship_f = observation_space["ships"].shape[1]
        rock_f = observation_space["asteroids"].shape[1]
        glob = observation_space["global"].shape[0]
        self._latent = latent
        self.ship_net = nn.Sequential(nn.Linear(ship_f, ship_dim), nn.ReLU(), nn.Linear(ship_dim, ship_dim), nn.ReLU())
        self.enemy_net = nn.Sequential(nn.Linear(ship_f, 32), nn.ReLU())
        self.rock_net = nn.Sequential(nn.Linear(rock_f, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(ship_dim + 32 + 64 + glob, latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: dict) -> EncoderOutput:
        ships = self.ship_net(obs["ships"])                                    # [B, 3, E]
        enemies = self.enemy_net(obs["enemy_ships"]).mean(dim=1)
        rocks = masked_mean(self.rock_net(obs["asteroids"]), obs["asteroid_mask"])
        latent = self.mlp(torch.cat([ships.mean(dim=1), enemies, rocks, obs["global"]], dim=-1))
        return EncoderOutput(latent=latent, aux={"ships": ships})


class MinersPolicy(BasePolicy):
    """One ``UnitsHead`` over ``[ship embedding, core features]`` per ship; the action is a bare
    ``Units`` space, so its params go to ``make_distribution`` unwrapped."""

    def __init__(self, in_dim: int, action_spec, ship_dim: int = 64, unit_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.ships_head = UnitsHead(action_spec.groups[0], ship_dim + in_dim, unit_hidden)

    def forward(self, features: torch.Tensor, aux: dict):
        ships = aux["ships"]
        context = features.unsqueeze(1).expand(-1, ships.shape[1], -1)
        return make_distribution(self.action_spec, self.ships_head(torch.cat([ships, context], dim=-1)))


class MinersValue(BaseValue):
    def __init__(self, in_dim: int, value_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
