"""Networks for the tic-tac-toe example on the SP2 model protocol (parts of a ComposedModel)."""

from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.core.specs import ActionSpec
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.dist import Distribution, make_distribution


class TicTacToeEncoder(BaseEncoder):
    """MLP over the 3x3x3 board (own marks, opponent marks, empty)."""

    def __init__(self, hidden: int = 128, latent: int = 64, **kwargs) -> None:
        super().__init__()
        self._latent = latent
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(27, hidden), nn.ReLU(), nn.Linear(hidden, latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float())


class TicTacToePolicy(BasePolicy):
    """Logits over the cells. ``in_dim`` is the core's output size and ``action_spec`` the role's
    action spec (both injected by ``build_model``); the spec must be a single ``Discrete``."""

    def __init__(self, action_spec: ActionSpec, in_dim: int = 64, **kwargs) -> None:
        super().__init__()
        groups = action_spec.groups
        if len(groups) != 1 or groups[0].kind != "discrete" or groups[0].path:
            raise ValueError(f"TicTacToePolicy needs a single Discrete action space, got groups "
                             f"{[(g.path, g.kind) for g in groups]}")
        self._spec = action_spec
        self.net = nn.Linear(in_dim, groups[0].nvec[0])

    def forward(self, features: torch.Tensor, aux: dict[str, torch.Tensor]) -> Distribution:
        return make_distribution(self._spec, self.net(features))


class TicTacToeValue(BaseValue):
    def __init__(self, in_dim: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
