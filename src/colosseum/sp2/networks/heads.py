"""Policy-head helpers: ``UnitsHead`` (per-unit features -> ``UnitsDist`` parameters) and GridNet layout."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from colosseum.sp2.core.specs import ActionGroup
from colosseum.sp2.networks.dist import make_distribution

__all__ = ["UnitsHead", "gridnet_to_units", "make_distribution"]


class UnitsHead(nn.Module):
    """Unit features ``[B, U, F]`` -> parameters of one units group: discrete component -> logits
    ``[B, U, n]``; box component -> ``{"mean": [B, U, d], "log_std": [d]}`` (a learned parameter).
    ``hidden > 0`` adds a shared ``Linear + ReLU`` layer per unit first."""

    def __init__(self, group: ActionGroup, in_dim: int, hidden: int = 0) -> None:
        super().__init__()
        if group.kind != "units":
            raise ValueError(f"UnitsHead needs a units action group, got {group.kind!r}")
        self.group = group
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU()) if hidden > 0 else nn.Identity()
        width = hidden if hidden > 0 else in_dim
        self.heads = nn.ModuleDict()
        self.log_std = nn.ParameterDict()
        for c in group.units.components:
            self.heads[c.name] = nn.Linear(width, c.size)
            if c.kind == "box":
                self.log_std[c.name] = nn.Parameter(torch.zeros(c.size))

    def forward(self, unit_features: Tensor) -> dict[str, Tensor | dict[str, Tensor]]:
        h = self.trunk(unit_features)
        out: dict[str, Tensor | dict[str, Tensor]] = {}
        for c in self.group.units.components:
            y = self.heads[c.name](h)
            out[c.name] = y if c.kind == "discrete" else {"mean": y, "log_std": self.log_std[c.name]}
        return out


def gridnet_to_units(x: Tensor) -> Tensor:
    """``[B, C, H, W]`` -> ``[B, H*W, C]``: every grid cell is a unit (row-major cell order)."""
    return x.flatten(2).transpose(1, 2)
