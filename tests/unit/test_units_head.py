"""UnitsHead and the GridNet layout helper (SP2 T2.3)."""
import torch
from gymnasium.spaces import Box, Dict, Discrete

from colosseum.core.specs import ActionSpec
from colosseum.envs.spaces import Units
from colosseum.networks.dist import UnitsDist
from colosseum.networks.heads import UnitsHead, gridnet_to_units


def test_units_head_parameters_per_component():
    group = ActionSpec.from_space(Units(5, Dict([("move", Discrete(4)), ("thrust", Box(-1.0, 1.0, (2,)))]))).groups[0]
    head = UnitsHead(group, in_dim=6, hidden=8)
    params = head(torch.randn(3, 5, 6))
    assert list(params) == ["move", "thrust"]
    assert params["move"].shape == (3, 5, 4)
    assert params["thrust"]["mean"].shape == (3, 5, 2) and params["thrust"]["log_std"].shape == (2,)
    dist = UnitsDist(group, params)
    assert dist.sample()["thrust"].shape == (3, 5, 2)


def test_gridnet_to_units_is_row_major_cells():
    x = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)   # [B, C, H, W]
    units = gridnet_to_units(x)
    assert units.shape == (2, 8, 3)
    assert torch.equal(units[1, 1 * 4 + 2], x[1, :, 1, 2])
