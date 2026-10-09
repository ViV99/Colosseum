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


def _natural_names_role():
    """A role whose ``Units`` components use names that collide with ``nn.Module`` attributes or contain ``.``."""
    from colosseum.envs.game import RoleSpec

    units = Units(3, Dict([("type", Discrete(3)), ("to", Discrete(2)), ("a.b", Box(-1.0, 1.0, (2,))),
                           ("move", Discrete(4))]))
    return RoleSpec(observation_space=Box(-1.0, 1.0, (4,)), action_space=units)


def test_units_head_accepts_natural_component_names():
    group = ActionSpec.from_space(_natural_names_role().action_space).groups[0]
    head = UnitsHead(group, in_dim=6, hidden=8)
    params = head(torch.randn(2, 3, 6))
    assert list(params) == ["type", "to", "a.b", "move"]         # the output keeps the user's names
    assert params["type"].shape == (2, 3, 3) and params["to"].shape == (2, 3, 2)
    assert params["a.b"]["mean"].shape == (2, 3, 2) and params["a.b"]["log_std"].shape == (2,)
    assert params["move"].shape == (2, 3, 4)


def test_natural_component_names_build_step_unroll_and_round_trip_the_state_dict():
    from colosseum.core.tree import tree_index, tree_map, tree_stack, tree_to_torch
    from colosseum.networks.model import act
    from game_helpers import make_test_model

    torch.manual_seed(0)
    role = _natural_names_role()
    spec = ActionSpec.from_space(role.action_space)
    model = make_test_model(role, core="lstm")
    S, B = 4, 2
    role.observation_space.seed(0)
    obs = tree_to_torch(tree_stack([tree_stack([role.observation_space.sample() for _ in range(B)])
                                    for _ in range(S)]))
    mask = tree_to_torch(spec.full_mask((S, B)))
    reset_after = torch.zeros(S, B, dtype=torch.bool)
    state = model.initial_state(B)
    actions, unit_lps = [], []
    for s in range(S):
        out = act(model, tree_index(obs, s), state, tree_index(mask, s))
        assert set(out.actions) == {"type", "to", "a.b", "move"}
        actions.append(out.actions)
        unit_lps.append(out.unit_log_probs)
        state = out.state
    flat_actions = tree_map(lambda *xs: torch.cat(xs, dim=0), actions[0], *actions[1:])
    unit_lps = torch.cat(unit_lps)
    out = model.unroll(obs, model.initial_state(B), reset_after, mask)
    assert torch.allclose(out.dist.unit_log_prob(flat_actions), unit_lps, atol=1e-5)

    clone = make_test_model(role, core="lstm")
    clone.load_state_dict(model.state_dict())
    again = clone.unroll(obs, clone.initial_state(B), reset_after, mask)
    assert torch.allclose(again.dist.unit_log_prob(flat_actions), unit_lps, atol=1e-5)
