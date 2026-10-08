"""UnitsDist: per-unit deciders, masks, only_if gating by the given action, no NaN (SP2 T2.2)."""
import math

import pytest
import torch
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.tree import tree_to_torch
from colosseum.sp2.envs.spaces import Units
from colosseum.sp2.networks.dist import UnitsDist, make_distribution

LOG2, LOG3 = math.log(2), math.log(3)

# move (3) | target (2); target counts only when move == 2.
MOVE_TARGET = Units(3, Dict([("move", Discrete(3)), ("target", Discrete(2))]), only_if={"target": ("move", {2})})
GROUP = ActionSpec.from_space(MOVE_TARGET).groups[0]


def _dist(mask=None, batch=1):
    params = {"move": torch.zeros(batch, 3, 3), "target": torch.zeros(batch, 3, 2)}
    return UnitsDist(GROUP, params, mask)


def _actions(move, target):
    return {"move": torch.tensor([move]), "target": torch.tensor([target])}


def test_exact_log_prob_entropy_and_validity_with_only_if():
    dist = _dist({"unit": torch.tensor([[True, True, False]])})
    a = _actions([2, 0, 1], [1, 1, 1])
    assert dist.num_deciders == 3 and dist.batch_size == 1
    # unit 0: move=2 -> target counts; unit 1: move=0 -> target gated off; unit 2: absent.
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-LOG3 - LOG2, -LOG3, 0.0]]))
    assert torch.allclose(dist.log_prob(a), torch.tensor([-2 * LOG3 - LOG2]))
    assert torch.allclose(dist.unit_entropy(a), torch.tensor([[LOG3 + LOG2, LOG3, 0.0]]))
    assert dist.unit_valid(a).tolist() == [[True, True, False]]
    # The gate follows the given action: with move=2 on unit 1 its target counts too.
    assert torch.allclose(dist.unit_entropy(_actions([2, 2, 1], [0, 0, 0])), torch.tensor([[LOG3 + LOG2] * 2 + [0.0]]))


def test_action_mask_rows_and_empty_rows():
    # unit 0: move row [T, F, T], target row [F, T]; unit 1: empty move row -> move invalid,
    # target gated by an invalid parent -> invalid -> the whole unit is invalid.
    action = torch.tensor([[[True, False, True, False, True],
                            [False, False, False, True, True],
                            [True, True, True, True, True]]])
    dist = _dist({"unit": torch.tensor([[True, True, False]]), "action": action})
    a = _actions([2, 2, 0], [1, 0, 0])
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-LOG2 + 0.0, 0.0, 0.0]]))
    assert dist.unit_valid(a).tolist() == [[True, False, False]]
    assert torch.allclose(dist.unit_entropy(a), torch.tensor([[LOG2, 0.0, 0.0]]))


def test_dead_units_and_empty_rows_give_no_nan_and_no_gradient():
    move = torch.randn(2, 3, 3, requires_grad=True)
    target = torch.randn(2, 3, 2, requires_grad=True)
    action = torch.zeros(2, 3, 5, dtype=torch.bool)
    action[:, 0] = True
    dist = UnitsDist(GROUP, {"move": move, "target": target},
                     {"unit": torch.tensor([[True, False, True]] * 2), "action": action})
    a = {"move": torch.tensor([[2, 1, 0]] * 2), "target": torch.tensor([[1, 0, 1]] * 2)}
    loss = (dist.log_prob(a).sum() + dist.unit_entropy(a).sum()) * 65536.0
    other = UnitsDist(GROUP, {"move": torch.zeros(2, 3, 3), "target": torch.zeros(2, 3, 2)})
    loss = loss + dist.unit_kl(other, a).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(move.grad).all() and torch.isfinite(target.grad).all()
    assert torch.all(move.grad[:, 1:] == 0) and torch.all(target.grad[:, 1:] == 0)
    assert dist.unit_valid(a).tolist() == [[True, False, False]] * 2


def test_sample_and_mode_zero_absent_units_and_respect_masks():
    action = torch.ones(1, 3, 5, dtype=torch.bool)
    action[0, 0, :3] = torch.tensor([False, True, False])
    dist = _dist({"unit": torch.tensor([[True, True, False]]), "action": action})
    torch.manual_seed(0)
    for _ in range(20):
        s = dist.sample()
        assert s["move"].shape == (1, 3) and s["move"].dtype == torch.int64
        assert s["move"][0, 0].item() == 1 and s["move"][0, 2].item() == 0 and s["target"][0, 2].item() == 0
    assert dist.mode()["move"].tolist() == [[1, 0, 0]]


def test_kl_is_gated_like_the_entropy():
    p = _dist()
    q = UnitsDist(GROUP, {"move": torch.zeros(1, 3, 3), "target": torch.log(torch.tensor([[[0.25, 0.75]] * 3]))})
    a = _actions([2, 0, 2], [0, 0, 0])
    kl_target = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert torch.allclose(p.unit_kl(q, a), torch.tensor([[kl_target, 0.0, kl_target]]), atol=1e-6)


def test_multi_discrete_and_box_components():
    md = ActionSpec.from_space(Units(2, MultiDiscrete([2, 4]), only_if={"1": ("0", {1})})).groups[0]
    dist = UnitsDist(md, {"0": torch.zeros(1, 2, 2), "1": torch.zeros(1, 2, 4)})
    a = torch.tensor([[[1, 3], [0, 3]]])
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-LOG2 - math.log(4), -LOG2]]))
    assert dist.sample().shape == (1, 2, 2)
    box = ActionSpec.from_space(Units(2, Dict([("kind", Discrete(2)), ("thrust", Box(-1.0, 1.0, (2,)))]),
                                      only_if={"thrust": ("kind", {1})})).groups[0]
    dist = UnitsDist(box, {"kind": torch.zeros(1, 2, 2),
                           "thrust": {"mean": torch.zeros(1, 2, 2), "log_std": torch.zeros(2)}})
    a = {"kind": torch.tensor([[1, 0]]), "thrust": torch.zeros(1, 2, 2)}
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-LOG2 - math.log(2 * math.pi), -LOG2]]))
    s = dist.sample()
    assert s["thrust"].shape == (1, 2, 2) and s["thrust"].dtype == torch.float32


def test_box_only_units_are_valid_when_present():
    group = ActionSpec.from_space(Units(2, Box(-1.0, 1.0, (1,)))).groups[0]
    dist = UnitsDist(group, {"0": {"mean": torch.zeros(1, 2, 1), "log_std": torch.zeros(1)}},
                     {"unit": torch.tensor([[True, False]])})
    a = torch.zeros(1, 2, 1)
    assert dist.unit_valid(a).tolist() == [[True, False]]
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-0.5 * math.log(2 * math.pi), 0.0]]))


def test_apply_mask_and_cat():
    dist = _dist(batch=2).apply_mask({"unit": torch.tensor([[True, False, True], [False, False, True]])})
    dist = dist.apply_mask({"unit": torch.tensor([[True, True, False], [True, True, True]])})
    assert dist.unit_mask.tolist() == [[True, False, False], [False, False, True]]
    with_action = dist.apply_mask({"action": torch.ones(2, 3, 5, dtype=torch.bool)})
    joined = UnitsDist.cat([with_action, _dist()])
    assert joined.batch_size == 3 and joined.action_mask.shape == (3, 3, 5) and joined.action_mask.all()
    assert joined.unit_mask[2].tolist() == [True, True, True]


def test_parameter_errors():
    with pytest.raises(ValueError, match="params for components"):
        UnitsDist(GROUP, {"move": torch.zeros(1, 3, 3)})
    with pytest.raises(ValueError, match=r"needs logits \[B, 3, 2\]"):
        UnitsDist(GROUP, {"move": torch.zeros(1, 3, 3), "target": torch.zeros(1, 3, 3)})
    with pytest.raises(ValueError, match=r"action mask must be \[B, 3, 5\]"):
        _dist({"action": torch.ones(1, 3, 4, dtype=torch.bool)})


def test_tree_dist_lays_out_deciders():
    spec = ActionSpec.from_space(Dict([("base", Discrete(3)), ("workers", MOVE_TARGET),
                                       ("scouts", Units(2, Discrete(2)))]))
    assert spec.num_deciders == 1 + 3 + 2
    params = {"base": torch.zeros(1, 3), "workers": {"move": torch.zeros(1, 3, 3), "target": torch.zeros(1, 3, 2)},
              "scouts": {"0": torch.zeros(1, 2, 2)}}
    mask = spec.full_mask((1,))
    mask["workers"]["unit"][0, 2] = False
    mask["scouts"]["unit"][0, 0] = False
    dist = make_distribution(spec, params).apply_mask(tree_to_torch(mask))
    a = {"base": torch.tensor([1]), "workers": _actions([2, 0, 0], [0, 0, 0]), "scouts": torch.tensor([[1, 0]])}
    expected = torch.tensor([[-LOG3, -LOG3 - LOG2, -LOG3, 0.0, 0.0, -LOG2]])
    assert torch.allclose(dist.unit_log_prob(a), expected)
    assert torch.allclose(dist.log_prob(a), expected.sum(-1))
    assert dist.unit_valid(a).tolist() == [[True, True, True, False, False, True]]
    s = dist.sample()
    assert list(s) == ["base", "workers", "scouts"] and s["scouts"].shape == (1, 2)


def test_k1_units_is_one_decider():
    spec = ActionSpec.from_space(Units(1, Dict([("kind", Discrete(2)), ("target", Discrete(4))]),
                                       only_if={"target": ("kind", {1})}))
    dist = make_distribution(spec, {"kind": torch.zeros(1, 1, 2), "target": torch.zeros(1, 1, 4)})
    assert dist.num_deciders == 1
    a = {"kind": torch.tensor([[1]]), "target": torch.tensor([[3]])}
    assert torch.allclose(dist.log_prob(a), torch.tensor([-LOG2 - math.log(4)]))
