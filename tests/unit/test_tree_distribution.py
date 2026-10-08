"""TreeDist and make_distribution over Dict action spaces (SP2 T2.1)."""
import math

import pytest
import torch
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.networks.dist import CategoricalDist, TreeDist, make_distribution

SPEC = ActionSpec.from_space(Dict([("move", Discrete(3)), ("aim", Box(-1.0, 1.0, (2,))),
                                   ("build", MultiDiscrete([2, 2]))]))


def _params(batch=2):
    return {"move": torch.zeros(batch, 3), "aim": {"mean": torch.zeros(batch, 2), "log_std": torch.zeros(2)},
            "build": torch.zeros(batch, 4)}


def test_non_units_groups_form_one_decider():
    dist = make_distribution(SPEC, _params())
    assert isinstance(dist, TreeDist) and dist.num_deciders == 1 and dist.batch_size == 2
    actions = {"move": torch.tensor([0, 2]), "aim": torch.zeros(2, 2), "build": torch.tensor([[1, 0], [0, 1]])}
    expected = -math.log(3) - math.log(2 * math.pi) - 2 * math.log(2)
    assert torch.allclose(dist.unit_log_prob(actions), torch.full((2, 1), expected))
    assert torch.allclose(dist.log_prob(actions), torch.full((2,), expected))
    assert dist.unit_valid(actions).tolist() == [[True], [True]]
    entropy = math.log(3) + 1.0 + math.log(2 * math.pi) + 2 * math.log(2)
    assert torch.allclose(dist.unit_entropy(actions), torch.full((2, 1), entropy))
    assert torch.allclose(dist.unit_kl(dist, actions), torch.zeros(2, 1), atol=1e-6)


def test_sample_and_mode_are_trees_in_natural_order():
    dist = make_distribution(SPEC, _params(batch=5))
    sample = dist.sample()
    assert list(sample) == ["move", "aim", "build"]
    assert sample["move"].shape == (5,) and sample["move"].dtype == torch.int64
    assert sample["aim"].shape == (5, 2) and sample["aim"].dtype == torch.float32
    assert sample["build"].shape == (5, 2) and sample["build"].dtype == torch.int64
    assert list(dist.mode()) == ["move", "aim", "build"]


def test_single_group_spaces_use_the_bare_value():
    dist = make_distribution(ActionSpec.from_space(Discrete(4)), torch.zeros(3, 4))
    assert dist.sample().shape == (3,)
    assert torch.allclose(dist.log_prob(torch.tensor([0, 1, 2])), torch.full((3,), -math.log(4)))


def test_apply_mask_tree_reaches_each_group():
    dist = make_distribution(SPEC, _params()).apply_mask(
        {"move": torch.tensor([[True, False, False], [True, True, True]]), "build": torch.ones(2, 4, dtype=bool)})
    assert dist.mode()["move"].tolist() == [0, 0]
    actions = {"move": torch.tensor([0, 1]), "aim": torch.zeros(2, 2), "build": torch.zeros(2, 2, dtype=torch.long)}
    lp = dist.unit_log_prob(actions)[:, 0] + math.log(2 * math.pi) + 2 * math.log(2)
    assert torch.allclose(lp, torch.tensor([0.0, -math.log(3)]), atol=1e-6)
    plain = make_distribution(SPEC, _params())
    assert plain.apply_mask(None) is plain


def test_cat_matches_separate_evaluation():
    a, b = make_distribution(SPEC, _params(1)), make_distribution(SPEC, _params(2))
    joined = TreeDist.cat([a, b])
    assert joined.batch_size == 3
    actions = {"move": torch.tensor([0, 1, 2]), "aim": torch.zeros(3, 2), "build": torch.zeros(3, 2, dtype=torch.long)}
    first = {"move": actions["move"][:1], "aim": actions["aim"][:1], "build": actions["build"][:1]}
    assert torch.allclose(joined.log_prob(actions)[:1], a.log_prob(first))


def test_parameter_errors():
    with pytest.raises(ValueError, match="needs 3 logits, got 4"):
        make_distribution(SPEC, {**_params(), "move": torch.zeros(2, 4)})
    with pytest.raises(ValueError, match="no parameters for action group build"):
        make_distribution(SPEC, {"move": torch.zeros(2, 3), "aim": _params()["aim"]})
    with pytest.raises(ValueError, match="parts for groups"):
        TreeDist(SPEC, {("move",): CategoricalDist(torch.zeros(2, 3))})
    with pytest.raises(ValueError, match="disagree on the batch size"):
        make_distribution(SPEC, {**_params(), "move": torch.zeros(3, 3)})


def test_kl_needs_the_same_action_space():
    dist = make_distribution(SPEC, _params())
    other = make_distribution(ActionSpec.from_space(Discrete(3)), torch.zeros(2, 3))
    with pytest.raises(TypeError):
        dist.unit_kl(other, None)
