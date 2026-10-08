"""ActionSpec / CompositeDist component order (T4.5)."""
import collections

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.core.action_spec import ActionSpec
from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.registry import validate_config
from colosseum.networks.distributions import CategoricalDist
from colosseum.networks.model import act
from helpers import (
    TWELVE_NVEC,
    MisorderedTwelveHeadPolicy,
    TwelveHeadPolicy,
    TwelveUnitEnv,
    twelve_unit_model,
)

Box, Discrete = gymnasium.spaces.Box, gymnasium.spaces.Discrete


def test_multidiscrete_components_follow_index_order():
    spec = ActionSpec.from_space(gymnasium.spaces.MultiDiscrete(np.array(TWELVE_NVEC)))
    assert spec.component_names == tuple(str(i) for i in range(12))
    assert [c.offset for c in spec.components] == list(range(12))
    assert [c.num_categories for c in spec.components] == list(TWELVE_NVEC)
    np.testing.assert_array_equal(spec.decode(np.arange(12, dtype=np.float32)), np.arange(12))


def test_multidiscrete_mask_segments_follow_index_order():
    env = TwelveUnitEnv(use_mask=True)
    _, info = env.reset()
    spec = ActionSpec.from_space(env.action_space)
    flat = spec.flatten_mask(info[0]["action_mask"])
    for i, comp in enumerate(spec.components):
        segment = flat[comp.mask_offset:comp.mask_offset + comp.mask_size]
        assert segment.sum() == 1 and int(np.argmax(segment)) == i


def test_tuple_components_follow_index_order():
    spec = ActionSpec.from_space(gymnasium.spaces.Tuple([Discrete(i + 2) for i in range(11)]))
    assert spec.component_names == tuple(str(i) for i in range(11))
    assert spec.decode(np.arange(11, dtype=np.float32)) == tuple(range(11))


def test_dict_components_follow_space_order():
    ordered = gymnasium.spaces.Dict(collections.OrderedDict([
        ("speed", Box(0.0, 1.0, (1,))), ("direction", Discrete(4)),
    ]))
    spec = ActionSpec.from_space(ordered)
    assert spec.component_names == ("speed", "direction")
    assert [c.offset for c in spec.components] == [0, 1]
    plain = gymnasium.spaces.Dict({"speed": Box(0.0, 1.0, (1,)), "direction": Discrete(4)})
    assert ActionSpec.from_space(plain).component_names == tuple(plain.spaces)


def test_composite_dist_keeps_insertion_order():
    dist = TwelveHeadPolicy(16)(torch.zeros(3, 16))
    assert dist.keys == [str(i) for i in range(12)]
    assert [offset for _name, offset, _size, _disc in dist.components] == list(range(12))
    np.testing.assert_array_equal(dist.mode()[0].numpy(), np.arange(12))


def test_check_distribution_accepts_matching_and_rejects_mismatches():
    spec = ActionSpec.from_space(TwelveUnitEnv().action_space)
    spec.check_distribution(TwelveHeadPolicy(16)(torch.zeros(2, 16)))
    with pytest.raises(ValueError, match="action space"):
        spec.check_distribution(MisorderedTwelveHeadPolicy(16)(torch.zeros(2, 16)))
    with pytest.raises(ValueError, match="CompositeDist"):
        spec.check_distribution(CategoricalDist(torch.zeros(2, 3)))
    ActionSpec.from_space(Discrete(3)).check_distribution(CategoricalDist(torch.zeros(2, 3)))
    with pytest.raises(ValueError, match="CompositeDist"):
        ActionSpec.from_space(Discrete(3)).check_distribution(TwelveHeadPolicy(16)(torch.zeros(2, 16)))


def test_twelve_masked_heads_step_and_decode():
    model = twelve_unit_model(peaked=False)
    env = TwelveUnitEnv(use_mask=True)
    obs, info = env.reset()
    spec = ActionSpec.from_space(env.action_space)
    mask = torch.as_tensor(spec.flatten_mask(info[0]["action_mask"]))[None]
    out = act(model, torch.as_tensor(obs[0])[None], model.initial_state(1), mask)
    np.testing.assert_array_equal(spec.decode(out.actions[0].numpy()), np.arange(12))


def _twelve_config(policy_class: str) -> ColosseumConfig:
    # use_mask=False: a misordered policy under the natural-order mask could get an
    # all-illegal head and fail inside the dummy step before the layout check runs.
    return ColosseumConfig.model_validate({
        "env": {"env_class": "helpers.TwelveUnitEnv", "num_players": 1, "kwargs": {"use_mask": False}},
        "networks": {
            "encoder_class": "helpers.SimpleEncoder",
            "policy_class": policy_class,
            "value_class": "helpers.SimpleValue",
        },
    })


def test_validate_config_rejects_misordered_composite_policy():
    validate_config(_twelve_config("helpers.TwelveHeadPolicy"))
    with pytest.raises(ConfigError, match="action space"):
        validate_config(_twelve_config("helpers.MisorderedTwelveHeadPolicy"))


def test_chase_example_policy_matches_its_action_space():
    from examples.composite_action import networks as chase_networks
    from examples.composite_action.env import ChaseEnv

    spec = ActionSpec.from_space(ChaseEnv().action_space)
    spec.check_distribution(chase_networks.ChasePolicy()(torch.zeros(2, chase_networks._LATENT)))


def test_space_miners_example_policy_matches_its_action_space():
    pytest.importorskip("Box2D")
    from examples.space_miners import networks as sm_networks
    from examples.space_miners.env import SpaceMinersEnv

    spec = ActionSpec.from_space(SpaceMinersEnv().action_space)
    spec.check_distribution(sm_networks.SpaceMinersPolicy()(torch.zeros(2, sm_networks._LATENT)))
