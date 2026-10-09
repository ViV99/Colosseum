"""ObsSpec / ActionSpec v2: leaves, allocation, mask layout, empty-row rules, deciders (SP2 T1.3)."""
import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple

from colosseum.core.errors import EnvContractError
from colosseum.core.specs import ActionSpec, LeafSpec, ObsSpec
from colosseum.envs.spaces import Units

OBS = Dict([
    ("grid", Box(0, 255, (3, 3), dtype=np.uint8)),
    ("entities", Dict([("feat", Box(-1.0, 1.0, (4, 2))), ("mask", MultiBinary(4))])),
    ("turn", Discrete(9)),
    ("cards", MultiDiscrete([3, 3])),
])

ACT = Dict([
    ("base", Discrete(3)),
    ("aim", Box(-1.0, 1.0, (2,))),
    ("workers", Units(2, Dict([("move", Discrete(4)), ("target", Discrete(3)), ("push", Box(-1.0, 1.0, (1,)))]),
                      only_if={"target": ("move", {3})})),
    ("build", MultiDiscrete([2, 3])),
])


def test_obs_spec_leaves_in_natural_order_with_dtypes():
    spec = ObsSpec.from_space(OBS)
    assert spec.is_dict
    assert spec.leaves == (
        LeafSpec(("grid",), (3, 3), np.dtype(np.uint8)),
        LeafSpec(("entities", "feat"), (4, 2), np.dtype(np.float32)),
        LeafSpec(("entities", "mask"), (4,), np.dtype(np.int8)),
        LeafSpec(("turn",), (), np.dtype(np.int64)),
        LeafSpec(("cards",), (2,), np.dtype(np.int64)),
    )
    assert spec.signature() == ("grid:uint8[3,3];entities/feat:float32[4,2];entities/mask:int8[4];"
                                "turn:int64[];cards:int64[2]")
    bare = ObsSpec.from_space(Box(-1.0, 1.0, (5,)))
    assert not bare.is_dict and bare.leaves == (LeafSpec((), (5,), np.dtype(np.float32)),)


def test_obs_spec_allocate_preserves_dtypes_and_order():
    zeros = ObsSpec.from_space(OBS).allocate((7, 2))
    assert list(zeros) == ["grid", "entities", "turn", "cards"]
    assert zeros["grid"].shape == (7, 2, 3, 3) and zeros["grid"].dtype == np.uint8
    assert zeros["entities"]["mask"].dtype == np.int8 and zeros["turn"].shape == (7, 2)
    assert ObsSpec.from_space(Box(-1.0, 1.0, (5,))).allocate((3,)).shape == (3, 5)


def test_obs_spec_rejects_unsupported_spaces():
    with pytest.raises(TypeError, match="unsupported observation space Tuple"):
        ObsSpec.from_space(Tuple([Discrete(2)]))
    with pytest.raises(ValueError, match="empty Dict"):
        ObsSpec.from_space(Dict([]))


def test_obs_spec_check_structure_and_shapes():
    spec = ObsSpec.from_space(OBS)
    good = OBS.sample()
    spec.check(good, "seat 0: observation")
    with pytest.raises(EnvContractError, match=r"seat 0: observation: keys at <root> do not match.*'cards'"):
        spec.check({k: v for k, v in good.items() if k != "cards"}, "seat 0: observation")
    with pytest.raises(EnvContractError, match=r"leaf entities/feat has shape \(4, 3\), expected \(4, 2\)"):
        spec.check({**good, "entities": {"feat": np.zeros((4, 3)), "mask": good["entities"]["mask"]}}, "obs")
    with pytest.raises(EnvContractError, match="expected a dict"):
        spec.check({**good, "entities": np.zeros(3)}, "obs")
    bare = ObsSpec.from_space(Box(-1.0, 1.0, (5,)))
    bare.check(np.zeros(5, dtype=np.float64), "obs")  # dtype is not checked here (cast on write)
    with pytest.raises(EnvContractError, match="got a dict"):
        bare.check({"x": np.zeros(5)}, "obs")


def test_obs_spec_check_rejects_ragged_and_missing_leaves():
    bare = ObsSpec.from_space(Box(-1.0, 1.0, (2, 2)))
    with pytest.raises(EnvContractError, match=r"^seat 1: observation: leaf <root> is not an array"):
        bare.check([[0.0, 1.0], [2.0]], "seat 1: observation")             # ragged nested list
    scalar = ObsSpec.from_space(Dict([("turn", Discrete(5)), ("x", Box(-1.0, 1.0, (2,)))]))
    scalar.check({"turn": 3, "x": np.zeros(2)}, "obs")
    with pytest.raises(EnvContractError, match=r"^seat 0: observation: leaf turn is None"):
        scalar.check({"turn": None, "x": np.zeros(2)}, "seat 0: observation")
    with pytest.raises(EnvContractError, match=r"leaf turn is a dict"):
        scalar.check({"turn": {"value": 3}, "x": np.zeros(2)}, "obs")
    with pytest.raises(EnvContractError, match=r"leaf <root> is None"):
        ObsSpec.from_space(Discrete(3)).check(None, "obs")


def test_action_groups_and_deciders():
    spec = ActionSpec.from_space(ACT)
    assert [g.path for g in spec.groups] == [("base",), ("aim",), ("workers",), ("build",)]
    assert [g.kind for g in spec.groups] == ["discrete", "box", "units", "multi_discrete"]
    assert [g.mask_size for g in spec.groups] == [3, 0, 7, 5]
    assert spec.groups[1].box_dim == 2 and spec.groups[3].nvec == (2, 3)
    assert spec.is_dict and spec.has_units and spec.has_masks
    assert spec.num_deciders == 1 + 2      # decider 0 = base+aim+build, then 2 workers
    assert ActionSpec.from_space(Discrete(5)).num_deciders == 1
    assert ActionSpec.from_space(Units(8, Discrete(3))).num_deciders == 8
    two_groups = ActionSpec.from_space(Dict([("a", Units(3, Discrete(2))), ("b", Units(2, Discrete(2)))]))
    assert two_groups.num_deciders == 5 and two_groups.groups[0].path == ("a",)
    box = ActionSpec.from_space(Box(-1.0, 1.0, (3,)))
    assert not box.has_masks and box.full_mask() is None and box.boot_mask() is None


def test_nested_dict_action_space_paths():
    spec = ActionSpec.from_space(Dict([("army", Dict([("stance", Discrete(2)), ("units", Units(2, Discrete(3)))]))]))
    assert [g.path for g in spec.groups] == [("army", "stance"), ("army", "units")]
    mask = spec.full_mask()
    assert list(mask["army"]) == ["stance", "units"] and mask["army"]["units"]["action"].shape == (2, 3)


def test_allocate_actions_native_dtypes():
    acts = ActionSpec.from_space(ACT).allocate_actions((4,))
    assert acts["base"].shape == (4,) and acts["base"].dtype == np.int64
    assert acts["aim"].shape == (4, 2) and acts["aim"].dtype == np.float32
    assert acts["workers"]["move"].shape == (4, 2) and acts["workers"]["push"].shape == (4, 2, 1)
    assert acts["build"].shape == (4, 2) and acts["build"].dtype == np.int64
    assert ActionSpec.from_space(Units(3, MultiDiscrete([2, 5]))).allocate_actions(()).shape == (3, 2)
    assert ActionSpec.from_space(Units(3, Box(-1.0, 1.0, (2,)))).allocate_actions((2,)).dtype == np.float32


def test_full_and_boot_masks():
    spec = ActionSpec.from_space(ACT)
    full = spec.full_mask((2,))
    assert list(full) == ["base", "workers", "build"]          # no leaf for the box group
    assert full["base"].shape == (2, 3) and full["base"].all()
    assert full["workers"]["unit"].shape == (2, 2) and full["workers"]["unit"].all()
    assert full["workers"]["action"].shape == (2, 2, 7) and full["build"].shape == (2, 5)
    boot = spec.boot_mask()
    assert not boot["workers"]["unit"].any() and boot["workers"]["action"].all() and boot["base"].all()


def test_normalize_mask_fills_missing_leaves_and_copies():
    spec = ActionSpec.from_space(ACT)
    base = np.array([True, False, True])
    raw = {"base": base, "workers": {"unit": np.array([True, False])}}
    mask = spec.normalize_mask(raw, "seat 1")
    assert mask["base"].tolist() == [True, False, True] and mask["base"] is not base
    assert mask["workers"]["unit"].tolist() == [True, False]
    assert mask["workers"]["action"].all() and mask["build"].all()
    assert spec.normalize_mask(None, "seat 1")["base"].all()
    flat = ActionSpec.from_space(Discrete(3))
    assert flat.normalize_mask(np.array([False, True, False]), "s").tolist() == [False, True, False]


@pytest.mark.parametrize("raw, message", [
    ({"base": np.array([1, 0, 1], dtype=np.int8)}, "must be a bool array, got dtype int8"),
    ({"base": np.ones(4, dtype=bool)}, r"action mask base has shape \(4,\), expected \(3,\)"),
    ({"aim": np.ones(2, dtype=bool)}, "unexpected key aim"),
    ({"bogus": np.ones(2, dtype=bool)}, "unexpected key bogus"),
    ({"workers": np.ones(2, dtype=bool)}, "must be a dict with keys 'unit' and/or 'action'"),
    ({"workers": {"units": np.ones(2, dtype=bool)}}, "must be a dict with keys 'unit' and/or 'action'"),
    ({"workers": {"action": np.ones((2, 6), dtype=bool)}}, r"workers/action has shape \(2, 6\), expected \(2, 7\)"),
    (np.ones(3, dtype=bool), "must be a dict"),
])
def test_normalize_mask_errors(raw, message):
    with pytest.raises(EnvContractError, match=message):
        ActionSpec.from_space(ACT).normalize_mask(raw, "worker 0, env 1, seat 2")


def test_box_only_action_space_takes_no_mask():
    with pytest.raises(EnvContractError, match="takes no action mask"):
        ActionSpec.from_space(Box(-1.0, 1.0, (2,))).normalize_mask(np.ones(2, dtype=bool), "seat 0")


def test_empty_row_rule_for_acting_seats():
    spec = ActionSpec.from_space(ACT)
    mask = spec.full_mask()
    # Units: an absent unit and an empty component row are allowed (invalid deciders, no error).
    mask["workers"]["unit"][:] = False
    mask["workers"]["action"][0, :4] = False
    spec.check_acting_mask(mask, "seat 0")
    mask["build"][2:] = False                     # sub-action 1 of build (3 values) has no legal value
    with pytest.raises(EnvContractError, match=r"seat 0: action mask build \(sub-action 1\) has no legal action"):
        spec.check_acting_mask(mask, "seat 0")
    flat = ActionSpec.from_space(Discrete(3))
    with pytest.raises(EnvContractError, match="action mask <root> has no legal action"):
        flat.check_acting_mask(np.zeros(3, dtype=bool), "seat 0")
    spec.check_acting_mask(None, "seat 0")


def test_action_spec_rejects_unsupported_spaces():
    with pytest.raises(TypeError, match="unsupported action space"):
        ActionSpec.from_space(Tuple([Discrete(2)]))
    with pytest.raises(ValueError, match="1-D float"):
        ActionSpec.from_space(Box(-1.0, 1.0, (2, 2)))
    with pytest.raises(ValueError, match="start"):
        ActionSpec.from_space(Discrete(3, start=1))


def test_signatures_identify_spaces():
    a = ActionSpec.from_space(ACT)
    assert a == ActionSpec.from_space(ACT)
    assert a.signature().startswith("base:discrete(3);aim:box(2);workers:units(2,dict,")
    assert "target<-move[3]" in a.signature()
    assert ActionSpec.from_space(Discrete(3)) != ActionSpec.from_space(Discrete(4))
    assert ObsSpec.from_space(OBS) == ObsSpec.from_space(OBS)
