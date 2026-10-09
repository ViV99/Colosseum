"""``Units`` action space: components, natural Dict order, only_if checks, sample/contains (SP2 T1.2)."""
import pickle

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

from colosseum.envs.spaces import UnitComponent, Units


def _move_target(only_if=None):
    per_unit = Dict([("move", Discrete(4)), ("target", Discrete(3)), ("thrust", Box(-1.0, 1.0, (2,)))])
    return Units(3, per_unit, only_if=only_if, seed=0)


def test_components_for_each_per_unit_kind():
    assert Units(5, Discrete(4)).components == (UnitComponent("0", "discrete", 4),)
    assert Units(5, Discrete(4)).per_unit_kind == "discrete"
    md = Units(5, MultiDiscrete([3, 7]))
    assert md.per_unit_kind == "multi_discrete"
    assert md.components == (UnitComponent("0", "discrete", 3), UnitComponent("1", "discrete", 7))
    box = Units(5, Box(-1.0, 1.0, (2,)))
    assert box.per_unit_kind == "box" and box.components == (UnitComponent("0", "box", 2),)
    d = _move_target()
    assert d.per_unit_kind == "dict"
    assert [c.name for c in d.components] == ["move", "target", "thrust"]
    assert [c.kind for c in d.components] == ["discrete", "discrete", "box"]


def test_dict_component_order_is_the_spaces_order():
    # gymnasium sorts the keys of a plain dict; a list of pairs keeps the given order.
    sorted_keys = Units(2, Dict({"move": Discrete(5), "attack": Discrete(3)}))
    kept = Units(2, Dict([("move", Discrete(5)), ("attack", Discrete(3))]))
    assert [c.name for c in sorted_keys.components] == ["attack", "move"]
    assert [c.name for c in kept.components] == ["move", "attack"]


@pytest.mark.parametrize("per_unit, message", [
    (Dict([("a", MultiDiscrete([2, 2]))]), "only be Discrete or 1-D Box"),
    (Dict([("a", Dict([("b", Discrete(2))]))]), "only be Discrete or 1-D Box"),
    (Box(-1.0, 1.0, (2, 2)), "1-D"),
    (Box(0, 3, (2,), dtype=np.int64), "float dtype"),
    (Discrete(3, start=1), "start"),
    (MultiBinary(3), "must be Discrete, MultiDiscrete, Box or Dict"),
    (Dict([]), "empty Dict"),
])
def test_invalid_per_unit_spaces(per_unit, message):
    with pytest.raises(ValueError, match=message):
        Units(2, per_unit)


def test_max_units_must_be_positive():
    with pytest.raises(ValueError, match="max_units"):
        Units(0, Discrete(2))


@pytest.mark.parametrize("only_if, message", [
    ({"nope": ("move", {3})}, "unknown component 'nope'"),
    ({"target": ("nope", {3})}, "unknown parent 'nope'"),
    ({"target": ("thrust", {0})}, "must be a discrete component"),
    ({"target": ("target", {0})}, "cannot depend on itself"),
    ({"target": ("move", set())}, "empty"),
    ({"target": ("move", {4})}, "outside parent 'move' range"),
    ({"target": ("move", {0}), "move": ("target", {0})}, "cycle"),
    ({"target": "move"}, "must be \\(parent, values\\)"),
])
def test_invalid_only_if(only_if, message):
    with pytest.raises(ValueError, match=message):
        _move_target(only_if)


def test_only_if_is_normalized_to_frozensets():
    space = _move_target({"target": ("move", [3, 3, 1])})
    assert space.only_if == {"target": ("move", frozenset({1, 3}))}
    assert Units(2, MultiDiscrete([3, 4]), only_if={"1": ("0", {2})}).only_if == {"1": ("0", frozenset({2}))}


def test_sample_respects_unit_and_action_masks():
    space = _move_target({"target": ("move", {3})})
    mask = {
        "unit": np.array([True, True, False]),
        # move (4) | target (3); unit 1 has an empty move row, unit 2 is absent
        "action": np.array([[0, 0, 0, 1, 0, 1, 0],
                            [0, 0, 0, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1]], dtype=bool),
    }
    for _ in range(20):
        a = space.sample(mask)
        assert a["move"].dtype == np.int64 and a["move"].shape == (3,)
        assert a["move"][0] == 3 and a["target"][0] == 1
        assert a["move"][1] == 0          # empty row -> 0
        assert a["move"][2] == 0 and a["target"][2] == 0 and np.all(a["thrust"][2] == 0.0)
        assert a["thrust"].dtype == np.float32 and a["thrust"].shape == (3, 2)
        assert space.contains(a)


def test_sample_layouts_per_kind():
    assert Units(4, Discrete(3), seed=1).sample().shape == (4,)
    md = Units(4, MultiDiscrete([3, 5]), seed=1).sample()
    assert md.shape == (4, 2) and md.dtype == np.int64 and np.all(md[:, 1] < 5)
    box = Units(4, Box(-2.0, 2.0, (3,)), seed=1).sample()
    assert box.shape == (4, 3) and box.dtype == np.float32 and np.all(np.abs(box) <= 2.0)


def test_contains_checks_structure_dtype_and_range():
    space = Units(2, MultiDiscrete([3, 5]))
    assert space.contains(np.array([[2, 4], [0, 0]]))
    assert not space.contains(np.array([[3, 0], [0, 0]]))            # out of range
    assert not space.contains(np.array([[0, 0]]))                     # wrong U
    assert not space.contains(np.array([[0.0, 0.0], [0.0, 0.0]]))     # float for discrete
    d = _move_target()
    good = {"move": np.zeros(3, np.int64), "target": np.zeros(3, np.int64), "thrust": np.zeros((3, 2), np.float32)}
    assert d.contains(good)
    assert not d.contains({"move": good["move"], "target": good["target"]})            # missing key
    assert not d.contains({**good, "thrust": np.full((3, 2), 5.0, np.float32)})        # outside the Box


def test_equality_and_pickle():
    a = _move_target({"target": ("move", {3})})
    b = _move_target({"target": ("move", [3])})
    assert a == b and a != _move_target() and a != Units(3, Discrete(4))
    assert pickle.loads(pickle.dumps(a)) == a
    assert "Units(3" in repr(a)


def test_equality_respects_component_order_and_hash_agrees():
    # gymnasium's Dict equality ignores key order, but the order fixes the mask layout.
    ab = Units(2, Dict([("a", Discrete(2)), ("b", Discrete(3))]))
    ba = Units(2, Dict([("b", Discrete(3)), ("a", Discrete(2))]))
    assert ab != ba
    same = Units(2, Dict([("a", Discrete(2)), ("b", Discrete(3))]))
    assert ab == same and hash(ab) == hash(same)
    assert len({ab, same, ba}) == 2
