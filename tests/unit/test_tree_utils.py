"""Tree utilities for observation / action / mask trees (SP2 T1.1)."""
import numpy as np
import pytest
import torch

from colosseum.core.tree import (
    tree_assign,
    tree_get,
    tree_index,
    tree_leaves,
    tree_map,
    tree_paths,
    tree_same_structure,
    tree_stack,
    tree_to_numpy,
    tree_to_torch,
)


def _obs():
    # Insertion order (grid before vec, b before a) is deliberately not alphabetical.
    return {
        "grid": np.arange(4, dtype=np.uint8).reshape(2, 2),
        "vec": {"b": np.array([1.0, 2.0], dtype=np.float32), "a": np.array(3, dtype=np.int64)},
    }


def test_leaves_and_paths_follow_insertion_order():
    obs = _obs()
    assert tree_paths(obs) == [("grid",), ("vec", "b"), ("vec", "a")]
    leaves = tree_leaves(obs)
    assert [leaf.dtype for leaf in leaves] == [np.uint8, np.float32, np.int64]
    assert tree_paths(np.zeros(3)) == [()]
    assert len(tree_leaves(np.zeros(3))) == 1


def test_tree_map_aligns_leaves_and_keeps_the_first_trees_order():
    a = {"x": np.array([1, 2]), "y": np.array([3])}
    b = {"y": np.array([10]), "x": np.array([20, 30])}
    out = tree_map(lambda u, v: u + v, a, b)
    assert list(out) == ["x", "y"]
    assert out["x"].tolist() == [21, 32] and out["y"].tolist() == [13]


def test_tree_map_rejects_different_structures_with_the_path():
    with pytest.raises(ValueError, match="vec"):
        tree_map(lambda u, v: u, _obs(), {"grid": np.zeros(1), "vec": np.zeros(1)})
    with pytest.raises(ValueError, match="keys"):
        tree_map(lambda u, v: u, {"a": 1}, {"b": 1})


def test_tree_get():
    obs = _obs()
    assert tree_get(obs, ("vec", "a")) == 3
    assert tree_get(obs, ()) is obs
    with pytest.raises(KeyError, match="vec/c"):
        tree_get(obs, ("vec", "c"))


def test_tree_stack_numpy_preserves_dtypes_and_torch_stacks_tensors():
    stacked = tree_stack([_obs(), _obs(), _obs()])
    assert stacked["grid"].shape == (3, 2, 2) and stacked["grid"].dtype == np.uint8
    assert stacked["vec"]["a"].shape == (3,) and stacked["vec"]["a"].dtype == np.int64
    t = tree_stack([{"x": torch.zeros(2)}, {"x": torch.ones(2)}], axis=1)
    assert isinstance(t["x"], torch.Tensor) and t["x"].shape == (2, 2)
    assert t["x"][:, 1].tolist() == [1.0, 1.0]
    with pytest.raises(ValueError):
        tree_stack([])


def test_tree_index_and_assign():
    buf = {"grid": np.zeros((5, 2, 2), dtype=np.uint8),
           "vec": {"b": np.zeros((5, 2), dtype=np.float32), "a": np.zeros(5, dtype=np.int64)}}
    tree_assign(buf, 3, _obs())
    row = tree_index(buf, 3)
    assert row["grid"].tolist() == [[0, 1], [2, 3]] and row["grid"].dtype == np.uint8
    assert row["vec"]["b"].tolist() == [1.0, 2.0] and int(row["vec"]["a"]) == 3
    assert buf["grid"][2].sum() == 0
    with pytest.raises(TypeError):
        tree_assign(np.zeros(3), 0, np.ones(()))


def test_torch_numpy_round_trip_preserves_dtypes():
    t = tree_to_torch(_obs())
    assert t["grid"].dtype == torch.uint8 and t["vec"]["a"].dtype == torch.int64
    back = tree_to_numpy(t)
    assert back["grid"].dtype == np.uint8 and back["vec"]["b"].dtype == np.float32
    assert np.array_equal(back["grid"], _obs()["grid"])
    assert tree_to_torch(None) is None and tree_to_numpy(None) is None


def test_tree_same_structure_is_order_sensitive():
    assert tree_same_structure(_obs(), _obs())
    assert not tree_same_structure({"a": 1, "b": 2}, {"b": 2, "a": 1})
    assert not tree_same_structure({"a": 1}, np.zeros(1))
    assert tree_same_structure(np.zeros(1), torch.zeros(2))
