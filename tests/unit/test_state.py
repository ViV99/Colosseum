"""Unit tests for colosseum.networks.state (State pytree utilities)."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest
import torch

from colosseum.networks.state import (
    batch_size_of,
    cat_batch,
    slice_batch,
    state_from_numpy,
    state_to,
    state_to_numpy,
    tree_leaves,
    tree_map,
    where_done,
)

Pair = namedtuple("Pair", ["a", "b"])


def _nested(batch: int = 3) -> dict:
    return {
        "h": torch.arange(batch * 2 * 4, dtype=torch.float32).reshape(batch, 2, 4),
        "pair": Pair(torch.ones(batch, 1), [torch.zeros(batch, dtype=torch.long)]),
        "none": None,
    }


def _assert_tree_equal(x, y):
    lx, ly = tree_leaves(x), tree_leaves(y)
    assert len(lx) == len(ly)
    for a, b in zip(lx, ly):
        assert a.dtype == b.dtype
        assert torch.equal(a, b)


def test_tree_map_keeps_structure_and_namedtuples():
    s = _nested()
    out = tree_map(lambda t: t * 2, s)
    assert set(out) == {"h", "pair", "none"}
    assert isinstance(out["pair"], Pair)
    assert isinstance(out["pair"].b, list)
    assert out["none"] is None
    assert torch.equal(out["h"], s["h"] * 2)
    assert tree_map(lambda t: t, None) is None


def test_tree_map_rejects_unknown_nodes():
    with pytest.raises(TypeError):
        tree_map(lambda t: t, {"x": 1.0})


def test_tree_leaves_order_is_insertion_order():
    s = _nested()
    leaves = tree_leaves(s)
    assert [tuple(t.shape) for t in leaves] == [(3, 2, 4), (3, 1), (3,)]
    assert tree_leaves(None) == []


def test_batch_size_of():
    assert batch_size_of(_nested(5)) == 5
    assert batch_size_of(None) is None
    assert batch_size_of({"x": None}) is None
    with pytest.raises(ValueError, match="batch size"):
        batch_size_of({"a": torch.zeros(2, 3), "b": torch.zeros(3, 3)})
    with pytest.raises(ValueError, match="0-dim"):
        batch_size_of({"a": torch.tensor(1.0)})


def test_slice_batch_int_keeps_batch_dim():
    s = _nested(3)
    row = slice_batch(s, 1)
    assert row["h"].shape == (1, 2, 4)
    assert torch.equal(row["h"][0], s["h"][1])
    last = slice_batch(s, -1)
    assert torch.equal(last["h"][0], s["h"][2])
    assert slice_batch(None, 0) is None


def test_slice_batch_sequence_and_tensor_index():
    s = _nested(4)
    sub = slice_batch(s, [3, 0])
    assert sub["h"].shape == (2, 2, 4)
    assert torch.equal(sub["h"][0], s["h"][3])
    sub_t = slice_batch(s, torch.tensor([1, 2]))
    assert torch.equal(sub_t["pair"].a, s["pair"].a[1:3])
    sub_mask = slice_batch(s, torch.tensor([True, False, True, False]))
    assert sub_mask["h"].shape[0] == 2


def test_slice_batch_integral_and_0dim_tensor_index_keep_batch_dim():
    s = _nested(3)
    for idx in (np.int64(2), torch.tensor(2), torch.tensor(2, dtype=torch.int32)):
        row = slice_batch(s, idx)
        assert row["h"].shape == (1, 2, 4)
        assert torch.equal(row["h"][0], s["h"][2])
        assert row["pair"].b[0].shape == (1,)


def test_slice_batch_rejects_bool_index():
    s = _nested(3)
    for idx in (True, False, torch.tensor(True)):
        with pytest.raises(TypeError, match="bool"):
            slice_batch(s, idx)


def test_slice_batch_accepts_numpy_and_range_indices():
    s = _nested(4)
    assert torch.equal(slice_batch(s, np.array([3, 0]))["h"], s["h"][[3, 0]])
    assert torch.equal(slice_batch(s, np.array([True, False, True, False]))["h"], s["h"][[0, 2]])
    assert slice_batch(s, np.array(2))["h"].shape == (1, 2, 4)
    assert torch.equal(slice_batch(s, range(1, 3))["h"], s["h"][1:3])
    with pytest.raises(TypeError, match="dtype"):
        slice_batch(s, np.array([0.0, 1.0]))
    with pytest.raises(TypeError, match="single bool"):
        slice_batch(s, np.array(True))


def test_slice_batch_rejects_numpy_bool_index_with_clear_message():
    s = _nested(3)
    for idx in (np.bool_(True), np.bool_(False)):
        with pytest.raises(TypeError, match="single bool is not a valid batch index"):
            slice_batch(s, idx)


@pytest.mark.parametrize("idx", [
    torch.tensor(1.0),
    torch.tensor(1.0, dtype=torch.float16),
    torch.tensor(1.0 + 0j),
    torch.tensor([0.0, 2.0]),
])
def test_slice_batch_rejects_non_integer_index_tensors(idx):
    with pytest.raises(TypeError, match="dtype"):
        slice_batch(_nested(3), idx)


@pytest.mark.parametrize("idx", [1.0, np.float64(1.0), [0, 1.5], [True, 2]])
def test_slice_batch_rejects_non_integer_indices(idx):
    with pytest.raises(TypeError, match="integer"):
        slice_batch(_nested(3), idx)


def test_cat_batch_roundtrip_of_rows():
    s = _nested(4)
    rows = [slice_batch(s, i) for i in range(4)]
    _assert_tree_equal(cat_batch(rows), s)
    assert isinstance(cat_batch(rows)["pair"], Pair)


def test_cat_batch_none_and_mismatch():
    assert cat_batch([None, None]) is None
    with pytest.raises(ValueError):
        cat_batch([None, {"h": torch.zeros(1, 2)}])
    with pytest.raises(ValueError):
        cat_batch([{"h": torch.zeros(1, 2)}, {"c": torch.zeros(1, 2)}])
    with pytest.raises(ValueError):
        cat_batch([])


def test_cat_batch_requires_identical_node_types():
    a = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="node types"):
        cat_batch([Pair(a, a), (a, a)])
    with pytest.raises(ValueError, match="node types"):
        cat_batch([(a, a), Pair(a, a)])
    Other = namedtuple("Other", ["a", "b"])
    with pytest.raises(ValueError, match="node types"):
        cat_batch([Pair(a, a), Other(a, a)])
    with pytest.raises(ValueError, match="node types"):
        cat_batch([[a], (a,)])


def test_where_done_validates_done_shape():
    state = {"h": torch.ones(3, 2)}
    reset = {"h": torch.zeros(3, 2)}
    for bad in (torch.tensor([True, False]), torch.zeros(3, 1, dtype=torch.bool),
                torch.tensor(True)):
        with pytest.raises(ValueError, match="done"):
            where_done(bad, reset, state)


def test_where_done_replaces_only_done_rows():
    state = {"mem": torch.full((3, 2), 5.0), "len": torch.tensor([4, 4, 4])}
    reset = {"mem": torch.zeros(3, 2), "len": torch.zeros(3, dtype=torch.long)}
    out = where_done(torch.tensor([False, True, False]), reset, state)
    assert torch.equal(out["mem"][1], torch.zeros(2))
    assert torch.equal(out["mem"][0], torch.full((2,), 5.0))
    assert out["len"].tolist() == [4, 0, 4]
    assert out["len"].dtype == torch.long
    assert where_done(torch.tensor([True]), None, None) is None


def test_numpy_roundtrip_preserves_structure_dtype_values():
    s = _nested(2)
    payload = state_to_numpy(s)
    assert isinstance(payload["h"], np.ndarray)
    assert isinstance(payload["pair"], Pair)
    assert payload["pair"].b[0].dtype == np.int64
    assert payload["none"] is None
    back = state_from_numpy(payload)
    _assert_tree_equal(back, s)
    assert state_to_numpy(None) is None
    assert state_from_numpy(None) is None


def test_state_to_numpy_is_a_copy():
    t = torch.zeros(2, 2)
    payload = state_to_numpy({"x": t})
    t.add_(1.0)
    assert payload["x"].sum() == 0.0


def test_state_from_numpy_copies_read_only_arrays_without_warning():
    """gRPC/np.frombuffer arrays are read-only: copy them, never warn (T2.2)."""
    import warnings

    ro = np.frombuffer(np.arange(6, dtype=np.float32).tobytes(), dtype=np.float32).reshape(1, 6)
    assert not ro.flags.writeable
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = state_from_numpy({"h": ro, "pair": Pair(ro, [ro])})
    assert torch.equal(back["h"], torch.arange(6, dtype=torch.float32).view(1, 6))
    back["h"].add_(1.0)  # writable, and does not alias the read-only buffer
    assert ro[0, 0] == 0.0
    assert isinstance(back["pair"], Pair)


def test_state_from_numpy_shares_writable_contiguous_arrays():
    """Documented aliasing: a writable C-contiguous array is used zero-copy on CPU."""
    arr = np.zeros((1, 2), np.float32)
    back = state_from_numpy(arr)
    back.add_(1.0)
    assert arr[0, 0] == 1.0
    strided = np.zeros((1, 4), np.float32)[:, ::2]
    state_from_numpy(strided).add_(1.0)  # non-contiguous: copied
    assert strided.sum() == 0.0


def test_state_to_numpy_upcasts_bfloat16_to_float32():
    h = torch.randn(2, 3).to(torch.bfloat16)
    payload = state_to_numpy({"h": h, "n": torch.tensor([[1]])})
    assert payload["h"].dtype == np.float32 and payload["n"].dtype == np.int64
    back = state_from_numpy(payload)
    assert back["h"].dtype == torch.float32 and torch.equal(back["h"], h.float())


def test_state_to_device():
    s = _nested(2)
    moved = state_to(s, "cpu")
    _assert_tree_equal(moved, s)
