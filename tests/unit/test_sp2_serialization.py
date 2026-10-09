"""gRPC serialization of numpy payloads with chunk v2 (SP1 guarantees, T6.4; wire format changes in SP5)."""
import json
import sys
from collections import namedtuple

import numpy as np
import pytest
import torch

from colosseum.core.types import TrajectoryChunk
from colosseum.networks.state import state_to_numpy
from colosseum.transport.serialization import (
    deserialize_chunk,
    deserialize_chunk_payload,
    deserialize_state_dict,
    pack_payload,
    serialize_chunk,
    serialize_chunk_payload,
    serialize_state_dict,
    unpack_payload,
    validate_chunk_payload,
)
from game_helpers import chunk_v2_payload

HC = namedtuple("HC", ["h", "c"])


@pytest.mark.parametrize("compress", [True, False])
def test_pack_unpack_nested_structures(compress):
    obj = {
        "a": np.arange(6, dtype=np.int64).reshape(2, 3),
        3: [np.float32(1.5), None, True, "s"],
        "t": (np.zeros(2, dtype=bool), 7),
        "n": {"x": np.ones((1, 1, 4), np.float32)},
    }
    data, compressed = pack_payload(obj, compress)
    assert compressed is compress
    back = unpack_payload(data, compressed)
    assert np.array_equal(back["a"], obj["a"]) and back["a"].dtype == np.int64
    assert back[3] == [1.5, None, True, "s"]
    assert isinstance(back["t"], tuple) and back["t"][1] == 7 and back["t"][0].dtype == bool
    assert back["n"]["x"].shape == (1, 1, 4)


def test_pack_rejects_tensors_object_arrays_and_container_keys():
    with pytest.raises(TypeError):
        pack_payload({(1, 2): 3})
    with pytest.raises(TypeError):
        pack_payload({"x": torch.zeros(1)})
    with pytest.raises(TypeError):
        pack_payload({"x": np.array([object()], dtype=object)})


def test_chunk_payload_bytes_roundtrip_with_state():
    payload = chunk_v2_payload(version=5)
    payload["initial_state"] = {"mem": np.ones((1, 2, 3), np.float32), "len": np.array([2])}
    data, compressed = serialize_chunk_payload(payload)
    back = deserialize_chunk_payload(data, compressed)
    validate_chunk_payload(back)
    chunk = TrajectoryChunk.from_payload(back)
    assert chunk.policy_version == 5
    assert torch.equal(chunk.initial_state["len"], torch.tensor([2]))
    assert chunk.action_masks.dtype == torch.bool and chunk.kind.dtype == torch.int8


def test_serialize_chunk_convenience_roundtrip():
    chunk = TrajectoryChunk.from_payload(chunk_v2_payload(version=1))
    data, compressed = serialize_chunk(chunk)
    back = deserialize_chunk("a", 9, data, compressed)
    assert torch.equal(back.obs, chunk.obs) and torch.equal(back.kind, chunk.kind)
    assert back.policy_version == 9 and back.agent_id == "a"


def test_state_dict_roundtrip():
    sd = {"w": np.random.randn(3, 3).astype(np.float32), "b": np.zeros(3, np.float32)}
    data, compressed = serialize_state_dict(sd)
    back = deserialize_state_dict(data, compressed)
    assert set(back) == {"w", "b"} and np.array_equal(back["w"], sd["w"])


def test_namedtuple_state_survives_the_wire():
    """Every State node type round-trips, namedtuples included."""
    state = {"core": HC(torch.randn(1, 1, 8), torch.randn(1, 1, 8)),
             "extra": [torch.ones(1, 2), (torch.zeros(1),)]}
    payload = chunk_v2_payload()
    payload["initial_state"] = state_to_numpy(state)
    back = TrajectoryChunk.from_payload(deserialize_chunk_payload(*serialize_chunk_payload(payload))).initial_state
    assert type(back["core"]) is HC
    assert torch.equal(back["core"].h, state["core"].h) and torch.equal(back["core"].c, state["core"].c)
    assert isinstance(back["extra"], list) and isinstance(back["extra"][1], tuple)
    assert torch.equal(back["extra"][0], state["extra"][0])


def test_unresolvable_namedtuple_is_rejected_with_a_clear_error():
    Local = namedtuple("Local", ["h"])  # not reachable by module path
    with pytest.raises(TypeError, match=r"namedtuple .*Local.* module level"):
        pack_payload({"s": Local(np.zeros(1))})


def test_unpack_does_not_import_unknown_namedtuple_modules():
    """Bytes from the network name a module that is not loaded: refuse, never import it."""
    data, _ = pack_payload(HC(np.zeros(1), np.ones(1)), compress=False)
    assert b'"test_sp2_serialization"' in data
    tampered = data.replace(b'"test_sp2_serialization"', b'"no_such_module_xyz1234"')  # same length
    with pytest.raises(ValueError, match="no_such_module_xyz1234"):
        unpack_payload(tampered, False)


# ---------------------------------------------------------------------------
# Fix round 1: hostile / malformed bytes from the network
# ---------------------------------------------------------------------------

def _raw_payload(skeleton, arrays_blob: bytes | None = None) -> bytes:
    """Uncompressed wire bytes with a hand-written skeleton (and an optional npz blob)."""
    import io
    import struct

    if arrays_blob is None:
        buf = io.BytesIO()
        np.savez(buf)
        arrays_blob = buf.getvalue()
    text = json.dumps(skeleton).encode()
    return struct.pack("<Q", len(text)) + text + arrays_blob


def test_oversized_decompressed_payload_is_rejected():
    data, compressed = pack_payload({"x": np.zeros(2**20, np.float32)})  # 4 MB, compresses to KBs
    assert len(data) < 2**20
    with pytest.raises(ValueError, match="exceeds the cap of 1048576 bytes"):
        unpack_payload(data, compressed, max_bytes=2**20)
    assert unpack_payload(data, compressed, max_bytes=2**23)["x"].shape == (2**20,)


def test_truncated_lz4_frame_is_rejected():
    data, compressed = pack_payload({"x": np.arange(1000)})
    with pytest.raises(ValueError, match="truncated"):
        unpack_payload(data[: len(data) // 2], compressed)


def test_array_header_declaring_a_huge_shape_is_rejected_before_allocation():
    import io
    import zipfile

    npy = io.BytesIO()
    np.lib.format.write_array_header_1_0(npy, {"descr": "<f4", "fortran_order": False, "shape": (10**12,)})
    npy.write(b"\0" * 16)  # far less data than the header claims
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as zf:
        zf.writestr("arr_0.npy", npy.getvalue())
    with pytest.raises(ValueError, match="declare 4000000000000 bytes"):
        unpack_payload(_raw_payload({"__nd__": 0}, blob.getvalue()), False)


@pytest.mark.parametrize("skeleton", [
    {"__nd__": 0}, {"__nd__": -1}, {"__nd__": True}, {"__nd__": "0"}, {"__nd__": 0.0},
    {"__dict__": 5}, {"__dict__": [[{"__list__": []}, 1]]}, {"__list__": 3}, {"what": 1},
    {"__namedtuple__": ["m"], "items": []}, {"__namedtuple__": ["m", "C", []]}, {"__tuple__": None},
])
def test_malformed_skeletons_raise_value_error(skeleton):
    with pytest.raises(ValueError):
        unpack_payload(_raw_payload(skeleton), False)


def test_namedtuple_lookup_never_triggers_module_getattr(monkeypatch):
    import types

    calls = []
    lazy = types.ModuleType("lazy_mod_t22")
    lazy.__getattr__ = lambda name: calls.append(name) or HC  # PEP 562 lazy import hook
    monkeypatch.setitem(sys.modules, "lazy_mod_t22", lazy)
    skeleton = {"__namedtuple__": ["lazy_mod_t22", "HC", ["h", "c"]], "items": [1, 2]}
    with pytest.raises(ValueError, match="lazy_mod_t22.HC"):
        unpack_payload(_raw_payload(skeleton), False)
    assert calls == []


def test_negative_dimension_cannot_cancel_a_huge_array_out_of_the_cap():
    """arr_1 (-10**10,) stored before arr_0 (10**10,): the running total stays <= 0."""
    import io
    import zipfile

    def npy(shape) -> bytes:
        f = io.BytesIO()
        np.lib.format.write_array_header_1_0(f, {"descr": "<f4", "fortran_order": False, "shape": shape})
        return f.getvalue() + b"\0" * 16

    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as zf:
        zf.writestr("arr_1.npy", npy((-(10**10),)))
        zf.writestr("arr_0.npy", npy((10**10,)))
    raw = _raw_payload({"__list__": [{"__nd__": 0}, {"__nd__": 1}]}, blob.getvalue())
    with pytest.raises(ValueError, match="invalid array shape"):
        unpack_payload(raw, False, max_bytes=2**20)


@pytest.mark.parametrize("value", [
    np.array(["a", "b"]), np.array([b"x"]), np.zeros(2, dtype="V4"), np.array(["2020-01-01"], dtype="datetime64[D]"),
])
def test_weights_payload_rejects_non_numeric_dtypes(value):
    from colosseum.transport.serialization import validate_state_dict_payload

    numeric = {"w": np.zeros(2, np.float32), "flag": np.array([True]), "n": np.arange(2, dtype=np.uint8),
               "c": np.zeros(1, np.complex64)}
    validate_state_dict_payload(numeric)  # bool/int/uint/float/complex are accepted
    with pytest.raises(ValueError, match="'bad'.*dtype"):
        validate_state_dict_payload({"w": np.zeros(2, np.float32), "bad": value})


def test_dimension_too_large_for_numpy_raises_value_error():
    """(0, 10**30) declares 0 bytes, but the dim exceeds sys.maxsize: rejected by the
    shape check before numpy would raise OverflowError."""
    import io
    import zipfile

    npy = io.BytesIO()
    np.lib.format.write_array_header_1_0(npy, {"descr": "<f4", "fortran_order": False, "shape": (0, 10**30)})
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as zf:
        zf.writestr("arr_0.npy", npy.getvalue())
    with pytest.raises(ValueError, match=r"invalid array shape \(0, 10{30}\) in arr_0\.npy"):
        unpack_payload(_raw_payload({"__nd__": 0}, blob.getvalue()), False)


# ---------------------------------------------------------------------------
# Chunk v2 payload validation (the servicer's guard)
# ---------------------------------------------------------------------------


def test_validate_chunk_payload_accepts_trees_and_rejects_bad_fields():
    good = chunk_v2_payload()  # S = 4
    good["obs"] = {"grid": np.zeros((4, 2, 2), np.uint8), "vec": np.zeros((4, 3), np.float32)}
    validate_chunk_payload(good)
    for field, value, message in [
        ("kind", None, "kind"),
        ("reward", np.zeros(3, np.float32), "reward"),
        ("obs", {"grid": np.zeros((3, 2), np.uint8)}, "obs"),
        ("obs", [np.zeros(4)], "obs"),
        ("policy_version", "1", "policy_version"),
        ("behavior_unit_logp", np.zeros(4, np.float32), "behavior_unit_logp"),
    ]:
        bad = {**chunk_v2_payload(), field: value}
        with pytest.raises(ValueError, match=message):
            validate_chunk_payload(bad)


def test_validate_chunk_payload_rejects_a_broken_slot_structure():
    """A chunk from the network that breaks the chunk v2 slot rules is refused by the guard
    (INVALID_ARGUMENT at the servicer) instead of crashing the learner's collect_batch."""
    validate_chunk_payload(chunk_v2_payload(pattern="ATPP"))
    bad = chunk_v2_payload(pattern="AAAB")
    bad["kind"] = np.array([0, 0, 0, 0], np.int8)  # an ACT in the last slot
    with pytest.raises(ValueError, match="an ACT in the last slot"):
        validate_chunk_payload(bad)
