"""gRPC serialization of numpy payloads (T2.2; wire format changes in SP5)."""
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
)
from dataflow_helpers import chunk_payload

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


def test_pack_rejects_tensors_and_object_arrays():
    with pytest.raises(TypeError):
        pack_payload({"x": torch.zeros(1)})
    with pytest.raises(TypeError):
        pack_payload({"x": np.array([object()], dtype=object)})


def test_chunk_payload_bytes_roundtrip_with_state():
    payload = chunk_payload(T=4, version=5)
    payload["initial_state"] = {"mem": np.ones((1, 2, 3), np.float32), "len": np.array([2])}
    payload["action_masks"] = np.ones((4, 3), dtype=bool)
    data, compressed = serialize_chunk_payload(payload)
    back = deserialize_chunk_payload(data, compressed)
    chunk = TrajectoryChunk.from_payload(back)
    assert chunk.behavior_policy_version == 5
    assert torch.equal(chunk.initial_state["len"], torch.tensor([2]))
    assert chunk.action_masks.dtype == torch.bool


def test_serialize_chunk_convenience_roundtrip():
    chunk = TrajectoryChunk.from_payload(chunk_payload(T=4, version=1))
    data, compressed = serialize_chunk(chunk)
    back = deserialize_chunk("a", 9, data, compressed)
    assert torch.equal(back.observations, chunk.observations)
    assert back.behavior_policy_version == 9


def test_state_dict_roundtrip():
    sd = {"w": np.random.randn(3, 3).astype(np.float32), "b": np.zeros(3, np.float32)}
    data, compressed = serialize_state_dict(sd)
    back = deserialize_state_dict(data, compressed)
    assert set(back) == {"w", "b"} and np.array_equal(back["w"], sd["w"])


def test_namedtuple_state_survives_the_wire():
    """Every State node type round-trips, namedtuples included (T1.5 concern)."""
    state = {"core": HC(torch.randn(1, 1, 8), torch.randn(1, 1, 8)),
             "extra": [torch.ones(1, 2), (torch.zeros(1),)]}
    payload = chunk_payload(T=4)
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
    assert b'"test_serialization"' in data
    tampered = data.replace(b'"test_serialization"', b'"no_such_mod_xyz123"')  # same length
    with pytest.raises(ValueError, match="no_such_mod_xyz123"):
        unpack_payload(tampered, False)
