"""Serialization for the gRPC transport: numpy payloads <-> bytes.

Wire format (SP1, unchanged in SP2; replaced in SP5):
    8-byte little-endian length N | N bytes of JSON skeleton | np.savez archive
The skeleton mirrors the payload structure; every numpy array is replaced by
``{"__nd__": i}`` (index into the archive). Dicts, lists, tuples and
namedtuples are tagged so that int dict keys, tuples and namedtuple types
survive JSON. Arrays are loaded with ``allow_pickle=False`` and no ``pickle``
is involved, so bytes from the network cannot execute code: a namedtuple is
rebuilt only from a class that is already loaded in the receiving process
(looked up by module and qualified name in the module ``__dict__``, never
imported). The whole blob is optionally lz4-compressed.

Decoding treats the bytes as hostile (the gRPC servers listen on ``[::]``):
the decompressed size and the total array size declared by the ``.npy``
headers are checked against ``max_bytes`` before anything is allocated, and
every malformed input raises ``ValueError``. The cap is derived from the gRPC
message limit: ``payload_byte_cap(grpc_max_message_mb)`` =
``DECOMPRESSED_SIZE_FACTOR`` (8) times the message limit; the servers and the
weight-store client pass it, other callers get ``DEFAULT_MAX_PAYLOAD_BYTES``
(the cap for the default 64 MB message limit, 512 MB).
"""

from __future__ import annotations

import io
import json
import math
import struct
import sys
import zipfile
from collections.abc import Mapping
from typing import Any

import lz4.frame
import numpy as np
import torch

from colosseum.core.ipc import is_namedtuple
from colosseum.sp2.core.types import TrajectoryChunk, validate_slot_structure

_HEADER = struct.Struct("<Q")
_KEY_TYPES = (str, int, float, bool, type(None))  # dict keys allowed in a payload

#: A decoded payload may be at most this many times the gRPC message limit.
DECOMPRESSED_SIZE_FACTOR = 8


def payload_byte_cap(max_message_mb: int) -> int:
    """Largest decoded payload (bytes) accepted for a gRPC message limit of ``max_message_mb``."""
    return int(max_message_mb) * 2**20 * DECOMPRESSED_SIZE_FACTOR


DEFAULT_MAX_PAYLOAD_BYTES = payload_byte_cap(64)


def _lookup_class(module: str, qualname: str) -> Any:
    """Class ``module.qualname`` from ``sys.modules``, or None.

    Walks ``__dict__`` entries only: never imports and never triggers a module
    ``__getattr__`` (PEP 562 lazy imports) or descriptors.
    """
    obj: Any = sys.modules.get(module)
    if obj is None or "<locals>" in qualname:
        return None
    for part in qualname.split("."):
        namespace = getattr(obj, "__dict__", None)
        if not isinstance(namespace, Mapping):
            return None
        obj = namespace.get(part)
        if obj is None:
            return None
    return obj


def _encode_namedtuple(obj: tuple, arrays: list[np.ndarray]) -> dict:
    cls = type(obj)
    if _lookup_class(cls.__module__, cls.__qualname__) is not cls:
        raise TypeError(
            f"cannot serialize namedtuple {cls.__module__}.{cls.__qualname__}: the receiver "
            f"rebuilds it by module path, so define it at module level of an importable module"
        )
    return {
        "__namedtuple__": [cls.__module__, cls.__qualname__, list(cls._fields)],
        "items": [_encode(v, arrays) for v in obj],
    }


def _decode_namedtuple(spec: Any, items: list) -> tuple:
    if not (isinstance(spec, list) and len(spec) == 3 and isinstance(spec[0], str)
            and isinstance(spec[1], str) and isinstance(spec[2], list)):
        raise ValueError(f"malformed namedtuple spec in payload skeleton: {spec!r}")
    module, qualname, fields = spec
    cls = _lookup_class(module, qualname)
    if cls is None:
        raise ValueError(
            f"payload contains namedtuple {module}.{qualname}, but it is not loaded in this "
            f"process (payload decoding never imports modules; import {module!r} first)"
        )
    if not (isinstance(cls, type) and issubclass(cls, tuple) and list(getattr(cls, "_fields", ())) == fields):
        raise ValueError(f"payload namedtuple {module}.{qualname} does not match the local class (fields {fields})")
    return cls(*items)


def _encode(obj: Any, arrays: list[np.ndarray]) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            raise TypeError("object arrays cannot be serialized")
        arrays.append(obj)
        return {"__nd__": len(arrays) - 1}
    if isinstance(obj, torch.Tensor):
        raise TypeError("torch.Tensor in payload; convert with to_payload()/to_numpy_tree()")
    if isinstance(obj, dict):
        bad = [k for k in obj if not isinstance(k, _KEY_TYPES)]
        if bad:
            raise TypeError(f"payload dict keys must be str/int/float/bool/None, got {type(bad[0]).__name__}")
        return {"__dict__": [[_encode(k, arrays), _encode(v, arrays)] for k, v in obj.items()]}
    if is_namedtuple(obj):
        return _encode_namedtuple(obj, arrays)
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v, arrays) for v in obj]}
    if isinstance(obj, list):
        return {"__list__": [_encode(v, arrays) for v in obj]}
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    raise TypeError(f"cannot serialize {type(obj).__name__} in a payload")


def _tagged_list(obj: dict, tag: str) -> list:
    value = obj.get(tag)
    if not isinstance(value, list):
        raise ValueError(f"malformed payload skeleton: {tag} must hold a list, got {type(value).__name__}")
    return value


def _decode(obj: Any, arrays: list[np.ndarray]) -> Any:
    if isinstance(obj, list):
        raise ValueError("malformed payload skeleton: untagged list")
    if not isinstance(obj, dict):
        return obj
    if "__nd__" in obj:
        i = obj["__nd__"]
        if not (type(i) is int and 0 <= i < len(arrays)):
            raise ValueError(f"malformed payload skeleton: bad array index {i!r} ({len(arrays)} arrays)")
        return arrays[i]
    if "__dict__" in obj:
        out = {}
        for pair in _tagged_list(obj, "__dict__"):
            if not (isinstance(pair, list) and len(pair) == 2):
                raise ValueError(f"malformed payload skeleton: dict entry {pair!r}")
            key = _decode(pair[0], arrays)
            if not isinstance(key, _KEY_TYPES):
                raise ValueError(f"malformed payload skeleton: dict key of type {type(key).__name__}")
            out[key] = _decode(pair[1], arrays)
        return out
    if "__namedtuple__" in obj:
        return _decode_namedtuple(obj["__namedtuple__"], [_decode(v, arrays) for v in _tagged_list(obj, "items")])
    if "__tuple__" in obj:
        return tuple(_decode(v, arrays) for v in _tagged_list(obj, "__tuple__"))
    if "__list__" in obj:
        return [_decode(v, arrays) for v in _tagged_list(obj, "__list__")]
    raise ValueError(f"malformed payload skeleton: {obj!r}")


def _read_npy_header(f: Any) -> tuple[tuple[int, ...], np.dtype]:
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(f)
    else:
        raise ValueError(f"unsupported .npy format version {version}")
    return shape, dtype


def _load_arrays(blob: bytes, max_bytes: int) -> list[np.ndarray]:
    """Arrays ``arr_0..arr_{n-1}`` of an ``np.savez`` blob, size-checked before loading."""
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            names = zf.namelist()
            if sorted(names) != sorted(f"arr_{i}.npy" for i in range(len(names))):
                raise ValueError(f"unexpected array archive members: {names[:5]}")
            total = 0
            for name in names:
                with zf.open(name) as f:
                    shape, dtype = _read_npy_header(f)
                if dtype.hasobject:
                    raise ValueError("object arrays are not allowed in a payload")
                if not (isinstance(shape, tuple) and all(type(d) is int and 0 <= d <= sys.maxsize for d in shape)):
                    raise ValueError(f"invalid array shape {shape!r} in {name}")
                total += math.prod(shape) * dtype.itemsize
                if total > max_bytes:
                    raise ValueError(f"payload arrays declare {total} bytes, over the cap of {max_bytes} bytes")
        with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
            return [npz[f"arr_{i}"] for i in range(len(names))]
    except (zipfile.BadZipFile, EOFError, OSError, MemoryError, OverflowError) as e:
        # Whatever a malformed archive triggers surfaces as ValueError (INVALID_ARGUMENT).
        raise ValueError(f"malformed payload array archive: {type(e).__name__}: {e}") from e


def pack_payload(obj: Any, compress: bool = True) -> tuple[bytes, bool]:
    """Serialize a payload (numpy arrays + primitives in dict/list/tuple/namedtuple)."""
    arrays: list[np.ndarray] = []
    skeleton = json.dumps(_encode(obj, arrays)).encode("utf-8")
    buf = io.BytesIO()
    np.savez(buf, *arrays)
    data = _HEADER.pack(len(skeleton)) + skeleton + buf.getvalue()
    if compress:
        data = lz4.frame.compress(data)
    return data, compress


def _decompress(data: bytes, max_bytes: int) -> bytes:
    decompressor = lz4.frame.LZ4FrameDecompressor()
    try:
        out = decompressor.decompress(data, max_length=max_bytes)
    except RuntimeError as e:
        raise ValueError(f"corrupt lz4 payload: {e}") from e
    if not decompressor.eof:
        if decompressor.needs_input:
            raise ValueError("truncated lz4 payload")
        raise ValueError(f"decompressed payload exceeds the cap of {max_bytes} bytes")
    if decompressor.unused_data:
        raise ValueError("trailing bytes after the lz4 payload")
    return out


def unpack_payload(data: bytes, compressed: bool, max_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> Any:
    """Inverse of :func:`pack_payload`; raises ValueError on malformed or oversized input."""
    if compressed:
        data = _decompress(data, max_bytes)
    elif len(data) > max_bytes:
        raise ValueError(f"payload of {len(data)} bytes exceeds the cap of {max_bytes} bytes")
    if len(data) < _HEADER.size:
        raise ValueError("truncated payload header")
    (n,) = _HEADER.unpack_from(data, 0)
    if n > len(data) - _HEADER.size:
        raise ValueError("truncated payload skeleton")
    arrays = _load_arrays(data[_HEADER.size + n:], max_bytes)
    try:
        skeleton = json.loads(data[_HEADER.size:_HEADER.size + n].decode("utf-8"))
        return _decode(skeleton, arrays)
    except RecursionError as e:
        raise ValueError("payload skeleton nested too deeply") from e
    except UnicodeDecodeError as e:
        raise ValueError(f"payload skeleton is not UTF-8: {e}") from e


def serialize_state_dict(state_dict: dict[str, np.ndarray], compress: bool = True) -> tuple[bytes, bool]:
    """Serialize a numpy state_dict (``WeightPayload.state_dict``)."""
    return pack_payload(dict(state_dict), compress)


def validate_state_dict_payload(state_dict: Any) -> None:
    """Raise ValueError unless ``state_dict`` is a ``dict[str, np.ndarray]`` with numeric dtypes."""
    if not isinstance(state_dict, dict):
        raise ValueError(f"weights payload must be a dict, got {type(state_dict).__name__}")
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ValueError(f"weights payload key {key!r} is not a str")
        if not isinstance(value, np.ndarray):
            raise ValueError(f"weights payload entry {key!r} is {type(value).__name__}, not a numpy array")
        if value.dtype.kind not in "biufc":  # bool/int/uint/float/complex: what torch.from_numpy accepts
            raise ValueError(f"weights payload entry {key!r} has non-numeric dtype {value.dtype}")


def deserialize_state_dict(
    data: bytes, compressed: bool, max_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> dict[str, np.ndarray]:
    """Inverse of :func:`serialize_state_dict`; raises ValueError unless the result is well formed."""
    state_dict = unpack_payload(data, compressed, max_bytes)
    validate_state_dict_payload(state_dict)
    return state_dict


def serialize_chunk_payload(payload: dict[str, Any], compress: bool = True) -> tuple[bytes, bool]:
    """Serialize a chunk payload (``TrajectoryChunk.to_payload()``)."""
    return pack_payload(payload, compress)


_CHUNK_FLAT = ("kind", "reward", "terminal", "reset_after", "behavior_logp")
_CHUNK_TREES = ("obs", "actions")
_CHUNK_OPTIONAL_TREES = ("global_state", "action_masks")


def _tree_arrays(tree: Any, name: str) -> list[np.ndarray]:
    """Leaves of a numpy payload tree (dicts with str keys); ValueError on anything else."""
    if isinstance(tree, np.ndarray):
        return [tree]
    if isinstance(tree, dict) and tree and all(isinstance(k, str) for k in tree):
        return [leaf for value in tree.values() for leaf in _tree_arrays(value, name)]
    raise ValueError(f"chunk payload field {name!r} must be a numpy array or a dict tree of them, "
                     f"got {type(tree).__name__}")


def validate_chunk_payload(payload: Any) -> None:
    """Raise ValueError unless ``payload`` is a well-formed chunk v2 payload.

    Checks the fields, their types and the common slot dimension ``S`` of every flat array and
    tree leaf, then builds the chunk once (which checks ``initial_state`` node types) and
    checks its slot structure (``validate_slot_structure``), so a hostile or broken chunk is
    refused here instead of stopping the learner.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"chunk payload must be a dict, got {type(payload).__name__}")
    if not isinstance(payload.get("agent_id"), str) or type(payload.get("policy_version")) is not int:
        raise ValueError("chunk payload needs a str agent_id and an int policy_version")
    for name in _CHUNK_FLAT:
        value = payload.get(name)
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise ValueError(f"chunk payload field {name!r} must be a 1-D numpy array [S]")
    num_slots = payload["kind"].shape[0]
    for name in _CHUNK_FLAT:
        if payload[name].shape[0] != num_slots:
            raise ValueError(f"chunk payload field {name!r} has shape {payload[name].shape}, expected [S={num_slots}]")
    trees = [(name, payload.get(name)) for name in _CHUNK_TREES]
    trees += [(name, payload.get(name)) for name in _CHUNK_OPTIONAL_TREES if payload.get(name) is not None]
    unit_logp = payload.get("behavior_unit_logp")
    if unit_logp is not None:
        if not isinstance(unit_logp, np.ndarray) or unit_logp.ndim != 2:
            raise ValueError("chunk payload field 'behavior_unit_logp' must be None or a numpy array [S, K]")
        trees.append(("behavior_unit_logp", unit_logp))
    for name, tree in trees:
        if tree is None:
            raise ValueError(f"chunk payload field {name!r} is missing")
        for leaf in _tree_arrays(tree, name):
            if leaf.ndim == 0 or leaf.shape[0] != num_slots:
                raise ValueError(f"chunk payload field {name!r} has a leaf of shape {leaf.shape}, "
                                 f"expected S={num_slots} first")
    try:
        chunk = TrajectoryChunk.from_payload(payload)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed chunk payload: {type(e).__name__}: {e}") from e
    validate_slot_structure(chunk)


def deserialize_chunk_payload(
    data: bytes, compressed: bool, max_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    """Inverse of :func:`serialize_chunk_payload` (not validated: the sender's
    ``agent_id`` / version travel in the proto; see :func:`validate_chunk_payload`)."""
    return unpack_payload(data, compressed, max_bytes)


def serialize_chunk(chunk: TrajectoryChunk, compress: bool = True) -> tuple[bytes, bool]:
    """Serialize a TrajectoryChunk via its numpy payload."""
    return serialize_chunk_payload(chunk.to_payload(), compress)


def deserialize_chunk(
    agent_id: str,
    policy_version: int,
    data: bytes,
    compressed: bool,
) -> TrajectoryChunk:
    """Deserialize bytes from :func:`serialize_chunk` into a TrajectoryChunk."""
    payload = deserialize_chunk_payload(data, compressed)
    payload["agent_id"] = agent_id
    payload["policy_version"] = int(policy_version)
    return TrajectoryChunk.from_payload(payload)
