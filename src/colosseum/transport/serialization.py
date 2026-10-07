"""Serialization for the gRPC transport: numpy payloads <-> bytes.

SP1 wire format (replaced in SP5):
    8-byte little-endian length N | N bytes of JSON skeleton | np.savez archive
The skeleton mirrors the payload structure; every numpy array is replaced by
``{"__nd__": i}`` (index into the archive). Dicts, lists, tuples and
namedtuples are tagged so that int dict keys, tuples and namedtuple types
survive JSON. Arrays are loaded with ``allow_pickle=False`` and no ``pickle``
is involved, so bytes from the network cannot execute code: a namedtuple is
rebuilt only from a class that is already loaded in the receiving process
(looked up by module and qualified name, never imported). The whole blob is
optionally lz4-compressed.
"""

from __future__ import annotations

import io
import json
import struct
import sys
from typing import Any

import lz4.frame
import numpy as np
import torch

from colosseum.core.types import TrajectoryChunk

_HEADER = struct.Struct("<Q")


def _lookup_class(module: str, qualname: str) -> Any:
    """Class ``module.qualname`` from ``sys.modules`` (never imports), or None."""
    obj: Any = sys.modules.get(module)
    if obj is None or "<locals>" in qualname:
        return None
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
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


def _decode_namedtuple(spec: list, items: list) -> tuple:
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
        return {"__dict__": [[_encode(k, arrays), _encode(v, arrays)] for k, v in obj.items()]}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
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


def _decode(obj: Any, arrays: list[np.ndarray]) -> Any:
    if isinstance(obj, dict):
        if "__nd__" in obj:
            return arrays[obj["__nd__"]]
        if "__dict__" in obj:
            return {_decode(k, arrays): _decode(v, arrays) for k, v in obj["__dict__"]}
        if "__namedtuple__" in obj:
            return _decode_namedtuple(obj["__namedtuple__"], [_decode(v, arrays) for v in obj["items"]])
        if "__tuple__" in obj:
            return tuple(_decode(v, arrays) for v in obj["__tuple__"])
        if "__list__" in obj:
            return [_decode(v, arrays) for v in obj["__list__"]]
        raise ValueError(f"malformed payload skeleton: {obj!r}")
    return obj


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


def unpack_payload(data: bytes, compressed: bool) -> Any:
    """Inverse of :func:`pack_payload`."""
    if compressed:
        data = lz4.frame.decompress(data)
    (n,) = _HEADER.unpack_from(data, 0)
    skeleton = json.loads(data[_HEADER.size:_HEADER.size + n].decode("utf-8"))
    with np.load(io.BytesIO(data[_HEADER.size + n:]), allow_pickle=False) as npz:
        arrays = [npz[f"arr_{i}"] for i in range(len(npz.files))]
    return _decode(skeleton, arrays)


def serialize_state_dict(state_dict: dict[str, np.ndarray], compress: bool = True) -> tuple[bytes, bool]:
    """Serialize a numpy state_dict (``WeightPayload.state_dict``)."""
    return pack_payload(dict(state_dict), compress)


def deserialize_state_dict(data: bytes, compressed: bool) -> dict[str, np.ndarray]:
    """Inverse of :func:`serialize_state_dict`."""
    return unpack_payload(data, compressed)


def serialize_chunk_payload(payload: dict[str, Any], compress: bool = True) -> tuple[bytes, bool]:
    """Serialize a chunk payload (``TrajectoryChunk.to_payload()``)."""
    return pack_payload(payload, compress)


def deserialize_chunk_payload(data: bytes, compressed: bool) -> dict[str, Any]:
    """Inverse of :func:`serialize_chunk_payload`."""
    return unpack_payload(data, compressed)


def serialize_chunk(chunk: TrajectoryChunk, compress: bool = True) -> tuple[bytes, bool]:
    """Serialize a TrajectoryChunk via its numpy payload."""
    return serialize_chunk_payload(chunk.to_payload(), compress)


def deserialize_chunk(
    agent_id: str,
    behavior_policy_version: int,
    data: bytes,
    compressed: bool,
) -> TrajectoryChunk:
    """Deserialize bytes from :func:`serialize_chunk` into a TrajectoryChunk."""
    payload = deserialize_chunk_payload(data, compressed)
    payload["agent_id"] = agent_id
    payload["behavior_policy_version"] = int(behavior_policy_version)
    return TrajectoryChunk.from_payload(payload)
