"""The array frame (SP5 spec block 4.9): trees of numpy arrays + JSON metadata in one buffer, decoded without pickle.

Every chunk, weights set, snapshot, frozen / teacher / init weights, checkpoint and recovery point crosses a process
or a machine boundary as one frame::

    offset 0   4 B  magic b"CSTF"
           4   2 B  uint16 LE format version (1)
           6   2 B  uint16 LE flags (bit 0: the data region is one lz4 frame)
           8   8 B  uint64 LE header length H (<= MAX_HEADER_BYTES)
          16   H B  UTF-8 JSON header
      align64(16 + H)  data region: raw little-endian buffers, each at a 64-aligned offset (or one lz4 frame of it)

Header: ``{"v": 1, "kind", "meta", "tree", "leaves", "data_bytes", "lz4"}``. ``tree`` mirrors the payload:
``{"t": "nd", "i": k}`` (leaf k), ``{"t": "dict", "k": [keys], "v": [nodes]}`` (keys str / int / float / bool /
null), ``{"t": "list" | "tuple", "v": [nodes]}``, ``{"t": "nt", "cls": "module:Qualname", "f": [fields], "v":
[nodes]}`` (a namedtuple) and ``{"t": "py", "v": scalar}`` (None / bool / int / float / str). ``leaves[k]`` =
``{"dtype", "shape", "offset", "nbytes"}`` with offsets relative to the data region; ``data_bytes`` is the raw size of
the data region and ``lz4`` is null or ``{"raw_bytes": data_bytes}``.

Decoding treats the bytes as hostile and raises :class:`FrameError` (a ``ValueError`` with a short ``reason``)
before it allocates anything large: the header length, the declared sizes against the caller's ``cap``, each leaf's
``nbytes == prod(shape) * itemsize``, 64-aligned non-overlapping offsets inside the data region, a dtype whitelist
(bool and the little-endian integer / float types; never ``object``), the leaf count and the nesting depth, every
leaf referenced exactly once, no NaN / Infinity in the header, and a namedtuple only as a class that is already
loaded with the same fields (looked up in ``sys.modules`` through ``__dict__``; never imported). lz4 decompression is
bounded by ``data_bytes``. Arrays are zero-copy ``np.frombuffer`` views of the buffer (read-only for ``bytes``)
unless ``copy=True`` or the view would be misaligned.

lz4 is a flag of the frame; :class:`Lz4Probe` decides it per (sender, agent): compress the first probe frames,
keep compressing when the ratio is at least ``LZ4_MIN_RATIO``, probe again every ``LZ4_REPROBE_SEC``.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

import lz4.frame
import numpy as np

FRAME_MAGIC = b"CSTF"
FRAME_VERSION = 1
FRAME_ALIGN = 64
FLAG_LZ4 = 1
MAX_HEADER_BYTES = 1 << 20
MAX_LEAVES = 100_000
MAX_DEPTH = 64
MAX_NDIM = 32
DECOMPRESSED_SIZE_FACTOR = 8             # payload_byte_cap(mb) = 8 x mb MiB (SP1 rule)
LZ4_MIN_RATIO = 1.3
LZ4_PROBE_FRAMES = 4
LZ4_REPROBE_SEC = 600.0
FRAME_KINDS = ("tree", "chunk", "weights", "checkpoint")

_PREFIX = struct.Struct("<4sHHQ")         # magic, version, flags, header length: 16 bytes
_HEADER_KEYS = {"v", "kind", "meta", "tree", "leaves", "data_bytes", "lz4"}
_LEAF_KEYS = {"dtype", "shape", "offset", "nbytes"}
_KEY_TYPES = (str, int, float, bool, type(None))
# Little-endian (or byte-order-free) numeric dtypes and bool; never object, str, bytes, void, datetime or complex.
DTYPES = frozenset({"|b1", "|i1", "|u1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f2", "<f4", "<f8"})
_DTYPE_OBJECTS = {name: np.dtype(name) for name in DTYPES}


class FrameError(ValueError):
    """A malformed or oversized frame. ``reason`` is a short machine-readable tag (counters, tests)."""

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(f"{reason}: {message}" if message else reason)
        self.reason = reason


def payload_byte_cap(max_message_mb: int) -> int:
    """Largest decoded payload (bytes) accepted for a message limit of ``max_message_mb`` MiB."""
    return int(max_message_mb) * 2**20 * DECOMPRESSED_SIZE_FACTOR


def _align(n: int) -> int:
    return (n + FRAME_ALIGN - 1) // FRAME_ALIGN * FRAME_ALIGN


def lookup_class(module: str, qualname: str) -> Any:
    """Class ``module.qualname`` from ``sys.modules``, or None. Walks ``__dict__`` only: never imports and never
    triggers a module ``__getattr__`` (PEP 562 lazy imports) or a descriptor."""
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


def _is_namedtuple(x: Any) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_fields")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _encode_node(obj: Any, leaves: list[np.ndarray], depth: int) -> dict:
    if depth > MAX_DEPTH:
        raise TypeError(f"payload nested deeper than {MAX_DEPTH} levels")
    if isinstance(obj, np.ndarray):
        dtype = obj.dtype
        if dtype.byteorder == ">":
            obj = obj.astype(dtype.newbyteorder("<"))
            dtype = obj.dtype
        if dtype.str not in DTYPES:
            raise TypeError(f"arrays of dtype {dtype} cannot be framed (bool and numeric little-endian only)")
        if obj.ndim > MAX_NDIM:
            raise TypeError(f"arrays with more than {MAX_NDIM} dimensions cannot be framed")
        leaves.append(obj)
        return {"t": "nd", "i": len(leaves) - 1}
    if isinstance(obj, dict):
        bad = [k for k in obj if not isinstance(k, _KEY_TYPES)]
        if bad:
            raise TypeError(f"payload dict keys must be str/int/float/bool/None, got {type(bad[0]).__name__}")
        return {"t": "dict", "k": list(obj), "v": [_encode_node(v, leaves, depth + 1) for v in obj.values()]}
    if _is_namedtuple(obj):
        cls = type(obj)
        if lookup_class(cls.__module__, cls.__qualname__) is not cls:
            raise TypeError(f"cannot frame namedtuple {cls.__module__}.{cls.__qualname__}: the receiver rebuilds it "
                            f"by module path, so define it at module level of an importable module")
        return {"t": "nt", "cls": f"{cls.__module__}:{cls.__qualname__}", "f": list(cls._fields),
                "v": [_encode_node(v, leaves, depth + 1) for v in obj]}
    if isinstance(obj, tuple):
        return {"t": "tuple", "v": [_encode_node(v, leaves, depth + 1) for v in obj]}
    if isinstance(obj, list):
        return {"t": "list", "v": [_encode_node(v, leaves, depth + 1) for v in obj]}
    if isinstance(obj, np.generic):
        return {"t": "py", "v": obj.item()}
    if obj is None or isinstance(obj, bool | int | float | str):
        return {"t": "py", "v": obj}
    raise TypeError(f"cannot frame {type(obj).__name__} (numpy arrays, primitives, dict / list / tuple / namedtuple "
                    f"only; convert tensors with to_payload() / state_dict_to_numpy())")


def encode_frame(tree: Any, *, kind: str = "tree", meta: Mapping[str, Any] | None = None,
                 compress: bool = False) -> bytes:
    """One frame of ``tree`` (numpy arrays + primitives in dict / list / tuple / namedtuple) with JSON ``meta``.

    Raises TypeError for anything that cannot be framed (a tensor, an object array, a local namedtuple, bytes,
    NaN in ``meta`` or in a scalar)."""
    if kind not in FRAME_KINDS:
        raise ValueError(f"frame kind must be one of {FRAME_KINDS}, got {kind!r}")
    leaves: list[np.ndarray] = []
    node = _encode_node(tree, leaves, 0)
    descs = []
    parts: list[Any] = []                         # the data region: padding and raw leaf bytes
    offset = 0
    for leaf in leaves:
        arr = leaf if leaf.flags.c_contiguous else np.array(leaf, order="C")   # keeps 0-d leaves 0-d
        start = _align(offset)
        if start > offset:
            parts.append(bytes(start - offset))
        descs.append({"dtype": arr.dtype.str, "shape": list(arr.shape), "offset": start, "nbytes": int(arr.nbytes)})
        if arr.nbytes:
            parts.append(arr.reshape(-1).view(np.uint8).data)
        offset = start + int(arr.nbytes)
    data_bytes = offset
    header = {"v": FRAME_VERSION, "kind": kind, "meta": dict(meta or {}), "tree": node, "leaves": descs,
              "data_bytes": data_bytes, "lz4": {"raw_bytes": data_bytes} if compress else None}
    try:
        text = json.dumps(header, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except ValueError as e:                       # NaN / Infinity
        raise TypeError(f"frame header holds a non-finite number: {e}") from None
    if len(text) > MAX_HEADER_BYTES:
        raise TypeError(f"frame header of {len(text)} bytes exceeds {MAX_HEADER_BYTES} bytes (too many leaves?)")
    if compress:
        parts = [lz4.frame.compress(b"".join(parts))]
    head = _PREFIX.pack(FRAME_MAGIC, FRAME_VERSION, FLAG_LZ4 if compress else 0, len(text)) + text
    return b"".join([head, bytes(_align(len(head)) - len(head)), *parts])


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _reject_constant(name: str) -> Any:
    raise FrameError("header_json", f"non-finite number {name} in the header")


def _finite_float(text: str) -> float:
    """json's float parser, refusing literals that overflow to infinity (``1e999``)."""
    value = float(text)
    if not math.isfinite(value):
        raise FrameError("header_json", f"number {text[:40]} in the header is not finite")
    return value


def _byte_view(buf: bytes | bytearray | memoryview) -> memoryview:
    """A flat, C-contiguous byte view of ``buf``; FrameError("buffer") for anything else."""
    try:
        mv = memoryview(buf)
        if not mv.c_contiguous:
            raise FrameError("buffer", "the frame buffer is not C-contiguous")
        return mv if mv.format == "B" and mv.ndim == 1 else mv.cast("B")
    except (TypeError, BufferError, ValueError) as e:
        raise FrameError("buffer", f"{type(e).__name__}: {e}") from None


def _read_header(mv: memoryview) -> tuple[dict, int, int]:
    """``(header, flags, data start)`` of a frame; checks the prefix and the header's top-level shape."""
    if len(mv) < _PREFIX.size:
        raise FrameError("truncated", f"{len(mv)} bytes, shorter than the {_PREFIX.size}-byte prefix")
    magic, version, flags, hlen = _PREFIX.unpack_from(mv, 0)
    if magic != FRAME_MAGIC:
        raise FrameError("magic", f"not a Colosseum frame (magic {bytes(magic)!r})")
    if version != FRAME_VERSION:
        raise FrameError("version", f"frame format version {version}; this code reads version {FRAME_VERSION}")
    if flags & ~FLAG_LZ4:
        raise FrameError("flags", f"unknown flags {flags:#06x}")
    if hlen > MAX_HEADER_BYTES:
        raise FrameError("header_too_large", f"header of {hlen} bytes exceeds {MAX_HEADER_BYTES} bytes")
    if _PREFIX.size + hlen > len(mv):
        raise FrameError("truncated", f"header of {hlen} bytes runs past the frame's {len(mv)} bytes")
    try:
        header = json.loads(str(mv[_PREFIX.size:_PREFIX.size + hlen], "utf-8"), parse_constant=_reject_constant,
                            parse_float=_finite_float)
    except FrameError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise FrameError("header_json", f"{type(e).__name__}: {e}") from None
    if not isinstance(header, dict) or set(header) != _HEADER_KEYS:
        keys = sorted(header) if isinstance(header, dict) else type(header).__name__
        raise FrameError("header", f"header keys {keys}, expected {sorted(_HEADER_KEYS)}")
    if header["v"] != FRAME_VERSION or type(header["v"]) is not int:
        raise FrameError("version", f"header version {header['v']!r}")
    if header["kind"] not in FRAME_KINDS:
        raise FrameError("kind", f"unknown frame kind {header['kind']!r}")
    if not isinstance(header["meta"], dict):
        raise FrameError("header", "meta must be an object")
    return header, flags, _align(_PREFIX.size + hlen)


def _check_int(value: Any, what: str) -> int:
    if type(value) is not int or value < 0:
        raise FrameError("header", f"{what} must be a non-negative integer, got {value!r}")
    return value


def _check_leaves(header: dict, cap: int) -> list[tuple[np.dtype, tuple[int, ...], int, int]]:
    data_bytes = _check_int(header["data_bytes"], "data_bytes")
    if data_bytes > cap:
        raise FrameError("too_large", f"data region of {data_bytes} bytes exceeds the cap of {cap} bytes")
    lz4_info = header["lz4"]
    if lz4_info is not None and (not isinstance(lz4_info, dict) or set(lz4_info) != {"raw_bytes"}
                                 or lz4_info["raw_bytes"] != data_bytes or type(lz4_info["raw_bytes"]) is not int):
        raise FrameError("header", f"malformed lz4 entry {lz4_info!r}")
    raw = header["leaves"]
    if not isinstance(raw, list):
        raise FrameError("header", "leaves must be a list")
    if len(raw) > MAX_LEAVES:
        raise FrameError("too_many_leaves", f"{len(raw)} leaves, more than {MAX_LEAVES}")
    out = []
    for k, desc in enumerate(raw):
        if type(desc) is not dict or desc.keys() != _LEAF_KEYS:
            raise FrameError("header", f"leaf {k}: malformed descriptor {desc!r}"[:200])
        name, shape, offset, nbytes = desc["dtype"], desc["shape"], desc["offset"], desc["nbytes"]
        dtype = _DTYPE_OBJECTS.get(name) if type(name) is str else None
        if dtype is None:
            raise FrameError("dtype", f"leaf {k}: dtype {name!r} is not allowed")
        if type(shape) is not list or len(shape) > MAX_NDIM \
                or not all(type(d) is int and d >= 0 for d in shape):
            raise FrameError("shape", f"leaf {k}: shape must be a list of at most {MAX_NDIM} non-negative "
                                      f"integers, got {shape!r}"[:200])
        if type(offset) is not int or type(nbytes) is not int or offset < 0 or nbytes < 0:
            raise FrameError("header", f"leaf {k}: offset and nbytes must be non-negative integers")
        if nbytes != math.prod(shape) * dtype.itemsize:
            raise FrameError("nbytes", f"leaf {k}: nbytes {nbytes} != prod({shape}) x {dtype.itemsize}")
        if math.prod(d for d in shape if d) * dtype.itemsize > cap:      # an empty leaf with huge other dims
            raise FrameError("shape", f"leaf {k}: shape {shape} exceeds the cap of {cap} bytes"[:200])
        if offset % FRAME_ALIGN:
            raise FrameError("offset", f"leaf {k}: offset {offset} is not {FRAME_ALIGN}-aligned")
        if offset + nbytes > data_bytes:
            raise FrameError("offset", f"leaf {k}: bytes [{offset}, {offset + nbytes}) run past the data region of "
                                       f"{data_bytes} bytes")
        out.append((dtype, tuple(shape), offset, nbytes))
    spans = sorted((o, o + n) for _, _, o, n in out if n)
    for (_, prev_end), (start, _) in zip(spans, spans[1:]):
        if start < prev_end:
            raise FrameError("overlap", f"leaf bytes overlap at offset {start}")
    return out


def _decompress(data: memoryview, raw_bytes: int) -> bytes:
    decompressor = lz4.frame.LZ4FrameDecompressor()
    try:
        out = decompressor.decompress(data, max_length=raw_bytes)
    except RuntimeError as e:
        raise FrameError("lz4", f"corrupt lz4 data: {e}") from None
    if not decompressor.eof:
        if decompressor.needs_input:
            raise FrameError("lz4", "truncated lz4 data")
        raise FrameError("lz4", f"lz4 data decompresses to more than the declared {raw_bytes} bytes")
    if decompressor.unused_data or len(out) != raw_bytes:
        raise FrameError("lz4", f"lz4 data decompresses to {len(out)} bytes (+{len(decompressor.unused_data)} "
                                f"trailing), declared {raw_bytes}")
    return out


class _Decoder:
    def __init__(self, arrays: list[np.ndarray]) -> None:
        self.arrays = arrays
        self.used = [False] * len(arrays)

    def node(self, obj: Any, depth: int) -> Any:
        if depth > MAX_DEPTH:
            raise FrameError("depth", f"tree nested deeper than {MAX_DEPTH} levels")
        tag = obj.get("t") if type(obj) is dict else None
        if tag == "nd" and len(obj) == 2:
            i = obj.get("i")
            if type(i) is not int or not 0 <= i < len(self.arrays) or self.used[i]:
                raise FrameError("leaf_ref", f"bad or repeated leaf reference {i!r}")
            self.used[i] = True
            return self.arrays[i]
        if tag == "dict" and len(obj) == 3:
            keys, values = obj.get("k"), obj.get("v")
            if type(keys) is list and type(values) is list and len(keys) == len(values):
                if not all(isinstance(k, _KEY_TYPES) for k in keys):
                    raise FrameError("dict_key", "dict keys must be str/int/float/bool/null")
                out = {k: self.node(v, depth + 1) for k, v in zip(keys, values)}
                if len(out) != len(keys):
                    raise FrameError("dict_key", "repeated dict key")
                return out
        elif tag == "py" and len(obj) == 2 and "v" in obj:
            if not isinstance(obj["v"], _KEY_TYPES):
                raise FrameError("node", f"scalar node holds a {type(obj['v']).__name__}")
            return obj["v"]
        elif tag in ("list", "tuple") and len(obj) == 2 and type(obj.get("v")) is list:
            items = [self.node(v, depth + 1) for v in obj["v"]]
            return items if tag == "list" else tuple(items)
        elif tag == "nt" and len(obj) == 4 and type(obj.get("cls")) is str and type(obj.get("f")) is list \
                and type(obj.get("v")) is list:
            module, _, qualname = obj["cls"].partition(":")
            cls = lookup_class(module, qualname)
            if cls is None:
                raise FrameError("namedtuple", f"namedtuple {module}.{qualname} is not loaded in this process "
                                               f"(frames never import modules; import {module!r} first)")
            if not (isinstance(cls, type) and issubclass(cls, tuple) and list(getattr(cls, "_fields", ())) == obj["f"]
                    and len(obj["v"]) == len(obj["f"])):
                raise FrameError("namedtuple", f"namedtuple {module}.{qualname} does not match the local class")
            return cls(*(self.node(v, depth + 1) for v in obj["v"]))
        raise FrameError("node", f"malformed tree node {obj!r}"[:200])


def decode_frame(buf: bytes | bytearray | memoryview, *, cap: int, copy: bool = False,
                 kind: str | None = None) -> tuple[Any, dict]:
    """``(tree, meta)`` of a frame. Arrays are views of ``buf`` (read-only for ``bytes``), or with ``copy`` writable
    views of one private copy of the data region (torch then shares them without another copy); raises
    :class:`FrameError` for anything malformed or larger than ``cap`` bytes, before allocating it, and (with ``kind``)
    for a frame of another kind."""
    mv = _byte_view(buf)
    header, flags, start = _read_header(mv)
    if kind is not None and header["kind"] != kind:
        raise FrameError("kind", f"expected a {kind} frame, got {header['kind']!r}")
    leaves = _check_leaves(header, int(cap))
    data_bytes = header["data_bytes"]
    if bool(flags & FLAG_LZ4) != (header["lz4"] is not None):
        raise FrameError("flags", "the lz4 flag and the header disagree")
    if start > len(mv):
        raise FrameError("truncated", "the frame ends inside the header padding")
    region: Any = mv[start:]
    if header["lz4"] is not None:
        region = _decompress(region, data_bytes)
    elif len(region) != data_bytes:
        raise FrameError("size", f"data region of {len(region)} bytes, declared {data_bytes}")
    if copy:
        region = bytearray(region)                 # one private copy: every array a writable view of it
    arrays = []
    for k, (dtype, shape, offset, nbytes) in enumerate(leaves):
        try:
            if nbytes == 0:
                arr = np.zeros(shape, dtype)
            else:
                arr = np.ndarray(shape, dtype, buffer=region, offset=offset)
                if not arr.flags.aligned:
                    arr = arr.copy()
        except (ValueError, TypeError) as e:          # numpy refuses some shapes only when it builds the array
            raise FrameError("shape", f"leaf {k}: shape {list(shape)}: {e}"[:200]) from None
        arrays.append(arr)
    decoder = _Decoder(arrays)
    tree = decoder.node(header["tree"], 0)
    if not all(decoder.used):
        raise FrameError("leaf_ref", f"leaf {decoder.used.index(False)} is never referenced")
    return tree, header["meta"]


def peek_meta(buf: bytes | bytearray | memoryview) -> tuple[str, dict]:
    """``(kind, meta)`` from the header only (routing): no arrays, no size cap beyond the header's."""
    mv = _byte_view(buf)
    header, _, _ = _read_header(mv)
    return header["kind"], header["meta"]


def frame_sizes(buf: bytes | bytearray | memoryview) -> tuple[int, int]:
    """``(raw data bytes, stored data bytes)`` of a well-formed frame (for :meth:`Lz4Probe.observe`)."""
    mv = _byte_view(buf)
    header, _, start = _read_header(mv)
    return _check_int(header["data_bytes"], "data_bytes"), max(0, len(mv) - start)


class Lz4Probe:
    """Adaptive lz4 for one (sender, agent) (spec 4.9).

    The first ``probe_frames`` frames are sent compressed and measured; compression stays on iff their total raw
    size is at least ``min_ratio`` times their total compressed size. Every ``reprobe_sec`` (sender's monotonic
    clock) the probe repeats, so a stream whose content changes is re-decided. Weights never use it (float32 does not
    compress: spec 4.9)."""

    def __init__(self, *, min_ratio: float = LZ4_MIN_RATIO, probe_frames: int = LZ4_PROBE_FRAMES,
                 reprobe_sec: float = LZ4_REPROBE_SEC, clock: Callable[[], float] = time.monotonic) -> None:
        if probe_frames < 1:
            raise ValueError(f"probe_frames must be >= 1, got {probe_frames}")
        self._min_ratio = float(min_ratio)
        self._probe_frames = int(probe_frames)
        self._reprobe_sec = float(reprobe_sec)
        self._clock = clock
        self._left = self._probe_frames          # probe frames still to measure
        self._raw = 0
        self._stored = 0
        self._enabled = False
        self._next_probe: float | None = None

    @property
    def enabled(self) -> bool:
        """The decision of the last finished probe (False before the first one ends)."""
        return self._enabled

    def compress(self) -> bool:
        """Whether the next frame should be compressed."""
        if self._left == 0 and self._next_probe is not None and self._clock() >= self._next_probe:
            self._left, self._raw, self._stored = self._probe_frames, 0, 0
        return True if self._left > 0 else self._enabled

    def observe(self, raw_bytes: int, compressed_bytes: int | None) -> None:
        """Sizes of a sent frame's data region: ``compressed_bytes`` None = it was sent uncompressed."""
        if compressed_bytes is None or self._left == 0:
            return
        self._raw += int(raw_bytes)
        self._stored += int(compressed_bytes)
        self._left -= 1
        if self._left == 0:
            self._enabled = self._raw >= self._min_ratio * self._stored
            self._next_probe = self._clock() + self._reprobe_sec
