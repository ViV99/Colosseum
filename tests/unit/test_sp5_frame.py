"""The array frame (SP5 T1.1, spec block 4.9 and §6 "Формат кадра"): round trips of every payload node type, the byte
layout, zero-copy views, lz4, hostile input refused before allocation (ported from test_sp2_serialization and
extended), peek_meta, frame_sizes and the adaptive lz4 probe."""
from __future__ import annotations

import json
import struct
import sys
import types
from collections import namedtuple

import lz4.frame
import numpy as np
import pytest
import torch

from colosseum.net.frame import (
    DTYPES,
    FRAME_ALIGN,
    FRAME_MAGIC,
    MAX_DEPTH,
    MAX_HEADER_BYTES,
    MAX_LEAVES,
    FrameError,
    Lz4Probe,
    decode_frame,
    encode_frame,
    frame_sizes,
    payload_byte_cap,
    peek_meta,
)

HC = namedtuple("HC", ["h", "c"])
CAP = payload_byte_cap(64)


def _header(buf: bytes) -> tuple[dict, int]:
    (hlen,) = struct.unpack_from("<Q", buf, 8)
    return json.loads(buf[16:16 + hlen]), hlen


def _forge(header: dict, region: bytes = b"", flags: int = 0, version: int = 1) -> bytes:
    """A frame with a hand-written header (and data region)."""
    text = json.dumps(header).encode()
    head = struct.pack("<4sHHQ", FRAME_MAGIC, version, flags, len(text)) + text
    pad = (-len(head)) % FRAME_ALIGN
    return head + bytes(pad) + region


def _base(**changes) -> dict:
    header = {"v": 1, "kind": "tree", "meta": {}, "tree": {"t": "nd", "i": 0},
              "leaves": [{"dtype": "<f4", "shape": [4], "offset": 0, "nbytes": 16}], "data_bytes": 16, "lz4": None}
    header.update(changes)
    return header


@pytest.mark.parametrize("compress", [False, True])
def test_every_node_type_round_trips_with_its_dtype(compress):
    tree = {
        "a": np.arange(6, dtype=np.int64).reshape(2, 3),
        3: [np.float32(1.5), None, True, "s", 7, 2.5],
        "t": (np.zeros(2, dtype=bool), 7),
        "n": {"x": np.ones((1, 1, 4), np.float32), "u8": np.arange(5, dtype=np.uint8)},
        "hc": HC(np.full((1, 2), 3.0, np.float16), [np.zeros((0, 3), np.int32)]),
        1.5: np.array(4.0),                  # a 0-d leaf
        None: np.zeros(3, np.float64),
        False: np.array([-1], np.int8),
    }
    back, meta = decode_frame(encode_frame(tree, meta={"x": 1, "s": "y"}, compress=compress), cap=CAP)
    assert meta == {"x": 1, "s": "y"}
    assert list(back) == list(tree)
    assert np.array_equal(back["a"], tree["a"]) and back["a"].dtype == np.int64
    assert back[3] == [1.5, None, True, "s", 7, 2.5]
    assert isinstance(back["t"], tuple) and back["t"][1] == 7 and back["t"][0].dtype == np.bool_
    assert back["n"]["x"].shape == (1, 1, 4) and back["n"]["u8"].tolist() == [0, 1, 2, 3, 4]
    assert type(back["hc"]) is HC and back["hc"].h.dtype == np.float16 and back["hc"].c[0].shape == (0, 3)
    assert back[1.5].shape == () and float(back[1.5]) == 4.0
    assert back[None].dtype == np.float64 and back[False].tolist() == [-1]


def test_the_byte_layout_is_the_contracts():
    buf = encode_frame({"w": np.arange(3, dtype=np.float32), "b": np.ones(5, np.int8)}, kind="weights",
                       meta={"agent_id": "a"})
    magic, version, flags, hlen = struct.unpack_from("<4sHHQ", buf, 0)
    assert (magic, version, flags) == (b"CSTF", 1, 0)
    header, _ = _header(buf)
    assert set(header) == {"v", "kind", "meta", "tree", "leaves", "data_bytes", "lz4"}
    assert header["kind"] == "weights" and header["lz4"] is None
    assert [leaf["offset"] for leaf in header["leaves"]] == [0, 64]           # 64-aligned
    assert [leaf["dtype"] for leaf in header["leaves"]] == ["<f4", "|i1"]
    start = (16 + hlen + FRAME_ALIGN - 1) // FRAME_ALIGN * FRAME_ALIGN
    assert len(buf) == start + header["data_bytes"] == start + 64 + 5
    assert np.frombuffer(buf, np.float32, 3, start).tolist() == [0.0, 1.0, 2.0]


def test_arrays_are_views_of_the_buffer_unless_copied():
    buf = bytearray(encode_frame({"x": np.arange(16, dtype=np.float32)}))
    view, _ = decode_frame(buf, cap=CAP)
    assert view["x"].base is not None and view["x"].flags.writeable       # a bytearray: writable views
    buf[-4:] = np.float32(99.0).tobytes()
    assert view["x"][-1] == 99.0                                           # shares the buffer
    copied, _ = decode_frame(buf, cap=CAP, copy=True)
    buf[-4:] = np.float32(5.0).tobytes()
    assert copied["x"][-1] == 99.0
    read_only, _ = decode_frame(bytes(buf), cap=CAP)
    assert not read_only["x"].flags.writeable


def test_big_endian_arrays_are_written_little_endian():
    back, _ = decode_frame(encode_frame(np.arange(3, dtype=">i4")), cap=CAP)
    assert back.dtype == np.dtype("<i4") and back.tolist() == [0, 1, 2]


@pytest.mark.parametrize("value, message", [
    (torch.zeros(1), "Tensor"),
    (np.array([object()], dtype=object), "dtype"),
    (np.array(["a"]), "dtype"),
    (np.zeros(1, np.complex64), "dtype"),
    (b"bytes", "bytes"),
    ({(1, 2): 3}, "keys"),
    ({"x": float("nan")}, "non-finite"),
])
def test_encoding_refuses_what_cannot_be_framed(value, message):
    with pytest.raises(TypeError, match=message):
        encode_frame(value)


def test_a_local_namedtuple_cannot_be_framed():
    Local = namedtuple("Local", ["h"])
    with pytest.raises(TypeError, match=r"namedtuple .*Local.* module level"):
        encode_frame({"s": Local(np.zeros(1))})


def test_peek_meta_reads_only_the_header_and_frame_sizes_report_the_region():
    plain = encode_frame({"x": np.zeros(1000, np.float32)}, kind="chunk", meta={"agent_id": "a"})
    packed = encode_frame({"x": np.zeros(1000, np.float32)}, kind="chunk", meta={"agent_id": "a"}, compress=True)
    assert peek_meta(plain) == peek_meta(packed) == ("chunk", {"agent_id": "a"})
    _, hlen = _header(plain)
    assert peek_meta(plain[:16 + hlen]) == ("chunk", {"agent_id": "a"})    # the body is not read
    assert frame_sizes(plain) == (4000, 4000)
    raw, stored = frame_sizes(packed)
    assert raw == 4000 and stored < 200


# ---------------------------------------------------------------------------
# Hostile input (spec 4.9; SP2's serialization tests ported and extended)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("buf, reason", [
    (b"", "truncated"),
    (b"XXXX" + bytes(12), "magic"),
    (struct.pack("<4sHHQ", FRAME_MAGIC, 2, 0, 0), "version"),
    (struct.pack("<4sHHQ", FRAME_MAGIC, 1, 6, 0), "flags"),
    (struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, MAX_HEADER_BYTES + 1), "header_too_large"),
    (struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, 100) + b"{}", "truncated"),
    (struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, 2) + b"\xff\xfe", "header_json"),
    (struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, 3) + b"[1]", "header"),
])
def test_malformed_prefixes_and_headers_are_refused(buf, reason):
    with pytest.raises(FrameError) as info:
        decode_frame(buf, cap=CAP)
    assert info.value.reason == reason and isinstance(info.value, ValueError)


def test_a_huge_header_length_is_refused_before_reading_it():
    buf = struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, 2**63)
    with pytest.raises(FrameError) as info:
        decode_frame(buf, cap=CAP)
    assert info.value.reason == "header_too_large"


@pytest.mark.parametrize("changes, reason", [
    ({"data_bytes": 10**12, "leaves": [{"dtype": "<f4", "shape": [250 * 10**9], "offset": 0, "nbytes": 10**12}]},
     "too_large"),
    ({"leaves": [{"dtype": "<f4", "shape": [-4], "offset": 0, "nbytes": -16}]}, "shape"),
    ({"leaves": [{"dtype": "<f4", "shape": [True], "offset": 0, "nbytes": 4}]}, "shape"),
    ({"leaves": [{"dtype": "<f4", "shape": [4], "offset": 0, "nbytes": 12}]}, "nbytes"),
    ({"leaves": [{"dtype": "<f4", "shape": [4], "offset": 8, "nbytes": 16}]}, "offset"),
    ({"leaves": [{"dtype": "<f4", "shape": [4], "offset": 64, "nbytes": 16}]}, "offset"),
    ({"leaves": [{"dtype": "|O", "shape": [4], "offset": 0, "nbytes": 32}]}, "dtype"),
    ({"leaves": [{"dtype": ">f4", "shape": [4], "offset": 0, "nbytes": 16}]}, "dtype"),
    ({"leaves": [{"dtype": "<U1", "shape": [4], "offset": 0, "nbytes": 16}]}, "dtype"),
    ({"leaves": [{"dtype": "<f4", "shape": [4], "offset": 0, "nbytes": 16, "x": 1}]}, "header"),
    ({"leaves": [{"dtype": "<f4", "shape": [1] * 33, "offset": 0, "nbytes": 4}]}, "shape"),
    ({"leaves": [{"dtype": "<f4", "shape": [0, 2**40, 2**40], "offset": 0, "nbytes": 0}]}, "shape"),   # P-PF1-1
    ({"leaves": [{"dtype": "<f4", "shape": [0, 2**63], "offset": 0, "nbytes": 0}]}, "shape"),
    ({"tree": {"t": "nd", "i": 1}}, "leaf_ref"),
    ({"tree": {"t": "list", "v": [{"t": "nd", "i": 0}, {"t": "nd", "i": 0}]}}, "leaf_ref"),
    ({"tree": {"t": "py", "v": None}}, "leaf_ref"),                       # leaf 0 never referenced
    ({"tree": {"t": "py", "v": [1]}}, "node"),
    ({"tree": {"t": "what"}}, "node"),
    ({"tree": [1]}, "node"),
    ({"tree": {"t": "dict", "k": [[1]], "v": [{"t": "nd", "i": 0}]}}, "dict_key"),
    ({"tree": {"t": "dict", "k": ["a", "a"], "v": [{"t": "nd", "i": 0}, {"t": "py", "v": 1}]}}, "dict_key"),
    ({"tree": {"t": "dict", "k": ["a"], "v": []}}, "node"),
    ({"kind": "evil"}, "kind"),
    ({"meta": [1]}, "header"),
    ({"v": 2}, "version"),
    ({"lz4": {"raw_bytes": 15}}, "header"),
])
def test_malformed_descriptors_and_trees_are_refused(changes, reason):
    with pytest.raises(FrameError) as info:
        decode_frame(_forge(_base(**changes), bytes(16)), cap=CAP)
    assert info.value.reason == reason


def test_overlapping_leaves_are_refused():
    header = _base(tree={"t": "list", "v": [{"t": "nd", "i": 0}, {"t": "nd", "i": 1}]},
                   leaves=[{"dtype": "<f4", "shape": [32], "offset": 0, "nbytes": 128},
                           {"dtype": "<f4", "shape": [4], "offset": 64, "nbytes": 16}], data_bytes=128)
    with pytest.raises(FrameError) as info:
        decode_frame(_forge(header, bytes(128)), cap=CAP)
    assert info.value.reason == "overlap"


def test_a_negative_dimension_cannot_cancel_a_huge_one():
    header = _base(tree={"t": "list", "v": [{"t": "nd", "i": 0}, {"t": "nd", "i": 1}]},
                   leaves=[{"dtype": "<f4", "shape": [-(10**10)], "offset": 0, "nbytes": 0},
                           {"dtype": "<f4", "shape": [10**10], "offset": 0, "nbytes": 0}], data_bytes=0)
    with pytest.raises(FrameError) as info:
        decode_frame(_forge(header), cap=2**20)
    assert info.value.reason == "shape"


def test_declared_sizes_over_the_cap_are_refused_before_allocation():
    buf = encode_frame({"x": np.zeros(2**18, np.float32)})              # 1 MiB of data
    assert decode_frame(buf, cap=2**20)[0]["x"].shape == (2**18,)
    with pytest.raises(FrameError) as info:
        decode_frame(buf, cap=2**20 - 1)
    assert info.value.reason == "too_large"


def test_a_truncated_or_padded_data_region_is_refused():
    buf = encode_frame({"x": np.arange(100, dtype=np.int64)})
    for bad in (buf[:-1], buf + b"\0"):
        with pytest.raises(FrameError) as info:
            decode_frame(bad, cap=CAP)
        assert info.value.reason == "size"


def test_an_lz4_bomb_is_refused_at_the_cap():
    """4 MiB of zeros compress to a few KB: the declared raw size is checked against the cap first, and a frame that
    lies about it decompresses only up to the declared size."""
    buf = encode_frame({"x": np.zeros(2**20, np.float32)}, compress=True)
    assert len(buf) < 2**16
    with pytest.raises(FrameError) as info:
        decode_frame(buf, cap=2**20)
    assert info.value.reason == "too_large"
    header, _ = _header(buf)
    region = lz4.frame.compress(bytes(2**22))
    lying = _base(tree={"t": "nd", "i": 0}, leaves=[{"dtype": "<f4", "shape": [4], "offset": 0, "nbytes": 16}],
                  data_bytes=16, lz4={"raw_bytes": 16})
    with pytest.raises(FrameError) as info:
        decode_frame(_forge(lying, region, flags=1), cap=CAP)
    assert info.value.reason == "lz4"


@pytest.mark.parametrize("damage", ["truncate", "garbage", "flag_without_header"])
def test_corrupt_lz4_regions_are_refused(damage):
    buf = encode_frame({"x": np.arange(1000)}, compress=True)
    if damage == "truncate":
        bad = buf[: len(buf) - 10]
    elif damage == "garbage":
        bad = buf[:-20] + bytes(20)
    else:
        bad = bytearray(encode_frame({"x": np.arange(1000)}))
        bad[6] = 1                                                          # flag set, header says no lz4
        bad = bytes(bad)
    with pytest.raises(FrameError) as info:
        decode_frame(bad, cap=CAP)
    assert info.value.reason in ("lz4", "flags")


def test_non_finite_numbers_in_the_header_are_refused():
    text = json.dumps(_base(meta={"x": 1})).replace('"x": 1', '"x": NaN').encode()
    head = struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, len(text)) + text
    with pytest.raises(FrameError) as info:
        decode_frame(head + bytes((-len(head)) % FRAME_ALIGN) + bytes(16), cap=CAP)
    assert info.value.reason == "header_json"


def test_too_many_leaves_and_too_deep_trees_are_refused(monkeypatch):
    import colosseum.net.frame as frame

    assert MAX_LEAVES == 100_000
    monkeypatch.setattr(frame, "MAX_LEAVES", 10)                            # 100 001 leaves exceed the header cap
    leaves = [{"dtype": "<f4", "shape": [0], "offset": 0, "nbytes": 0}] * 11
    with pytest.raises(FrameError) as info:
        decode_frame(_forge(_base(leaves=leaves, data_bytes=0)), cap=CAP)
    assert info.value.reason == "too_many_leaves"
    node: dict = {"t": "nd", "i": 0}
    for _ in range(MAX_DEPTH + 1):
        node = {"t": "list", "v": [node]}
    with pytest.raises(FrameError) as info:
        decode_frame(_forge(_base(tree=node), bytes(16)), cap=CAP)
    assert info.value.reason == "depth"
    deep = "[" * 100_000 + "]" * 100_000                                    # json's own recursion limit
    head = struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, len(deep)) + deep.encode()
    with pytest.raises(FrameError) as info:
        decode_frame(head, cap=CAP)
    assert info.value.reason == "header_json"


def test_unknown_namedtuples_are_refused_and_never_imported(monkeypatch):
    calls = []
    lazy = types.ModuleType("lazy_mod_sp5")
    lazy.__getattr__ = lambda name: calls.append(name) or HC              # PEP 562 lazy import hook
    monkeypatch.setitem(sys.modules, "lazy_mod_sp5", lazy)
    for cls, fields, reason in [("no_such_module_sp5:HC", ["h", "c"], "namedtuple"),
                                ("lazy_mod_sp5:HC", ["h", "c"], "namedtuple"),
                                (f"{__name__}:HC", ["h", "x"], "namedtuple"),
                                ("builtins:int", [], "namedtuple")]:
        tree = {"t": "nt", "cls": cls, "f": fields, "v": [{"t": "nd", "i": 0}, {"t": "py", "v": 1}][:len(fields)]}
        header = _base(tree=tree) if fields else _base(tree=tree, leaves=[], data_bytes=0)
        with pytest.raises(FrameError) as info:
            decode_frame(_forge(header, bytes(16) if fields else b""), cap=CAP)
        assert info.value.reason == reason
    assert calls == [] and "no_such_module_sp5" not in sys.modules


def test_the_dtype_whitelist_is_numeric_and_bool_only():
    assert DTYPES == {"|b1", "|i1", "|u1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f2", "<f4", "<f8"}


# ---------------------------------------------------------------------------
# Adaptive lz4 (spec 4.9)
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_the_probe_keeps_lz4_on_for_compressible_frames_and_reprobes_on_its_clock():
    clock = _Clock()
    probe = Lz4Probe(min_ratio=1.3, probe_frames=4, reprobe_sec=600.0, clock=clock)
    for _ in range(4):
        assert probe.compress()
        probe.observe(1000, 500)                    # ratio 2.0
    assert probe.enabled and probe.compress()
    probe.observe(1000, 900)                        # outside a probe: ignored
    clock.t = 599.0
    assert probe.compress() and probe.enabled
    clock.t = 600.0
    for _ in range(4):                              # a new probe: the stream became incompressible
        assert probe.compress()
        probe.observe(1000, 990)
    assert not probe.enabled and not probe.compress()
    probe.observe(1000, None)                       # sent plain: nothing to learn
    assert not probe.compress()


def test_the_probe_switches_lz4_off_below_the_ratio():
    probe = Lz4Probe(min_ratio=1.3, probe_frames=2, clock=_Clock())
    probe.observe(1000, 800)                        # 1.25
    probe.observe(1000, 800)
    assert not probe.enabled and not probe.compress()
    with pytest.raises(ValueError):
        Lz4Probe(probe_frames=0)
