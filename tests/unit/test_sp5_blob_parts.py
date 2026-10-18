"""Blobs in parts (SP5 T2.3, spec 4.1 "Большие данные", 3.4): round trips of bytes and files, empty blobs, the
cap checked at the first part before anything is allocated, gaps / overlaps / short streams / digest mismatches
refused, a file target written atomically (nothing left on error), ``not_modified`` answers."""
from __future__ import annotations

import os

import pytest

from colosseum.net.grpc_util import PART_BYTES, BlobError, blob_digest, not_modified_part, receive_parts, send_parts


@pytest.mark.parametrize("size", [0, 1, PART_BYTES - 1, PART_BYTES, 3 * PART_BYTES + 5])
def test_bytes_round_trip_in_parts(size):
    data = os.urandom(size)
    parts = list(send_parts(data, meta_json='{"k": 1}'))
    assert len(parts) == max(1, -(-size // PART_BYTES)) and parts[-1].last and not any(p.last for p in parts[:-1])
    assert parts[0].digest == blob_digest(data) and all(not p.digest for p in parts[1:])
    assert receive_parts(parts, cap=size + 1) == (data, '{"k": 1}')


def test_a_memoryview_and_a_given_digest_are_sent_as_they_are():
    data = os.urandom(PART_BYTES + 3)
    parts = list(send_parts(memoryview(data), digest="d" * 32))
    assert parts[0].digest == "d" * 32 and b"".join(p.data for p in parts) == data


def test_a_file_round_trips_into_a_file(tmp_path):
    d = tmp_path / "blobs"
    d.mkdir()
    src = d / "src.frame"
    src.write_bytes(os.urandom(2 * PART_BYTES + 7))
    out, meta = receive_parts(send_parts(src, meta_json="m"), cap=10 * PART_BYTES, into=d / "dst.frame")
    assert out == d / "dst.frame" and out.read_bytes() == src.read_bytes() and meta == "m"
    assert sorted(p.name for p in d.iterdir()) == ["dst.frame", "src.frame"]


def test_an_empty_file_is_one_part(tmp_path):
    src = tmp_path / "empty.frame"
    src.write_bytes(b"")
    parts = list(send_parts(src))
    assert len(parts) == 1 and parts[0].last and parts[0].digest == blob_digest(b"")
    assert receive_parts(parts, cap=0) == (b"", "")


def test_the_cap_is_checked_at_the_first_part():
    parts = send_parts(os.urandom(3 * PART_BYTES))
    with pytest.raises(BlobError) as e:
        receive_parts(parts, cap=PART_BYTES)
    assert e.value.reason == "too_large"


@pytest.mark.parametrize("mutate,reason", [
    (lambda ps: [ps[0], ps[2]], "offset"),                        # a gap
    (lambda ps: [ps[0], ps[0]], "offset"),                        # a repeated part
    (lambda ps: ps[:2], "size"),                                  # cut short
    (lambda ps: [type(ps[0])(data=b"x" + ps[0].data[1:], offset=0, total_size=ps[0].total_size,
                              digest=ps[0].digest, last=False)] + ps[1:], "digest"),
])
def test_inconsistent_streams_are_refused_and_leave_no_file(tmp_path, mutate, reason):
    d = tmp_path / "blobs"
    d.mkdir()
    parts = list(send_parts(os.urandom(2 * PART_BYTES + 3)))
    with pytest.raises(BlobError) as e:
        receive_parts(mutate(parts), cap=10 * PART_BYTES, into=d / "x.frame")
    assert e.value.reason == reason and list(d.iterdir()) == []


def test_an_empty_stream_and_a_not_modified_answer():
    with pytest.raises(BlobError) as e:
        receive_parts([], cap=10)
    assert e.value.reason == "empty"
    assert receive_parts([not_modified_part('{"version": 4}')], cap=10) == (None, '{"version": 4}')
