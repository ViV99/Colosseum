"""Recovery points in the hub (SP5 T4.2, spec 5.2 and 7.2): accepted only from the current generation and never
older than the stored one, header-checked without decoding, moved into place atomically, overwritten, with a
meta.json copy; unreadable files are ignored with a WARNING."""
from __future__ import annotations

import json
import logging

import numpy as np

from colosseum.hub.recovery import RECOVERY_META, RecoveryStore, read_frame_meta
from colosseum.net.frame import encode_frame


def _frame(agent="main", generation=2, version=40, kind="recovery") -> bytes:
    meta = {"kind": kind, "agent_id": agent, "generation": generation, "policy_version": version, "final": False}
    tree = {"w": np.arange(6, dtype=np.float32), "trainer_state_bytes": np.frombuffer(b"\x80\x02xyz", np.uint8)}
    return encode_frame(tree, kind="checkpoint", meta=meta)


def _spool(tmp_path, data: bytes, name: str = "up-1") -> object:
    path = tmp_path / ".incoming" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    return path


def test_accept_moves_the_frame_into_place_and_writes_meta(tmp_path):
    store = RecoveryStore(tmp_path)
    spooled = _spool(tmp_path, _frame())
    assert store.accept("main", spooled, generation=2, version=40, env_steps=1234, current_generation=2) == (True, "")
    assert not spooled.exists()
    point = store.get("main")
    assert (point.generation, point.version, point.env_steps) == (2, 40, 1234)
    assert point.path == tmp_path / "recovery" / "main" / "recovery.frame"
    assert json.loads((point.path.parent / RECOVERY_META).read_text()) == {"generation": 2, "version": 40,
                                                                             "env_steps": 1234}
    assert read_frame_meta(point.path) == ("checkpoint", {"kind": "recovery", "agent_id": "main", "generation": 2,
                                                          "policy_version": 40, "final": False})


def test_stale_generation_and_older_points_are_refused(tmp_path):
    store = RecoveryStore(tmp_path)
    store.accept("main", _spool(tmp_path, _frame(version=40)), generation=2, version=40, env_steps=1,
                 current_generation=2)
    stale = _spool(tmp_path, _frame(generation=1, version=99), "up-2")
    assert store.accept("main", stale, generation=1, version=99, env_steps=2, current_generation=2) == (
        False, "stale_generation")
    assert not stale.exists()
    older = _spool(tmp_path, _frame(version=30), "up-3")
    assert store.accept("main", older, generation=2, version=30, env_steps=3, current_generation=2) == (
        False, "not_newer")
    newer = _spool(tmp_path, _frame(generation=3, version=55), "up-4")
    assert store.accept("main", newer, generation=3, version=55, env_steps=4, current_generation=3) == (True, "")
    assert (store.get("main").generation, store.get("main").version) == (3, 55)


def test_a_header_that_does_not_match_the_upload_is_invalid(tmp_path):
    store = RecoveryStore(tmp_path)
    for i, data in enumerate((_frame(agent="other"), _frame(version=41), _frame(kind="periodic"), b"not a frame")):
        spooled = _spool(tmp_path, data, f"bad-{i}")
        assert store.accept("main", spooled, generation=2, version=40, env_steps=0, current_generation=2) == (
            False, "invalid")
    assert store.get("main") is None


def test_the_frame_header_wins_over_a_stale_meta_copy(tmp_path):
    store = RecoveryStore(tmp_path)
    store.accept("main", _spool(tmp_path, _frame(version=40)), generation=2, version=40, env_steps=500,
                 current_generation=2)
    (tmp_path / "recovery" / "main" / RECOVERY_META).write_text(json.dumps(
        {"generation": 2, "version": 10, "env_steps": 9}))           # a crash between the two writes
    point = store.get("main")
    assert (point.version, point.env_steps) == (40, 0)


def test_an_unreadable_point_is_ignored_with_a_warning(tmp_path, caplog):
    path = tmp_path / "recovery" / "main" / "recovery.frame"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"CSTF\x01\x00")
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.recovery"):
        assert RecoveryStore(tmp_path).get("main") is None
    assert "unreadable" in caplog.text
