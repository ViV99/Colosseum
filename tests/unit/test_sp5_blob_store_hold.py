"""The hub's blob store and the evicted-snapshot hold (SP5 T3.1, spec 4.2): a held snapshot is converted from
``checkpoints/.evicted/``; a released one is forgotten (its converted frame removed, then NOT_FOUND); a conversion
that finishes after ``forget`` leaves no frame."""
from __future__ import annotations

import numpy as np

from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.run_dir import RunDir
from colosseum.hub import server
from colosseum.hub.server import BlobStore
from colosseum.net.chunks import decode_weights

STATE = {"w": np.arange(6, dtype=np.float32), "b": np.ones(2, np.float32)}


def _store(tmp_path, *, hold=True):
    run_dir = RunDir(tmp_path / "run")
    m = CheckpointManager(run_dir.checkpoints, keep_last=1)
    m.set_eviction_hold(lambda agent, ckpt: hold)
    m.save("main", 10, STATE)
    m.save("main", 20, STATE)                                   # v10 is evicted (held or not)
    return m, BlobStore(run_dir)


def _frames(store):
    return sorted(p.name for p in store.dir.glob("snapshot-*")) if store.dir.exists() else []


def test_a_held_snapshot_is_served_from_the_evicted_dir(tmp_path):
    m, store = _store(tmp_path)
    assert not (m.base_dir / "main" / "ckpt_v10").exists() and m.held_dir("main", "ckpt_v10").is_dir()
    path = store.frame_path("snapshot", "main", "ckpt_v10")
    state, meta = decode_weights(path.read_bytes(), cap=1 << 20)
    assert (meta.kind, meta.checkpoint_id, meta.policy_version) == ("snapshot", "ckpt_v10", 10)
    assert all(np.array_equal(state[k], STATE[k]) for k in STATE)


def test_a_released_snapshot_is_not_found(tmp_path):
    m, store = _store(tmp_path)
    assert store.frame_path("snapshot", "main", "ckpt_v10") is not None
    m.release_held("main", "ckpt_v10")
    store.forget("snapshot", "main", "ckpt_v10")
    assert _frames(store) == [] and store.frame_path("snapshot", "main", "ckpt_v10") is None
    assert store.frame_path("snapshot", "main", "ckpt_v20") is not None       # stored snapshots are unaffected


def test_an_evicted_snapshot_that_was_not_held_is_not_found(tmp_path):
    _m, store = _store(tmp_path, hold=False)
    assert store.frame_path("snapshot", "main", "ckpt_v10") is None


def test_forget_during_a_conversion_leaves_no_frame(tmp_path, monkeypatch):
    m, store = _store(tmp_path)
    encode = server.encode_weights

    def encode_and_release(state, meta):                       # the main loop releases while a servicer converts
        frame = encode(state, meta)
        m.release_held("main", "ckpt_v10")
        store.forget("snapshot", "main", "ckpt_v10")
        return frame

    monkeypatch.setattr(server, "encode_weights", encode_and_release)
    assert store.frame_path("snapshot", "main", "ckpt_v10") is None
    assert _frames(store) == []


def test_a_re_saved_id_is_served_again_after_forget(tmp_path):
    m, store = _store(tmp_path)
    m.release_held("main", "ckpt_v10")
    store.forget("snapshot", "main", "ckpt_v10")
    m.save("main", 10, STATE)                                  # the id exists again (e.g. after a resume)
    assert store.frame_path("snapshot", "main", "ckpt_v10") is not None


def test_a_snapshot_evicted_while_it_is_read_is_served_from_its_held_copy(tmp_path, monkeypatch):
    """Review round 1: a conversion that chose the stored ``model.pt`` and finds it gone (evicted meanwhile) reads the
    held copy instead of answering NOT_FOUND."""
    run_dir = RunDir(tmp_path / "run")
    m = CheckpointManager(run_dir.checkpoints, keep_last=1)
    m.set_eviction_hold(lambda agent, ckpt: True)
    m.save("main", 10, STATE)
    store = BlobStore(run_dir)
    read = server.read_weights_file
    calls = []

    def read_after_eviction(path):                             # the main loop evicts v10 just as it is read
        calls.append(path)
        if len(calls) == 1:
            m.save("main", 20, STATE)
        return read(path)

    monkeypatch.setattr(server, "read_weights_file", read_after_eviction)
    path = store.frame_path("snapshot", "main", "ckpt_v10")
    assert path is not None and len(calls) == 2 and ".evicted" in str(calls[1])
    _state, meta = decode_weights(path.read_bytes(), cap=1 << 20)
    assert meta.checkpoint_id == "ckpt_v10"
