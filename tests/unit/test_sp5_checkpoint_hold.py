"""The checkpoint manager's eviction hold (SP5 T3.1, spec 4.2): a snapshot evicted while the hub says a granted
lineup names it is kept (hard links) under checkpoints/.evicted/<agent>/<id>/ until the hub releases it; the hold
never fails a save and is invisible to the pool, rescans, run-dir listings and imports."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from colosseum.coordinator.checkpoint_manager import EVICTED_DIR, CheckpointManager, list_snapshots
from colosseum.core.errors import ConfigError

STATE = {"w": np.ones(3, np.float32)}


def _manager(tmp_path, **kw):
    return CheckpointManager(tmp_path / "run" / "checkpoints", keep_last=1, **kw)


def test_an_evicted_snapshot_named_by_a_lineup_is_kept_under_evicted(tmp_path):
    evicted = []
    m = _manager(tmp_path, on_evict=lambda a, c: evicted.append((c, m.held_dir(a, c).is_dir())))
    m.set_eviction_hold(lambda agent, ckpt: ckpt == "ckpt_v10")
    m.save("main", 10, STATE)
    m.save("main", 20, STATE)
    held = m.held_dir("main", "ckpt_v10")
    assert held == m.base_dir / EVICTED_DIR / "main" / "ckpt_v10"
    assert evicted == [("ckpt_v10", True)]                    # the copy exists when the listeners run
    assert sorted(p.name for p in held.iterdir()) == ["meta.json", "model.pt"]
    assert [c.checkpoint_id for c in m.list_checkpoints("main")] == ["ckpt_v20"]
    assert list(list_snapshots(tmp_path / "run")) == ["main"]  # .evicted is not an agent
    assert not (m.base_dir / "main" / "ckpt_v10").exists()
    m.save("main", 30, STATE)                                 # v20 is not named by a lineup: deleted
    assert not m.held_dir("main", "ckpt_v20").exists()
    m.release_held("main", "ckpt_v10")
    assert not held.exists()
    assert CheckpointManager(m.base_dir, keep_last=1).agents == ["main"]   # a rescan ignores .evicted


def test_a_failing_hold_check_never_fails_the_save(tmp_path, caplog):
    m = _manager(tmp_path)
    m.set_eviction_hold(lambda a, c: 1 / 0)
    m.save("main", 10, STATE)
    with caplog.at_level(logging.WARNING):
        m.save("main", 20, STATE)
    assert not m.held_dir("main", "ckpt_v10").exists()
    assert "not held" in caplog.text


def test_clear_held_removes_every_leftover(tmp_path):
    m = _manager(tmp_path)
    m.set_eviction_hold(lambda a, c: True)
    for v in (10, 20, 30):
        m.save("main", v, STATE)
    assert m.held_dir("main", "ckpt_v20").is_dir()
    m.clear_held()
    assert not (m.base_dir / EVICTED_DIR).exists()


def test_a_run_dir_import_skips_the_evicted_dir(tmp_path):
    parent = _manager(tmp_path / "parent")
    parent.set_eviction_hold(lambda a, c: True)
    for v in (10, 20):
        parent.save("main", v, STATE)
    child = CheckpointManager(tmp_path / "child" / "checkpoints", keep_last=5)
    assert child.import_snapshots(parent.base_dir, "main") == ["ckpt_v20"]


def test_bad_ids_are_refused(tmp_path):
    with pytest.raises(ConfigError):
        _manager(tmp_path).held_dir("main", "../x")
