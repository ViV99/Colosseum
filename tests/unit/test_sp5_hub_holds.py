"""The hub's evicted-snapshot hold (SP5 T3.1, spec 4.2): a snapshot evicted while a granted lineup names it is held
(counted in ``snapshot_holds``) and served from ``checkpoints/.evicted/`` until no grant names it; an evicted snapshot
nobody names is forgotten at once; held copies of an earlier hub process are cleared at start."""
from __future__ import annotations

import time

import numpy as np

from colosseum.coordinator.checkpoint_manager import EVICTED_DIR
from colosseum.hub.state import CellKey
from game_helpers import make_test_config, scripted_agent
from hub_kit import lineup_hub

STATE = {"w": np.arange(4, dtype=np.float32)}
CELL = CellKey(1, 0, 0)


def _hub(tmp_path):
    config = make_test_config("turns", agents={"a": {}, "random": scripted_agent()},
                              checkpoint={"keep_last": 1, "keep_every": 0})
    return lineup_hub(config, tmp_path)


def test_a_snapshot_named_by_a_grant_is_held_until_no_grant_names_it(tmp_path):
    hub = _hub(tmp_path)
    manager = hub._coordinator.checkpoint_manager
    manager.save("a", 10, STATE)
    hub._holds.grant(CELL, {("a", "ckpt_v10")})                # a granted lineup names v10
    manager.save("a", 20, STATE)                               # v10 evicted while named: held
    assert hub._book.get("snapshot_holds") == 1 and manager.held_dir("a", "ckpt_v10").is_dir()
    assert hub._blobs.frame_path("snapshot", "a", "ckpt_v10") is not None
    hub._holds.grant(CELL, ())
    hub._holds.grant(CELL, ())                                 # its last two grants no longer name v10
    hub._release_holds(time.monotonic())
    assert not manager.held_dir("a", "ckpt_v10").exists()
    assert hub._blobs.frame_path("snapshot", "a", "ckpt_v10") is None


def test_an_evicted_snapshot_nobody_names_is_forgotten_at_once(tmp_path):
    hub = _hub(tmp_path)
    manager = hub._coordinator.checkpoint_manager
    manager.save("a", 10, STATE)
    assert hub._blobs.frame_path("snapshot", "a", "ckpt_v10") is not None     # converted once
    manager.save("a", 20, STATE)
    assert hub._book.get("snapshot_holds") == 0 and not manager.held_dir("a", "ckpt_v10").exists()
    assert hub._blobs.frame_path("snapshot", "a", "ckpt_v10") is None
    assert not list(hub._blobs.dir.glob("snapshot-*-ckpt_v10.frame"))


def test_a_lost_node_releases_its_holds_and_the_stop_clears_every_copy(tmp_path):
    hub = _hub(tmp_path)
    manager = hub._coordinator.checkpoint_manager
    manager.save("a", 10, STATE)
    hub._holds.grant(CELL, {("a", "ckpt_v10")})
    manager.save("a", 20, STATE)
    hub._holds.drop_node(1)                                    # what the hub does when node 1 is lost
    hub._release_holds(time.monotonic())
    assert not manager.held_dir("a", "ckpt_v10").exists()
    manager.save("a", 30, STATE)
    hub._holds.grant(CELL, {("a", "ckpt_v30")})
    manager.save("a", 40, STATE)
    hub._release_holds()                                       # the stop
    assert not (manager.base_dir / EVICTED_DIR).exists()


def test_held_copies_of_an_earlier_hub_are_cleared_at_start(tmp_path):
    leftover = tmp_path / "runs" / "hub-run" / "checkpoints" / EVICTED_DIR / "a" / "ckpt_v5"
    hub = _hub(tmp_path)
    leftover.mkdir(parents=True)
    (leftover / "model.pt").write_bytes(b"stale")
    hub._prepare()                                             # a second start of the same run dir
    assert not leftover.exists()
