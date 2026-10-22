"""The hub's evicted-snapshot hold (SP5 T3.1, spec 4.2), driven through the real hub handlers: a snapshot evicted while
a lineup granted by ``_on_lineup_request`` names it is held (``snapshot_holds``) and served from
``checkpoints/.evicted/`` until no grant names it; a lost or removed node, a new incarnation of a cell and a disabled
worker slot drop their references; an evicted snapshot nobody names is forgotten at once; nothing is held from the
stop on; held copies of an earlier hub process are cleared at start."""
from __future__ import annotations

import time

import numpy as np
import pytest

from colosseum.coordinator.checkpoint_manager import EVICTED_DIR
from colosseum.net.proto import colosseum_v1_pb2 as pb
from game_helpers import make_test_config, scripted_agent
from hub_kit import accept_version, lineup_hub, node_message, open_node, request_cells

STATE = {"w": np.arange(4, dtype=np.float32)}
SNAPSHOTS_ONLY = {"opponents": {"latest": 0.0, "snapshots": 1.0, "rivals": 0.0, "anchors": 0.0}}


def _hub(tmp_path):
    config = make_test_config("turns", agents={"a": {}, "random": scripted_agent()},
                              checkpoint={"keep_last": 1, "keep_every": 0}, matchmaking=SNAPSHOTS_ONLY,
                              rollout={"match_refresh_interval_sec": 1.0})
    hub = lineup_hub(config, tmp_path)
    node = open_node(hub, name="n", workers=1)
    accept_version(hub, "a")
    return hub, node, hub._coordinator.checkpoint_manager


def _granted(hub, node, worker=0, env=0, incarnation=1):
    """The snapshots the hub's grant for one cell names (through ``_on_lineup_request``)."""
    [grant] = request_cells(hub, node, [(worker, env, 0, incarnation)])
    return {s.network_id for s in grant.lineup.seats if s.network_id.startswith("ckpt_v")}


def _released(hub, manager, ckpt="ckpt_v10"):
    hub._release_holds(time.monotonic())
    return not manager.held_dir("a", ckpt).exists() and hub._blobs.frame_path("snapshot", "a", ckpt) is None


def test_a_snapshot_a_granted_lineup_names_is_held_until_no_grant_names_it(tmp_path):
    hub, node, manager = _hub(tmp_path)
    manager.save("a", 10, STATE)
    assert _granted(hub, node) == {"ckpt_v10"}
    manager.save("a", 20, STATE)                               # v10 evicted while named: held
    assert hub._book.get("snapshot_holds") == 1 and manager.held_dir("a", "ckpt_v10").is_dir()
    assert hub._blobs.frame_path("snapshot", "a", "ckpt_v10") is not None
    assert not _released(hub, manager)
    for _ in range(2):                                          # the cell's next two grants name v20 only
        assert _granted(hub, node) == {"ckpt_v20"}
    assert _released(hub, manager)


@pytest.mark.parametrize("reason", ["lost", "removed"])
def test_a_lost_or_removed_node_drops_its_references(tmp_path, reason):
    hub, node, manager = _hub(tmp_path)
    manager.save("a", 10, STATE)
    _granted(hub, node)
    manager.save("a", 20, STATE)
    assert not _released(hub, manager)
    hub._on_node_lost(hub._registry.by_id(node.node_id), reason=reason)
    assert _released(hub, manager)


def test_a_new_incarnation_drops_the_dead_ones_references(tmp_path):
    hub, node, manager = _hub(tmp_path)
    manager.save("a", 10, STATE)
    _granted(hub, node, incarnation=1)
    manager.save("a", 20, STATE)
    assert _granted(hub, node, incarnation=2) == {"ckpt_v20"}  # the restarted worker's first grant
    assert _released(hub, manager)                             # the dead incarnation's v10 is no longer named


def test_a_disabled_worker_slot_drops_its_references(tmp_path):
    hub, node, manager = _hub(tmp_path)
    manager.save("a", 10, STATE)
    _granted(hub, node, worker=0, env=0)
    _granted(hub, node, worker=0, env=1)
    manager.save("a", 20, STATE)
    assert not _released(hub, manager)
    hub._on_message(node_message(node, event=pb.Event(kind="slot_disabled", proc="worker-0")))
    assert _released(hub, manager)


def test_an_evicted_snapshot_nobody_names_is_forgotten_at_once(tmp_path):
    hub, _node, manager = _hub(tmp_path)
    manager.save("a", 10, STATE)
    assert hub._blobs.frame_path("snapshot", "a", "ckpt_v10") is not None     # converted once
    manager.save("a", 20, STATE)
    assert hub._book.get("snapshot_holds") == 0 and not manager.held_dir("a", "ckpt_v10").exists()
    assert hub._blobs.frame_path("snapshot", "a", "ckpt_v10") is None
    assert not list(hub._blobs.dir.glob("snapshot-*-ckpt_v10.frame"))


def test_from_the_stop_on_nothing_is_held(tmp_path):
    """The stop releases every hold; a snapshot a late upload evicts afterwards (the shutdown still drains the inbox)
    leaves no held copy behind (review round 1)."""
    hub, node, manager = _hub(tmp_path)
    manager.save("a", 10, STATE)
    _granted(hub, node)
    manager.save("a", 20, STATE)
    hub._release_holds()                                       # the stop
    assert not (manager.base_dir / EVICTED_DIR).exists()
    _granted(hub, node)                                         # named by a grant, evicted by a late final upload
    manager.save("a", 30, STATE)
    assert not (manager.base_dir / EVICTED_DIR).exists()


def test_held_copies_of_an_earlier_hub_are_cleared_at_start(tmp_path):
    hub, _node, _manager = _hub(tmp_path)
    leftover = hub._run_dir.checkpoints / EVICTED_DIR / "a" / "ckpt_v5"
    leftover.mkdir(parents=True)
    (leftover / "model.pt").write_bytes(b"stale")
    hub._prepare()                                             # a second start of the same run dir
    assert not leftover.exists()
