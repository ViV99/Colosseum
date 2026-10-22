"""The hub's learner methods (SP5 T4.1, ``hub.learners``; review round 1): placement only on nodes whose stream is
open, parked frames served only for agents this hub process parked, parking after a lost node's stream is gone, the
version floor's checkpoint identity (R38), a failing placement pass on the metrics tick."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from colosseum.coordinator.checkpoint_manager import CheckpointInfo
from colosseum.hub.server import BlobStore, SessionClosed
from game_helpers import agent_role_of, make_test_config
from hub_kit import lineup_hub, open_node


def _config(agents=None, **sections):
    return make_test_config("turns", agents=agents or {"a": {}, "b": {}}, **sections)


def test_no_learner_is_placed_on_a_node_whose_stream_is_down(tmp_path):
    hub = lineup_hub(_config(), tmp_path)
    node = open_node(hub, name="n1", cpu_learners=1)
    assert hub._learner_node_of("a") is not None and hub._learners["b"].status == "pending"
    hub._learners.fail("a", "test")                                       # its slot is free (no placement pass)
    hub._on_session_closed(SessionClosed(node.node_id, node.session_id, "cut", node))
    hub._place_learners()
    assert hub._learners["b"].status == "pending"                         # never started on the cut stream
    back = open_node(hub, name="n1", node_id=node.node_id, session_id=node.session_id, cpu_learners=1)
    starts = [c.learner for c in back.of("command") if c.kind == "start_learner"]
    assert [(la.agent_id, la.generation) for la in starts] == [("b", 1)]


def test_parked_frames_are_served_only_for_agents_this_process_parked(tmp_path):
    from colosseum.hub.hub import Hub, HubOptions

    first = lineup_hub(_config(), tmp_path)
    store = BlobStore(first._run_dir)
    stale = store.parked_path("a")
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"left by an earlier hub process")
    assert store.parked_file("a") is None                                 # not marked: never served
    store.write_parked("b", b"frame")
    assert store.parked_file("b") == store.parked_path("b")
    store.drop_parked("b")
    assert store.parked_file("b") is None and not store.parked_path("b").exists()
    again = Hub(first._config, first._run_dir, options=HubOptions(mode="hub", local_node=None,
                                                                   identity_dir=Path(tmp_path) / "hub-identity"))
    again._prepare()                                                       # a later hub process on the same run dir
    assert not stale.exists() and again._blobs.parked_file("a") is None


def test_a_lost_nodes_learners_are_parked_after_its_stream_is_gone(tmp_path):
    from colosseum.core.registry import build_model

    base = _config({"a": {}})
    pt = tmp_path / "init.pt"
    torch.save(build_model(base.get_agent_config("a"), agent_role_of(base, "a")[1]).state_dict(), pt)
    hub = lineup_hub(_config({"a": {"init": {"from": str(pt)}}}), tmp_path)
    lost = open_node(hub, name="lost", cpu_learners=1)
    other = open_node(hub, name="other", cpu_learners=0, no_learners=True)
    assert hub._learner_node_of("a") == lost.of("welcome")[0].node_number
    before = len(lost.sent)
    entry = hub._registry.by_id(lost.node_id)
    entry.state = "lost"
    hub._on_node_lost(entry)
    assert hub._learners["a"].status == "relocating" and hub._versions.parked("a") is not None
    assert lost.closed_reason == "lost" and [m.WhichOneof("m") for m in lost.sent[before:]] == []
    parked = [v for m in other.of("announce") for v in m.versions if v.agent_id == "a"]
    assert [(v.node, v.generation) for v in parked] == [(-1, 0)]


def test_the_version_floor_spares_only_the_start_points_own_checkpoint(tmp_path):
    """R38: a start point equal to the newest stored snapshot (checkpoint id and run id) may publish its version."""
    hub = lineup_hub(_config({"a": {}}), tmp_path)
    info = CheckpointInfo("ckpt_v7", "a", 7, Path(tmp_path), time.time(), meta={"run_id": "parent"})
    hub._coordinator.checkpoint_manager.latest = lambda agent: info
    assert hub._version_floor("a", start=("ckpt_v7", "parent")) == 7
    assert hub._version_floor("a", start=("ckpt_v7", "other")) == 8
    assert hub._version_floor("a", start=("ckpt_v6", "parent")) == 8
    assert hub._version_floor("a") == 8


def test_a_failing_placement_pass_on_the_tick_fails_the_run(tmp_path):
    hub = lineup_hub(_config({"a": {}}), tmp_path)
    node = open_node(hub, name="n1", cpu_learners=0, no_learners=True)
    assert hub._learners["a"].status == "pending"
    hub._registry.by_id(node.node_id).slots = {"cpu_learners": 1}

    def full_disk(*, fsync, **_kw):
        if fsync:
            raise OSError(28, "No space left on device")

    hub._save_hub_state = full_disk
    hub._tick_placement()                                                  # no traceback out of the main loop
    assert "a learner placement pass failed (OSError" in hub._failure
    assert hub._learners["a"].status == "pending"
