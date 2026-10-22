"""The hub's side of chunk routing between machines (SP5 T3.2, spec 4.7, 4.8): with the hub relay on a remote learner's
seats collect on every node; with ``distributed.relay: false`` ``Heartbeat.paths`` decide (only on a node that
reported a direct path) and a node that cannot reach a learner's data port is named once in a WARNING; malformed path
reports are counted, a lost node's paths go; the learner node's loopback data addresses reach only nodes on its own
machine (carried decision 3a of T3.1). The real ``Hub`` handlers, driven in-process through ``hub_kit``'s recording
session handles."""
from __future__ import annotations

import logging

from colosseum.net.messages import VersionInfo, lineup_from_proto, version_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from game_helpers import make_test_config
from hub_kit import accept_version, lineup_hub, node_message, open_node, request_cells

LATEST_ONLY = {"opponents": {"latest": 1.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.0}}


def _hub(tmp_path, relay=False):
    config = make_test_config("turns", agents={"a": {}}, matchmaking=LATEST_ONLY, rollout={"envs_per_worker": 4},
                              distributed={"relay": relay})
    return lineup_hub(config, tmp_path)


def _collecting(grants) -> list[set[str]]:
    return [{s.agent_id for s in lineup_from_proto(g.lineup).seats if s.collect} for g in grants]


def _paths(hub, handle, **paths):
    hub._on_message(node_message(handle, heartbeat=pb.Heartbeat(
        paths=[pb.PathState(node=int(n[1:]), path=p) for n, p in paths.items()])))


def test_a_remote_learners_seats_collect_only_where_a_direct_path_was_reported(tmp_path, caplog):
    hub = _hub(tmp_path)
    first = open_node(hub, name="learner-node", workers=1)
    second = open_node(hub, name="worker-node", workers=1, cpu_learners=0, no_learners=True)
    a, b = (h.of("welcome")[0].node_number for h in (first, second))
    assert hub._learner_node == {"a": a}
    accept_version(hub, "a")
    cells = [(0, e, 0, 0) for e in range(4)]
    assert _collecting(request_cells(hub, second, cells)) == [set()] * 4      # no path reported yet: nothing collects
    assert _collecting(request_cells(hub, first, cells)) == [{"a"}] * 4       # the learner's own node

    _paths(hub, second, **{f"n{a}": "direct"})
    grants = request_cells(hub, second, cells)
    assert _collecting(grants) == [{"a"}] * 4
    assert hub._paths == {b: {a: "direct"}}

    with caplog.at_level(logging.WARNING, logger="colosseum.hub.hub"):
        for _ in range(2):
            _paths(hub, second, **{f"n{a}": "down"})
    grants = request_cells(hub, second, cells)
    assert _collecting(grants) == [set()] * 4
    said = [r.getMessage() for r in caplog.records if "cannot reach the data port" in r.getMessage()]
    assert len(said) == 1 and f"node {b} " in said[0] and f"node {a}" in said[0] and "['a']" in said[0]
    assert "relay is off (distributed.relay: false)" in said[0]


def test_with_the_hub_relay_on_a_remote_learners_seats_collect_on_every_node(tmp_path, caplog):
    hub = _hub(tmp_path, relay=True)
    first = open_node(hub, name="learner-node", workers=1)
    second = open_node(hub, name="worker-node", workers=1, cpu_learners=0, no_learners=True)
    a = first.of("welcome")[0].node_number
    accept_version(hub, "a")
    cells = [(0, e, 0, 0) for e in range(4)]
    assert _collecting(request_cells(hub, second, cells)) == [{"a"}] * 4      # before any path report: relayed
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.hub"):
        _paths(hub, second, **{f"n{a}": "relay"})
    assert _collecting(request_cells(hub, second, cells)) == [{"a"}] * 4
    assert not [r for r in caplog.records if "cannot reach the data port" in r.getMessage()]


def test_malformed_path_reports_are_counted_and_a_lost_nodes_paths_go(tmp_path):
    hub = _hub(tmp_path)
    first = open_node(hub, name="learner-node", workers=1)
    second = open_node(hub, name="worker-node", workers=1, cpu_learners=0, no_learners=True)
    a, b = (h.of("welcome")[0].node_number for h in (first, second))
    hub._on_message(node_message(second, heartbeat=pb.Heartbeat(paths=[
        pb.PathState(node=a, path="direct"), pb.PathState(node=9, path="teleport")])))
    assert hub._paths == {b: {a: "direct"}}
    assert hub._book.totals().get("node_messages.invalid") == 1
    _paths(hub, second)                                                   # an empty report: no remote target any more
    assert hub._paths == {b: {}}
    _paths(hub, second, **{f"n{a}": "direct"})
    hub._on_node_lost(hub._registry.get(b))
    assert b not in hub._paths


def _register(hub, name, machine_id):
    record, _secret = hub.identity.nodes.register(machine_id=machine_id, name=name, tags=[], node_id=None)
    return record.node_id


def test_loopback_data_addresses_reach_only_nodes_on_the_learner_nodes_machine(tmp_path):
    hub = _hub(tmp_path)
    addresses = ["127.0.0.1:47102", "10.0.0.5:47102"]
    learner = open_node(hub, name="learner-node", workers=1, data_addresses=addresses,
                        node_id=_register(hub, "learner-node", "machine-1"))
    twin = open_node(hub, name="twin", workers=1, no_learners=True, node_id=_register(hub, "twin", "machine-1"))
    other = open_node(hub, name="other", workers=1, no_learners=True, node_id=_register(hub, "other", "machine-2"))
    number = learner.of("welcome")[0].node_number
    info = VersionInfo(agent_id="a", generation=hub._versions.generation("a"), version=0, size=10, digest="d" * 32,
                       node=number)
    hub._on_version(hub._registry.get(number), version_to_proto(info))
    hub._broadcast_announce()
    [seen_twin] = twin.of("announce")[-1].versions
    [seen_other] = other.of("announce")[-1].versions
    assert list(seen_twin.addresses) == addresses and not seen_twin.relay_only
    assert list(seen_other.addresses) == ["10.0.0.5:47102"] and not seen_other.relay_only
    late = open_node(hub, name="late", workers=1, no_learners=True, node_id=_register(hub, "late", "machine-3"))
    [welcomed] = late.of("welcome")[0].versions
    assert list(welcomed.addresses) == ["10.0.0.5:47102"]


def test_the_relay_follows_the_online_nodes_and_versions_and_its_counters_reach_the_book(tmp_path):
    hub = _hub(tmp_path, relay=True)
    first = open_node(hub, name="learner-node", workers=1)
    second = open_node(hub, name="worker-node", workers=1, cpu_learners=0, no_learners=True)
    a, b = (h.of("welcome")[0].node_number for h in (first, second))
    assert (hub._relay.number_of(first.node_id), hub._relay.number_of(second.node_id)) == (a, b)
    info = VersionInfo(agent_id="a", generation=hub._versions.generation("a"), version=3, size=10, digest="d" * 32,
                       node=a)
    hub._on_version(hub._registry.get(a), version_to_proto(info))
    assert hub._latest.latest("a").version == 3 and hub._latest.latest("a").relay_only    # no data address
    assert hub._relay.put(b, a, "frame", 10) is None
    inbox = hub._relay.attach(a, initial_credits=1)
    assert hub._relay.next_item(inbox, timeout=1.0) == ("chunk", "frame")
    hub._report_relay()
    assert hub._book.by_source_prefix("hub/") == {"relay_frames": 1, "relay_bytes": 10}
    assert f"hub/{hub._incarnation}" in hub._book.state()["sources"]
    hub._registry.remove(second.node_id)
    hub._on_node_lost(hub._registry.by_id(second.node_id), reason="removed")
    assert hub._relay.number_of(second.node_id) is None
