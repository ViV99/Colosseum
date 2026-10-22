"""The node's routing section (SP5 T3.2, spec 4.7, 4.8): learners on other nodes become remote routes as soon as their
versions are known; the path to each is probed before any frame and reported (``NodeAgent.paths()``,
``Heartbeat.paths``); workers are routed via the node unless the path is down; a down path is one WARNING and one
``Event(warning)`` per target; a parked version (node -1) is no route (pre-flight ruling P-PF3-13).

The node runs in a thread against ``node_kit.FakeHub`` (insecure) with no worker (no loaded process); node 7's data
port is a real ``DataServer`` (plain + the token secret) started by the test."""
from __future__ import annotations

import logging

from cluster_kit import cluster_config
from colosseum.net.local import LearnerRoute
from colosseum.net.messages import VersionInfo, version_to_proto
from colosseum.node.dataport import DataServer
from colosseum.node.router import Counters, LocalDelivery
from node_kit import FakeHub, run_node_in_thread, toy_node_options, wait_for


def _config(tmp_path):
    return cluster_config(tmp_path, {"agents.main": "{}", "rollout.envs_per_worker": 2})


def _version(node, addresses=(), relay_only=False):
    return version_to_proto(VersionInfo(agent_id="main", generation=1, version=0, size=1, digest="0" * 32, node=node,
                                        addresses=tuple(addresses), relay_only=relay_only))


def _heartbeat_paths(hub):
    beats = hub.messages("heartbeat")
    return {} if not beats else {p.node: p.path for p in beats[-1].paths}


def test_an_unreachable_learner_node_is_down_named_once_and_its_seats_are_not_routed(tmp_path, caplog):
    hub = FakeHub(tmp_path, _config(tmp_path))
    node = run_node_in_thread(toy_node_options(tmp_path, hub, workers=0))
    try:
        with caplog.at_level(logging.WARNING, logger="colosseum.node"):
            wait_for(lambda: node.agent.run_id is not None, timeout=30)
            hub.announce(versions=[_version(7, relay_only=True)])
            wait_for(lambda: _heartbeat_paths(hub) == {7: "down"}, timeout=30)
            assert node.agent.paths() == {7: "down"}
            assert node.agent._learner_routes["main"] == LearnerRoute("main", None, False, 1)
            hub.announce(versions=[_version(7, relay_only=True)])               # nothing new: no second warning
            wait_for(lambda: len(hub.messages("heartbeat")) >= 3, timeout=30)
        said = [r.getMessage() for r in caplog.records if "data port of node 7" in r.getMessage()]
        events = [e.message for e in hub.node_events("warning") if "node 7" in e.message]
        assert len(said) == 1 and "relay is off" in said[0] and "['main']" in said[0]
        assert len(events) == 1
        assert node.stop() == 0
    finally:
        if node.thread.is_alive():
            node.stop()
        hub.stop()


def test_a_reachable_learner_node_is_direct_and_its_seats_go_via_the_node(tmp_path):
    hub = FakeHub(tmp_path, _config(tmp_path))
    port = DataServer(cache=None, own_agents=lambda: set(), max_message_mb=64, insecure=True,
                      token_secret=hub.token.secret, node=7, delivery=LocalDelivery(Counters()))
    port.start(0, credentials=None, host="127.0.0.1")
    node = run_node_in_thread(toy_node_options(tmp_path, hub, workers=0))
    try:
        wait_for(lambda: node.agent.run_id is not None, timeout=30)
        hub.announce(versions=[_version(7, [f"127.0.0.1:{port.port}"])])
        wait_for(lambda: _heartbeat_paths(hub) == {7: "direct"}, timeout=30)
        assert node.agent._learner_routes["main"] == LearnerRoute("main", None, True, 1)
        hub.announce(versions=[version_to_proto(VersionInfo(agent_id="main", generation=0, version=1, size=1,
                                                            digest="0" * 32, node=-1, relay_only=True))])
        wait_for(lambda: node.agent.paths() == {}, timeout=30)                 # parked weights: no route (P-PF3-13)
        assert node.agent._learner_routes["main"] == LearnerRoute("main", None, False, 0)
        assert node.stop() == 0
    finally:
        if node.thread.is_alive():
            node.stop()
        port.stop()
        hub.stop()
