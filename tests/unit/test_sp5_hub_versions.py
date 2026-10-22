"""The hub's version fan-out (SP5 T3.1, spec 4.5, 3.1): an accepted version goes to every node with the learner
node's data addresses (``Announce``, ``Welcome``); new addresses re-announce the latest version at once; malformed
addresses are dropped and counted; evictions wait for a node marked lost and reach it with its next ``Welcome``."""
from __future__ import annotations

import itertools
import logging
import time

from colosseum.net.messages import VersionInfo, version_from_proto, version_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from game_helpers import make_test_config, scripted_agent
from hub_kit import lineup_hub, node_message, open_node


def _hub(tmp_path):
    return lineup_hub(make_test_config("turns", agents={"a": {}, "random": scripted_agent()}), tmp_path)


def _announce(hub, handle, version, node):
    info = VersionInfo(agent_id="a", generation=hub._versions.generation("a"), version=version, size=10,
                       digest="d" * 32, node=node, addresses=("6.6.6.6:1",), teacher_active=True)
    hub._on_message(node_message(handle, version=pb.VersionAnnounce(info=version_to_proto(info))))


def _addresses(hub, handle, *addresses):
    hub._on_message(node_message(handle, addresses=pb.DataAddresses(data_addresses=list(addresses))))


def _announced(handle):
    return [version_from_proto(v) for a in handle.of("announce") for v in a.versions]


def test_an_accepted_version_reaches_every_node_with_the_learner_nodes_addresses(tmp_path):
    hub = _hub(tmp_path)
    learner = open_node(hub, name="learner")
    other = open_node(hub, name="other", cpu_learners=0, no_learners=True)
    number = hub._learner_node_of("a")
    _addresses(hub, learner, "10.0.0.2:47102", "192.168.1.2:47102")
    hub._broadcast_announce()
    _announce(hub, learner, 3, number)
    hub._broadcast_announce()
    [info] = _announced(other)
    assert (info.version, info.node, info.addresses, info.relay_only, info.teacher_active) == (
        3, number, ("10.0.0.2:47102", "192.168.1.2:47102"), False, True)
    assert hub._versions.latest("a") == info                            # Welcome.versions too
    late = open_node(hub, name="late", cpu_learners=0, no_learners=True)
    assert [version_from_proto(v) for v in late.of("welcome")[0].versions] == [info]
    hub._broadcast_announce()
    assert _announced(other) == [info]                                  # announced once


def test_a_learner_node_without_addresses_is_relay_only_until_it_announces_them(tmp_path):
    hub = _hub(tmp_path)
    learner = open_node(hub, name="learner")
    other = open_node(hub, name="other", cpu_learners=0, no_learners=True)
    number = hub._learner_node_of("a")
    _announce(hub, learner, 0, number)
    hub._broadcast_announce()
    _addresses(hub, learner, "10.0.0.2:47199")                          # its data port was bound after Welcome
    hub._broadcast_announce()
    first, second = _announced(other)
    assert (first.version, first.relay_only, first.addresses) == (0, True, ())
    assert (second.version, second.relay_only, second.addresses) == (0, False, ("10.0.0.2:47199",))
    assert hub._registry.by_id(learner.node_id).data_addresses == ("10.0.0.2:47199",)


def test_malformed_data_addresses_are_dropped_and_counted(tmp_path, caplog):
    hub = _hub(tmp_path)
    node = open_node(hub, name="n")
    _addresses(hub, node, "10.0.0.2:1")
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.hub"):
        _addresses(hub, node, *[f"10.0.0.{i}:1" for i in range(17)])     # more than 16
        _addresses(hub, node, "x" * 256 + ":1")                          # too long
        _addresses(hub, node, "")
    assert hub._registry.by_id(node.node_id).data_addresses == ("10.0.0.2:1",)
    assert hub._book.get("node_messages.invalid") == 3
    assert caplog.text.count("node_messages.invalid") == 1


def test_evictions_wait_for_a_lost_node_and_reach_it_with_its_welcome(tmp_path):
    hub = _hub(tmp_path)
    node = open_node(hub, name="away", cpu_learners=0)
    for entry in hub._registry.lost(time.monotonic() + 3600.0, 10.0):  # silent too long: marked lost
        hub._on_node_lost(entry)
    assert hub._registry.by_id(node.node_id).state == "lost"
    evictions = itertools.chain([{"a": ["ckpt_v5"]}], itertools.repeat({}))   # T4.1's placement announces too
    hub._coordinator.take_evictions = lambda: next(evictions)
    hub._broadcast_announce()
    back = open_node(hub, node_id=node.node_id, session_id=node.session_id)
    assert [(e.agent_id, e.checkpoint_id) for e in back.of("welcome")[0].evictions] == [("a", "ckpt_v5")]
    hub._broadcast_announce()
    assert not back.of("announce")[-1].evictions


def test_hello_data_addresses_have_the_same_bounds(tmp_path, caplog):
    hub = _hub(tmp_path)
    learner = open_node(hub, name="learner", data_addresses=["10.0.0.2:47102"])
    other = open_node(hub, name="other", cpu_learners=0, no_learners=True)
    number = hub._learner_node_of("a")
    _announce(hub, learner, 0, number)
    hub._broadcast_announce()
    assert _announced(other)[-1].addresses == ("10.0.0.2:47102",)
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.hub"):     # the learner node reconnects
        back = open_node(hub, node_id=learner.node_id, session_id=learner.session_id,
                         data_addresses=[f"10.0.0.{i}:1" for i in range(17)])
    hub._broadcast_announce()
    assert hub._registry.by_id(back.node_id).data_addresses == ()
    assert hub._book.get("node_messages.invalid") == 1 and "node_messages.invalid" in caplog.text
    last = _announced(other)[-1]
    assert (last.version, last.relay_only, last.addresses) == (0, True, ())   # nothing over the bound goes out
