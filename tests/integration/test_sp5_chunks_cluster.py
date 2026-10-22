"""Chunks between machines with a real hub and two real nodes (SP5 T3.2, spec 4.7, 4.8): node b's worker collects for
the learner on node a; its frames leave on its route connection, go through node b's router in one mutual-TLS stream
to node a's data port and reach the learner over node a's own chunk connection, undecoded on the way. The learner
trains only on them (node a runs no worker).

When node a's data port does not answer (it is announced behind a stalled ``TcpFaultProxy``), chunks and the latest
weights go through the hub relay; with ``distributed.relay: false`` node b gets no collecting seat (and, unable to
fetch the learner's weights, plays nothing: W03-10).

``cluster_kit``: the hub and both nodes in threads of the test process (TLS hub port, mutual-TLS data ports); the
loaded processes are node a's learner and node b's worker (2)."""
from __future__ import annotations

import contextlib
import logging

from cluster_kit import (
    cluster,
    cluster_config,
    hub_counter,
    hub_version,
    journal_rows,
    start_hub,
    start_node,
    wait_until,
)
from fault_proxy import TcpFaultProxy

NODES = [dict(name="a", cpu_learners=1, workers=0), dict(name="b", workers=1, no_learners=True)]


def _drops(counters: dict) -> dict:
    return {k: v for k, v in counters.items() if k.startswith("dropped_chunks.") and v}


def _trains_over_the_direct_path(tmp_path, overrides):
    config = cluster_config(tmp_path, {"agents.main": {}, "training.total_timesteps": 10**9, **overrides})
    with cluster(tmp_path, config, NODES, learner_first="main") as (hub, (a, b)):
        assert wait_until(lambda: (hub_version(hub, "main") or 0) >= 4, 180), "the learner did not train"
        # a stream's frames are counted sent when it ends (STREAM_ROTATE_SEC)
        assert wait_until(lambda: b.agent.counters().get("chunks_sent.direct", 0) > 0, 30), b.agent.counters()
        b_counters, a_counters = b.agent.counters(), a.agent.counters()
        assert b_counters.get("chunks_sent.direct", 0) > 0 and "chunks_sent.relay" not in b_counters
        assert b_counters.get("chunks_routed", 0) >= b_counters["chunks_sent.direct"]
        assert a_counters.get("chunks_delivered", 0) > 0
        assert _drops(a_counters) == {} and _drops(b_counters) == {}
        assert b.agent.paths() == {a.agent.number: "direct"}
        assert a.agent.paths() == {}                          # its own learner: no remote target
        assert len(b.agent.fetch_history()["main"]) >= 2      # b follows the versions it trains

        def played_new_weights() -> bool:
            rows = [r for r in journal_rows(hub) if f"/n{b.agent.number}/" in r.uid]
            return any(s.version_end >= 1 for r in rows for s in r.seats if s.agent_id == "main")
        assert wait_until(played_new_weights, 60)
        assert hub.stop() == 0


def test_a_worker_on_another_node_trains_the_learner_directly(tmp_path):
    _trains_over_the_direct_path(tmp_path, {})


def test_relay_off_a_reachable_learner_collects_directly(tmp_path):
    """Pre-flight ruling P-PF3-1: with ``distributed.relay: false`` the node probes its path before any frame, reports
    ``direct`` and is granted collecting seats."""
    _trains_over_the_direct_path(tmp_path, {"distributed.relay": False})


@contextlib.contextmanager
def proxied_cluster(tmp_path, overrides, *, a_workers=0):
    """Node a (the learner) announces its data port as a ``TcpFaultProxy`` (stalled until the test restores it), then
    node b starts once a's learner announced a version."""
    config = cluster_config(tmp_path, {"agents.main": {}, "training.total_timesteps": 10**9, **overrides})
    proxy = TcpFaultProxy()
    proxy.stall()
    hub = start_hub(tmp_path, config)
    nodes = []
    try:
        a = start_node(tmp_path, hub, "a", cpu_learners=1, workers=a_workers, data_addresses=[proxy.address])
        nodes.append(a)
        assert wait_until(lambda: a.agent.data_port() is not None and hub_version(hub, "main") is not None, 120)
        proxy.set_target(("127.0.0.1", a.agent.data_port()))
        b = start_node(tmp_path, hub, "b", workers=1, no_learners=True)
        nodes.append(b)
        yield hub, a, b, proxy
    finally:
        try:
            hub.stop()
        finally:
            for node in nodes:
                node.stop()
            proxy.close()


def test_relay_carries_chunks_and_weights_when_the_data_port_is_unreachable(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="colosseum.node"), \
            proxied_cluster(tmp_path, {}) as (hub, a, b, _proxy):
        assert wait_until(lambda: (hub_version(hub, "main") or 0) >= 4, 180), "the learner did not train"
        assert wait_until(lambda: b.agent.counters().get("chunks_sent.relay", 0) > 0, 30), b.agent.counters()
        assert wait_until(lambda: len(b.agent.fetch_history().get("main", [])) >= 2, 60)
        b_counters, a_counters = b.agent.counters(), a.agent.counters()
        assert "chunks_sent.direct" not in b_counters
        assert a_counters.get("relay_frames_received", 0) > 0 and a_counters.get("chunks_delivered", 0) > 0
        assert b.agent.paths() == {a.agent.number: "relay"}
        assert list((hub.root / "blobs" / "relay").glob("latest-*-v*.frame"))      # relayed weights, uploaded once
        assert wait_until(lambda: hub_counter(hub, "relay_bytes") > 0, 30)        # after a metrics tick
        assert hub_counter(hub, "relay_uploads") >= 1
        assert _drops(a_counters) == {} and _drops(b_counters) == {}
        assert {k: v for k, v in hub.hub._relay.stats().items() if k.startswith("dropped_chunks.") and v} == {}
        warned = [r.getMessage() for r in caplog.records
                  if r.name == "colosseum.node.routing" and "through the hub" in r.getMessage()]
        assert len(warned) == 1 and f"node {a.agent.number}" in warned[0]
        assert hub.stop() == 0


def test_relay_off_without_a_snapshot_an_unreachable_node_waits_quietly(tmp_path, caplog):
    """Controller ruling R37 (spec 4.8), ``distributed.relay: false``: node b cannot reach the learner's data port, so
    ``main`` plays there only from a stored snapshot; it has none yet (node a has no worker: the learner never trains),
    so b's cells get no lineup and ask again quietly: no collecting seat, no chunk, no ``lineup_fetch_failed``; the
    hub and the node name the pair once, with the remedy. (Playing from a snapshot: the slow suite.)"""
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.hub"), \
            proxied_cluster(tmp_path, {"distributed.relay": False}) as (hub, a, b, _proxy):
        assert wait_until(lambda: b.agent.paths() == {a.agent.number: "down"}, 60)
        assert wait_until(lambda: [r for r in caplog.records if r.name == "colosseum.hub.hub"
                                   and "cannot reach the data port" in r.getMessage()], 30)
        # time passes by counters: b retries the learner's weights every sync interval (and fails: unreachable)
        assert wait_until(lambda: b.agent.counters().get("weight_fetch_failed", 0) >= 4, 60)
        counters = b.agent.counters()
        assert counters.get("lineup_fetch_failed", 0) == 0 and counters.get("chunks_routed", 0) == 0
        assert not [r for r in journal_rows(hub) if f"/n{b.agent.number}/" in r.uid]
        assert hub_version(hub, "main") == 0                                     # nothing collects anywhere
        said = [r.getMessage() for r in caplog.records if r.name == "colosseum.hub.hub"
                and "cannot reach the data port" in r.getMessage()]
        assert len(said) == 1 and f"node {a.agent.number}" in said[0] and "distributed.relay: true" in said[0]
        node_said = [r.getMessage() for r in caplog.records if r.name == "colosseum.node.routing"]
        assert len(node_said) == 1 and "only from their stored snapshots" in node_said[0]
        assert hub.stop() == 0
