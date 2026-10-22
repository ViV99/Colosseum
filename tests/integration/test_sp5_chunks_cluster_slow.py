"""Chunk routing under network faults (SP5 T3.2, spec 4.7, 4.8, 7.1 "разрыв сети при живой машине" for the data
plane): a cut of the direct path moves the chunks to the hub relay and back, with every chunk accounted for; a
learner reachable only through the relay survives the loss of its relay inbox.

``cluster_kit``: the hub and both nodes in threads of the test process; node a runs the learner (no worker), node b
one worker (2 loaded processes). Node a's data port is announced behind a ``TcpFaultProxy``. Slow: each test runs
for tens of seconds."""
from __future__ import annotations

import contextlib
import time

import pytest

from cluster_kit import cluster_config, hub_version, start_hub, start_node, wait_until
from colosseum.node import router as router_mod
from colosseum.node.router import SENDER_QUEUE_FRAMES, STREAM_ROTATE_FRAMES, STREAM_ROTATE_SEC
from fault_proxy import TcpFaultProxy

pytestmark = pytest.mark.slow
CONNECT_SEC = 1.0                                     # cluster_kit's distributed.connect_timeout_sec


def _drops(counters: dict) -> int:
    return sum(v for k, v in counters.items() if k.startswith("dropped_chunks."))


@contextlib.contextmanager
def _cluster(tmp_path, *, stalled: bool):
    config = cluster_config(tmp_path, {"agents.main": {}, "training.total_timesteps": 10**9})
    proxy = TcpFaultProxy()
    if stalled:
        proxy.stall()
    hub = start_hub(tmp_path, config)
    nodes = []
    try:
        a = start_node(tmp_path, hub, "a", cpu_learners=1, workers=0, data_addresses=[proxy.address])
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
                if node.thread.is_alive():
                    node.stop()
            proxy.close()


def test_a_cut_is_survived_and_every_chunk_is_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(router_mod, "DIRECT_RETRY_SEC", 3.0)       # the nodes run in this process
    with _cluster(tmp_path, stalled=False) as (hub, a, b, proxy):
        assert wait_until(lambda: b.agent.counters().get("chunks_sent.direct", 0) > 0, 120), b.agent.counters()
        version = hub_version(hub, "main")
        proxy.cut(5.0)
        assert wait_until(lambda: b.agent.counters().get("chunks_sent.relay", 0) > 0,
                          CONNECT_SEC + STREAM_ROTATE_SEC + 4.0), b.agent.counters()
        assert wait_until(lambda: hub_version(hub, "main") > version, 5.0)          # training goes on in the cut
        direct = b.agent.counters()["chunks_sent.direct"]
        assert wait_until(lambda: b.agent.paths() == {a.agent.number: "direct"}
                          and b.agent.counters()["chunks_sent.direct"] > direct,
                          5.0 + 3.0 + STREAM_ROTATE_SEC + 2.0 + 10.0), (b.agent.paths(), b.agent.counters())
        assert b.stop() == 0                                 # its router counts everything still queued
        b_final = b.agent.counters()
        assert wait_until(lambda: _settled(a), 30)
        a_final = a.agent.counters()
        relay_drops = sum(v for k, v in hub.hub._relay.stats().items() if k.startswith("dropped_chunks."))
        routed = b_final["chunks_routed"]
        sent = b_final.get("chunks_sent.direct", 0) + b_final.get("chunks_sent.relay", 0)
        assert routed == sent + _drops(b_final)              # every frame b read is sent or counted
        assert b_final.get("dropped_chunks.disconnect", 0) <= STREAM_ROTATE_FRAMES + SENDER_QUEUE_FRAMES
        # what b sent arrived at a's learner or was counted on the way (a broken stream counts an upper bound)
        assert routed <= a_final.get("chunks_delivered", 0) + _drops(a_final) + relay_drops + _drops(b_final)
        assert hub.stop() == 0


def _settled(node) -> bool:
    """Nothing more arrived at ``node``'s learners within one second (what was in flight has landed)."""
    before = node.agent.counters().get("chunks_delivered", 0)
    time.sleep(1.0)
    return node.agent.counters().get("chunks_delivered", 0) == before


def test_a_relay_only_learner_survives_a_cut_of_the_hub_relay_stream(tmp_path):
    with _cluster(tmp_path, stalled=True) as (hub, a, b, _proxy):
        assert wait_until(lambda: (hub_version(hub, "main") or 0) >= 2, 180)
        assert wait_until(lambda: b.agent.counters().get("chunks_sent.relay", 0) > 0, 30)
        relay, number = hub.hub._relay, a.agent.number
        old = relay._dest(number).inbox
        assert old is not None
        relay.set_online({e.node_id: e.number for e in hub.hub._registry.online() if e.number != number})
        assert old.detached                                  # a's inbox and its pair buffers are gone
        hub.hub._publish_online()                            # a is back on the next pass
        assert wait_until(lambda: relay._dest(number).inbox not in (None, old), 30)   # a's inbox reattached
        version = hub_version(hub, "main")
        assert wait_until(lambda: hub_version(hub, "main") >= version + 2, 120)      # training goes on
        assert hub.stop() == 0
