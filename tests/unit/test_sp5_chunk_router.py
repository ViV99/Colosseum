"""The node chunk router without gRPC (SP5 T3.2, spec 4.7, 4.8): frames are routed by their header, unknown agents
and malformed frames are counted, a slow target holds the worker back, a down path drops and counts, paths are
probed before any frame (pre-flight ruling P-PF3-1), a failing stream is retried with a backoff (P-PF3-7), a replaced
learner connection is retired, never closed under a writer (P-PF3-6)."""
from __future__ import annotations

import multiprocessing as mp
import threading
import time

import grpc
import numpy as np

from colosseum.net.frame import encode_frame
from colosseum.node import router as router_mod
from colosseum.node.router import ChunkRouter, Counters, LocalDelivery, RemoteRoute


def _frame(agent, gen=1, size=100):
    """A frame whose header looks like a chunk frame's (``peek_meta`` is all the router reads)."""
    meta = {"agent_id": agent, "generation": gen, "policy_version": 0, "node": 1, "worker": 0}
    return encode_frame({"x": np.zeros(size, np.uint8)}, kind="chunk", meta=meta)


class FakeRelay:
    def __init__(self, delay=0.0):
        self.got = []
        self.delay = delay

    def __call__(self):
        def call(frames):
            for f in frames:
                time.sleep(self.delay)
                self.got.append(f)
            return None
        return call


class Unavailable(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.02)
    return pred()


def test_frames_go_to_their_agents_target_and_unknown_agents_are_counted():
    relay = FakeRelay()
    paths = []
    router = ChunkRouter(make_frame=lambda a, g, f, t: (a, g, t, len(f)), make_channel=None, direct_stub=None,
                         relay_call=relay, connect_timeout_sec=0.5, on_path=lambda t, p: paths.append((t, p)))
    router.set_routes({"m": RemoteRoute(node=2, generation=1, addresses=(), relay_only=True)})
    worker, node_side = mp.Pipe()
    router.add_worker(node_side, worker=0)
    for _ in range(5):
        worker.send_bytes(_frame("m"))
    worker.send_bytes(_frame("ghost"))
    worker.send_bytes(b"garbage")
    assert _wait(lambda: len(relay.got) == 5 and router.counters().get("dropped_chunks.invalid") == 1)
    assert relay.got[0][:3] == ("m", 1, 2)
    assert router.counters()["dropped_chunks.no_learner"] == 1
    assert router.counters()["chunks_routed"] == 7                     # every frame read (P-PF3-15)
    assert paths == [(2, "relay")] and router.paths() == {2: "relay"}
    router.stop()


def test_a_slow_target_blocks_the_worker_instead_of_dropping():
    relay = FakeRelay(delay=0.05)
    router = ChunkRouter(make_frame=lambda a, g, f, t: a, make_channel=None, direct_stub=None, relay_call=relay,
                         connect_timeout_sec=0.5)
    router.set_routes({"m": RemoteRoute(2, 1, (), True)})
    worker, node_side = mp.Pipe()
    router.add_worker(node_side, worker=0)
    sent = []

    def produce():
        for _ in range(60):
            worker.send_bytes(_frame("m", size=200_000))     # larger than the pipe buffer
            sent.append(1)
    t = threading.Thread(target=produce, daemon=True)
    t.start()
    time.sleep(0.5)
    assert len(sent) < 60                      # the worker is held back (the sender queue is full)
    assert _wait(lambda: len(relay.got) == 60, timeout=15)
    assert "dropped_chunks.disconnect" not in router.counters()
    router.stop()


def test_without_relay_and_without_a_direct_path_frames_are_dropped_and_the_path_is_down():
    paths = []
    router = ChunkRouter(make_frame=lambda a, g, f, t: a, make_channel=None, direct_stub=None, relay_call=None,
                         connect_timeout_sec=0.2, on_path=lambda t, p: paths.append(p))
    router.set_routes({"m": RemoteRoute(2, 1, (), True)})
    worker, node_side = mp.Pipe()
    router.add_worker(node_side, worker=0)
    for _ in range(3):
        worker.send_bytes(_frame("m"))
    assert _wait(lambda: router.counters().get("dropped_chunks.disconnect", 0) == 3)
    assert paths[0] == "down"
    router.stop()


def test_every_target_is_probed_before_any_frame_relay_on_or_off():
    """P-PF3-1: ``set_routes`` starts a sender per remote target at once, so the node reports a path
    (``Heartbeat.paths``) before the hub grants it collecting seats."""
    for relay, expected in ((None, "down"), (FakeRelay(), "relay")):
        _probe_eagerly(relay, expected)


def _probe_eagerly(relay, expected):
    paths = []
    router = ChunkRouter(make_frame=lambda a, g, f, t: a, make_channel=None, direct_stub=None, relay_call=relay,
                         connect_timeout_sec=0.2, on_path=lambda t, p: paths.append((t, p)))
    router.set_routes({"m": RemoteRoute(2, 1, (), True), "k": RemoteRoute(3, 1, (), True)})
    assert _wait(lambda: router.paths() == {2: expected, 3: expected})     # no worker, no frame
    assert sorted(paths) == [(2, expected), (3, expected)]
    router.set_routes({"m": RemoteRoute(2, 1, (), True)})                    # node 3 left: its sender goes
    assert _wait(lambda: router.paths() == {2: expected})
    router.stop()


def test_a_failing_stream_is_retried_with_a_backoff_and_its_frames_are_counted(monkeypatch):
    """P-PF3-7: after every failed stream the sender waits (RETRY_MIN_SEC doubling); waiting frames stay queued (the
    worker is held back); only the frames handed to a failed stream are counted ``disconnect``."""
    monkeypatch.setattr(router_mod, "RETRY_MIN_SEC", 0.3)
    calls, pulled = [], []

    def relay_call():
        def call(frames):
            calls.append(time.monotonic())
            pulled.append(next(iter(frames)))          # one frame taken, then the stream fails
            raise Unavailable()
        return call

    router = ChunkRouter(make_frame=lambda a, g, f, t: a, make_channel=None, direct_stub=None, relay_call=relay_call,
                         connect_timeout_sec=0.2)
    router.set_routes({"m": RemoteRoute(2, 1, (), True)})
    worker, node_side = mp.Pipe()
    router.add_worker(node_side, worker=0)
    sent = []

    def produce():
        try:
            for _ in range(200):
                worker.send_bytes(_frame("m", size=20_000))      # 4 MB in all: more than the pipe and the queue
                sent.append(1)
        except OSError:                                                  # the router stopped
            pass
    threading.Thread(target=produce, daemon=True).start()
    assert _wait(lambda: len(calls) >= 3, timeout=10)
    gaps = [b - a for a, b in zip(calls, calls[1:])]
    assert all(g >= 0.25 for g in gaps[:2])                              # spaced by the backoff, never a tight loop
    assert len(sent) < 200                                               # the rest waits: backpressure, not loss
    router.stop()
    counters = router.counters()
    assert counters["dropped_chunks.disconnect"] >= len(pulled)
    assert counters.get("chunks_sent.relay", 0) == 0


def test_a_replaced_learner_connection_is_retired_not_closed_under_a_writer():
    """P-PF3-6: a delivery blocked on a learner that does not read; ``set_learner`` swaps in a new connection from
    another thread without closing the old one under the writer; the old reader goes away -> that delivery is a
    ``disconnect``; the next frame goes to the new connection."""
    counters = Counters()
    delivery = LocalDelivery(counters)
    old_learner, old_node = mp.Pipe()
    new_learner, new_node = mp.Pipe()
    delivery.set_learner("m", old_node)
    outcome = []
    blocked = threading.Thread(target=lambda: outcome.append(delivery.deliver("m", b"x" * (4 << 20))), daemon=True)
    blocked.start()
    time.sleep(0.3)
    assert blocked.is_alive()                                            # the old learner does not read
    delivery.set_learner("m", new_node)                                  # returns at once (the writer holds the lock)
    assert not old_node.closed
    old_learner.close()                                                  # the old learner process is gone
    blocked.join(5)
    assert outcome == ["disconnect"]
    assert delivery.deliver("m", b"frame") is None
    assert new_learner.recv_bytes() == b"frame"
    assert _wait(lambda: old_node.closed)                                # retired once its writer let go
    assert counters.snapshot() == {"dropped_chunks.disconnect": 1, "chunks_delivered": 1}


def test_a_closed_learner_connection_is_a_disconnect_and_then_no_learner():
    counters = Counters()
    delivery = LocalDelivery(counters)
    learner, node_side = mp.Pipe()
    delivery.set_learner("m", node_side)
    assert delivery.deliver("m", b"a") is None and learner.recv_bytes() == b"a"
    node_side.close()                                                    # broken on this side (TypeError / OSError)
    assert delivery.deliver("m", b"b") == "disconnect"
    assert delivery.deliver("m", b"c") == "no_learner" and not delivery.connected("m")
    assert counters.snapshot() == {"chunks_delivered": 1, "dropped_chunks.disconnect": 1,
                                   "dropped_chunks.no_learner": 1}
