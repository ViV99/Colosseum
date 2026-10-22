"""``NodeData.Chunks`` on a node's data port (SP5 T3.2, spec 4.7): frames reach the local learner's chunk connection
undecoded, frames without a learner (or for another node) and frames to a dead learner are counted, a slow learner
holds the sending node back through gRPC flow control, and the call needs the token (``--insecure``) or a node
certificate (covered by ``test_sp5_data_port.py``)."""
from __future__ import annotations

import multiprocessing as mp
import threading
import time

import grpc
import numpy as np
import pytest

from colosseum.net.frame import encode_frame
from colosseum.net.grpc_util import channel_options
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.proto import colosseum_v1_pb2_grpc as pbg
from colosseum.net.security import MD_TOKEN
from colosseum.node.dataport import DataServer
from colosseum.node.router import Counters, LocalDelivery

MAX_MB = 64
SECRET = "sekret"


def _frame(agent="main", size=100, k=0):
    meta = {"agent_id": agent, "generation": 1, "policy_version": k, "node": 1, "worker": 0}
    return encode_frame({"x": np.full(size, k % 251, np.uint8)}, kind="chunk", meta=meta)


@pytest.fixture
def port():
    counters = Counters()
    delivery = LocalDelivery(counters)
    srv = DataServer(cache=None, own_agents=lambda: set(), max_message_mb=MAX_MB, insecure=True,
                     token_secret=SECRET, node=2, delivery=delivery)
    srv.start(0, credentials=None, host="127.0.0.1")
    channel = grpc.insecure_channel(f"127.0.0.1:{srv.port}", channel_options(MAX_MB))
    try:
        yield srv, delivery, counters, pbg.NodeDataStub(channel)
    finally:
        channel.close()
        srv.stop()


def _call(stub, frames, **kw):
    return stub.Chunks(iter(frames), metadata=[(MD_TOKEN, SECRET)], timeout=30, **kw)


def test_frames_reach_the_local_learner_undecoded(port):
    _srv, delivery, counters, stub = port
    learner, node_side = mp.Pipe()
    delivery.set_learner("main", node_side)
    frames = [_frame(k=k) for k in range(10)]
    summary = _call(stub, [pb.ChunkFrame(agent_id="main", target_node=2, generation=1, frame=f) for f in frames])
    assert [learner.recv_bytes() for _ in range(10)] == frames
    assert (summary.received, summary.dropped) == (10, 0)
    assert counters.snapshot()["chunks_delivered"] == 10


def test_frames_without_a_learner_or_with_a_dead_learner_are_counted(port):
    _srv, delivery, counters, stub = port
    learner, node_side = mp.Pipe()
    delivery.set_learner("main", node_side)
    learner.close()                                               # the learner is gone
    frames = ([pb.ChunkFrame(agent_id="ghost", target_node=2, frame=_frame("ghost"))] * 2
              + [pb.ChunkFrame(agent_id="main", target_node=3, frame=_frame())]       # another node's: never forwarded
              + [pb.ChunkFrame(agent_id="main", target_node=2, frame=_frame())] * 3)
    summary = _call(stub, frames)
    got = counters.snapshot()
    assert got["dropped_chunks.disconnect"] == 1                  # the first send that fails
    assert got["dropped_chunks.no_learner"] == 2 + 1 + 2          # then the entry is gone
    assert (summary.received, summary.dropped) == (6, 6) and "chunks_delivered" not in got


def test_a_slow_learner_backpressures_the_sending_node(port):
    _srv, delivery, counters, stub = port
    learner, node_side = mp.Pipe()
    delivery.set_learner("main", node_side)
    received = []

    def read():
        while len(received) < 100:
            received.append(learner.recv_bytes())
            time.sleep(0.02)
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    pulled = []

    def frames():
        for k in range(100):
            pulled.append(k)
            yield pb.ChunkFrame(agent_id="main", target_node=2, frame=_frame(size=200_000, k=k))
    result = []
    caller = threading.Thread(target=lambda: result.append(_call(stub, frames())), daemon=True)
    caller.start()
    time.sleep(0.5)
    assert len(pulled) < 60                                       # gRPC window + the pipe buffer: held back
    caller.join(30)
    reader.join(30)
    assert len(received) == 100 and (result[0].received, result[0].dropped) == (100, 0)
    assert counters.snapshot() == {"chunks_delivered": 100}


def test_chunks_need_the_token_or_a_node_certificate(port):
    _srv, _delivery, _counters, stub = port
    with pytest.raises(grpc.RpcError) as e:
        stub.Chunks(iter([pb.ChunkFrame(agent_id="main", target_node=2, frame=_frame())]), timeout=10)
    assert e.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_a_data_port_without_delivery_refuses_chunks(tmp_path):
    srv = DataServer(cache=None, own_agents=lambda: set(), max_message_mb=MAX_MB, insecure=True, token_secret=SECRET,
                     node=2)
    srv.start(0, credentials=None, host="127.0.0.1")
    channel = grpc.insecure_channel(f"127.0.0.1:{srv.port}", channel_options(MAX_MB))
    try:
        with pytest.raises(grpc.RpcError) as e:
            _call(pbg.NodeDataStub(channel), [pb.ChunkFrame(agent_id="main", target_node=2, frame=_frame())])
        assert e.value.code() == grpc.StatusCode.UNIMPLEMENTED
        delivery = LocalDelivery(Counters())
        srv.set_delivery(delivery)                                # the node's run started its router
        assert _call(pbg.NodeDataStub(channel), [pb.ChunkFrame(agent_id="x", target_node=2, frame=_frame("x"))]
                     ).dropped == 1
    finally:
        channel.close()
        srv.stop()
