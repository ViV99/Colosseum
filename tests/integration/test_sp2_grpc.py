"""gRPC weight store, trajectory transport with chunk v2, and the distributed adapters (T6.4)."""
from __future__ import annotations

import queue
import socket
import time

import numpy as np
import pytest
import torch

from colosseum.sp2.core.types import TrajectoryChunk, WeightPayload
from game_helpers import chunk_v2_payload

grpc = pytest.importorskip("grpc")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _state_dict() -> dict[str, np.ndarray]:
    return {"w": np.random.randn(4, 4).astype(np.float32), "b": np.random.randn(4).astype(np.float32)}


def test_weight_store_roundtrip_and_adapters():
    from colosseum.sp2.distributed import GRPCWeightSink, GRPCWeightSource
    from colosseum.sp2.weight_store.grpc_store import GRPCWeightStore, serve_weight_store

    port = _free_port()
    server = serve_weight_store(port=port)
    time.sleep(0.3)
    store = GRPCWeightStore(f"localhost:{port}")
    try:
        assert store.get("agent_0") is None and store.get_version("agent_0") == -1
        sink, source = GRPCWeightSink(store, "agent_0"), GRPCWeightSource(store, "agent_0")
        with pytest.raises(queue.Empty):
            source.get_nowait()
        sd1 = _state_dict()
        sink.put_nowait(WeightPayload("agent_0", 1, sd1))
        payload = source.get_nowait()
        assert payload.policy_version == 1 and np.allclose(payload.state_dict["w"], sd1["w"])
        with pytest.raises(queue.Empty):  # no newer version: do not re-pull
            source.get_nowait()
        sink.put_nowait(WeightPayload("agent_0", 2, _state_dict()))
        assert source.get_nowait().policy_version == 2 and store.get_version("agent_0") == 2
    finally:
        store.close()
        server.stop(0)


def test_trajectory_transport_carries_chunk_v2_trees():
    from colosseum.sp2.distributed import GRPCTrajectorySink
    from colosseum.sp2.transport.grpc_transport import GRPCTransport, serve_trajectory_receiver

    port = _free_port()
    chunk_queue: queue.Queue = queue.Queue(maxsize=16)
    server = serve_trajectory_receiver(chunk_queue, port=port)
    time.sleep(0.3)
    transport = GRPCTransport(f"localhost:{port}")
    try:
        chunk = TrajectoryChunk.from_payload(chunk_v2_payload(version=7, agent_id="agent_0"))
        chunk.obs = {"grid": torch.randint(0, 255, (4, 2, 2), dtype=torch.uint8), "vec": chunk.obs}
        GRPCTrajectorySink(transport, "agent_0").put(chunk, timeout=1.0)
        received = TrajectoryChunk.from_payload(chunk_queue.get(timeout=2.0))
        assert received.policy_version == 7 and received.agent_id == "agent_0"
        assert received.obs["grid"].dtype == torch.uint8 and torch.equal(received.obs["grid"], chunk.obs["grid"])
        assert torch.equal(received.kind, chunk.kind) and torch.equal(received.action_masks, chunk.action_masks)
        assert transport.send_chunks_batch("agent_0", [chunk_v2_payload(), chunk_v2_payload(version=1)]) == 2
    finally:
        transport.close()
        server.stop(0)


def test_servicer_rejects_a_chunk_without_obs():
    from colosseum.sp2.transport.grpc_transport import GRPCTransport, serve_trajectory_receiver

    port = _free_port()
    chunk_queue: queue.Queue = queue.Queue(maxsize=8)
    server = serve_trajectory_receiver(chunk_queue, port=port)
    transport = GRPCTransport(f"localhost:{port}")
    try:
        bad = chunk_v2_payload()
        del bad["obs"]
        with pytest.raises(grpc.RpcError) as err:
            transport.send_chunk("agent_0", bad)
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT and "obs" in err.value.details()
        assert chunk_queue.empty()
        transport.send_chunk("agent_0", chunk_v2_payload())  # a good chunk still goes through
        assert TrajectoryChunk.from_payload(chunk_queue.get(timeout=2.0)).num_slots == 4
    finally:
        transport.close()
        server.stop(0)


def test_trajectory_sink_tolerates_a_dead_learner():
    from colosseum.sp2.distributed import GRPCTrajectorySink
    from colosseum.sp2.transport.grpc_transport import GRPCTransport

    transport = GRPCTransport(f"localhost:{_free_port()}")  # nothing listening
    GRPCTrajectorySink(transport, "agent_0").put(chunk_v2_payload(), timeout=0.5)
    transport.close()


def test_weight_store_rejects_non_array_weights():
    from colosseum.sp2.weight_store.grpc_store import GRPCWeightStore, serve_weight_store

    port = _free_port()
    server = serve_weight_store(port=port)
    client = GRPCWeightStore(f"localhost:{port}")
    try:
        with pytest.raises(grpc.RpcError) as err:
            client.put("agent_0", WeightPayload("agent_0", 1, {"w": np.zeros((2, 2), np.float32), "b": [1.0]}))
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT and "'b'" in err.value.details()
        assert client.get("agent_0") is None
    finally:
        client.close()
        server.stop(0)
