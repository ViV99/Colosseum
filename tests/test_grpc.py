"""Tests for gRPC transport layer: serialization, weight store, trajectory transport."""
import os
import queue
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from colosseum.core.types import TrajectoryChunk, WeightPayload
from colosseum.transport.serialization import (
    deserialize_chunk,
    deserialize_state_dict,
    serialize_chunk,
    serialize_state_dict,
)


def _free_port():
    """Get a free TCP port from the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _make_state_dict():
    return {"w": torch.randn(10, 10), "b": torch.randn(10)}


def _make_chunk():
    return TrajectoryChunk(
        agent_id="test",
        observations=torch.randn(32, 3, 3, 3),
        actions=torch.randint(0, 9, (32,)),
        action_log_probs=torch.randn(32),
        rewards=torch.randn(32),
        dones=torch.zeros(32),
        values=torch.randn(32),
        bootstrap_value=torch.tensor(0.0),
        behavior_policy_version=5,
    )


def test_serialize_state_dict_compressed():
    sd = _make_state_dict()
    data, compressed = serialize_state_dict(sd, compress=True)
    assert compressed is True
    sd2 = deserialize_state_dict(data, compressed)
    assert torch.allclose(sd["w"], sd2["w"])


def test_serialize_state_dict_uncompressed():
    sd = _make_state_dict()
    data, compressed = serialize_state_dict(sd, compress=False)
    assert compressed is False
    sd2 = deserialize_state_dict(data, compressed)
    assert torch.allclose(sd["b"], sd2["b"])


def test_serialize_chunk_roundtrip():
    chunk = _make_chunk()
    data, compressed = serialize_chunk(chunk)
    chunk2 = deserialize_chunk("test", 5, data, compressed)
    assert torch.allclose(chunk.observations, chunk2.observations)
    assert torch.allclose(chunk.rewards, chunk2.rewards)
    assert chunk2.behavior_policy_version == 5


def test_grpc_weight_store():
    from colosseum.weight_store.grpc_store import GRPCWeightStore, serve_weight_store

    port = _free_port()
    server = serve_weight_store(port=port)
    time.sleep(0.3)

    try:
        client = GRPCWeightStore(f"localhost:{port}")

        # No weights yet
        assert client.get("agent_0") is None
        assert client.get_version("agent_0") == -1

        # Put and get
        sd = _make_state_dict()
        client.put("agent_0", WeightPayload("agent_0", 1, sd))
        result = client.get("agent_0")
        assert result is not None
        assert result.policy_version == 1
        assert torch.allclose(result.state_dict["w"], sd["w"])
        assert client.get_version("agent_0") == 1

        # Update
        sd2 = _make_state_dict()
        client.put("agent_0", WeightPayload("agent_0", 2, sd2))
        assert client.get_version("agent_0") == 2

        client.close()
    finally:
        server.stop(0)


def test_grpc_trajectory_transport():
    from colosseum.transport.grpc_transport import GRPCTransport, serve_trajectory_receiver

    port = _free_port()
    chunk_queue = queue.Queue(maxsize=64)
    server = serve_trajectory_receiver(chunk_queue, port=port)
    time.sleep(0.3)

    try:
        transport = GRPCTransport(f"localhost:{port}")
        transport.create_channel("agent_0")

        chunk = _make_chunk()

        # Send single
        transport.send_chunk("agent_0", chunk)
        received = chunk_queue.get(timeout=2.0)
        assert torch.allclose(received.observations, chunk.observations)

        # Send batch
        n = transport.send_chunks_batch("agent_0", [chunk, chunk])
        assert n == 2
        for _ in range(2):
            r = chunk_queue.get(timeout=2.0)
            assert torch.allclose(r.observations, chunk.observations)

        transport.close()
    finally:
        server.stop(0)
