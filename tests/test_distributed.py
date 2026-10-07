"""Tests for the distributed (gRPC) adapters and orchestration (C2)."""
import os
import queue
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from colosseum.core.types import TrajectoryChunk, WeightPayload


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _state_dict():
    return {"w": torch.randn(4, 4), "b": torch.randn(4)}


# ---------------------------------------------------------------------------
# Weight adapters: learner publishes via GRPCWeightSink, worker pulls via
# GRPCWeightSource (only newer versions, else Empty).
# ---------------------------------------------------------------------------

def test_grpc_weight_sink_and_source():
    from colosseum.distributed import GRPCWeightSink, GRPCWeightSource
    from colosseum.weight_store.grpc_store import GRPCWeightStore, serve_weight_store

    port = _free_port()
    server = serve_weight_store(port=port)
    time.sleep(0.3)
    try:
        store = GRPCWeightStore(f"localhost:{port}")
        sink = GRPCWeightSink(store, "agent_0")
        source = GRPCWeightSource(store, "agent_0")

        # Nothing published yet → Empty.
        try:
            source.get_nowait()
            assert False, "expected queue.Empty before any weights"
        except queue.Empty:
            pass

        # Learner publishes v1.
        sd1 = _state_dict()
        sink.put_nowait(WeightPayload("agent_0", 1, sd1))
        payload = source.get_nowait()
        assert payload.policy_version == 1
        assert torch.allclose(payload.state_dict["w"], sd1["w"])

        # No new version → Empty (don't re-pull the same weights).
        try:
            source.get_nowait()
            assert False, "expected Empty when version unchanged"
        except queue.Empty:
            pass

        # Learner publishes v2 → source returns it.
        sink.put_nowait(WeightPayload("agent_0", 2, _state_dict()))
        assert source.get_nowait().policy_version == 2

        store.close()
    finally:
        server.stop(0)


# ---------------------------------------------------------------------------
# Trajectory sink ships chunks (with masks — C9 over the wire) and tolerates
# transient RPC failures.
# ---------------------------------------------------------------------------

def test_grpc_trajectory_sink_sends_chunk_with_masks():
    from colosseum.distributed import GRPCTrajectorySink
    from colosseum.transport.grpc_transport import GRPCTransport, serve_trajectory_receiver

    port = _free_port()
    chunk_queue: queue.Queue = queue.Queue(maxsize=16)
    server = serve_trajectory_receiver(chunk_queue, port=port)
    time.sleep(0.3)
    try:
        transport = GRPCTransport(f"localhost:{port}")
        sink = GRPCTrajectorySink(transport, "agent_0")

        masks = torch.zeros(8, 5, dtype=torch.bool)
        masks[:, 1] = True
        chunk = TrajectoryChunk(
            agent_id="agent_0",
            observations=torch.randn(8, 4),
            actions=torch.randint(0, 5, (8,)),
            action_log_probs=torch.randn(8),
            rewards=torch.randn(8),
            dones=torch.zeros(8),
            values=torch.randn(8),
            bootstrap_value=torch.tensor(0.0),
            behavior_policy_version=7,
            action_masks=masks,
        )
        sink.put(chunk, timeout=1.0)

        received = chunk_queue.get(timeout=2.0)
        assert received.action_masks is not None
        assert torch.equal(received.action_masks, masks)
        transport.close()
    finally:
        server.stop(0)


def test_grpc_trajectory_sink_tolerates_dead_learner():
    """A send to a non-existent learner must not crash the worker."""
    from colosseum.distributed import GRPCTrajectorySink
    from colosseum.transport.grpc_transport import GRPCTransport

    transport = GRPCTransport(f"localhost:{_free_port()}")  # nothing listening
    sink = GRPCTrajectorySink(transport, "agent_0")
    chunk = TrajectoryChunk(
        agent_id="agent_0",
        observations=torch.randn(4, 4),
        actions=torch.randint(0, 5, (4,)),
        action_log_probs=torch.randn(4),
        rewards=torch.randn(4),
        dones=torch.zeros(4),
        values=torch.randn(4),
        bootstrap_value=torch.tensor(0.0),
        behavior_policy_version=0,
    )
    # Should swallow the RpcError, not raise.
    sink.put(chunk, timeout=0.5)
    transport.close()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
