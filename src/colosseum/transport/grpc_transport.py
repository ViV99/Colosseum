"""gRPC-based transport for trajectory chunks (Workers → Learners).

Provides:
- TrajectoryServer: receives streaming chunks from workers
- GRPCTransport client: sends chunks from worker to learner
"""

from __future__ import annotations

import logging
import queue
from concurrent import futures

import grpc

from colosseum.core.types import TrajectoryChunk
from colosseum.transport import colosseum_pb2, colosseum_pb2_grpc
from colosseum.transport.base import BaseTransport
from colosseum.transport.serialization import deserialize_chunk_payload, serialize_chunk_payload

logger = logging.getLogger(__name__)


# =====================================================================
# Server (runs in the learner process)
# =====================================================================


class TrajectoryServicer(colosseum_pb2_grpc.TrajectoryServiceServicer):
    """gRPC servicer that receives trajectory chunks from workers.

    Items put on ``chunk_queue`` are chunk payload dicts
    (``TrajectoryChunk.to_payload()`` form), never tensors.
    """

    def __init__(self, chunk_queue: queue.Queue, max_queue_size: int = 64) -> None:
        self._queue = chunk_queue

    def SendChunks(self, request_iterator, context):
        """Decode each chunk into its numpy payload dict and queue it for the learner."""
        count = 0
        for proto_chunk in request_iterator:
            payload = deserialize_chunk_payload(proto_chunk.tensor_data, proto_chunk.compressed)
            payload["agent_id"] = proto_chunk.agent_id
            payload["behavior_policy_version"] = int(proto_chunk.behavior_policy_version)
            try:
                self._queue.put(payload, timeout=5.0)
                count += 1
            except queue.Full:
                logger.warning("Trajectory queue full, dropping chunk")
        return colosseum_pb2.SendChunksResponse(chunks_received=count)


def serve_trajectory_receiver(
    chunk_queue: queue.Queue,
    port: int = 50052,
    max_workers: int = 4,
    max_message_mb: int = 64,
) -> grpc.Server:
    """Create and start a gRPC trajectory receiver server.

    Returns the server (call server.wait_for_termination() to block).
    """
    max_bytes = max_message_mb * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", max_bytes),
            ("grpc.max_receive_message_length", max_bytes),
        ],
    )
    colosseum_pb2_grpc.add_TrajectoryServiceServicer_to_server(
        TrajectoryServicer(chunk_queue), server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"TrajectoryService gRPC server started on port {port}")
    return server


# =====================================================================
# Client (used by workers)
# =====================================================================


class GRPCTransport(BaseTransport):
    """gRPC transport client for sending trajectory chunks to learners."""

    def __init__(self, address: str = "localhost:50052", max_message_mb: int = 64) -> None:
        max_bytes = max_message_mb * 1024 * 1024
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", max_bytes),
                ("grpc.max_receive_message_length", max_bytes),
            ],
        )
        self._stub = colosseum_pb2_grpc.TrajectoryServiceStub(self._channel)
        self._channels: set[str] = set()

    def create_channel(self, agent_id: str) -> None:
        self._channels.add(agent_id)

    @staticmethod
    def _to_proto(agent_id: str, chunk: TrajectoryChunk | dict) -> colosseum_pb2.TrajectoryChunkProto:
        payload = chunk.to_payload() if isinstance(chunk, TrajectoryChunk) else chunk
        data, compressed = serialize_chunk_payload(payload)
        return colosseum_pb2.TrajectoryChunkProto(
            agent_id=agent_id,
            behavior_policy_version=int(payload["behavior_policy_version"]),
            tensor_data=data,
            compressed=compressed,
        )

    def send_chunk(self, agent_id: str, chunk: TrajectoryChunk | dict) -> None:
        """Send one chunk (a TrajectoryChunk or its payload dict) in one streaming RPC."""
        self._stub.SendChunks(iter([self._to_proto(agent_id, chunk)]))

    def send_chunks_batch(self, agent_id: str, chunks: list) -> int:
        """Send several chunks (TrajectoryChunks or payload dicts) in one streaming RPC."""
        response = self._stub.SendChunks(self._to_proto(agent_id, c) for c in chunks)
        return response.chunks_received

    def recv_chunk(self, agent_id: str, timeout: float | None = None) -> TrajectoryChunk | None:
        """Not used on the client side — chunks are received by the server."""
        raise NotImplementedError("GRPCTransport.recv_chunk: use TrajectoryServicer on the server side")

    def close(self) -> None:
        self._channel.close()
