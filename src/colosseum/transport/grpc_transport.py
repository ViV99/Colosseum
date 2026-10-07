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
from colosseum.transport.serialization import deserialize_chunk, serialize_chunk

logger = logging.getLogger(__name__)


# =====================================================================
# Server (runs in the learner process)
# =====================================================================


class TrajectoryServicer(colosseum_pb2_grpc.TrajectoryServiceServicer):
    """gRPC servicer that receives trajectory chunks from workers."""

    def __init__(self, chunk_queue: queue.Queue, max_queue_size: int = 64) -> None:
        self._queue = chunk_queue

    def SendChunks(self, request_iterator, context):
        count = 0
        for proto_chunk in request_iterator:
            chunk = deserialize_chunk(
                agent_id=proto_chunk.agent_id,
                behavior_policy_version=proto_chunk.behavior_policy_version,
                data=proto_chunk.tensor_data,
                compressed=proto_chunk.compressed,
            )
            try:
                self._queue.put(chunk, timeout=5.0)
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

    def send_chunk(self, agent_id: str, chunk: TrajectoryChunk) -> None:
        """Send a single chunk (opens a stream, sends one chunk, closes)."""
        data, compressed = serialize_chunk(chunk)
        proto = colosseum_pb2.TrajectoryChunkProto(
            agent_id=agent_id,
            behavior_policy_version=chunk.behavior_policy_version,
            tensor_data=data,
            compressed=compressed,
        )
        self._stub.SendChunks(iter([proto]))

    def send_chunks_batch(self, agent_id: str, chunks: list[TrajectoryChunk]) -> int:
        """Send multiple chunks in a single streaming RPC (more efficient)."""
        def chunk_generator():
            for chunk in chunks:
                data, compressed = serialize_chunk(chunk)
                yield colosseum_pb2.TrajectoryChunkProto(
                    agent_id=agent_id,
                    behavior_policy_version=chunk.behavior_policy_version,
                    tensor_data=data,
                    compressed=compressed,
                )

        response = self._stub.SendChunks(chunk_generator())
        return response.chunks_received

    def recv_chunk(self, agent_id: str, timeout: float | None = None) -> TrajectoryChunk | None:
        """Not used on the client side — chunks are received by the server."""
        raise NotImplementedError("GRPCTransport.recv_chunk: use TrajectoryServicer on the server side")

    def close(self) -> None:
        self._channel.close()
