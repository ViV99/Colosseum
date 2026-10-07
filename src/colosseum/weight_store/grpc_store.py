"""gRPC-based weight store for distributed training.

Provides a WeightStoreServer that wraps InMemoryWeightStore behind a gRPC
service, and a GRPCWeightStore client that implements BaseWeightStore.
"""

from __future__ import annotations

import logging
from concurrent import futures

import grpc

from colosseum.core.types import WeightPayload
from colosseum.transport import colosseum_pb2, colosseum_pb2_grpc
from colosseum.transport.serialization import deserialize_state_dict, serialize_state_dict
from colosseum.weight_store.base import BaseWeightStore
from colosseum.weight_store.shared_memory import InMemoryWeightStore

logger = logging.getLogger(__name__)


# =====================================================================
# Server
# =====================================================================


class WeightStoreServicer(colosseum_pb2_grpc.WeightStoreServiceServicer):
    """gRPC servicer wrapping an InMemoryWeightStore."""

    def __init__(self) -> None:
        self._store = InMemoryWeightStore()

    def PutWeights(self, request, context):
        state_dict = deserialize_state_dict(
            request.state_dict_bytes, request.compressed,
        )
        payload = WeightPayload(
            agent_id=request.agent_id,
            policy_version=request.policy_version,
            state_dict=state_dict,
        )
        self._store.put(request.agent_id, payload)
        logger.debug(
            f"WeightStore: stored v{request.policy_version} for {request.agent_id}"
        )
        return colosseum_pb2.PutWeightsResponse(success=True)

    def GetWeights(self, request, context):
        payload = self._store.get(request.agent_id)
        if payload is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"No weights for agent {request.agent_id}")
            return colosseum_pb2.WeightPayloadProto()

        data, compressed = serialize_state_dict(payload.state_dict)
        return colosseum_pb2.WeightPayloadProto(
            agent_id=payload.agent_id,
            policy_version=payload.policy_version,
            state_dict_bytes=data,
            compressed=compressed,
        )

    def GetVersion(self, request, context):
        version = self._store.get_version(request.agent_id)
        return colosseum_pb2.GetVersionResponse(policy_version=version)


def serve_weight_store(port: int = 50051, max_workers: int = 4, max_message_mb: int = 64) -> grpc.Server:
    """Create and start a gRPC weight store server.

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
    colosseum_pb2_grpc.add_WeightStoreServiceServicer_to_server(
        WeightStoreServicer(), server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"WeightStore gRPC server started on port {port}")
    return server


# =====================================================================
# Client
# =====================================================================


class GRPCWeightStore(BaseWeightStore):
    """gRPC client implementing BaseWeightStore."""

    def __init__(self, address: str = "localhost:50051", max_message_mb: int = 64) -> None:
        max_bytes = max_message_mb * 1024 * 1024
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", max_bytes),
                ("grpc.max_receive_message_length", max_bytes),
            ],
        )
        self._stub = colosseum_pb2_grpc.WeightStoreServiceStub(self._channel)

    def put(self, agent_id: str, payload: WeightPayload) -> None:
        data, compressed = serialize_state_dict(payload.state_dict)
        request = colosseum_pb2.WeightPayloadProto(
            agent_id=agent_id,
            policy_version=payload.policy_version,
            state_dict_bytes=data,
            compressed=compressed,
        )
        self._stub.PutWeights(request)

    def get(self, agent_id: str) -> WeightPayload | None:
        request = colosseum_pb2.GetWeightsRequest(agent_id=agent_id)
        try:
            response = self._stub.GetWeights(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise
        state_dict = deserialize_state_dict(
            response.state_dict_bytes, response.compressed,
        )
        return WeightPayload(
            agent_id=response.agent_id,
            policy_version=response.policy_version,
            state_dict=state_dict,
        )

    def get_version(self, agent_id: str) -> int:
        request = colosseum_pb2.GetVersionRequest(agent_id=agent_id)
        response = self._stub.GetVersion(request)
        return response.policy_version

    def close(self) -> None:
        self._channel.close()
