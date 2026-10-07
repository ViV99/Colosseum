"""Serialization utilities for gRPC transport.

Converts TrajectoryChunks and WeightPayloads to/from bytes using
torch.save() for tensors + optional lz4 compression.
"""

from __future__ import annotations

import io

import lz4.frame
import torch

from colosseum.core.types import TrajectoryChunk


def serialize_state_dict(state_dict: dict, compress: bool = True) -> tuple[bytes, bool]:
    """Serialize a state_dict to bytes, optionally lz4-compressed."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    data = buf.getvalue()
    if compress:
        data = lz4.frame.compress(data)
    return data, compress


def deserialize_state_dict(data: bytes, compressed: bool) -> dict:
    """Deserialize a state_dict from bytes."""
    if compressed:
        data = lz4.frame.decompress(data)
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=True)


def serialize_chunk(chunk: TrajectoryChunk, compress: bool = True) -> tuple[bytes, bool]:
    """Serialize trajectory chunk tensor data to bytes."""
    tensor_dict = {
        "observations": chunk.observations,
        "actions": chunk.actions,
        "action_log_probs": chunk.action_log_probs,
        "rewards": chunk.rewards,
        "dones": chunk.dones,
        "values": chunk.values,
        "bootstrap_value": chunk.bootstrap_value,
    }
    if chunk.lstm_hidden is not None:
        tensor_dict["lstm_h"] = chunk.lstm_hidden[0]
        tensor_dict["lstm_c"] = chunk.lstm_hidden[1]
    if chunk.action_masks is not None:
        tensor_dict["action_masks"] = chunk.action_masks

    buf = io.BytesIO()
    torch.save(tensor_dict, buf)
    data = buf.getvalue()
    if compress:
        data = lz4.frame.compress(data)
    return data, compress


def deserialize_chunk(
    agent_id: str,
    behavior_policy_version: int,
    data: bytes,
    compressed: bool,
) -> TrajectoryChunk:
    """Deserialize trajectory chunk from bytes."""
    if compressed:
        data = lz4.frame.decompress(data)
    buf = io.BytesIO(data)
    td = torch.load(buf, weights_only=True)

    lstm_hidden = None
    if "lstm_h" in td and "lstm_c" in td:
        lstm_hidden = (td["lstm_h"], td["lstm_c"])

    return TrajectoryChunk(
        agent_id=agent_id,
        observations=td["observations"],
        actions=td["actions"],
        action_log_probs=td["action_log_probs"],
        rewards=td["rewards"],
        dones=td["dones"],
        values=td["values"],
        bootstrap_value=td["bootstrap_value"],
        behavior_policy_version=behavior_policy_version,
        lstm_hidden=lstm_hidden,
        action_masks=td.get("action_masks"),
    )
