"""In-process and Manager-backed weight stores holding numpy state dicts.

``WeightPayload.state_dict`` is a ``dict[str, np.ndarray]`` (spec block 2), so the
stores keep it as is. Numpy arrays pickle safely across processes, unlike torch
tensors shared through file descriptors (R6-02).

``InMemoryWeightStore`` backs the gRPC weight store server. ``SharedMemoryWeightStore``
is unused in SP1; kept for SP5 (distribution).
"""

from __future__ import annotations

import threading

from colosseum.core.types import WeightPayload
from colosseum.weight_store.base import BaseWeightStore


class InMemoryWeightStore(BaseWeightStore):
    """Thread-safe in-memory weight store (backs the gRPC weight store server).

    ``get`` returns a new dict over the stored arrays (no array copy); callers
    must not modify the arrays in place.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._weights: dict[str, dict] = {}
        self._versions: dict[str, int] = {}

    def put(self, agent_id: str, payload: WeightPayload) -> None:
        with self._lock:
            self._weights[agent_id] = dict(payload.state_dict)
            self._versions[agent_id] = payload.policy_version

    def get(self, agent_id: str) -> WeightPayload | None:
        with self._lock:
            if agent_id not in self._weights:
                return None
            return WeightPayload(
                agent_id=agent_id,
                policy_version=self._versions[agent_id],
                state_dict=dict(self._weights[agent_id]),
            )

    def get_version(self, agent_id: str) -> int:
        with self._lock:
            return self._versions.get(agent_id, -1)


class SharedMemoryWeightStore(BaseWeightStore):
    """Weight store backed by a ``multiprocessing.Manager`` dict.

    Unused in SP1; kept for SP5 (distribution). Single-machine weights travel through
    newest-wins ``mp.Queue`` mailboxes (``core.ipc.put_latest``).
    """

    def __init__(self) -> None:
        import multiprocessing as mp

        self._manager = mp.Manager()
        self._weights = self._manager.dict()
        self._versions = self._manager.dict()
        self._lock = self._manager.Lock()

    def put(self, agent_id: str, payload: WeightPayload) -> None:
        with self._lock:
            self._weights[agent_id] = dict(payload.state_dict)
            self._versions[agent_id] = payload.policy_version

    def get(self, agent_id: str) -> WeightPayload | None:
        with self._lock:
            if agent_id not in self._weights:
                return None
            state_dict = self._weights[agent_id]
            version = self._versions[agent_id]
        return WeightPayload(agent_id=agent_id, policy_version=version, state_dict=dict(state_dict))

    def get_version(self, agent_id: str) -> int:
        with self._lock:
            return self._versions.get(agent_id, -1)
