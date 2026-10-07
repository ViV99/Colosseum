"""Weight store using in-process dict with threading lock.

For single-machine mode with multiprocessing, weights are serialized
via torch.save and stored in a multiprocessing-safe manner using a Manager.
For simplicity in the initial implementation, we use a process-local dict
that is shared between threads in the same process. Cross-process weight
sharing is handled by the worker/learner pulling weights from the learner
process directly via mp.Queue or by using torch.save/load on disk.

In the MVP (Milestone 1), the weight store runs in the main process and
both worker and learner access it through mp.Queue-based message passing
managed by the launcher.
"""

from __future__ import annotations

import io
import threading

import torch

from colosseum.core.types import WeightPayload
from colosseum.weight_store.base import BaseWeightStore


class InMemoryWeightStore(BaseWeightStore):
    """Thread-safe in-memory weight store.

    Stores serialized state_dicts to avoid sharing tensor memory
    between processes (which requires special handling).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._weights: dict[str, bytes] = {}
        self._versions: dict[str, int] = {}

    def put(self, agent_id: str, payload: WeightPayload) -> None:
        """Serialize and store weights."""
        buf = io.BytesIO()
        torch.save(payload.state_dict, buf)
        serialized = buf.getvalue()

        with self._lock:
            self._weights[agent_id] = serialized
            self._versions[agent_id] = payload.policy_version

    def get(self, agent_id: str) -> WeightPayload | None:
        """Deserialize and return latest weights."""
        with self._lock:
            if agent_id not in self._weights:
                return None
            serialized = self._weights[agent_id]
            version = self._versions[agent_id]

        buf = io.BytesIO(serialized)
        state_dict = torch.load(buf, weights_only=True)
        return WeightPayload(agent_id=agent_id, policy_version=version, state_dict=state_dict)

    def get_version(self, agent_id: str) -> int:
        with self._lock:
            return self._versions.get(agent_id, -1)


class SharedMemoryWeightStore(BaseWeightStore):
    """Weight store using multiprocessing shared state.

    Uses a mp.Manager dict for cross-process weight sharing.
    This is simpler than raw shared memory but slightly slower.
    For the MVP this is sufficient; raw shared_memory optimization
    can be added later.
    """

    def __init__(self) -> None:
        import multiprocessing as mp

        self._manager = mp.Manager()
        self._weights = self._manager.dict()
        self._versions = self._manager.dict()
        self._lock = self._manager.Lock()

    def put(self, agent_id: str, payload: WeightPayload) -> None:
        buf = io.BytesIO()
        torch.save(payload.state_dict, buf)
        serialized = buf.getvalue()

        with self._lock:
            self._weights[agent_id] = serialized
            self._versions[agent_id] = payload.policy_version

    def get(self, agent_id: str) -> WeightPayload | None:
        with self._lock:
            if agent_id not in self._weights:
                return None
            serialized = self._weights[agent_id]
            version = self._versions[agent_id]

        buf = io.BytesIO(serialized)
        state_dict = torch.load(buf, weights_only=True)
        return WeightPayload(agent_id=agent_id, policy_version=version, state_dict=state_dict)

    def get_version(self, agent_id: str) -> int:
        with self._lock:
            return self._versions.get(agent_id, -1)
