"""Local transport using multiprocessing queues.

For single-machine mode. Trajectory chunks flow from worker processes
to learner processes via mp.Queue.
"""

from __future__ import annotations

import multiprocessing as mp
from queue import Empty
from typing import Optional

from colosseum.core.types import TrajectoryChunk
from colosseum.transport.base import BaseTransport


class LocalTransport(BaseTransport):
    """Single-machine transport using multiprocessing.Queue."""

    def __init__(self, queue_size: int = 64) -> None:
        self._queue_size = queue_size
        self._queues: dict[str, mp.Queue] = {}

    def create_channel(self, agent_id: str) -> None:
        """Create a queue for an agent's trajectory chunks."""
        if agent_id not in self._queues:
            self._queues[agent_id] = mp.Queue(maxsize=self._queue_size)

    def send_chunk(self, agent_id: str, chunk: TrajectoryChunk) -> None:
        """Send chunk to the agent's queue. Blocks if queue is full."""
        if agent_id not in self._queues:
            raise KeyError(f"No channel for agent '{agent_id}'. Call create_channel() first.")
        self._queues[agent_id].put(chunk)

    def recv_chunk(self, agent_id: str, timeout: Optional[float] = None) -> Optional[TrajectoryChunk]:
        """Receive chunk from the agent's queue."""
        if agent_id not in self._queues:
            raise KeyError(f"No channel for agent '{agent_id}'. Call create_channel() first.")
        try:
            if timeout == 0:
                return self._queues[agent_id].get_nowait()
            else:
                return self._queues[agent_id].get(timeout=timeout)
        except Empty:
            return None

    def get_queue(self, agent_id: str) -> mp.Queue:
        """Get the underlying queue for direct use by worker/learner processes."""
        if agent_id not in self._queues:
            raise KeyError(f"No channel for agent '{agent_id}'.")
        return self._queues[agent_id]

    @property
    def agent_ids(self) -> list[str]:
        return list(self._queues.keys())
