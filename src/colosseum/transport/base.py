from __future__ import annotations

from abc import ABC, abstractmethod

from colosseum.core.types import TrajectoryChunk


class BaseTransport(ABC):
    """Abstract transport layer for trajectory chunk delivery.

    Workers send trajectory chunks to learners through this interface.
    The implementation can be local (mp.Queue) or remote (gRPC).
    """

    @abstractmethod
    def send_chunk(self, agent_id: str, chunk: TrajectoryChunk) -> None:
        """Send a trajectory chunk to the learner for the given agent."""
        ...

    @abstractmethod
    def recv_chunk(self, agent_id: str, timeout: float | None = None) -> TrajectoryChunk | None:
        """Receive a trajectory chunk for the given agent's learner.

        Args:
            agent_id: which agent's learner to receive for
            timeout: seconds to wait (None = block forever, 0 = non-blocking)

        Returns:
            TrajectoryChunk or None if timeout elapsed
        """
        ...

    @abstractmethod
    def create_channel(self, agent_id: str) -> None:
        """Create a communication channel for an agent (e.g., create a queue)."""
        ...
