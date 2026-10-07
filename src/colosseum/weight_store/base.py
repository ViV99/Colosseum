from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from colosseum.core.types import WeightPayload


class BaseWeightStore(ABC):
    """Abstract weight store for sharing model weights between learners and workers."""

    @abstractmethod
    def put(self, agent_id: str, payload: WeightPayload) -> None:
        """Publish new weights for an agent."""
        ...

    @abstractmethod
    def get(self, agent_id: str) -> Optional[WeightPayload]:
        """Get latest weights for an agent. Returns None if no weights published yet."""
        ...

    @abstractmethod
    def get_version(self, agent_id: str) -> int:
        """Get the current policy version without fetching full weights.

        Returns -1 if no weights have been published for this agent.
        """
        ...
