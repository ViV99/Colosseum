"""Agent pool: registry of all agents (trainable, frozen, scripted)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    TRAINABLE = "trainable"
    FROZEN = "frozen"
    SCRIPTED = "scripted"


@dataclass
class AgentHandle:
    """Represents an agent in the pool."""

    agent_id: str
    agent_type: AgentType

    # For trainable agents
    network_config: Optional[dict] = None

    # For frozen agents
    checkpoint_path: Optional[str] = None
    checkpoint_id: Optional[str] = None

    # For scripted agents
    scripted_class: Optional[str] = None

    # Ratings (populated later in Milestone 4)
    elo: float = 1200.0
    win_rates: dict[str, float] = field(default_factory=dict)
    match_counts: dict[str, int] = field(default_factory=dict)


class AgentPool:
    """Registry of all agents."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentHandle] = {}

    def register_trainable(
        self,
        agent_id: str,
        network_config: Optional[dict] = None,
    ) -> AgentHandle:
        """Register a trainable agent with its own learner."""
        handle = AgentHandle(
            agent_id=agent_id,
            agent_type=AgentType.TRAINABLE,
            network_config=network_config,
        )
        self._agents[agent_id] = handle
        logger.info(f"Registered trainable agent: {agent_id}")
        return handle

    def register_frozen(
        self,
        agent_id: str,
        checkpoint_path: str,
        checkpoint_id: Optional[str] = None,
    ) -> AgentHandle:
        """Register a frozen checkpoint agent (no training)."""
        handle = AgentHandle(
            agent_id=agent_id,
            agent_type=AgentType.FROZEN,
            checkpoint_path=checkpoint_path,
            checkpoint_id=checkpoint_id,
        )
        self._agents[agent_id] = handle
        logger.info(f"Registered frozen agent: {agent_id}")
        return handle

    def register_scripted(
        self,
        agent_id: str,
        scripted_class: str,
    ) -> AgentHandle:
        """Register a scripted (rule-based) agent."""
        handle = AgentHandle(
            agent_id=agent_id,
            agent_type=AgentType.SCRIPTED,
            scripted_class=scripted_class,
        )
        self._agents[agent_id] = handle
        logger.info(f"Registered scripted agent: {agent_id}")
        return handle

    def get(self, agent_id: str) -> AgentHandle:
        """Get agent handle by ID."""
        if agent_id not in self._agents:
            raise KeyError(f"Agent not found: {agent_id}")
        return self._agents[agent_id]

    def remove(self, agent_id: str) -> None:
        """Remove an agent from the pool."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info(f"Removed agent: {agent_id}")

    def list_trainable(self) -> list[AgentHandle]:
        """List all trainable agents."""
        return [a for a in self._agents.values() if a.agent_type == AgentType.TRAINABLE]

    def list_all(self) -> list[AgentHandle]:
        """List all agents."""
        return list(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents
