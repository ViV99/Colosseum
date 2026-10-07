"""Matchmaking: generate match configurations for different training phases.

- SimpleSelfPlayMatchmaker: all slots latest, no checkpoints
- SelfPlayMatchmaker: agent plays against its own historical checkpoints
- PFSPMatchmaker: agents play against each other in a league with prioritized opponent selection
"""

from __future__ import annotations

import logging
import random
import uuid
from abc import ABC, abstractmethod

from colosseum.coordinator.agent_pool import AgentHandle, AgentPool
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.coordinator.ratings import WinRateTracker
from colosseum.core.types import MatchConfig, PlayerSlot

logger = logging.getLogger(__name__)


class BaseMatchmaker(ABC):
    """Abstract matchmaker: generates match configurations."""

    @abstractmethod
    def generate_matches(
        self,
        agent_id: str,
        num_envs: int,
        num_players: int,
    ) -> list[MatchConfig]:
        """Generate match configs for a given agent's training.

        Args:
            agent_id: the trainable agent we're generating matches for
            num_envs: number of environments to fill
            num_players: players per environment

        Returns:
            list of MatchConfig, one per env
        """
        ...


class SimpleSelfPlayMatchmaker(BaseMatchmaker):
    """All player slots = same agent with latest weights. No checkpoints.

    This is the Milestone 1 matchmaker (used when no checkpoints exist yet).
    """

    def generate_matches(
        self,
        agent_id: str,
        num_envs: int,
        num_players: int,
    ) -> list[MatchConfig]:
        configs = []
        for _ in range(num_envs):
            slots = [
                PlayerSlot(agent_id=agent_id, checkpoint_id=None, collect_trajectories=True)
                for _ in range(num_players)
            ]
            configs.append(MatchConfig(
                match_id=str(uuid.uuid4())[:8],
                env_config={},
                player_slots=slots,
            ))
        return configs


class SelfPlayMatchmaker(BaseMatchmaker):
    """Self-play against pool of own historical checkpoints.

    For each env's N player slots:
    - Slot 0: always latest weights (collect trajectories)
    - Slots 1..N-1: with prob `latest_prob` use latest (collect trajectories),
      otherwise use random historical checkpoint (don't collect)

    This ensures at least one slot always collects trajectories while
    providing opponent diversity through historical checkpoints.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        latest_prob: float = 0.5,
    ) -> None:
        self._ckpt_mgr = checkpoint_manager
        self._latest_prob = latest_prob

    def generate_matches(
        self,
        agent_id: str,
        num_envs: int,
        num_players: int,
    ) -> list[MatchConfig]:
        checkpoints = self._ckpt_mgr.list_checkpoints(agent_id)
        configs = []

        for _ in range(num_envs):
            slots = []
            for player_idx in range(num_players):
                if player_idx == 0:
                    # First slot: always latest, always collects
                    slots.append(PlayerSlot(
                        agent_id=agent_id,
                        checkpoint_id=None,
                        collect_trajectories=True,
                    ))
                elif not checkpoints or random.random() < self._latest_prob:
                    # Use latest weights
                    slots.append(PlayerSlot(
                        agent_id=agent_id,
                        checkpoint_id=None,
                        collect_trajectories=True,
                    ))
                else:
                    # Use historical checkpoint
                    ckpt = random.choice(checkpoints)
                    slots.append(PlayerSlot(
                        agent_id=agent_id,
                        checkpoint_id=ckpt.checkpoint_id,
                        collect_trajectories=False,
                    ))

            configs.append(MatchConfig(
                match_id=str(uuid.uuid4())[:8],
                env_config={},
                player_slots=slots,
            ))

        return configs


class PFSPMatchmaker(BaseMatchmaker):
    """Prioritized Fictitious Self-Play matchmaker for league training.

    For each env, randomly decide between:
    - Solo match (prob = self_play_ratio): play against own checkpoints
      (same as SelfPlayMatchmaker)
    - Arena match (prob = 1 - self_play_ratio): play against another
      trainable agent from the pool, selected via PFSP priority

    PFSP priority function: f(win_rate) = (1 - win_rate)^p
    This focuses training on hard opponents (low win rate).

    In arena matches, ALL trainable agents collect trajectories (each slot
    feeds its own learner). If fewer agents than player slots, agents are
    duplicated to fill remaining slots.
    """

    def __init__(
        self,
        agent_pool: AgentPool,
        checkpoint_manager: CheckpointManager,
        win_rate_tracker: WinRateTracker,
        self_play_ratio: float = 0.5,
        pfsp_exponent: float = 1.0,
        latest_prob: float = 0.5,
    ) -> None:
        self._pool = agent_pool
        self._ckpt_mgr = checkpoint_manager
        self._win_rates = win_rate_tracker
        self._self_play_ratio = self_play_ratio
        self._pfsp_exponent = pfsp_exponent
        self._latest_prob = latest_prob

    def generate_matches(
        self,
        agent_id: str,
        num_envs: int,
        num_players: int,
    ) -> list[MatchConfig]:
        trainable = self._pool.list_trainable()
        other_agents = [a for a in trainable if a.agent_id != agent_id]

        configs = []
        for _ in range(num_envs):
            if not other_agents or random.random() < self._self_play_ratio:
                # Solo match: play against own checkpoints
                configs.append(self._solo_match(agent_id, num_players))
            else:
                # Arena match: play against another agent
                opponent = self._select_opponent(agent_id, other_agents)
                configs.append(self._arena_match(agent_id, opponent.agent_id, num_players))

        return configs

    def _solo_match(self, agent_id: str, num_players: int) -> MatchConfig:
        """Generate a solo self-play match against own checkpoints."""
        checkpoints = self._ckpt_mgr.list_checkpoints(agent_id)
        slots = []
        for i in range(num_players):
            if i == 0:
                slots.append(PlayerSlot(
                    agent_id=agent_id, checkpoint_id=None, collect_trajectories=True,
                ))
            elif not checkpoints or random.random() < self._latest_prob:
                slots.append(PlayerSlot(
                    agent_id=agent_id, checkpoint_id=None, collect_trajectories=True,
                ))
            else:
                ckpt = random.choice(checkpoints)
                slots.append(PlayerSlot(
                    agent_id=agent_id, checkpoint_id=ckpt.checkpoint_id,
                    collect_trajectories=False,
                ))
        return MatchConfig(
            match_id=str(uuid.uuid4())[:8], env_config={}, player_slots=slots,
        )

    def _arena_match(self, agent_id: str, opponent_id: str, num_players: int) -> MatchConfig:
        """Generate an arena match: both agents collect trajectories."""
        agents = [agent_id, opponent_id]
        slots = []
        for i in range(num_players):
            aid = agents[i % len(agents)]
            slots.append(PlayerSlot(
                agent_id=aid, checkpoint_id=None, collect_trajectories=True,
            ))
        return MatchConfig(
            match_id=str(uuid.uuid4())[:8], env_config={}, player_slots=slots,
        )

    def _select_opponent(self, agent_id: str, candidates: list[AgentHandle]) -> AgentHandle:
        """Select opponent using PFSP priority: f(wr) = (1-wr)^p."""
        if not candidates:
            raise ValueError("No candidates for PFSP selection")

        weights = []
        for c in candidates:
            wr = self._win_rates.get_win_rate(agent_id, c.agent_id)
            priority = max(1e-6, (1.0 - wr) ** self._pfsp_exponent)
            weights.append(priority)

        total = sum(weights)
        if total == 0:
            probs = [1.0 / len(weights)] * len(weights)
        else:
            probs = [w / total for w in weights]

        return random.choices(candidates, weights=probs, k=1)[0]
