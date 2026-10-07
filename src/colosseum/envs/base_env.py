from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import gymnasium
import numpy as np


class BaseEnv(ABC):
    """N-player symmetric environment interface.

    All players share the same observation_space and action_space.
    The env processes all players' actions in a single step() call.

    Players are identified by integer indices 0..N-1.
    All return values are dicts keyed by player index.

    For turn-based games: only the active player's action matters;
    other players can pass any valid action (ignored by env).
    """

    @property
    @abstractmethod
    def num_players(self) -> int:
        """Number of players (fixed for this env type)."""
        ...

    @property
    @abstractmethod
    def observation_space(self) -> gymnasium.spaces.Space:
        """Observation space (same for all players)."""
        ...

    @property
    @abstractmethod
    def action_space(self) -> gymnasium.spaces.Space:
        """Action space (same for all players)."""
        ...

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None
    ) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
        """Reset environment.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            observations: dict mapping player_index -> observation
            infos: dict mapping player_index -> info dict
        """
        ...

    @abstractmethod
    def step(
        self, actions: dict[int, Any]
    ) -> tuple[
        dict[int, np.ndarray],  # observations per player
        dict[int, float],  # rewards per player
        dict[int, bool],  # terminated per player
        dict[int, bool],  # truncated per player
        dict[int, dict],  # infos per player
    ]:
        """Execute one timestep.

        Args:
            actions: dict mapping player_index -> action.

        For turn-based games: only the active player's action matters;
        other players can pass any valid action (ignored by env).

        An episode is done when terminated[i] is True for any player i.
        When the episode is done, terminated should be True for ALL players.

        Returns:
            observations: dict mapping player_index -> observation
            rewards: dict mapping player_index -> reward (float)
            terminated: dict mapping player_index -> terminated flag
            truncated: dict mapping player_index -> truncated flag
            infos: dict mapping player_index -> info dict
        """
        ...

    def close(self) -> None:
        """Clean up resources."""
        pass

    def render(self) -> Optional[Any]:
        """Optional rendering."""
        return None
