from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium
import numpy as np


class BaseEnv(ABC):
    """N-player symmetric environment interface.

    All players share the same observation_space and action_space.
    The env processes all players' actions in a single step() call.

    Players are identified by integer indices 0..N-1.
    All return values are dicts keyed by player index.

    Turn-based convention (used by the rollout worker):
    - ``info[p]["active"]`` (bool) marks the players who act on the next step;
      players without the key act. A non-acting player runs no inference, sends
      a zero action (ignored by the env) and its model state does not advance.
    - ``info[p]["action_mask"]`` may be all-false for a non-acting player; an
      acting player must have at least one legal action (else EnvContractError).
    - A player's rewards are credited to its last action until it acts again; at
      episode end the last transition of every collecting player gets done=True.
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
        self, seed: int | None = None
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

    def render(self) -> Any | None:
        """Optional rendering."""
        return None
