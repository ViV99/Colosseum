"""Tic-Tac-Toe environment for testing Colosseum."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium
import numpy as np

from colosseum.envs.base_env import BaseEnv


class TicTacToeEnv(BaseEnv):
    """2-player Tic-Tac-Toe environment.

    Observation: 3x3x3 tensor (channels: current player's marks, opponent's marks, empty)
    Action: integer 0-8 (board position)
    Turn-based: only the active player's action is used.

    Rewards:
    - Win: +1 for winner, -1 for loser
    - Draw: 0 for both
    - Illegal move: -1 for the player, +1 for opponent (game ends)
    - Per-step: 0
    """

    WINNING_LINES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6],              # diags
    ]

    @property
    def num_players(self) -> int:
        return 2

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(low=0.0, high=1.0, shape=(3, 3, 3), dtype=np.float32)

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        return gymnasium.spaces.Discrete(9)

    def __init__(self) -> None:
        self._board = np.zeros(9, dtype=np.int8)  # 0=empty, 1=player0, 2=player1
        self._current_player = 0
        self._done = False
        self._step_count = 0

    def reset(self, seed: Optional[int] = None) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
        if seed is not None:
            np.random.seed(seed)
        self._board = np.zeros(9, dtype=np.int8)
        self._current_player = 0
        self._done = False
        self._step_count = 0

        obs = {i: self._get_obs(i) for i in range(2)}
        infos = {i: {"current_player": self._current_player} for i in range(2)}
        return obs, infos

    def step(self, actions: dict[int, Any]) -> tuple[
        dict[int, np.ndarray],
        dict[int, float],
        dict[int, bool],
        dict[int, bool],
        dict[int, dict],
    ]:
        if self._done:
            obs = {i: self._get_obs(i) for i in range(2)}
            return obs, {0: 0.0, 1: 0.0}, {0: True, 1: True}, {0: False, 1: False}, {0: {}, 1: {}}

        # Get active player's action
        action = int(actions[self._current_player])
        other_player = 1 - self._current_player

        rewards = {0: 0.0, 1: 0.0}
        terminated = {0: False, 1: False}
        truncated = {0: False, 1: False}

        # Check for illegal move
        if action < 0 or action >= 9 or self._board[action] != 0:
            # Illegal move: punish the player, reward opponent
            rewards[self._current_player] = -1.0
            rewards[other_player] = 1.0
            terminated = {0: True, 1: True}
            self._done = True
        else:
            # Place the mark
            self._board[action] = self._current_player + 1
            self._step_count += 1

            # Check for win
            if self._check_win(self._current_player + 1):
                rewards[self._current_player] = 1.0
                rewards[other_player] = -1.0
                terminated = {0: True, 1: True}
                self._done = True
            # Check for draw
            elif self._step_count >= 9:
                terminated = {0: True, 1: True}
                self._done = True
            else:
                # Switch player
                self._current_player = other_player

        obs = {i: self._get_obs(i) for i in range(2)}
        infos = {i: {"current_player": self._current_player} for i in range(2)}
        return obs, rewards, terminated, truncated, infos

    def _get_obs(self, player: int) -> np.ndarray:
        """Get observation from player's perspective.

        Channel 0: current player's marks
        Channel 1: opponent's marks
        Channel 2: empty squares
        """
        my_mark = player + 1
        opp_mark = 2 - player
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        board_2d = self._board.reshape(3, 3)
        obs[0] = (board_2d == my_mark).astype(np.float32)
        obs[1] = (board_2d == opp_mark).astype(np.float32)
        obs[2] = (board_2d == 0).astype(np.float32)
        return obs

    def _check_win(self, mark: int) -> bool:
        for line in self.WINNING_LINES:
            if all(self._board[i] == mark for i in line):
                return True
        return False
