"""Tic-Tac-Toe environment for Colosseum (turn-based, 2 players)."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from colosseum.envs.base_env import BaseEnv


class TicTacToeEnv(BaseEnv):
    """2-player Tic-Tac-Toe.

    Observation: 3x3x3 from the player's own perspective (own marks, opponent marks, empty).
    Action: 0-8 (board cell).

    Turn-based: every info dict carries
    - ``active``: True for the player whose move it is (the worker runs inference and
      records transitions only for active players);
    - ``action_mask``: bool[9], the empty cells. A full board gives all True, because no one
      acts on it and the worker requires a legal action in an acting player's mask.

    Rewards: win +1 / loss -1 / draw 0. An illegal move (impossible with the mask) loses: -1, +1 to the opponent.
    """

    WINNING_LINES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6],
    ]

    def __init__(self) -> None:
        self._board = np.zeros(9, dtype=np.int8)  # 0 empty, 1 player 0, 2 player 1
        self._current_player = 0
        self._done = False
        self._step_count = 0

    @property
    def num_players(self) -> int:
        return 2

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(low=0.0, high=1.0, shape=(3, 3, 3), dtype=np.float32)

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        return gymnasium.spaces.Discrete(9)

    def reset(self, seed: int | None = None) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
        if seed is not None:
            np.random.seed(seed)
        self._board = np.zeros(9, dtype=np.int8)
        self._current_player = 0
        self._done = False
        self._step_count = 0
        return self._all_obs(), self._infos()

    def step(self, actions: dict[int, Any]) -> tuple[
        dict[int, np.ndarray], dict[int, float], dict[int, bool], dict[int, bool], dict[int, dict],
    ]:
        if self._done:
            return (self._all_obs(), {0: 0.0, 1: 0.0}, {0: True, 1: True}, {0: False, 1: False}, self._infos())

        action = int(actions[self._current_player])
        other = 1 - self._current_player
        rewards = {0: 0.0, 1: 0.0}
        terminated = {0: False, 1: False}
        truncated = {0: False, 1: False}

        if action < 0 or action >= 9 or self._board[action] != 0:
            rewards[self._current_player] = -1.0
            rewards[other] = 1.0
            terminated = {0: True, 1: True}
            self._done = True
        else:
            self._board[action] = self._current_player + 1
            self._step_count += 1
            if self._check_win(self._current_player + 1):
                rewards[self._current_player] = 1.0
                rewards[other] = -1.0
                terminated = {0: True, 1: True}
                self._done = True
            elif self._step_count >= 9:
                terminated = {0: True, 1: True}
                self._done = True
            else:
                self._current_player = other

        return self._all_obs(), rewards, terminated, truncated, self._infos()

    def _infos(self) -> dict[int, dict]:
        legal = self._board == 0
        if not legal.any():
            legal = np.ones(9, dtype=bool)
        return {
            p: {"current_player": self._current_player,
                "active": p == self._current_player,
                "action_mask": legal.copy()}
            for p in range(2)
        }

    def _all_obs(self) -> dict[int, np.ndarray]:
        return {p: self._get_obs(p) for p in range(2)}

    def _get_obs(self, player: int) -> np.ndarray:
        my_mark = player + 1
        opp_mark = 2 - player
        board = self._board.reshape(3, 3)
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        obs[0] = board == my_mark
        obs[1] = board == opp_mark
        obs[2] = board == 0
        return obs

    def _check_win(self, mark: int) -> bool:
        return any(all(self._board[i] == mark for i in line) for line in self.WINNING_LINES)
