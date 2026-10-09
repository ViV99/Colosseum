"""Tic-tac-toe on the SP2 contract: a turn-based two-player ``MultiAgentEnv`` with action masks."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult

WINNING_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


class TicTacToeGame(MultiAgentEnv):
    """Two seats (layout ``"2p"``, role ``"player"``; team = seat), seat 0 moves first.

    - Only the player to move acts: ``acting`` is ``{current}``; it gets the observation and the
      mask of the empty cells.
    - Observation ``float32[3, 3, 3]`` from the mover's side: own marks, opponent marks, empty.
    - Action ``Discrete(9)``: the cell.
    - The last step gives +1 / -1 to the winner / loser (both seats, so the waiting loser's last
      move is credited too) or 0 / 0 for a draw, with ``Outcome.team_rank``. An illegal move
      (impossible with the mask) loses.
    """

    OBSERVATION_SPACE = gymnasium.spaces.Box(0.0, 1.0, (3, 3, 3), np.float32)
    ACTION_SPACE = gymnasium.spaces.Discrete(9)
    spec = GameSpec.symmetric(2, OBSERVATION_SPACE, ACTION_SPACE)

    def __init__(self) -> None:
        self._board = np.zeros(9, dtype=np.int8)  # 0 empty, 1 seat 0, 2 seat 1
        self._current = 0

    def reset(self, seed: int | None, layout: str) -> StepResult:
        if layout != "2p":
            raise ValueError(f"TicTacToeGame has the single layout '2p', got {layout!r}")
        self._board[:] = 0
        self._current = 0
        return self._turn()

    def step(self, actions: dict[int, Any]) -> StepResult:
        mover, other = self._current, 1 - self._current
        cell = int(actions[mover])
        if not (0 <= cell < 9) or self._board[cell] != 0:
            return self._end(winner=other)
        self._board[cell] = mover + 1
        if any(all(self._board[i] == mover + 1 for i in line) for line in WINNING_LINES):
            return self._end(winner=mover)
        if not (self._board == 0).any():
            return self._end(winner=None)
        self._current = other
        return self._turn()

    def _turn(self) -> StepResult:
        seat = self._current
        return StepResult(acting={seat}, obs={seat: self._obs(seat)}, action_masks={seat: self._board == 0})

    def _end(self, winner: int | None) -> StepResult:
        if winner is None:
            rewards, ranks = {0: 0.0, 1: 0.0}, {0: 1.0, 1: 1.0}
        else:
            rewards = {winner: 1.0, 1 - winner: -1.0}
            ranks = {winner: 1.0, 1 - winner: 2.0}
        return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True,
                          outcome=Outcome(team_rank=ranks))

    def _obs(self, seat: int) -> np.ndarray:
        board = self._board.reshape(3, 3)
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        obs[0] = board == seat + 1
        obs[1] = board == 2 - seat
        obs[2] = board == 0
        return obs
