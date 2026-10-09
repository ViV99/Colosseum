"""Coin grid: a solo game (spec block 10).

One player walks on a ``size x size`` grid and collects coins; a collected coin reappears on a
random free cell. There is no rule-based end: the episode is cut by an artificial step limit, so
every episode ends with ``truncated=True`` and a ``final_obs`` (the learner bootstraps from it).

Exercises: ``GameSpec.solo``, a Dict observation with a ``uint8`` grid leaf and a float vector
leaf, action masks (moves off the grid are illegal), truncation with ``final_obs``.
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, StepResult

# stay, up, down, left, right as (drow, dcol)
MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)


class CoinGridGame(MultiAgentEnv):
    def __init__(self, size: int = 7, num_coins: int = 5, max_steps: int = 50) -> None:
        if size < 3 or not 1 <= num_coins < size * size:
            raise ValueError(f"need size >= 3 and 1 <= num_coins < size*size, got {size}, {num_coins}")
        self.size, self.num_coins, self.max_steps = size, num_coins, max_steps
        obs_space = gymnasium.spaces.Dict(
            [("grid", gymnasium.spaces.Box(0, 1, (2, size, size), np.uint8)),   # agent, coins
             ("vec", gymnasium.spaces.Box(0.0, 1.0, (1,), np.float32))])          # steps left / max_steps
        self.spec = GameSpec.solo(obs_space, gymnasium.spaces.Discrete(5))
        self._rng = np.random.default_rng()
        self._pos = np.zeros(2, np.int64)
        self._coins = np.zeros((size, size), bool)
        self._t = 0

    def _free_cell(self) -> np.ndarray:
        free = ~self._coins
        free[self._pos[0], self._pos[1]] = False
        cells = np.argwhere(free)
        return cells[self._rng.integers(len(cells))]

    def _obs(self) -> dict:
        grid = np.zeros((2, self.size, self.size), np.uint8)
        grid[0, self._pos[0], self._pos[1]] = 1
        grid[1] = self._coins
        vec = np.array([(self.max_steps - self._t) / self.max_steps], np.float32)
        return {"grid": grid, "vec": vec}

    def _mask(self) -> np.ndarray:
        nxt = self._pos + MOVES
        return ((nxt >= 0) & (nxt < self.size)).all(axis=1)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._coins[:] = False
        self._pos = self._rng.integers(self.size, size=2)
        for _ in range(self.num_coins):
            r, c = self._free_cell()
            self._coins[r, c] = True
        return StepResult(acting={0}, obs={0: self._obs()}, action_masks={0: self._mask()})

    def step(self, actions: dict) -> StepResult:
        move = int(actions[0])
        self._pos = np.clip(self._pos + MOVES[move], 0, self.size - 1)
        self._t += 1
        reward = 0.0
        if self._coins[self._pos[0], self._pos[1]]:
            reward = 1.0
            self._coins[self._pos[0], self._pos[1]] = False
            r, c = self._free_cell()
            self._coins[r, c] = True
        if self._t >= self.max_steps:
            return StepResult(acting=set(), obs={}, rewards={0: reward}, episode_over=True, truncated=True,
                              final_obs={0: self._obs()})
        return StepResult(acting={0}, obs={0: self._obs()}, action_masks={0: self._mask()}, rewards={0: reward})


def greedy_action(obs: dict) -> int:
    """Scripted reference: one step towards the nearest coin (used to sanity-check the env)."""
    agent = np.argwhere(obs["grid"][0])[0]
    coins = np.argwhere(obs["grid"][1])
    target = coins[np.abs(coins - agent).sum(axis=1).argmin()]
    d = target - agent
    if d[0] != 0:
        return 1 if d[0] < 0 else 2
    if d[1] != 0:
        return 3 if d[1] < 0 else 4
    return 0
