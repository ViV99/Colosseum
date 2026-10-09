"""Chase: a 1v1 game with a tree action (reference example).

Each player moves on a continuous ``10 x 10`` field towards its own fixed target. An action is a
``Dict`` (built from a list, so the order is ``direction``, ``speed``): ``direction``:
``Discrete(4)`` (up, right, down, left) and ``speed``: ``Box(0, 1, (1,))``. After 20 steps the
player closer to its target wins (+1 / -1, equal distance is a draw). The end is a rule
(``truncated=False``); without an ``Outcome`` the framework derives the result from the returns.

Observation ``float32[5]``: own x, y, target x, y (all / 10), steps left fraction.
Layout ``2p`` (``GameSpec.symmetric(2, ...)``), simultaneous moves, no masks.
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, StepResult

DIRS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.float32)   # up, right, down, left
GRID = 10.0
MAX_STEPS = 20


def _scalar(value, name: str, player: int):
    """The one element of a scalar or size-1 array action component."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"player {player}: action component {name!r} must have one element, got shape {arr.shape}")
    return arr.reshape(-1)[0]


class ChaseGame(MultiAgentEnv):
    def __init__(self) -> None:
        obs_space = gymnasium.spaces.Box(0.0, 1.0, (5,), np.float32)
        act_space = gymnasium.spaces.Dict([
            ("direction", gymnasium.spaces.Discrete(4)),
            ("speed", gymnasium.spaces.Box(0.0, 1.0, (1,), np.float32)),
        ])
        self.spec = GameSpec.symmetric(2, obs_space, act_space)
        self._rng = np.random.default_rng()
        self._pos = np.zeros((2, 2), np.float32)
        self._target = np.zeros((2, 2), np.float32)
        self._t = 0

    def _obs(self, p: int) -> np.ndarray:
        return np.array([*(self._pos[p] / GRID), *(self._target[p] / GRID), 1.0 - self._t / MAX_STEPS], np.float32)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._pos = (self._rng.random((2, 2)) * GRID).astype(np.float32)
        self._target = (self._rng.random((2, 2)) * GRID).astype(np.float32)
        self._t = 0
        return StepResult(acting={0, 1}, obs={p: self._obs(p) for p in (0, 1)})

    def step(self, actions: dict) -> StepResult:
        for p in (0, 1):
            d = int(_scalar(actions[p]["direction"], "direction", p))
            s = float(np.clip(_scalar(actions[p]["speed"], "speed", p), 0.0, 1.0))
            self._pos[p] = np.clip(self._pos[p] + DIRS[d] * s, 0.0, GRID)
        self._t += 1
        if self._t < MAX_STEPS:
            return StepResult(acting={0, 1}, obs={p: self._obs(p) for p in (0, 1)})
        dist = np.linalg.norm(self._pos - self._target, axis=1)
        sign = float(np.sign(dist[1] - dist[0]))          # +1 when player 0 is closer
        return StepResult(acting=set(), obs={}, rewards={0: sign, 1: -sign}, episode_over=True)
