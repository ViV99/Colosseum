"""Chase game: 2-player grid with composite actions (direction + speed).

Each player picks a discrete direction (up/right/down/left) and a continuous
speed ∈ [0, 1].  Players move on a 10×10 grid.  The first player chases a
fixed target; the second player chases a different target.  Whoever is closer
to their target at episode end (20 steps) wins.

Demonstrates Dict action space: {"direction": Discrete(4), "speed": Box(1)}.
"""

import gymnasium
import numpy as np

from colosseum.envs.base_env import BaseEnv

_DIRS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.float32)  # URDL
_GRID = 10.0
_MAX_STEPS = 20


class ChaseEnv(BaseEnv):
    """Two players chase fixed targets on a grid with composite actions."""

    def __init__(self):
        self._positions = None
        self._targets = None
        self._step_count = 0

    @property
    def num_players(self) -> int:
        return 2

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        # [pos_x, pos_y, target_x, target_y, steps_remaining_normalized]
        return gymnasium.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

    @property
    def action_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Dict({
            "direction": gymnasium.spaces.Discrete(4),
            "speed": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    def _obs(self) -> dict[int, np.ndarray]:
        obs = {}
        for p in range(2):
            obs[p] = np.array([
                self._positions[p, 0] / _GRID,
                self._positions[p, 1] / _GRID,
                self._targets[p, 0] / _GRID,
                self._targets[p, 1] / _GRID,
                1.0 - self._step_count / _MAX_STEPS,
            ], dtype=np.float32)
        return obs

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self._positions = np.random.rand(2, 2).astype(np.float32) * _GRID
        self._targets = np.random.rand(2, 2).astype(np.float32) * _GRID
        self._step_count = 0
        return self._obs(), {0: {}, 1: {}}

    def step(self, actions):
        self._step_count += 1

        for p in range(2):
            a = actions[p]
            d = int(a["direction"])
            s = float(np.clip(a["speed"], 0.0, 1.0))
            self._positions[p] += _DIRS[d] * s
            self._positions[p] = np.clip(self._positions[p], 0.0, _GRID)

        done = self._step_count >= _MAX_STEPS
        dist = [np.linalg.norm(self._positions[p] - self._targets[p]) for p in range(2)]

        if done:
            if dist[0] < dist[1]:
                rewards = {0: 1.0, 1: -1.0}
            elif dist[1] < dist[0]:
                rewards = {0: -1.0, 1: 1.0}
            else:
                rewards = {0: 0.0, 1: 0.0}
        else:
            rewards = {0: 0.0, 1: 0.0}

        terminated = {0: done, 1: done}
        truncated = {0: False, 1: False}
        return self._obs(), rewards, terminated, truncated, {0: {}, 1: {}}

    def close(self):
        pass
