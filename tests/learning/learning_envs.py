"""Tiny single-player envs that APPO must solve in seconds (spec §3.3).

The model for them is ``helpers.make_simple_model`` (an MLP ComposedModel).
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.base_env import BaseEnv


class ContextualBandit(BaseEnv):
    """One step per episode. Observation: one-hot context c in {0, 1}. Reward 1 iff action == c."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng()
        self._context = 0

    @property
    def num_players(self) -> int:
        return 1

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(2)

    def _obs(self):
        obs = np.zeros(2, np.float32)
        obs[self._context] = 1.0
        return {0: obs}

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._context = int(self._rng.integers(2))
        return self._obs(), {0: {}}

    def step(self, actions):
        reward = 1.0 if int(actions[0]) == self._context else 0.0
        return self._obs(), {0: reward}, {0: True}, {0: False}, {0: {}}


class ShortChain(BaseEnv):
    """Combination lock: states 0..4, start at 0, actions 0/1. The correct action at state i
    is ``KEY[i]`` and advances to i+1; a wrong action steps back to max(0, i-1). Reaching 4
    gives +1 and ends the episode; 20 steps truncate.

    The key alternates, so no state-independent bias solves it, and only the last transition
    is rewarded: solving it needs credit assignment over several steps (gamma > 0)."""

    KEY = (1, 0, 1, 0)
    LENGTH = len(KEY) + 1
    MAX_STEPS = 20

    def __init__(self) -> None:
        self._pos = 0
        self._t = 0

    @property
    def num_players(self) -> int:
        return 1

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(0.0, 1.0, (self.LENGTH,), np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(2)

    def _obs(self):
        obs = np.zeros(self.LENGTH, np.float32)
        obs[self._pos] = 1.0
        return {0: obs}

    def reset(self, seed=None):
        self._pos, self._t = 0, 0
        return self._obs(), {0: {}}

    def step(self, actions):
        self._t += 1
        if int(actions[0]) == self.KEY[self._pos]:
            self._pos += 1
        else:
            self._pos = max(0, self._pos - 1)
        reached = self._pos == self.LENGTH - 1
        reward = 1.0 if reached else 0.0
        truncated = (not reached) and self._t >= self.MAX_STEPS
        return self._obs(), {0: reward}, {0: reached}, {0: truncated}, {0: {}}
