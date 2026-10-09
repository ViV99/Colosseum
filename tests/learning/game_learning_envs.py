"""Tiny solo games that the SP2 pipeline must solve in seconds, and a small MLP model for them."""
from __future__ import annotations

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.networks.cores import NoCore
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, StepResult
from colosseum.sp2.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.sp2.networks.composed import ComposedModel
from colosseum.sp2.networks.dist import CategoricalDist


class MLPEncoder(BaseEncoder):
    """[B, obs_dim] -> Linear -> relu -> [B, hidden]."""

    def __init__(self, obs_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self._latent = hidden
        self.fc = nn.Linear(obs_dim, hidden)

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(obs.float()))


class MLPPolicy(BasePolicy):
    def __init__(self, in_dim: int, num_actions: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_actions)

    def forward(self, features: torch.Tensor, aux: dict) -> CategoricalDist:
        return CategoricalDist(self.fc(features))


class MLPValue(BaseValue):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features).squeeze(-1)


def make_mlp_model(obs_dim: int, num_actions: int, hidden: int = 32) -> ComposedModel:
    """SP1's ``make_simple_model`` shape: Linear+relu encoder, no core, linear heads."""
    return ComposedModel(MLPEncoder(obs_dim, hidden), NoCore(hidden), MLPPolicy(hidden, num_actions), MLPValue(hidden))


class MaskedChoiceGame(MultiAgentEnv):
    """Solo, 8 decisions per episode. Observation: one-hot context c in 0..3. Legal actions:
    {c, (c + 1) % 4}; the expert plays c (reward 1), the other legal action gives 0."""

    spec = GameSpec.solo(gymnasium.spaces.Box(0.0, 1.0, (4,), np.float32), gymnasium.spaces.Discrete(4))
    LENGTH = 8

    def __init__(self) -> None:
        self._rng = np.random.default_rng()
        self._context = 0
        self._t = 0

    def expert_action(self) -> int:
        return self._context

    def _turn(self, rewards: dict) -> StepResult:
        self._context = int(self._rng.integers(4))
        obs = np.zeros(4, np.float32)
        obs[self._context] = 1.0
        mask = np.zeros(4, dtype=bool)
        mask[[self._context, (self._context + 1) % 4]] = True
        return StepResult(acting={0}, obs={0: obs}, action_masks={0: mask}, rewards=rewards)

    def reset(self, seed, layout) -> StepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        return self._turn({})

    def step(self, actions) -> StepResult:
        reward = 1.0 if int(actions[0]) == self._context else 0.0
        self._t += 1
        if self._t >= self.LENGTH:
            return StepResult(acting=set(), obs={}, rewards={0: reward}, episode_over=True)
        return self._turn({0: reward})


class ContextualBanditGame(MultiAgentEnv):
    """Solo, one decision per episode. Observation: one-hot context c in {0, 1}; reward 1 iff action == c."""

    spec = GameSpec.solo(gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32), gymnasium.spaces.Discrete(2))

    def __init__(self) -> None:
        self._rng = np.random.default_rng()
        self._context = 0

    def reset(self, seed, layout) -> StepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._context = int(self._rng.integers(2))
        obs = np.zeros(2, np.float32)
        obs[self._context] = 1.0
        return StepResult(acting={0}, obs={0: obs})

    def step(self, actions) -> StepResult:
        reward = 1.0 if int(actions[0]) == self._context else 0.0
        return StepResult(acting=set(), obs={}, rewards={0: reward}, episode_over=True)


class ShortChainGame(MultiAgentEnv):
    """Combination lock: states 0..4, start at 0, actions 0/1. The correct action at state i is
    ``KEY[i]`` and advances to i+1; a wrong action steps back to max(0, i-1). Reaching 4 gives +1
    and ends the episode by the rules; 20 decisions truncate it (``truncated`` with ``final_obs``).

    The key alternates, so no state-independent bias solves it, and only the last transition is
    rewarded: solving it needs credit assignment over several decisions (gamma > 0)."""

    KEY = (1, 0, 1, 0)
    LENGTH = len(KEY) + 1
    MAX_STEPS = 20
    spec = GameSpec.solo(gymnasium.spaces.Box(0.0, 1.0, (LENGTH,), np.float32), gymnasium.spaces.Discrete(2))

    def __init__(self) -> None:
        self._pos = 0
        self._t = 0

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.LENGTH, np.float32)
        obs[self._pos] = 1.0
        return obs

    def reset(self, seed, layout) -> StepResult:
        self._pos, self._t = 0, 0
        return StepResult(acting={0}, obs={0: self._obs()})

    def step(self, actions) -> StepResult:
        self._t += 1
        if int(actions[0]) == self.KEY[self._pos]:
            self._pos += 1
        else:
            self._pos = max(0, self._pos - 1)
        if self._pos == self.LENGTH - 1:
            return StepResult(acting=set(), obs={}, rewards={0: 1.0}, episode_over=True)
        if self._t >= self.MAX_STEPS:
            return StepResult(acting=set(), obs={}, episode_over=True, truncated=True, final_obs={0: self._obs()})
        return StepResult(acting={0}, obs={0: self._obs()})
