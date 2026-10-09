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
