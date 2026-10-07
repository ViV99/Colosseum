"""Toy envs, probe models and queue helpers for data-flow / transition tests.

Imported by bare name (``from dataflow_helpers import ...``; ``tests/`` is on
``sys.path``, see ``conftest.py``), so spawned processes can unpickle the env
and model factories defined here.
"""

from __future__ import annotations

import os

import gymnasium
import numpy as np
import torch

from colosseum.envs.base_env import BaseEnv
from helpers import TinyMonolithicModel

OBS_DIM = 4
NUM_ACTIONS = 3


class TinyModel(TinyMonolithicModel):
    """Small stateless trainable model (linear policy + value) sized for the envs below."""

    def __init__(self, obs_dim: int = OBS_DIM, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__(obs_dim=obs_dim, num_actions=num_actions)


def make_tiny_model() -> TinyModel:
    return TinyModel()


class _Base(BaseEnv):
    """Shared plumbing: obs[p] = [env_id, ep, t, p], Discrete(NUM_ACTIONS)."""

    NUM_PLAYERS = 2

    def __init__(self, env_id: int = 0) -> None:
        self.env_id = env_id
        self.ep = -1
        self.t = 0
        self.log: list[dict] = []

    @property
    def num_players(self) -> int:
        return self.NUM_PLAYERS

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(-1e9, 1e9, (OBS_DIM,), np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(NUM_ACTIONS)

    def _obs(self) -> dict[int, np.ndarray]:
        return {p: np.array([self.env_id, self.ep, self.t, p], np.float32)
                for p in range(self.num_players)}


class ThreadProbeEnv(_Base):
    """1-player env whose observation reports the process's thread settings.

    obs = [torch.get_num_threads(), OMP_NUM_THREADS (or -1), torch.get_num_interop_threads(), t];
    4 steps per episode.
    """

    NUM_PLAYERS = 1

    def _probe(self) -> dict[int, np.ndarray]:
        omp = float(os.environ.get("OMP_NUM_THREADS", "-1"))
        return {0: np.array([torch.get_num_threads(), omp, torch.get_num_interop_threads(), self.t],
                            np.float32)}

    def reset(self, seed=None):
        self.ep += 1
        self.t = 0
        return self._probe(), {0: {}}

    def step(self, actions):
        self.t += 1
        done = self.t >= 4
        return self._probe(), {0: 0.0}, {0: done}, {0: False}, {0: {}}
