"""Shared test helpers: toy environments and small networks/models."""

from pathlib import Path

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.envs.base_env import BaseEnv
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import CategoricalDist

REPO_ROOT = Path(__file__).resolve().parent.parent


def example_config(name: str) -> Path:
    """Absolute path of ``configs/examples/<name>`` (tests run with cwd = tmp_path)."""
    return REPO_ROOT / "configs" / "examples" / name


def make_simple_network(obs_dim=8, hidden_dim=16, num_actions=4):
    """Create a simple feedforward ActorCriticNetwork for testing."""
    from colosseum.networks.actor_critic import ActorCriticNetwork

    encoder = SimpleEncoder(obs_dim, hidden_dim)
    policy = SimplePolicy(hidden_dim, num_actions)
    value = SimpleValue(hidden_dim)
    return ActorCriticNetwork(encoder, policy, value)


class SimpleEncoder(BaseEncoder):
    def __init__(self, obs_dim=8, hidden_dim=16):
        super().__init__()
        self._latent_dim = hidden_dim
        self.fc = nn.Linear(obs_dim, hidden_dim)

    @property
    def latent_dim(self):
        return self._latent_dim

    def forward(self, obs):
        return torch.relu(self.fc(obs))


class SimplePolicy(BasePolicy):
    def __init__(self, hidden_dim=16, num_actions=4):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, latent):
        return CategoricalDist(self.fc(latent))


class SimpleValue(BaseValue):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, latent):
        return self.fc(latent).squeeze(-1)


# ---------------------------------------------------------------------------
# Deterministic toy environment for contract tests
# ---------------------------------------------------------------------------


class CountingEnv(BaseEnv):
    """Deterministic N-player simultaneous-move env for contract tests.

    Every episode lasts exactly ``episode_length`` steps and then terminates.
    At in-episode step ``t`` player ``p`` observes
    ``[t / episode_length, p, 1.0, 0.0, ...]`` (``obs_dim`` floats) and gets
    reward 1.0 if its action equals ``(t + p) % num_actions``, else 0.0.
    The step and player index can be recovered from any recorded observation:
    ``t = round(obs[0] * episode_length)``, ``p = round(obs[1])``.
    """

    def __init__(self, num_players: int = 2, episode_length: int = 5,
                 num_actions: int = 3, obs_dim: int = 4) -> None:
        if obs_dim < 3:
            raise ValueError("obs_dim must be >= 3")
        self._num_players = num_players
        self.episode_length = episode_length
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self._t = 0

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        return gymnasium.spaces.Discrete(self.num_actions)

    def _obs(self, p: int) -> np.ndarray:
        o = np.zeros(self.obs_dim, dtype=np.float32)
        o[0] = self._t / self.episode_length
        o[1] = float(p)
        o[2] = 1.0
        return o

    def reset(self, seed=None):
        self._t = 0
        players = range(self._num_players)
        return {p: self._obs(p) for p in players}, {p: {} for p in players}

    def step(self, actions):
        players = range(self._num_players)
        rewards = {p: 1.0 if int(actions[p]) == (self._t + p) % self.num_actions else 0.0 for p in players}
        self._t += 1
        done = self._t >= self.episode_length
        obs = {p: self._obs(p) for p in players}
        return obs, rewards, {p: done for p in players}, {p: False for p in players}, {p: {} for p in players}
