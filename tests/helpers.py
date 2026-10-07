"""Shared test helpers: toy environments and small networks/models."""

from pathlib import Path

import torch
import torch.nn as nn

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
