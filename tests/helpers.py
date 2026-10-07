"""Shared test helpers: toy environments and small networks/models."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.envs.base_env import BaseEnv
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import Core, GRUCore, LSTMCore, NoCore, WindowAttentionCore
from colosseum.networks.distributions import CategoricalDist, DiagGaussianDist
from colosseum.networks.model import PolicyModel, StepOutput, UnrollOutput

REPO_ROOT = Path(__file__).resolve().parent.parent


def example_config(name: str) -> Path:
    """Absolute path of ``configs/examples/<name>`` (tests run with cwd = tmp_path)."""
    return REPO_ROOT / "configs" / "examples" / name


# ---------------------------------------------------------------------------
# Small networks
# ---------------------------------------------------------------------------


class SimpleEncoder(BaseEncoder):
    def __init__(self, obs_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__()
        self._latent_dim = hidden_dim
        self.fc = nn.Linear(obs_dim, hidden_dim)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, obs):
        return torch.relu(self.fc(obs))


class SimplePolicy(BasePolicy):
    def __init__(self, in_dim: int = 16, num_actions: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_actions)

    def forward(self, latent):
        return CategoricalDist(self.fc(latent))


class SimpleValue(BaseValue):
    def __init__(self, in_dim: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, latent):
        return self.fc(latent).squeeze(-1)


class FixedInputPolicy(BasePolicy):
    """Policy head hard-wired to 64 input features (no ``in_dim``): mismatches other cores."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 9)

    def forward(self, latent):
        return CategoricalDist(self.fc(latent))


class BadShapeValue(BaseValue):
    """Value head that forgets to squeeze: returns [B, 1] (validate_config must reject it)."""

    def __init__(self, in_dim: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, latent):
        return self.fc(latent)


class TinyMonolithicModel(PolicyModel):
    """Stateless PolicyModel used through ``networks.model_class``."""

    def __init__(self, obs_dim: int = 27, num_actions: int = 9) -> None:
        super().__init__()
        self.pi = nn.Linear(obs_dim, num_actions)
        self.v = nn.Linear(obs_dim, 1)

    def step(self, obs, state, action_mask=None):
        x = obs.reshape(obs.shape[0], -1)
        dist = CategoricalDist(self.pi(x))
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        return StepOutput(dist, self.v(x).squeeze(-1), None)


HCState = namedtuple("HCState", ["h", "c"])


class NamedTupleStateModel(TinyMonolithicModel):
    """Stateful model whose state is a module-level namedtuple (crosses processes fine)."""

    state_cls = HCState

    def initial_state(self, batch_size, device="cpu"):
        return self.state_cls(torch.zeros(batch_size, 2, device=device), torch.zeros(batch_size, 2, device=device))

    def step(self, obs, state, action_mask=None):
        out = super().step(obs, None, action_mask)
        return StepOutput(out.dist, out.value, self.state_cls(state.h + 1.0, state.c))


class LocalNamedTupleStateModel(NamedTupleStateModel):
    """State namedtuple class is not reachable by module path (validate_config must reject it)."""

    def __init__(self, obs_dim: int = 27, num_actions: int = 9) -> None:
        super().__init__(obs_dim, num_actions)
        self.state_cls = namedtuple("UnreachableState", ["h", "c"])


class GaussianActionModel(TinyMonolithicModel):
    """Returns a 2-D Gaussian for a Discrete action space (validate_config: action-shape mismatch)."""

    def step(self, obs, state, action_mask=None):
        x = obs.reshape(obs.shape[0], -1)
        mean = self.pi(x)[:, :2]
        return StepOutput(DiagGaussianDist(mean, torch.zeros_like(mean)), self.v(x).squeeze(-1), None)


class BadUnrollModel(TinyMonolithicModel):
    """``unroll`` returns values shaped [T, B] instead of [T*B] (validate_config must reject it)."""

    def unroll(self, obs, state0, dones, action_mask=None):
        out = super().unroll(obs, state0, dones, action_mask)
        return UnrollOutput(out.dist, out.value.reshape(obs.shape[0], obs.shape[1]))


class LayerFirstLSTMCore(Core):
    """Core keeping nn.LSTM's native layer-first state [L, B, H] (validate_config must reject it).

    With ``num_layers=2`` the state of a batch of 2 is ``(2, 2, H)``, so dim 0
    alone cannot tell layers from batch.
    """

    def __init__(self, input_dim: int, hidden_size: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers)

    def initial_state(self, batch_size, device="cpu"):
        zeros = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return {"h": zeros, "c": zeros.clone()}

    def step(self, x, state):
        y, (h, c) = self.rnn(x.unsqueeze(0), (state["h"], state["c"]))
        return y.squeeze(0), {"h": h, "c": c}


CORE_KINDS = ("none", "lstm", "gru", "window")


def make_core(kind: str, input_dim: int) -> Core:
    """Small core of each kind; recurrent sizes differ from input_dim on purpose (R1-21)."""
    if kind == "none":
        return NoCore(input_dim)
    if kind == "lstm":
        return LSTMCore(input_dim, hidden_size=24)
    if kind == "gru":
        return GRUCore(input_dim, hidden_size=24)
    if kind == "window":
        return WindowAttentionCore(input_dim, d_model=16, window=3, num_heads=2)
    raise ValueError(f"unknown core kind {kind!r}")


def make_simple_model(obs_dim: int = 8, hidden_dim: int = 16, num_actions: int = 4,
                      core: str = "none") -> ComposedModel:
    """ComposedModel: SimpleEncoder -> core -> SimplePolicy / SimpleValue."""
    encoder = SimpleEncoder(obs_dim, hidden_dim)
    trunk = make_core(core, encoder.latent_dim)
    return ComposedModel(encoder, trunk, SimplePolicy(trunk.output_dim, num_actions),
                         SimpleValue(trunk.output_dim))


def make_ttt_model() -> ComposedModel:
    """Tic-tac-toe example networks as a stateless ComposedModel."""
    from examples.tic_tac_toe.networks import TicTacToeEncoder, TicTacToePolicy, TicTacToeValue

    encoder = TicTacToeEncoder()
    return ComposedModel(encoder, NoCore(encoder.latent_dim), TicTacToePolicy(), TicTacToeValue())


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


class WrongMaskEnv(CountingEnv):
    """CountingEnv whose reset info carries an action mask one entry too long."""

    def reset(self, seed=None):
        obs, _ = super().reset(seed)
        return obs, {p: {"action_mask": np.ones(self.num_actions + 1, dtype=bool)} for p in obs}


class ResetFailsEnv(CountingEnv):
    """CountingEnv whose ``reset`` raises (validate_config must wrap it in ConfigError)."""

    def reset(self, seed=None):
        raise RuntimeError("reset exploded")
