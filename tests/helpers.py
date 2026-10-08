"""Shared test helpers: toy environments and small networks/models."""

from __future__ import annotations

import copy
from collections import namedtuple
from pathlib import Path

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.core.types import TrajectoryChunk
from colosseum.envs.base_env import BaseEnv
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import Core, GRUCore, LSTMCore, NoCore, WindowAttentionCore
from colosseum.networks.distributions import CategoricalDist, CompositeDist, DiagGaussianDist
from colosseum.networks.model import PolicyModel, StepOutput, UnrollOutput
from colosseum.networks.normalization import NormalizeObs

REPO_ROOT = Path(__file__).resolve().parent.parent


def example_config(name: str) -> Path:
    """Absolute path of ``configs/examples/<name>`` (tests run with cwd = tmp_path)."""
    return REPO_ROOT / "configs" / "examples" / name


# ---------------------------------------------------------------------------
# Small networks
# ---------------------------------------------------------------------------


class SimpleEncoder(BaseEncoder):
    """[B, obs_dim] -> optional NormalizeObs -> Linear -> relu -> [B, hidden_dim]."""

    def __init__(self, obs_dim: int = 8, hidden_dim: int = 16, normalize: bool = False) -> None:
        super().__init__()
        self._latent_dim = hidden_dim
        self.norm = NormalizeObs(shape=(obs_dim,)) if normalize else None
        self.fc = nn.Linear(obs_dim, hidden_dim)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, obs):
        if self.norm is not None:
            obs = self.norm(obs)
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
    """Small core of each kind; every core width except "none" differs from input_dim on purpose (R1-21)."""
    if kind == "none":
        return NoCore(input_dim)
    if kind == "lstm":
        return LSTMCore(input_dim, hidden_size=24)
    if kind == "gru":
        return GRUCore(input_dim, hidden_size=24)
    if kind == "window":
        return WindowAttentionCore(input_dim, d_model=24, window=3, num_heads=2)
    raise ValueError(f"unknown core kind {kind!r}")


def make_simple_model(obs_dim: int = 8, hidden_dim: int = 16, num_actions: int = 4,
                      core: str = "none", normalize: bool = False,
                      seed: int | None = None) -> ComposedModel:
    """ComposedModel: SimpleEncoder -> core -> SimplePolicy / SimpleValue.

    ``core`` is one of ``CORE_KINDS``. ``normalize`` puts a ``NormalizeObs`` in
    front of the encoder. ``seed`` (if given) seeds torch first, so two calls
    with the same seed build identical weights.
    """
    if seed is not None:
        torch.manual_seed(seed)
    encoder = SimpleEncoder(obs_dim, hidden_dim, normalize)
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


# ---------------------------------------------------------------------------
# Algorithm / eval test kit (SP1 blocks 4 and 7)
# ---------------------------------------------------------------------------


class MaskedToyEnv(BaseEnv):
    """Solo env: random 4-vector observation, Discrete(4), two random legal actions.

    Each step exposes ``info["action_mask"]`` (if ``use_mask``) with exactly two
    legal actions. Episodes last ``EPISODE_LENGTH`` steps. The reward is 1 for the
    scripted expert's choice (the legal action with the largest observation
    entry), else 0. Every received action is appended to ``self.received``.
    """

    EPISODE_LENGTH = 5

    def __init__(self, use_mask: bool = True) -> None:
        self._use_mask = use_mask
        self._rng = np.random.default_rng(0)
        self._t = 0
        self._obs = np.zeros(4, dtype=np.float32)
        self._mask = np.ones(4, dtype=bool)
        self.received: list[int] = []

    @property
    def num_players(self) -> int:
        return 1

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(-1.0, 1.0, (4,), np.float32)

    @property
    def action_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(4)

    def expert_action(self) -> int:
        return int(np.argmax(np.where(self._mask, self._obs, -np.inf)))

    def _draw(self) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
        self._obs = self._rng.uniform(-1.0, 1.0, 4).astype(np.float32)
        self._mask = np.zeros(4, dtype=bool)
        self._mask[self._rng.choice(4, size=2, replace=False)] = True
        info = {"action_mask": self._mask.copy()} if self._use_mask else {}
        return {0: self._obs.copy()}, {0: info}

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        return self._draw()

    def step(self, actions):
        action = int(np.asarray(actions[0]).reshape(-1)[0])
        self.received.append(action)
        reward = 1.0 if action == self.expert_action() else 0.0
        self._t += 1
        done = self._t >= self.EPISODE_LENGTH
        obs, info = self._draw()
        return obs, {0: reward}, {0: done}, {0: False}, info


def rollout_chunks(
    model: PolicyModel,
    env_fn,
    num_chunks: int,
    chunk_length: int = 8,
    num_envs: int = 2,
    seed: int = 0,
) -> list[TrajectoryChunk]:
    """Collect ``num_chunks`` real chunks for agent "a" with an in-process RolloutLoop.

    The loop acts with a deep copy of ``model`` (same weights). Chunks go through
    ``to_payload()``/``from_payload()`` exactly as they would across processes.
    """
    # Local import: dataflow_helpers imports this module.
    from dataflow_helpers import make_loop, run_until_chunks

    loop, collected = make_loop(env_fn, lambda: copy.deepcopy(model), agent_ids=("a",),
                                num_envs=num_envs, chunk_length=chunk_length, seed=seed)
    try:
        chunks = run_until_chunks(loop, collected, num_chunks)
    finally:
        loop.close()
    return [TrajectoryChunk.from_payload(c.to_payload()) for c in chunks]


TWELVE_NVEC = tuple(range(2, 14))   # unit i has i + 2 actions
TWELVE_OBS_DIM = 8                  # SimpleEncoder's default obs_dim, so configs need no encoder kwargs


class TwelveUnitEnv(BaseEnv):
    """Solo env with a 12-unit MultiDiscrete action (unit i has i + 2 actions).

    With ``use_mask`` the flat mask (natural unit order) allows only action i for
    unit i. Every received action vector is appended to ``self.received``.
    Episodes last 5 steps; rewards are 0.
    """

    def __init__(self, use_mask: bool = True) -> None:
        self._use_mask = use_mask
        self._t = 0
        self.received: list[np.ndarray] = []

    @property
    def num_players(self) -> int:
        return 1

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(0.0, 1.0, (TWELVE_OBS_DIM,), np.float32)

    @property
    def action_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.MultiDiscrete(np.array(TWELVE_NVEC))

    def _infos(self) -> dict[int, dict]:
        if not self._use_mask:
            return {0: {}}
        mask = np.concatenate([np.arange(n) == i for i, n in enumerate(TWELVE_NVEC)])
        return {0: {"action_mask": mask}}

    def reset(self, seed=None):
        self._t = 0
        return {0: np.zeros(TWELVE_OBS_DIM, np.float32)}, self._infos()

    def step(self, actions):
        self.received.append(np.asarray(actions[0], dtype=np.int64).copy())
        self._t += 1
        done = self._t >= 5
        return {0: np.zeros(TWELVE_OBS_DIM, np.float32)}, {0: 0.0}, {0: done}, {0: False}, self._infos()


class TwelveHeadPolicy(BasePolicy):
    """One Categorical head per unit, keyed "0".."11" in unit order.

    ``peaked=True``: head i puts (almost) all mass on action i.
    ``peaked=False``: uniform logits (use it with the env's mask).
    """

    def __init__(self, in_dim: int = 16, peaked: bool = True) -> None:
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(in_dim, n) for n in TWELVE_NVEC)
        with torch.no_grad():
            for i, head in enumerate(self.heads):
                head.weight.zero_()
                head.bias.zero_()
                if peaked:
                    head.bias[i] = 50.0

    def _named_dists(self, latent: torch.Tensor) -> list[tuple[str, CategoricalDist]]:
        return [(str(i), CategoricalDist(head(latent))) for i, head in enumerate(self.heads)]

    def forward(self, latent: torch.Tensor) -> CompositeDist:
        return CompositeDist(dict(self._named_dists(latent)))


class MisorderedTwelveHeadPolicy(TwelveHeadPolicy):
    """Same heads inserted in string-sorted key order ("0", "1", "10", "11", "2", ...)."""

    def forward(self, latent: torch.Tensor) -> CompositeDist:
        return CompositeDist(dict(sorted(self._named_dists(latent), key=lambda kv: kv[0])))


def twelve_unit_model(peaked: bool = True) -> ComposedModel:
    torch.manual_seed(0)
    return ComposedModel(SimpleEncoder(TWELVE_OBS_DIM, 16), NoCore(input_dim=16),
                         TwelveHeadPolicy(16, peaked), SimpleValue(16))


def make_test_run_dir(config, tmp_path, name: str = "test-run"):
    """Point ``config.run`` at ``tmp_path`` and create the run dir (tests never write to cwd)."""
    from colosseum.core.run_dir import RunDir

    config.run.dir = str(tmp_path / "runs")
    config.run.name = name
    return RunDir.create(config)
