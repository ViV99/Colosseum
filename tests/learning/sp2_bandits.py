"""One-step games for the fast SP2 learning tests (spec section 6, "Учится": fast tests)."""
from __future__ import annotations

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.core.specs import ActionSpec
from colosseum.envs.game import GameSpec, MultiAgentEnv, StepResult
from colosseum.envs.spaces import Units
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue, EncoderOutput
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import NoCore
from colosseum.networks.heads import UnitsHead, make_distribution


class UnitsBandit(MultiAgentEnv):
    """Solo, one step. Between 1 and ``max_units`` units exist (random slots); unit ``u`` sees a
    one-hot context ``c_u`` in ``{0..arms-1}`` and should pick arm ``c_u``. The reward is the share
    of existing units that picked their own arm (one shared scalar). Random play scores ``1 / arms``.
    Solving it shows that a ``Units`` action learns end to end (per-unit features -> ``UnitsHead`` ->
    per-unit deciders -> loss); it is not a per-unit credit-assignment test, since the joint ratio
    solves it as well (the unit heads share weights)."""

    def __init__(self, max_units: int = 8, arms: int = 4) -> None:
        self.max_units, self.arms = max_units, arms
        obs_space = gymnasium.spaces.Dict([
            ("units", gymnasium.spaces.Box(0.0, 1.0, (max_units, arms), np.float32)),
            ("unit_mask", gymnasium.spaces.MultiBinary(max_units)),
        ])
        self.spec = GameSpec.solo(obs_space, Units(max_units, gymnasium.spaces.Discrete(arms)))
        self._rng = np.random.default_rng()
        self._alive = np.zeros(max_units, bool)
        self._ctx = np.zeros(max_units, np.int64)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        count = int(self._rng.integers(1, self.max_units + 1))
        self._alive = self._rng.permutation(self.max_units) < count
        self._ctx = self._rng.integers(self.arms, size=self.max_units)
        units = np.zeros((self.max_units, self.arms), np.float32)
        units[self._alive, self._ctx[self._alive]] = 1.0
        obs = {"units": units, "unit_mask": self._alive.astype(np.int8)}
        mask = {"unit": self._alive.copy(), "action": np.ones((self.max_units, self.arms), bool)}
        return StepResult(acting={0}, obs={0: obs}, action_masks={0: mask})

    def step(self, actions: dict) -> StepResult:
        picked = np.asarray(actions[0], np.int64).reshape(self.max_units)
        reward = float((picked[self._alive] == self._ctx[self._alive]).mean())
        return StepResult(acting=set(), obs={}, rewards={0: reward}, episode_over=True)


class CoopBandit(MultiAgentEnv):
    """One team of two seats, one step. Both seats see the same one-hot context ``c`` in
    ``{0..arms-1}`` plus their own seat id. The team scores 1 only if seat 0 picks ``c`` and seat 1
    picks ``arms - 1 - c`` in the same step (both seats get that reward), so neither seat can be
    rewarded without the other. Random play scores ``1 / arms**2``."""

    def __init__(self, arms: int = 4) -> None:
        self.arms = arms
        obs_space = gymnasium.spaces.Box(0.0, 1.0, (arms + 2,), np.float32)
        self.spec = GameSpec.teams_of([2], obs_space, gymnasium.spaces.Discrete(arms))
        self._rng = np.random.default_rng()
        self._ctx = 0

    def _obs(self, seat: int) -> np.ndarray:
        obs = np.zeros(self.arms + 2, np.float32)
        obs[self._ctx] = 1.0
        obs[self.arms + seat] = 1.0
        return obs

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._ctx = int(self._rng.integers(self.arms))
        return StepResult(acting={0, 1}, obs={s: self._obs(s) for s in (0, 1)})

    def step(self, actions: dict) -> StepResult:
        hit = int(actions[0]) == self._ctx and int(actions[1]) == self.arms - 1 - self._ctx
        reward = 1.0 if hit else 0.0
        return StepResult(acting=set(), obs={}, rewards={0: reward, 1: reward}, episode_over=True)


class _UnitsEncoder(BaseEncoder):
    """Per-unit MLP (the unit's own context reaches its own head through ``aux``)."""

    def __init__(self, arms: int, hidden: int) -> None:
        super().__init__()
        self._hidden = hidden
        self.unit_net = nn.Sequential(nn.Linear(arms, hidden), nn.Tanh())

    @property
    def latent_dim(self) -> int:
        return self._hidden

    def forward(self, obs: dict) -> EncoderOutput:
        units = self.unit_net(obs["units"])
        m = obs["unit_mask"].to(units.dtype).unsqueeze(-1)
        pooled = (units * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return EncoderOutput(latent=pooled, aux={"units": units})


class _UnitsPolicy(BasePolicy):
    def __init__(self, action_spec: ActionSpec, hidden: int) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.head = UnitsHead(action_spec.groups[0], hidden)

    def forward(self, features: torch.Tensor, aux: dict):
        return make_distribution(self.action_spec, self.head(aux["units"]))


class _Value(BaseValue):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def make_units_bandit_model(max_units: int = 8, arms: int = 4, hidden: int = 32) -> ComposedModel:
    spec = UnitsBandit(max_units, arms).spec
    action_spec = ActionSpec.from_space(spec.roles["player"].action_space)
    return ComposedModel(_UnitsEncoder(arms, hidden), NoCore(hidden), _UnitsPolicy(action_spec, hidden),
                         _Value(hidden))
