"""Tiny toy envs exercising different game types against the Colosseum pipeline."""
from __future__ import annotations

from typing import Any, Optional

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.envs.base_env import BaseEnv
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import CategoricalDist, CompositeDist, Distribution


# ---------------------------------------------------------------------------
# 1. Solo env: one player, reward maximisation.
# ---------------------------------------------------------------------------
class SoloEnv(BaseEnv):
    """1 player. obs = one-hot target; reward 1 if action == target. 10 steps."""

    def __init__(self, length: int = 10):
        self._len = length
        self._t = 0
        self._target = 0
        self._rng = np.random.default_rng(0)

    @property
    def num_players(self):
        return 1

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(4)

    def _obs(self):
        o = np.zeros(4, dtype=np.float32)
        o[self._target] = 1.0
        return {0: o}

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._target = int(self._rng.integers(4))
        return self._obs(), {0: {}}

    def step(self, actions):
        r = 1.0 if int(actions[0]) == self._target else 0.0
        self._t += 1
        self._target = int(self._rng.integers(4))
        done = self._t >= self._len
        return self._obs(), {0: r}, {0: done}, {0: False}, {0: {"score": r}}


# ---------------------------------------------------------------------------
# 2. 3-player FFA with mid-game elimination.
# ---------------------------------------------------------------------------
class FFAElimEnv(BaseEnv):
    """3 players. Player 0 is eliminated after step 2, player 1 after step 4,
    player 2 wins at step 6. Elimination reward -1 at the elimination step;
    final placement reward (rank 1:+1, 2:0, 3:-1) is given to ALL players at
    the final step (common Kaggle-style terminal reward).

    obs[0:3] = one-hot player id (so chunks can be attributed to seats),
    obs[3] = alive flag, obs[4] = t/6.

    mode:
      "per_player_term": terminated[p]=True only for the eliminated player.
      "active": terminated all False until the end; info["active"]=False once
                eliminated (the documented convention for non-acting players).
    """

    RANK = {0: 3, 1: 2, 2: 1}

    def __init__(self, mode: str = "active"):
        self.mode = mode
        self._t = 0
        self._alive = [True, True, True]

    @property
    def num_players(self):
        return 3

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(3)

    def _obs(self):
        out = {}
        for p in range(3):
            o = np.zeros(5, dtype=np.float32)
            o[p] = 1.0
            o[3] = float(self._alive[p])
            o[4] = self._t / 6.0
            out[p] = o
        return out

    def _infos(self):
        if self.mode == "active":
            return {p: {"active": self._alive[p]} for p in range(3)}
        return {p: {} for p in range(3)}

    def reset(self, seed=None):
        self._t = 0
        self._alive = [True, True, True]
        return self._obs(), self._infos()

    def step(self, actions):
        self._t += 1
        rew = {0: 0.0, 1: 0.0, 2: 0.0}
        term = {0: False, 1: False, 2: False}
        trunc = {0: False, 1: False, 2: False}
        if self._t == 2:
            self._alive[0] = False
            rew[0] -= 1.0
            if self.mode == "per_player_term":
                term[0] = True
        if self._t == 4:
            self._alive[1] = False
            rew[1] -= 1.0
            if self.mode == "per_player_term":
                term[1] = True
        final = self._t >= 6
        if final:
            for p in range(3):
                rew[p] += {1: 1.0, 2: 0.0, 3: -1.0}[self.RANK[p]]
                term[p] = True
        infos = self._infos()
        if final:
            for p in range(3):
                infos[p]["rank"] = self.RANK[p]
        return self._obs(), rew, term, trunc, infos


# ---------------------------------------------------------------------------
# 3. Turn-based 2-player env (alternating moves).
# ---------------------------------------------------------------------------
class AlternatingEnv(BaseEnv):
    """2 players alternate; 4 plies total (p0, p1, p0, p1). After the last ply
    (made by p1) the game ends: p0 gets +1, p1 gets -1 (fixed so it's checkable).

    use_active: put info["active"] (the documented turn-based convention).
    mask_inactive_all_false: give the inactive player an all-False action mask
        (a natural way to express "you cannot act now").
    """

    def __init__(self, use_active: bool = True, mask_inactive_all_false: bool = False,
                 with_mask: bool = False):
        self.use_active = use_active
        self.mask_all_false = mask_inactive_all_false
        self.with_mask = with_mask or mask_inactive_all_false
        self._t = 0

    @property
    def num_players(self):
        return 2

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(3)

    @property
    def current(self):
        return self._t % 2

    def _obs(self):
        out = {}
        for p in range(2):
            o = np.zeros(4, dtype=np.float32)
            o[p] = 1.0
            o[2] = float(self.current == p)
            o[3] = self._t / 4.0
            out[p] = o
        return out

    def _infos(self):
        infos = {}
        for p in range(2):
            d = {}
            if self.use_active:
                d["active"] = (self.current == p)
            if self.with_mask:
                if self.current == p or not self.mask_all_false:
                    d["action_mask"] = np.ones(3, dtype=bool)
                else:
                    d["action_mask"] = np.zeros(3, dtype=bool)
            infos[p] = d
        return infos

    def reset(self, seed=None):
        self._t = 0
        return self._obs(), self._infos()

    def step(self, actions):
        self._t += 1
        done = self._t >= 4
        rew = {0: 1.0 if done else 0.0, 1: -1.0 if done else 0.0}
        infos = self._infos()
        if done:
            infos[0]["outcome"] = 1.0
            infos[1]["outcome"] = 0.0
        return self._obs(), rew, {0: done, 1: done}, {0: False, 1: False}, infos


# ---------------------------------------------------------------------------
# 4. Dict observation env.
# ---------------------------------------------------------------------------
class DictObsEnv(SoloEnv):
    @property
    def num_players(self):
        return 2

    @property
    def observation_space(self):
        return gymnasium.spaces.Dict({
            "map": gymnasium.spaces.Box(0.0, 1.0, shape=(2, 4, 4), dtype=np.float32),
            "units": gymnasium.spaces.Box(-1.0, 1.0, shape=(8, 3), dtype=np.float32),
            "unit_mask": gymnasium.spaces.MultiBinary(8),
        })

    def _obs(self):
        o = self.observation_space.sample()
        return {0: o, 1: o}

    def reset(self, seed=None):
        super().reset(seed)
        return self._obs(), {0: {}, 1: {}}

    def step(self, actions):
        self._t += 1
        done = self._t >= self._len
        return self._obs(), {0: 0.0, 1: 0.0}, {0: done, 1: done}, {0: False, 1: False}, {0: {}, 1: {}}


# ---------------------------------------------------------------------------
# 5. Per-unit action env: up to K units, each with A actions, variable count.
# ---------------------------------------------------------------------------
class UnitsEnv(BaseEnv):
    """2 players, each controls up to K units (alive count changes each step).
    Action: MultiDiscrete([A]*K). Unit i alive iff i < n_alive(t).
    Dead/non-existent units: mask allows only action 0 (no-op) unless
    dead_mask_all_false=True, in which case their mask row is all False.
    Reward: +1/K per alive unit choosing action 1.
    obs: [alive flags (K), t/len]
    """

    def __init__(self, K: int = 8, A: int = 4, length: int = 8,
                 dead_mask_all_false: bool = False, nvec_2d: bool = False):
        self.K, self.A, self.len = K, A, length
        self.dead_all_false = dead_mask_all_false
        self.nvec_2d = nvec_2d
        self._t = 0

    @property
    def num_players(self):
        return 2

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(0.0, 1.0, shape=(self.K + 1,), dtype=np.float32)

    @property
    def action_space(self):
        if self.nvec_2d:
            return gymnasium.spaces.MultiDiscrete(np.full((2, self.K // 2), self.A))
        return gymnasium.spaces.MultiDiscrete([self.A] * self.K)

    def _n_alive(self):
        return 1 + (self._t * 3) % self.K

    def _obs(self):
        o = np.zeros(self.K + 1, dtype=np.float32)
        o[: self._n_alive()] = 1.0
        o[-1] = self._t / self.len
        return {0: o, 1: o.copy()}

    def _infos(self):
        n = self._n_alive()
        m = np.zeros((self.K, self.A), dtype=bool)
        m[:n] = True
        if not self.dead_all_false:
            m[n:, 0] = True
        return {p: {"action_mask": m.reshape(-1).copy()} for p in range(2)}

    def reset(self, seed=None):
        self._t = 0
        return self._obs(), self._infos()

    def step(self, actions):
        n = self._n_alive()
        rew = {}
        for p in range(2):
            a = np.asarray(actions[p]).reshape(-1)
            rew[p] = float((a[:n] == 1).sum()) / self.K
        self._t += 1
        done = self._t >= self.len
        return self._obs(), rew, {0: done, 1: done}, {0: False, 1: False}, self._infos()


# ---------------------------------------------------------------------------
# Generic networks
# ---------------------------------------------------------------------------
class MLPEncoder(BaseEncoder):
    def __init__(self, obs_dim: int = 4, hidden: int = 32, **kw):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(obs_dim, hidden), nn.Tanh())
        self._h = hidden

    def forward(self, obs):
        return self.net(obs.float())

    @property
    def latent_dim(self):
        return self._h


class CatPolicy(BasePolicy):
    def __init__(self, hidden: int = 32, n_actions: int = 4, **kw):
        super().__init__()
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, latent):
        return CategoricalDist(self.head(latent))


class ValueHead(BaseValue):
    def __init__(self, hidden: int = 32, **kw):
        super().__init__()
        self.head = nn.Linear(hidden, 1)

    def forward(self, latent):
        return self.head(latent).squeeze(-1)


class MultiUnitCompositePolicy(BasePolicy):
    """K independent categorical heads through the built-in CompositeDist."""

    def __init__(self, hidden: int = 32, K: int = 8, A: int = 4, **kw):
        super().__init__()
        self.K, self.A = K, A
        self.head = nn.Linear(hidden, K * A)

    def forward(self, latent):
        logits = self.head(latent).view(-1, self.K, self.A)
        return CompositeDist({str(i): CategoricalDist(logits[:, i]) for i in range(self.K)})
