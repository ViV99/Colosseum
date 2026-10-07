"""Toy envs, probe models and queue helpers for data-flow / transition tests.

Imported by bare name (``from dataflow_helpers import ...``; ``tests/`` is on
``sys.path``, see ``conftest.py``), so spawned processes can unpickle the env
and model factories defined here.
"""

from __future__ import annotations

import os
import queue

import gymnasium
import numpy as np
import torch

from colosseum.core.ipc import assert_no_tensors
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


class CheckedQueue(queue.Queue):
    """queue.Queue that rejects any item containing a torch.Tensor."""

    def put(self, item, block=True, timeout=None):
        assert_no_tensors(item, "queued item")
        super().put(item, block, timeout)

    def put_nowait(self, item):
        self.put(item, block=False)

    def cancel_join_thread(self) -> None:
        pass


def chunk_payload(T: int = 4, version: int = 0, agent_id: str = "a") -> dict:
    """A valid chunk payload with random content (for learner-loop tests only)."""
    rng = np.random.default_rng(version)
    return {
        "agent_id": agent_id,
        "observations": rng.standard_normal((T, OBS_DIM)).astype(np.float32),
        "actions": rng.integers(0, NUM_ACTIONS, T).astype(np.int64),
        "action_log_probs": np.full(T, -np.log(NUM_ACTIONS), np.float32),
        "rewards": np.zeros(T, np.float32),
        "dones": np.zeros(T, bool),
        "values": np.zeros(T, np.float32),
        "bootstrap_value": 0.0,
        "behavior_policy_version": int(version),
        "initial_state": None,
        "action_masks": None,
    }


class GridStepEnv(_Base):
    """Simultaneous-move env; every slot acts every step.

    obs[p] = [env_id, ep, t, p]; reward[p] = ep + 0.01 * t + 0.1 * p (or
    ``const_reward``). Episode ``ep`` lasts ``lengths[ep % len(lengths)]`` steps
    and ends with truncation if ``truncate_every`` and ``ep % truncate_every == 0``,
    else with termination. Every step is appended to ``self.log``.
    """

    def __init__(self, env_id: int = 0, lengths=(3,), truncate_every: int = 0,
                 const_reward: float | None = None, num_players: int = 2) -> None:
        super().__init__(env_id)
        self.lengths = tuple(lengths)
        self.truncate_every = truncate_every
        self.const_reward = const_reward
        self.NUM_PLAYERS = num_players

    def _length(self) -> int:
        return self.lengths[self.ep % len(self.lengths)]

    def reset(self, seed=None):
        self.ep += 1
        self.t = 0
        return self._obs(), {p: {} for p in range(self.num_players)}

    def step(self, actions):
        P = self.num_players
        pre_obs = self._obs()
        rew = {p: (self.const_reward if self.const_reward is not None
                   else self.ep + 0.01 * self.t + 0.1 * p) for p in range(P)}
        self.t += 1
        done = self.t >= self._length()
        trunc = done and self.truncate_every > 0 and self.ep % self.truncate_every == 0
        term = done and not trunc
        self.log.append({"ep": self.ep, "t": self.t - 1, "obs": pre_obs,
                         "actions": {p: int(actions[p]) for p in range(P)},
                         "active": {p: True for p in range(P)},
                         "rewards": rew, "done": done, "truncated": trunc})
        return (self._obs(), rew, {p: term for p in range(P)},
                {p: trunc for p in range(P)}, {p: {} for p in range(P)})


class EnvFactory:
    """Callable env factory: assigns env ids 0, 1, ... and keeps the instances."""

    def __init__(self, cls, **kwargs) -> None:
        self.cls = cls
        self.kwargs = kwargs
        self.created: list[BaseEnv] = []

    def __call__(self) -> BaseEnv:
        env = self.cls(env_id=len(self.created), **self.kwargs)
        self.created.append(env)
        return env
