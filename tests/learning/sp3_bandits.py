"""One-step-decision games for the fast SP3 learning tests (spec section 6, «Быстрые учится»)."""
from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
import torch

from colosseum.envs.game import GameSpec, MultiAgentEnv, StepResult
from colosseum.players import ScriptedBot

CONTEXTS = 4
OBS_SPACE = gymnasium.spaces.Box(0.0, 1.0, (CONTEXTS,), np.float32)
ACT_SPACE = gymnasium.spaces.Discrete(CONTEXTS)
OFFSET_TARGETS = torch.tensor([(c + 1) % CONTEXTS for c in range(CONTEXTS)])   # OffsetTeacher's action per context


def _one_hot(c: int) -> np.ndarray:
    return np.eye(CONTEXTS, dtype=np.float32)[int(c)]


class SilentBandit(MultiAgentEnv):
    """Solo, ``LENGTH`` decisions per episode on a one-hot context; the reward is always 0, so only a
    kickstart teacher can move the policy."""

    LENGTH = 4
    spec = GameSpec.solo(OBS_SPACE, ACT_SPACE)

    def __init__(self) -> None:
        self._rng = np.random.default_rng()
        self._t = 0
        self._ctx = 0

    def _turn(self, rewards: dict[int, float]) -> StepResult:
        return StepResult(acting={0}, obs={0: _one_hot(self._ctx)}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._ctx = int(self._rng.integers(CONTEXTS))
        return self._turn({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        self._t += 1
        if self._t >= self.LENGTH:
            return StepResult(acting=set(), obs={}, rewards={0: 0.0}, episode_over=True)
        self._ctx = int(self._rng.integers(CONTEXTS))
        return self._turn({0: 0.0})


class DuelBandit(MultiAgentEnv):
    """Two seats act simultaneously for ``LENGTH`` steps; each sees its own one-hot context and scores 1
    when it picks it. The default outcome (team score = return) decides the match; a uniformly random
    player scores ``LENGTH / 4`` on average, a perfect one ``LENGTH``."""

    LENGTH = 4
    spec = GameSpec.symmetric(2, OBS_SPACE, ACT_SPACE)

    def __init__(self) -> None:
        self._rng = np.random.default_rng()
        self._t = 0
        self._ctx = np.zeros(2, dtype=np.int64)

    def _turn(self, rewards: dict[int, float]) -> StepResult:
        return StepResult(acting={0, 1}, obs={s: _one_hot(self._ctx[s]) for s in (0, 1)}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._ctx = self._rng.integers(CONTEXTS, size=2)
        return self._turn({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        rewards = {s: float(int(actions[s]) == int(self._ctx[s])) for s in (0, 1)}
        self._t += 1
        if self._t >= self.LENGTH:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        self._ctx = self._rng.integers(CONTEXTS, size=2)
        return self._turn(rewards)


class OffsetTeacher(ScriptedBot):
    """Plays ``(context + 1) % 4``: an action no reward of ``SilentBandit`` points to."""

    def act(self, obs: Any, mask: Any, info: Any) -> int:
        return int((int(np.argmax(obs)) + 1) % CONTEXTS)
