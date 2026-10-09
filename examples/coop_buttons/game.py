"""Coop buttons: one team of two seats and no opponent (spec block 10).

Two bots on a ``size x size`` grid each have their own button. The team scores a point only when
both bots press while standing on their buttons in the same step; then both buttons move to new
random cells. A press anywhere else does nothing. The match lasts ``max_steps`` steps (a rule:
``truncated=False``); the team score is the number of points (``Outcome.team_score``).

Rewards: every point gives both seats +1. A seat that ends a step on its own button also gets
``button_reward`` (default 0.05): without it, uniformly random play almost never scores and the
team would get no learning signal. The score (and so every threshold) counts points only.

Layout ``coop2`` (``GameSpec.teams_of([2], ...)``), outcome kind ``score``.
Observation ``float32[9]``: own x, y; own button dx, dy; partner dx, dy; partner's button dx, dy
(relative to the partner); steps left fraction. Action ``Discrete(6)``: stay, up, down, left, right,
press. Moves off the board are masked.
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult

MOVES = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64)
PRESS = 5


class CoopButtonsGame(MultiAgentEnv):
    def __init__(self, size: int = 5, max_steps: int = 40, button_reward: float = 0.05) -> None:
        if size < 3:
            raise ValueError(f"size must be >= 3, got {size}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        self.size, self.max_steps, self.button_reward = size, max_steps, button_reward
        obs_space = gymnasium.spaces.Box(-1.0, 1.0, (9,), np.float32)
        self.spec = GameSpec.teams_of([2], obs_space, gymnasium.spaces.Discrete(6))
        self._rng = np.random.default_rng()
        self._pos = np.zeros((2, 2), np.int64)
        self._button = np.zeros((2, 2), np.int64)
        self._score = 0
        self._t = 0

    def _new_buttons(self) -> None:
        cells = self._rng.choice(self.size * self.size, size=2, replace=False)
        self._button = np.stack([cells // self.size, cells % self.size], axis=1).astype(np.int64)

    def _obs(self, seat: int) -> np.ndarray:
        scale, other = float(self.size - 1), 1 - seat
        return np.concatenate([
            self._pos[seat] / scale,
            (self._button[seat] - self._pos[seat]) / scale,
            (self._pos[other] - self._pos[seat]) / scale,
            (self._button[other] - self._pos[other]) / scale,
            [(self.max_steps - self._t) / self.max_steps],
        ]).astype(np.float32)

    def _mask(self, seat: int) -> np.ndarray:
        nxt = self._pos[seat] + MOVES
        return np.append(((nxt >= 0) & (nxt < self.size)).all(axis=1), True)

    def _result(self, rewards: dict[int, float]) -> StepResult:
        return StepResult(acting={0, 1}, obs={s: self._obs(s) for s in (0, 1)},
                          action_masks={s: self._mask(s) for s in (0, 1)}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t, self._score = 0, 0
        self._pos = self._rng.integers(self.size, size=(2, 2))
        self._new_buttons()
        return self._result({})

    def step(self, actions: dict) -> StepResult:
        acts = [int(actions[0]), int(actions[1])]
        on_button = [bool((self._pos[s] == self._button[s]).all()) for s in (0, 1)]
        point = all(acts[s] == PRESS and on_button[s] for s in (0, 1))
        for s in (0, 1):
            if acts[s] != PRESS:
                self._pos[s] = np.clip(self._pos[s] + MOVES[acts[s]], 0, self.size - 1)
        if point:
            self._score += 1
            self._new_buttons()
        self._t += 1
        on_button = [bool((self._pos[s] == self._button[s]).all()) for s in (0, 1)]
        rewards = {s: (1.0 if point else 0.0) + (self.button_reward if on_button[s] else 0.0) for s in (0, 1)}
        if self._t < self.max_steps:
            return self._result(rewards)
        return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True,
                          outcome=Outcome(team_score={0: float(self._score)}))


def scripted_action(obs: np.ndarray, size: int = 5) -> int:
    """Oracle used to set the learning threshold: walk to the own button, wait there, press when
    the partner stands on its button too (both see the same facts, so both press together)."""
    to_button = np.rint(obs[2:4] * (size - 1))
    partner_on = not np.rint(obs[6:8] * (size - 1)).any()
    if not to_button.any():
        return PRESS if partner_on else 0
    dx, dy = to_button
    if dx != 0:
        return 4 if dx > 0 else 3
    return 2 if dy > 0 else 1


def measure_baselines(episodes: int = 300, seed: int = 0, **env_kwargs) -> dict[str, float]:
    """Mean team score of the oracle and of uniformly random legal play (numpy only)."""
    env = CoopButtonsGame(**env_kwargs)
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("oracle", "random"):
        scores = []
        for ep in range(episodes):
            res = env.reset(seed + ep, "coop2")
            while not res.episode_over:
                if name == "oracle":
                    acts = {s: scripted_action(res.obs[s], env.size) for s in res.acting}
                else:
                    acts = {s: int(rng.choice(np.flatnonzero(res.action_masks[s]))) for s in res.acting}
                res = env.step(acts)
            scores.append(res.outcome.team_score[0])
        out[name] = float(np.mean(scores))
    return out
