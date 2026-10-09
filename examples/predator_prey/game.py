"""Predator and prey: one hunter against two prey, roles with different spaces (spec block 10).

On a ``size x size`` grid the hunter (team 0) and two prey (team 1) move simultaneously. After the
moves, every prey within ``catch_radius`` of the hunter (Chebyshev distance) is caught: it is
eliminated (``terminated``) with reward -1, and the hunter gets +0.5 per catch. The match ends by
rule when both prey are caught (the hunter wins) or after ``max_steps`` steps (the prey team wins
if at least one prey is still free; a surviving prey gets +1, the hunter -1).
``Outcome.team_rank`` = {winner: 1, loser: 2}.

Roles (layout ``1v2``: seat 0 hunter / team 0, seats 1-2 prey / team 1):
- ``hunter``: observation ``float32[9]``: own x, y; for both prey (nearest first) dx, dy and a free
  flag; steps left fraction. Action ``Discrete(5)``: stay, up, down, left, right.
- ``prey``: observation ``float32[7]``: own x, y; dx, dy to the hunter; dx, dy to the other prey and
  its free flag. Action ``Discrete(9)``: stay and the eight king moves.
Moves off the board are masked. The roles have different spaces, so they need separate agents
(``agents.<id>.roles``).
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, RoleSpec, SeatSpec, StepResult

HUNTER_MOVES = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64)
PREY_MOVES = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]],
                      dtype=np.int64)
HUNTER, PREY = 0, (1, 2)


class PredatorPreyGame(MultiAgentEnv):
    def __init__(self, size: int = 7, max_steps: int = 30, catch_radius: int = 1) -> None:
        if size < 4:
            raise ValueError(f"size must be >= 4, got {size}")
        self.size, self.max_steps, self.catch_radius = size, max_steps, catch_radius
        box = gymnasium.spaces.Box
        hunter = RoleSpec(box(-1.0, 1.0, (9,), np.float32), gymnasium.spaces.Discrete(5))
        prey = RoleSpec(box(-1.0, 1.0, (7,), np.float32), gymnasium.spaces.Discrete(9))
        self.spec = GameSpec(roles={"hunter": hunter, "prey": prey},
                             layouts={"1v2": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1))})
        self._rng = np.random.default_rng()
        self._pos = np.zeros((3, 2), np.int64)
        self._free = np.ones(3, bool)
        self._t = 0

    def _moves(self, seat: int) -> np.ndarray:
        return HUNTER_MOVES if seat == HUNTER else PREY_MOVES

    def _mask(self, seat: int) -> np.ndarray:
        nxt = self._pos[seat] + self._moves(seat)
        return ((nxt >= 0) & (nxt < self.size)).all(axis=1)

    def _obs(self, seat: int) -> np.ndarray:
        scale = float(self.size - 1)
        own = self._pos[seat] / scale
        if seat == HUNTER:
            prey = sorted(PREY, key=lambda p: (not self._free[p], np.abs(self._pos[p] - self._pos[0]).sum()))
            parts = [own]
            for p in prey:
                d = (self._pos[p] - self._pos[0]) / scale if self._free[p] else np.zeros(2)
                parts += [d, [float(self._free[p])]]
            parts.append([(self.max_steps - self._t) / self.max_steps])
        else:
            other = PREY[1] if seat == PREY[0] else PREY[0]
            d_other = (self._pos[other] - self._pos[seat]) / scale if self._free[other] else np.zeros(2)
            parts = [own, (self._pos[HUNTER] - self._pos[seat]) / scale, d_other, [float(self._free[other])]]
        return np.concatenate(parts).astype(np.float32)

    def _acting(self) -> set[int]:
        return {HUNTER} | {p for p in PREY if self._free[p]}

    def _result(self, rewards: dict[int, float], terminated: set[int]) -> StepResult:
        acting = self._acting()
        return StepResult(acting=acting, obs={s: self._obs(s) for s in acting},
                          action_masks={s: self._mask(s) for s in acting}, rewards=rewards, terminated=terminated)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._free[:] = True
        while True:
            self._pos = self._rng.integers(self.size, size=(3, 2))
            gap = np.abs(self._pos[1:] - self._pos[0]).max(axis=1)
            if (gap > self.catch_radius + 1).all():
                break
        return self._result({}, set())

    def step(self, actions: dict) -> StepResult:
        for seat, action in actions.items():
            self._pos[seat] = np.clip(self._pos[seat] + self._moves(seat)[int(action)], 0, self.size - 1)
        self._t += 1
        caught = {p for p in PREY if self._free[p]
                  and np.abs(self._pos[p] - self._pos[HUNTER]).max() <= self.catch_radius}
        rewards = {p: -1.0 for p in caught}
        rewards[HUNTER] = 0.5 * len(caught)
        for p in caught:
            self._free[p] = False
        if not self._free[1:].any():
            return StepResult(acting=set(), obs={}, rewards=rewards, terminated=caught, episode_over=True,
                              outcome=Outcome(team_rank={0: 1.0, 1: 2.0}))
        if self._t < self.max_steps:
            return self._result(rewards, caught)
        rewards[HUNTER] -= 1.0
        for p in PREY:
            if self._free[p]:
                rewards[p] = rewards.get(p, 0.0) + 1.0
        return StepResult(acting=set(), obs={}, rewards=rewards, terminated=caught, episode_over=True,
                          outcome=Outcome(team_rank={0: 2.0, 1: 1.0}))


def chase_action(obs: np.ndarray) -> int:
    """Reference hunter: step towards the nearest free prey."""
    dx, dy = obs[2], obs[3]
    if abs(dx) >= abs(dy) and dx != 0:
        return 4 if dx > 0 else 3
    if dy != 0:
        return 2 if dy > 0 else 1
    return 0


def flee_action(obs: np.ndarray, mask: np.ndarray, size: int = 7) -> int:
    """Reference prey: the legal move that maximizes the Chebyshev distance to the hunter."""
    to_hunter = np.rint(obs[2:4] * (size - 1))
    dist = [np.abs(to_hunter - PREY_MOVES[a]).max() if mask[a] else -1.0 for a in range(9)]
    return int(np.argmax(dist))
