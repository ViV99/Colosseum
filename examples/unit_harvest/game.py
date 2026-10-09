"""Unit harvest: one bot controls a base and a variable number of workers (spec block 10).

Two sides on a ``size x size`` grid move simultaneously. Each side has a base cell (seat 0 on the
left edge, seat 1 on the right edge) and up to ``max_units`` workers.
- A worker that stands on a resource cell with empty hands picks up one resource; a worker that
  carries a resource and stands on its own base deposits it (+1 score, +1 stock).
- The base may build a worker for ``build_cost`` stock; it appears on the base in the lowest free
  unit slot (units are born).
- After the moves, every cell that holds workers of both sides loses all of them (units die).
- The match lasts exactly ``max_steps`` steps (a rule, not a cut: ``truncated=False``). The side
  with the higher score wins (``Outcome.team_score``; equal scores are a draw).

Rewards: ``deposit_reward`` per deposit, plus +1 / -1 / 0 for a win / loss / draw at the end.

Observation (``Dict``, per seat, mirrored for seat 1 so both sides "start on the left"):
- ``units``: ``float32[max_units, 10]`` per own worker: x, y, carrying, direction to the nearest
  resource (dx, dy), to the own base (dx, dy), to the nearest enemy worker (dx, dy), enemy adjacent;
- ``unit_mask``: ``MultiBinary(max_units)``, 1 = the slot holds a live worker;
- ``enemies``: ``float32[max_units, 2]`` enemy worker positions + ``enemy_mask``;
- ``resources``: ``float32[num_resources, 2]`` resource positions + ``resource_mask`` (all ones:
  resources never run out; the mask follows the entity-list convention);
- ``base``: ``float32[4]``: stock / 10, own score / 10, opponent score / 10, steps left fraction.

Action (``Dict`` in this order): ``base``: ``Discrete(2)`` (0 idle, 1 build; build is masked when
the stock is short or every slot is used) and ``workers``: ``Units(max_units, Discrete(5))``
(stay, up, down, left, right; moves off the grid are masked). Deciders: 1 + ``max_units``.
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult
from colosseum.envs.spaces import Units

# stay, up, down, left, right as (dx, dy); x grows to the right, y grows down
MOVES = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64)
MIRROR_ACTION = np.array([0, 1, 2, 4, 3], dtype=np.int64)   # left <-> right for seat 1
UNIT_FEATURES = 10


class UnitHarvestGame(MultiAgentEnv):
    def __init__(self, max_units: int = 8, size: int = 8, num_resources: int = 6, initial_workers: int = 2,
                 build_cost: int = 1, max_steps: int = 50, deposit_reward: float = 0.1) -> None:
        if not 1 <= initial_workers <= max_units:
            raise ValueError(f"initial_workers must be in [1, max_units], got {initial_workers}")
        if num_resources % 2 or num_resources < 2 or size < 4:
            raise ValueError("num_resources must be even and >= 2, size >= 4")
        free = (size // 2 - 1) * size     # cells of the left half for resources: 1 <= x < size // 2
        if num_resources // 2 > free:
            raise ValueError(f"num_resources={num_resources} does not fit: each half of a size-{size} board "
                             f"has {free} free cells for num_resources // 2 resources")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        self.max_units, self.size, self.num_resources = max_units, size, num_resources
        self.initial_workers, self.build_cost, self.max_steps = initial_workers, build_cost, max_steps
        self.deposit_reward = deposit_reward
        U, R = max_units, num_resources
        box = gymnasium.spaces.Box
        obs_space = gymnasium.spaces.Dict([
            ("units", box(-1.0, 1.0, (U, UNIT_FEATURES), np.float32)),
            ("unit_mask", gymnasium.spaces.MultiBinary(U)),
            ("enemies", box(0.0, 1.0, (U, 2), np.float32)),
            ("enemy_mask", gymnasium.spaces.MultiBinary(U)),
            ("resources", box(0.0, 1.0, (R, 2), np.float32)),
            ("resource_mask", gymnasium.spaces.MultiBinary(R)),
            ("base", box(0.0, np.inf, (4,), np.float32)),
        ])
        act_space = gymnasium.spaces.Dict([
            ("base", gymnasium.spaces.Discrete(2)),
            ("workers", Units(U, gymnasium.spaces.Discrete(5))),
        ])
        self.spec = GameSpec.symmetric(2, obs_space, act_space)
        self._rng = np.random.default_rng()
        self._bases = np.array([[0, size // 2], [size - 1, size // 2]], dtype=np.int64)
        self._pos = np.zeros((2, U, 2), np.int64)       # [side, slot, (x, y)]
        self._alive = np.zeros((2, U), bool)
        self._carry = np.zeros((2, U), bool)
        self._resources = np.zeros((R, 2), np.int64)
        self._stock = np.zeros(2, np.int64)
        self._score = np.zeros(2, np.int64)
        self._t = 0

    # ----- helpers -------------------------------------------------------------------------
    def _place_resources(self) -> None:
        half = self.num_resources // 2
        cells = [(x, y) for x in range(1, self.size // 2) for y in range(self.size)]
        cells = [c for c in cells if (c[0], c[1]) != tuple(self._bases[0])]
        idx = self._rng.choice(len(cells), size=half, replace=False)
        left = np.array([cells[i] for i in idx], dtype=np.int64)
        right = left.copy()
        right[:, 0] = self.size - 1 - left[:, 0]
        self._resources = np.concatenate([left, right])

    def _spawn(self, side: int) -> None:
        slot = int(np.flatnonzero(~self._alive[side])[0])
        self._alive[side, slot] = True
        self._carry[side, slot] = False
        self._pos[side, slot] = self._bases[side]

    def _view(self, side: int, xy: np.ndarray) -> np.ndarray:
        """Grid coordinates as seen by ``side`` (seat 1 sees the board mirrored left-right)."""
        out = xy.copy()
        if side == 1:
            out[..., 0] = self.size - 1 - out[..., 0]
        return out

    @staticmethod
    def _nearest(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Per source cell: (dx, dy) to its nearest destination cell (Manhattan); zeros if none."""
        if len(dst) == 0:
            return np.zeros_like(src)
        diff = dst[None, :, :] - src[:, None, :]
        best = np.abs(diff).sum(axis=2).argmin(axis=1)
        return diff[np.arange(len(src)), best]

    def _obs(self, side: int) -> dict:
        U, scale = self.max_units, float(self.size - 1)
        own = self._view(side, self._pos[side])
        enemy_alive = self._alive[1 - side]
        enemy = self._view(side, self._pos[1 - side])[enemy_alive]
        res = self._view(side, self._resources)
        base = self._view(side, self._bases[side])
        units = np.zeros((U, UNIT_FEATURES), np.float32)
        units[:, 0:2] = own / scale
        units[:, 2] = self._carry[side]
        units[:, 3:5] = np.sign(self._nearest(own, res))
        units[:, 5:7] = np.sign(base[None, :] - own)
        to_enemy = self._nearest(own, enemy)
        units[:, 7:9] = np.sign(to_enemy)
        units[:, 9] = (np.abs(to_enemy).sum(axis=1) == 1) if len(enemy) else 0.0
        units[~self._alive[side]] = 0.0
        enemies = np.zeros((U, 2), np.float32)
        enemies[enemy_alive] = enemy / scale
        return {
            "units": units,
            "unit_mask": self._alive[side].astype(np.int8),
            "enemies": enemies,
            "enemy_mask": enemy_alive.astype(np.int8),
            "resources": (res / scale).astype(np.float32),
            "resource_mask": np.ones(self.num_resources, np.int8),
            "base": np.array([self._stock[side] / 10.0, self._score[side] / 10.0, self._score[1 - side] / 10.0,
                              (self.max_steps - self._t) / self.max_steps], np.float32),
        }

    def _mask(self, side: int) -> dict:
        can_build = self._stock[side] >= self.build_cost and not self._alive[side].all()
        own = self._view(side, self._pos[side])
        nxt = own[:, None, :] + MOVES[None, :, :]
        moves = ((nxt >= 0) & (nxt < self.size)).all(axis=2)
        moves[~self._alive[side]] = True
        return {"base": np.array([True, can_build]),
                "workers": {"unit": self._alive[side].copy(), "action": moves}}

    def _result(self, rewards: dict[int, float]) -> StepResult:
        return StepResult(acting={0, 1}, obs={s: self._obs(s) for s in (0, 1)},
                          action_masks={s: self._mask(s) for s in (0, 1)}, rewards=rewards)

    # ----- MultiAgentEnv -------------------------------------------------------------------
    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._alive[:] = False
        self._carry[:] = False
        self._stock[:] = 0
        self._score[:] = 0
        self._place_resources()
        for side in (0, 1):
            for _ in range(self.initial_workers):
                self._spawn(side)
        return self._result({})

    def step(self, actions: dict) -> StepResult:
        for side in (0, 1):
            act = actions[side]
            moves = np.asarray(act["workers"], np.int64).reshape(self.max_units)
            if side == 1:
                moves = MIRROR_ACTION[moves]
            alive = self._alive[side]
            self._pos[side, alive] = np.clip(self._pos[side, alive] + MOVES[moves[alive]], 0, self.size - 1)
            if int(act["base"]) == 1 and self._stock[side] >= self.build_cost and not alive.all():
                self._stock[side] -= self.build_cost
                self._spawn(side)
        # fights: every cell that holds workers of both sides loses all of them
        cell = self._pos[..., 0] * self.size + self._pos[..., 1]                  # [2, U]
        contested = np.intersect1d(cell[0, self._alive[0]], cell[1, self._alive[1]])
        if len(contested):
            hit = self._alive & np.isin(cell, contested)
            self._alive[hit] = False
            self._carry[hit] = False
        # harvest and deposit (vectorized over sides and slots)
        res_cell = self._resources[:, 0] * self.size + self._resources[:, 1]
        base_cell = self._bases[:, 0] * self.size + self._bases[:, 1]               # [2]
        at_base = self._alive & self._carry & (cell == base_cell[:, None])
        on_res = self._alive & ~self._carry & np.isin(cell, res_cell)
        deposits = at_base.sum(axis=1)
        self._carry[at_base] = False
        self._carry[on_res] = True
        self._score += deposits
        self._stock += deposits
        self._t += 1
        rewards = {s: self.deposit_reward * float(deposits[s]) for s in (0, 1)}
        if self._t < self.max_steps:
            return self._result(rewards)
        diff = int(self._score[0] - self._score[1])
        final = {0: float(np.sign(diff)), 1: float(-np.sign(diff))}
        rewards = {s: rewards[s] + final[s] for s in (0, 1)}
        return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True,
                          outcome=Outcome(team_score={0: float(self._score[0]), 1: float(self._score[1])}))


def scripted_action(obs: dict) -> dict:
    """Reference policy for sanity checks: carriers walk home, the others walk to the nearest
    resource; the base builds whenever it can (build legality is checked by the env)."""
    units = obs["units"]
    moves = np.zeros(len(units), np.int64)
    for i, f in enumerate(units):
        dx, dy = (f[5], f[6]) if f[2] > 0 else (f[3], f[4])
        if dx != 0:
            moves[i] = 4 if dx > 0 else 3
        elif dy != 0:
            moves[i] = 2 if dy > 0 else 1
    return {"base": 1 if obs["base"][0] * 10 >= 1 else 0, "workers": moves}
