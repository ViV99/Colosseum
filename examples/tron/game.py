"""Tron light cycles: free-for-all with elimination, 2 to 4 players (spec block 10).

Every live cycle moves one cell per step, all at once, and leaves a wall behind. A cycle crashes
(``terminated``) when it moves off the board, into any wall (its own included, current heads
included) or into the same cell as another cycle in the same step. The match ends when at most
one cycle is left or after ``size * size`` steps.

Ranks follow the elimination order: cycles that crash in the same step share the average of their
places (two cycles fighting for places 3 and 4 both get 3.5; a 2p head-on crash is 1.5 / 1.5, a
draw). Survivors at the end share the top places the same way. ``Outcome.team_rank`` carries the
ranks (team = seat). Each cycle gets one reward when its place is known (at its crash or at the
end): ``1 - 2 * (rank - 1) / (n - 1)``, so +1 for first, -1 for last.

Layouts ``2p``, ``3p``, ``4p`` (``GameSpec.symmetric([2, 3, 4], ...)``), one role.
Observation: ``uint8[2, 2r+1, 2r+1]``, a window around the head rotated so that the heading points
up (row 0): channel 0 = blocked (walls, trails, off-board), channel 1 = other cycles' heads.
Action: ``Discrete(3)``: 0 straight, 1 turn left, 2 turn right. No masks.
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult

# headings: 0 up, 1 right, 2 down, 3 left, as (drow, dcol)
HEADINGS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int64)
TURN = np.array([0, -1, 1], dtype=np.int64)   # straight, left, right


class TronGame(MultiAgentEnv):
    def __init__(self, size: int = 10, view_radius: int = 3, players: tuple[int, ...] = (2, 3, 4)) -> None:
        if size < 6 or not set(players) <= {2, 3, 4}:
            raise ValueError("size must be >= 6 and players a subset of {2, 3, 4}")
        if view_radius < 1:
            raise ValueError(f"view_radius must be >= 1, got {view_radius}")
        self.size, self.radius = size, view_radius
        side = 2 * view_radius + 1
        obs_space = gymnasium.spaces.Box(0, 1, (2, side, side), np.uint8)
        self.spec = GameSpec.symmetric(list(players), obs_space, gymnasium.spaces.Discrete(3))
        self._rng = np.random.default_rng()
        self._n = 0
        self._walls = np.zeros((size, size), bool)
        self._head = np.zeros((4, 2), np.int64)
        self._heading = np.zeros(4, np.int64)
        self._alive = np.zeros(4, bool)
        self._ranks: dict[int, float] = {}
        self._t = 0

    def _starts(self) -> tuple[np.ndarray, np.ndarray]:
        """Start cells on the four sides of the board, heading inwards, shuffled per episode."""
        s, q = self.size, self.size // 4
        j = self._rng.integers(-1, 2, size=4)
        cells = np.array([[s // 2 + j[0], q - 1], [q - 1, s // 2 + j[1]],
                          [s // 2 + j[2], s - q], [s - q, s // 2 + j[3]]], dtype=np.int64)
        headings = np.array([1, 2, 3, 0], dtype=np.int64)          # each faces the centre
        order = self._rng.permutation(4)[: self._n]
        return cells[order], headings[order]

    def _obs(self, seat: int) -> np.ndarray:
        r, side = self.radius, 2 * self.radius + 1
        padded = np.ones((self.size + 2 * r, self.size + 2 * r), bool)
        padded[r:-r, r:-r] = self._walls
        heads = np.zeros_like(padded)
        for other in np.flatnonzero(self._alive[: self._n]):
            if other != seat:
                heads[self._head[other, 0] + r, self._head[other, 1] + r] = True
        row, col = self._head[seat]
        win = np.stack([padded[row:row + side, col:col + side], heads[row:row + side, col:col + side]])
        return np.ascontiguousarray(np.rot90(win, k=int(self._heading[seat]), axes=(1, 2))).astype(np.uint8)

    def _live(self) -> list[int]:
        return [int(s) for s in np.flatnonzero(self._alive[: self._n])]

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._n = len(self.spec.layouts[layout])
        self._t = 0
        self._ranks = {}
        self._walls[:] = False
        self._alive[:] = False
        self._alive[: self._n] = True
        cells, headings = self._starts()
        self._head[: self._n], self._heading[: self._n] = cells, headings
        for seat in range(self._n):
            self._walls[cells[seat, 0], cells[seat, 1]] = True
        live = self._live()
        return StepResult(acting=set(live), obs={s: self._obs(s) for s in live})

    def _reward(self, rank: float) -> float:
        return 1.0 - 2.0 * (rank - 1.0) / (self._n - 1)

    def step(self, actions: dict) -> StepResult:
        live = self._live()
        targets = {}
        for seat in live:
            self._heading[seat] = (self._heading[seat] + TURN[int(actions[seat])]) % 4
            targets[seat] = self._head[seat] + HEADINGS[self._heading[seat]]
        crashed = set()
        for seat, (row, col) in targets.items():
            off = not (0 <= row < self.size and 0 <= col < self.size)
            if off or self._walls[row, col]:
                crashed.add(seat)
        cells = [tuple(t) for t in targets.values()]
        for seat, cell in zip(targets, cells):
            if cells.count(cell) > 1:
                crashed.add(seat)
        for seat in live:
            if seat not in crashed:
                self._head[seat] = targets[seat]
                self._walls[targets[seat][0], targets[seat][1]] = True
        self._t += 1
        survivors = [s for s in live if s not in crashed]
        rewards: dict[int, float] = {}
        if crashed:   # places len(survivors)+1 .. len(live), shared
            shared = len(survivors) + (len(crashed) + 1) / 2.0
            for seat in crashed:
                self._ranks[seat] = shared
                rewards[seat] = self._reward(shared)
                self._alive[seat] = False
        over = len(survivors) <= 1 or self._t >= self.size * self.size
        if not over:
            return StepResult(acting=set(survivors), obs={s: self._obs(s) for s in survivors},
                              rewards=rewards, terminated=crashed)
        if survivors:
            shared = (len(survivors) + 1) / 2.0
            for seat in survivors:
                self._ranks[seat] = shared
                rewards[seat] = self._reward(shared)
        return StepResult(acting=set(), obs={}, rewards=rewards, terminated=crashed, episode_over=True,
                          outcome=Outcome(team_rank=dict(self._ranks)))


def safe_action(obs: np.ndarray, rng: np.random.Generator) -> int:
    """Reference policy for sanity checks: a random action among those whose next cell is free."""
    r = obs.shape[1] // 2
    free = [a for a, (dr, dc) in enumerate(((-1, 0), (0, -1), (0, 1))) if not obs[0, r + dr, r + dc]]
    return int(rng.choice(free)) if free else 0
