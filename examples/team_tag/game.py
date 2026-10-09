"""Team tag: two teams of two, each seat is a separate bot (spec block 10).

Four bots on a ``size x size`` grid act simultaneously. A bot may tag (freeze) the enemies within
one cell of it (Chebyshev distance <= 1, its own cell included); all tags of a step resolve at
once, so two bots can freeze each other. A frozen bot stays on the board, stops acting and is
*not* terminated: it still receives its team's rewards and gets ``terminal`` at the end of the
episode (the "dead teammate" rule of the contract).

The match ends by rule when a team has no active bot or after ``max_steps`` steps
(``truncated=False``). ``Outcome.team_score`` = active bots per team; more wins, equal is a draw.
Rewards (shared by every member of a team, frozen or not): ``tag_reward`` for each enemy the team
froze this step, ``-tag_reward`` for each own bot frozen, and +1 / -1 / 0 at the end.

Each bot sees only a local window; the centralized critic gets the full map as ``global_state``.
Team 1 sees the board mirrored left-right (both teams "start on the left"); its actions are mirrored
back.
- Observation (``Dict``): ``window``: ``uint8[4, 2r+1, 2r+1]`` around the bot: allies (active),
  enemies (active), frozen bots, off-board; ``vec``: ``float32[4]``: x, y, steps left fraction,
  active teammates other than itself.
- ``global_state`` (if ``with_global_state``): ``uint8[4, size, size]``: own team active, enemies
  active, frozen bots, the bot itself.
- Action: ``Discrete(6)``: stay, up, down, left, right, tag. Moves off the board are masked; tag is
  masked unless an active enemy is within reach.
Layout ``2v2`` (``GameSpec.teams_of([2, 2], ...)``).
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult

# stay, up, down, left, right as (dx, dy); x grows to the right, y grows down
MOVES = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64)
MIRROR_ACTION = np.array([0, 1, 2, 4, 3, 5], dtype=np.int64)   # left <-> right for team 1
TAG = 5
TEAM = np.array([0, 0, 1, 1])


class TeamTagGame(MultiAgentEnv):
    def __init__(self, size: int = 7, view_radius: int = 2, max_steps: int = 30, tag_reward: float = 0.2,
                 with_global_state: bool = True) -> None:
        if size < 5:
            raise ValueError(f"size must be >= 5, got {size}")
        self.size, self.radius, self.max_steps, self.tag_reward = size, view_radius, max_steps, tag_reward
        w = 2 * view_radius + 1
        obs_space = gymnasium.spaces.Dict([
            ("window", gymnasium.spaces.Box(0, 1, (4, w, w), np.uint8)),
            ("vec", gymnasium.spaces.Box(0.0, 1.0, (4,), np.float32)),
        ])
        gs_space = gymnasium.spaces.Box(0, 1, (4, size, size), np.uint8) if with_global_state else None
        self.with_global_state = with_global_state
        self.spec = GameSpec.teams_of([2, 2], obs_space, gymnasium.spaces.Discrete(6), gs_space)
        self._rng = np.random.default_rng()
        self._pos = np.zeros((4, 2), np.int64)
        self._active = np.ones(4, bool)
        self._t = 0

    def _view(self, seat: int, xy: np.ndarray) -> np.ndarray:
        out = np.array(xy, copy=True)
        if TEAM[seat] == 1:
            out[..., 0] = self.size - 1 - out[..., 0]
        return out

    def _maps(self, seat: int) -> np.ndarray:
        """``bool[4, size, size]`` from ``seat``'s perspective: allies, enemies, frozen, self."""
        maps = np.zeros((4, self.size, self.size), bool)
        pos = self._view(seat, self._pos)
        for other in range(4):
            x, y = pos[other]
            if not self._active[other]:
                maps[2, y, x] = True
            elif TEAM[other] == TEAM[seat]:
                maps[0, y, x] = True
            else:
                maps[1, y, x] = True
        maps[3, pos[seat, 1], pos[seat, 0]] = True
        return maps

    def _obs(self, seat: int) -> dict:
        r, w = self.radius, 2 * self.radius + 1
        maps = self._maps(seat)
        padded = np.zeros((4, self.size + 2 * r, self.size + 2 * r), bool)
        padded[:3, r:-r, r:-r] = maps[:3]
        padded[3] = True
        padded[3, r:-r, r:-r] = False
        x, y = self._view(seat, self._pos[seat])
        mates = [s for s in range(4) if TEAM[s] == TEAM[seat] and s != seat and self._active[s]]
        vec = np.array([x / (self.size - 1), y / (self.size - 1), (self.max_steps - self._t) / self.max_steps,
                        float(len(mates))], np.float32)
        return {"window": padded[:, y:y + w, x:x + w].astype(np.uint8), "vec": vec}

    def _enemy_in_reach(self, seat: int) -> bool:
        enemies = (TEAM != TEAM[seat]) & self._active
        dist = np.abs(self._pos - self._pos[seat]).max(axis=1)
        return bool((enemies & (dist <= 1)).any())

    def _mask(self, seat: int) -> np.ndarray:
        nxt = self._view(seat, self._pos[seat]) + MOVES
        mask = np.ones(6, bool)
        mask[:5] = ((nxt >= 0) & (nxt < self.size)).all(axis=1)
        mask[TAG] = self._enemy_in_reach(seat)
        return mask

    def _acting_result(self, rewards: dict[int, float]) -> StepResult:
        acting = {int(s) for s in np.flatnonzero(self._active)}
        gs = {s: self._maps(s).astype(np.uint8) for s in acting} if self.with_global_state else None
        return StepResult(acting=acting, obs={s: self._obs(s) for s in acting},
                          action_masks={s: self._mask(s) for s in acting}, rewards=rewards, global_state=gs)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._active[:] = True
        rows = self._rng.choice(self.size, size=4, replace=False)
        self._pos = np.array([[0, rows[0]], [0, rows[1]], [self.size - 1, rows[2]], [self.size - 1, rows[3]]],
                             dtype=np.int64)
        return self._acting_result({})

    def step(self, actions: dict) -> StepResult:
        taggers = []
        for seat, action in actions.items():
            a = int(action)
            if TEAM[seat] == 1:
                a = int(MIRROR_ACTION[a])
            if a == TAG:
                taggers.append(seat)
        frozen_now = set()
        for seat in taggers:   # resolved against the positions before the moves, all at once
            dist = np.abs(self._pos - self._pos[seat]).max(axis=1)
            for other in np.flatnonzero((TEAM != TEAM[seat]) & self._active & (dist <= 1)):
                frozen_now.add(int(other))
        for seat, action in actions.items():
            a = int(action)
            if TEAM[seat] == 1:
                a = int(MIRROR_ACTION[a])
            if a != TAG and seat not in frozen_now:
                self._pos[seat] = np.clip(self._pos[seat] + MOVES[a], 0, self.size - 1)
        for seat in frozen_now:
            self._active[seat] = False
        self._t += 1
        lost = np.array([sum(1 for s in frozen_now if TEAM[s] == t) for t in (0, 1)])
        team_reward = {t: self.tag_reward * float(lost[1 - t] - lost[t]) for t in (0, 1)}
        active = np.array([int(self._active[TEAM == t].sum()) for t in (0, 1)])
        over = active.min() == 0 or self._t >= self.max_steps
        if not over:
            return self._acting_result({s: team_reward[TEAM[s]] for s in range(4)})
        final = {0: float(np.sign(active[0] - active[1])), 1: float(np.sign(active[1] - active[0]))}
        rewards = {s: team_reward[TEAM[s]] + final[TEAM[s]] for s in range(4)}
        return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True,
                          outcome=Outcome(team_score={0: float(active[0]), 1: float(active[1])}))


def chase_action(obs: dict, mask: np.ndarray) -> int:
    """Reference policy for sanity checks: tag when possible, else step towards the nearest
    visible enemy, else walk right (towards the enemy side)."""
    if mask[TAG]:
        return TAG
    enemies = np.argwhere(obs["window"][1])
    if len(enemies):
        r = obs["window"].shape[1] // 2
        dy, dx = (enemies[np.abs(enemies - r).sum(axis=1).argmin()] - r)
        if dx != 0 and mask[4 if dx > 0 else 3]:
            return 4 if dx > 0 else 3
        if dy != 0 and mask[2 if dy > 0 else 1]:
            return 2 if dy > 0 else 1
        return 0
    return 4 if mask[4] else 0
