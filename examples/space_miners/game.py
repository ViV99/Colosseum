"""Space Miners on the SP2 contract: a 1v1 competitive game with units (reference example).

Two players, three ships each, push asteroids into their own base to score (Box2D physics in
``game_engine.py``). Presets: ``Round 1`` (no energy or upgrades), ``Round 2`` (energy),
``Final Round`` (energy + upgrades; upgrades are not controlled by this wrapper). An unknown
preset is a ``ValueError``.

Requires Box2D (``pip install -e ".[examples]"``).

Contract features shown here:
- ``GameSpec.symmetric(2, ...)``, simultaneous moves, a rule-based end at ``max_ticks``
  (``truncated=False``) with ``Outcome.team_score`` = game scores (equal scores broken by the engine's
  "who scored first" rule through ``Outcome.team_rank``; a full tie gives both teams rank 1);
- ships as ``Units(3, Dict(accel=Box(2), push=Discrete(2)))``: a continuous component and a
  discrete one per unit (the ``Dict`` is built from a list, so the order is ``accel``, ``push``);
- asteroids as an entity list with a mask (``asteroids`` + ``asteroid_mask``) in a ``Dict``
  observation;
- masks inside ``Units``: ``push`` is legal only when an asteroid is within push range (otherwise
  the push would do nothing). With energy (``Round 2``, ``Final Round``) a legal push can still be
  unaffordable: when a ship's energy is below the cost of its acceleration plus push, the engine
  drops that ship's whole command for the tick, acceleration included.

Observation from the player's perspective (``Dict``, positions in [-1, 1], player 1 mirrored so
that both players' bases are on the left):
- ``ships``: ``float32[3, 9]`` own ships: x, y, vx, vy, energy, 4 upgrade levels;
- ``enemy_ships``: ``float32[3, 9]``;
- ``asteroids``: ``float32[max_asteroids, 7]``: x, y, vx, vy, size one-hot; ``asteroid_mask``;
- ``global``: ``float32[3]``: own score / 100, opponent score / 100, time fraction.
Rewards: (own score gain - opponent score gain) / 20 per step, +1 / -1 at the end for win / loss.
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult
from colosseum.envs.spaces import Units
from examples.space_miners.game_engine import (
    ASTEROID_RADIUS_UNITS,
    GAME_HEIGHT,
    GAME_WIDTH,
    MAX_ACCELERATION,
    MAX_VELOCITY,
    PPM,
    PRESET_CONFIGS,
    PUSH_RADIUS_UNITS,
    SHIP_RADIUS_UNITS,
    SpaceMinersGameState,
)

MAX_SHIPS = 3
SHIP_FEATURES = 9
ASTEROID_FEATURES = 7
ENGINE_MAX_ASTEROIDS = 21      # the engine keeps at most 21 (20 + one pair overshoot on respawn)
DEFAULT_MAX_ASTEROIDS = 24
SIZE_NAMES = ("small", "medium", "large")
UPGRADE_NAMES = ("max_speed", "max_accel", "push_force", "energy_efficiency")


class SpaceMinersGame(MultiAgentEnv):
    def __init__(self, preset: str = "Round 1", max_ticks: int = 1000,
                 max_asteroids: int = DEFAULT_MAX_ASTEROIDS) -> None:
        if preset not in PRESET_CONFIGS:
            raise ValueError(f"unknown preset {preset!r}; expected one of "
                             f"{', '.join(repr(p) for p in PRESET_CONFIGS)}")
        if max_ticks < 1:
            raise ValueError(f"max_ticks must be >= 1, got {max_ticks}")
        if max_asteroids < ENGINE_MAX_ASTEROIDS:
            raise ValueError(f"max_asteroids must be >= {ENGINE_MAX_ASTEROIDS} (the engine keeps up to "
                             f"{ENGINE_MAX_ASTEROIDS} asteroids), got {max_asteroids}")
        self._preset, self._max_ticks, self._max_asteroids = preset, max_ticks, max_asteroids
        box = gymnasium.spaces.Box
        obs_space = gymnasium.spaces.Dict([
            ("ships", box(-1.0, 1.0, (MAX_SHIPS, SHIP_FEATURES), np.float32)),
            ("enemy_ships", box(-1.0, 1.0, (MAX_SHIPS, SHIP_FEATURES), np.float32)),
            ("asteroids", box(-1.0, 1.0, (max_asteroids, ASTEROID_FEATURES), np.float32)),
            ("asteroid_mask", gymnasium.spaces.MultiBinary(max_asteroids)),
            ("global", box(-1.0, 1.0, (3,), np.float32)),
        ])
        per_ship = gymnasium.spaces.Dict([("accel", box(-1.0, 1.0, (2,), np.float32)),
                                          ("push", gymnasium.spaces.Discrete(2))])
        self.spec = GameSpec.symmetric(2, obs_space, Units(MAX_SHIPS, per_ship))
        self._game: SpaceMinersGameState | None = None

    # ----- observation ---------------------------------------------------------------------
    @staticmethod
    def _xy(pos: tuple[float, float], player: int) -> tuple[float, float]:
        x = float(np.clip(pos[0] / (GAME_WIDTH / 2) - 1.0, -1.0, 1.0))
        y = float(np.clip(pos[1] / (GAME_HEIGHT / 2) - 1.0, -1.0, 1.0))
        return (-x if player == 1 else x), y

    @staticmethod
    def _v(vel: tuple[float, float], player: int) -> tuple[float, float]:
        vx = float(np.clip(vel[0] / MAX_VELOCITY, -1.0, 1.0))
        return (-vx if player == 1 else vx), float(np.clip(vel[1] / MAX_VELOCITY, -1.0, 1.0))

    def _ships(self, owner: int, viewer: int) -> np.ndarray:
        game = self._game
        out = np.zeros((MAX_SHIPS, SHIP_FEATURES), np.float32)
        for i, ship in enumerate(game.players[owner].ships):
            out[i, 0:2] = self._xy(ship.pos_game, viewer)
            out[i, 2:4] = self._v(ship.vel_game, viewer)
            if game.energy_enabled:
                out[i, 4] = ship.energy / 50.0 - 1.0
            if game.upgrades_enabled:
                out[i, 5:9] = [min(ship.upgrades[k] / 5.0, 1.0) for k in UPGRADE_NAMES]
        return out

    def _obs(self, player: int) -> dict:
        game = self._game
        asteroids = np.zeros((self._max_asteroids, ASTEROID_FEATURES), np.float32)
        mask = np.zeros(self._max_asteroids, np.int8)
        for i, a in enumerate(game.asteroids):
            asteroids[i, 0:2] = self._xy(a.pos_game, player)
            asteroids[i, 2:4] = self._v(a.vel_game, player)
            asteroids[i, 4 + SIZE_NAMES.index(a.size)] = 1.0
            mask[i] = 1
        me, opp = game.players[player], game.players[1 - player]
        glob = np.array([np.clip(me.score / 100.0, -1, 1), np.clip(opp.score / 100.0, -1, 1),
                         game.tick / game.max_ticks * 2.0 - 1.0], np.float32)
        return {"ships": self._ships(player, player), "enemy_ships": self._ships(1 - player, player),
                "asteroids": asteroids, "asteroid_mask": mask, "global": glob}

    def _mask(self, player: int) -> dict:
        """``push=1`` is legal for a ship only when an asteroid is within the engine's push range."""
        reach = np.zeros(MAX_SHIPS, bool)
        for i, ship in enumerate(self._game.players[player].ships):
            sx, sy = ship.body.position
            for a in self._game.asteroids:
                ax, ay = a.body.position
                gap = np.hypot(ax - sx, ay - sy) - (SHIP_RADIUS_UNITS + ASTEROID_RADIUS_UNITS[a.size]) / PPM
                if gap <= PUSH_RADIUS_UNITS / PPM:
                    reach[i] = True
                    break
        action = np.ones((MAX_SHIPS, 2), bool)
        action[:, 1] = reach                       # push=1 only within reach; push=0 always legal
        return {"unit": np.ones(MAX_SHIPS, bool), "action": action}

    def _acting_result(self, rewards: dict[int, float]) -> StepResult:
        return StepResult(acting={0, 1}, obs={p: self._obs(p) for p in (0, 1)},
                          action_masks={p: self._mask(p) for p in (0, 1)}, rewards=rewards)

    # ----- MultiAgentEnv -------------------------------------------------------------------
    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._game = SpaceMinersGameState(preset=self._preset, max_ticks=self._max_ticks, seed=seed)
        return self._acting_result({})

    def _commands(self, player: int, action: dict) -> dict:
        accel = np.asarray(action["accel"], np.float32).reshape(MAX_SHIPS, 2)
        push = np.asarray(action["push"], np.int64).reshape(MAX_SHIPS)
        sign = -1.0 if player == 1 else 1.0         # undo the mirror of player 1
        return {"commands": [{"ship_id": i,
                              "acceleration": {"x": sign * float(np.clip(accel[i, 0], -1, 1)) * MAX_ACCELERATION,
                                               "y": float(np.clip(accel[i, 1], -1, 1)) * MAX_ACCELERATION},
                              "push": bool(push[i])} for i in range(MAX_SHIPS)]}

    def step(self, actions: dict) -> StepResult:
        game = self._game
        before = [p.score for p in game.players]
        game.update([self._commands(0, actions[0]), self._commands(1, actions[1])])
        gain = [game.players[i].score - before[i] for i in (0, 1)]
        rewards = {i: (gain[i] - gain[1 - i]) / 20.0 for i in (0, 1)}
        if not game.is_game_over():
            return self._acting_result(rewards)
        winner = game.get_winner_index()
        if winner >= 0:
            rewards[winner] += 1.0
            rewards[1 - winner] -= 1.0
        ranks = {0: 1.0, 1: 1.0} if winner < 0 else {winner: 1.0, 1 - winner: 2.0}   # ties share the min rank
        scores = {i: float(game.players[i].score) for i in (0, 1)}
        return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True,
                          outcome=Outcome(team_rank=ranks, team_score=scores))

    def close(self) -> None:
        self._game = None
