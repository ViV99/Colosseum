"""Space Miners Hard — Colosseum BaseEnv wrapper.

2-player simultaneous game: push asteroids to your base to score.
Supports Round 1 (no energy/upgrades), Round 2 (energy), Final Round (energy + upgrades).

Requires: pip install Box2D
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium
import numpy as np

from colosseum.envs.base_env import BaseEnv
from examples.space_miners.game_engine import (
    ASTEROID_RADIUS_UNITS,
    BASE_COLLECTION_RADIUS_UNITS,
    GAME_HEIGHT,
    GAME_WIDTH,
    MAX_ACCELERATION,
    MAX_VELOCITY,
    PPM,
    SpaceMinersGameState,
)

# Observation layout constants
SHIP_FEATURES = 9  # pos_x, pos_y, vel_x, vel_y, energy, 4 upgrades
MAX_SHIPS = 3
ASTEROID_FEATURES = 8  # pos_x, pos_y, vel_x, vel_y, is_small, is_med, is_large, exists
DEFAULT_MAX_ASTEROIDS = 20
GLOBAL_FEATURES = 4  # my_score, opp_score, turn_frac, base_side

SIZE_NAMES = ("small", "medium", "large")


def _obs_dim(max_asteroids: int = DEFAULT_MAX_ASTEROIDS) -> int:
    return (
        2 * MAX_SHIPS * SHIP_FEATURES
        + max_asteroids * ASTEROID_FEATURES
        + GLOBAL_FEATURES
    )


class SpaceMinersEnv(BaseEnv):
    """Space Miners Hard environment for Colosseum RL training.

    Observation: flat vector [own_ships | opp_ships | asteroids | global] normalized to ~[-1, 1].
    Action: Dict(accel=Box(6), push_0=Discrete(2), push_1=Discrete(2), push_2=Discrete(2)).
    """

    def __init__(
        self,
        preset: str = "Round 1",
        max_ticks: int = 1000,
        max_asteroids: int = DEFAULT_MAX_ASTEROIDS,
    ):
        self._preset = preset
        self._max_ticks = max_ticks
        self._max_asteroids = max_asteroids
        self._obs_size = _obs_dim(max_asteroids)

        self._game: Optional[SpaceMinersGameState] = None
        self._prev_scores = [0, 0]

    @property
    def num_players(self) -> int:
        return 2

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(self._obs_size,), dtype=np.float32
        )

    @property
    def action_space(self) -> gymnasium.spaces.Dict:
        return gymnasium.spaces.Dict(
            {
                "accel": gymnasium.spaces.Box(
                    low=-1.0, high=1.0, shape=(6,), dtype=np.float32
                ),
                "push_0": gymnasium.spaces.Discrete(2),
                "push_1": gymnasium.spaces.Discrete(2),
                "push_2": gymnasium.spaces.Discrete(2),
            }
        )

    def reset(
        self, seed: Optional[int] = None
    ) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
        self._game = SpaceMinersGameState(
            preset=self._preset,
            max_ticks=self._max_ticks,
            seed=seed,
        )
        self._prev_scores = [0, 0]

        obs = {i: self._build_obs(i) for i in range(2)}
        infos = {i: {} for i in range(2)}
        return obs, infos

    def step(
        self, actions: dict[int, Any]
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, float],
        dict[int, bool],
        dict[int, bool],
        dict[int, dict],
    ]:
        assert self._game is not None, "Call reset() before step()"

        # Save previous scores for reward
        prev = [p.score for p in self._game.players]

        # Convert actions to game commands
        game_actions = [
            self._decode_actions(0, actions[0]),
            self._decode_actions(1, actions[1]),
        ]

        # Step game
        self._game.update(game_actions)

        # Compute rewards
        rewards = {}
        done = self._game.is_game_over()
        for i in range(2):
            opp = 1 - i
            score_delta = (self._game.players[i].score - prev[i]) - (
                self._game.players[opp].score - prev[opp]
            )
            r = score_delta / 20.0

            if done:
                winner = self._game.get_winner_index()
                if winner == i:
                    r += 1.0
                elif winner == 1 - i:
                    r -= 1.0

            rewards[i] = r

        self._prev_scores = [p.score for p in self._game.players]

        obs = {i: self._build_obs(i) for i in range(2)}
        terminated = {0: done, 1: done}
        truncated = {0: False, 1: False}
        infos = {
            i: {
                "score": self._game.players[i].score,
                "opp_score": self._game.players[1 - i].score,
            }
            for i in range(2)
        }
        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        self._game = None

    def _build_obs(self, player_id: int) -> np.ndarray:
        """Build observation vector from player's perspective."""
        obs = np.zeros(self._obs_size, dtype=np.float32)
        game = self._game
        opp_id = 1 - player_id

        offset = 0

        # Own ships (3 × 9 = 27)
        for ship in game.players[player_id].ships:
            px, py = ship.pos_game
            vx, vy = ship.vel_game
            obs[offset + 0] = px / (GAME_WIDTH / 2) - 1.0  # [-1, 1]
            obs[offset + 1] = py / (GAME_HEIGHT / 2) - 1.0
            obs[offset + 2] = np.clip(vx / MAX_VELOCITY, -1.0, 1.0)
            obs[offset + 3] = np.clip(vy / MAX_VELOCITY, -1.0, 1.0)
            if game.energy_enabled:
                obs[offset + 4] = ship.energy / 50.0 - 1.0  # [0,100] → [-1, 1]
            if game.upgrades_enabled:
                obs[offset + 5] = ship.upgrades["max_speed"] / 5.0
                obs[offset + 6] = ship.upgrades["max_accel"] / 5.0
                obs[offset + 7] = ship.upgrades["push_force"] / 5.0
                obs[offset + 8] = ship.upgrades["energy_efficiency"] / 5.0
            offset += SHIP_FEATURES

        # Opponent ships (3 × 9 = 27)
        for ship in game.players[opp_id].ships:
            px, py = ship.pos_game
            vx, vy = ship.vel_game
            obs[offset + 0] = px / (GAME_WIDTH / 2) - 1.0
            obs[offset + 1] = py / (GAME_HEIGHT / 2) - 1.0
            obs[offset + 2] = np.clip(vx / MAX_VELOCITY, -1.0, 1.0)
            obs[offset + 3] = np.clip(vy / MAX_VELOCITY, -1.0, 1.0)
            if game.energy_enabled:
                obs[offset + 4] = ship.energy / 50.0 - 1.0
            if game.upgrades_enabled:
                obs[offset + 5] = ship.upgrades["max_speed"] / 5.0
                obs[offset + 6] = ship.upgrades["max_accel"] / 5.0
                obs[offset + 7] = ship.upgrades["push_force"] / 5.0
                obs[offset + 8] = ship.upgrades["energy_efficiency"] / 5.0
            offset += SHIP_FEATURES

        # Asteroids (max_asteroids × 8)
        for idx, asteroid in enumerate(game.asteroids[: self._max_asteroids]):
            base = offset + idx * ASTEROID_FEATURES
            px, py = asteroid.pos_game
            vx, vy = asteroid.vel_game
            obs[base + 0] = px / (GAME_WIDTH / 2) - 1.0
            obs[base + 1] = py / (GAME_HEIGHT / 2) - 1.0
            obs[base + 2] = np.clip(vx / MAX_VELOCITY, -1.0, 1.0)
            obs[base + 3] = np.clip(vy / MAX_VELOCITY, -1.0, 1.0)
            # Size one-hot
            for s_idx, s_name in enumerate(SIZE_NAMES):
                obs[base + 4 + s_idx] = 1.0 if asteroid.size == s_name else 0.0
            obs[base + 7] = 1.0  # exists flag
        offset += self._max_asteroids * ASTEROID_FEATURES

        # Global features (4)
        obs[offset + 0] = np.clip(
            game.players[player_id].score / 100.0, -1.0, 1.0
        )
        obs[offset + 1] = np.clip(
            game.players[opp_id].score / 100.0, -1.0, 1.0
        )
        obs[offset + 2] = game.tick / game.max_ticks * 2.0 - 1.0
        obs[offset + 3] = 1.0 if player_id == 0 else -1.0  # base side

        return obs

    def _decode_actions(self, player_id: int, action: Any) -> dict[str, Any]:
        """Convert Dict action → game command format."""
        accel = action["accel"]
        commands = []
        for i in range(MAX_SHIPS):
            ax = float(accel[i * 2]) * MAX_ACCELERATION
            ay = float(accel[i * 2 + 1]) * MAX_ACCELERATION
            push = bool(action[f"push_{i}"])
            commands.append(
                {
                    "ship_id": i,
                    "acceleration": {"x": ax, "y": ay},
                    "push": push,
                }
            )
        return {"commands": commands}
