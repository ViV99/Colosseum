"""Space Miners Hard — game engine adapted for Colosseum RL environment.

Adapted from the competition local runner. Uses Box2D for physics simulation.
Requires: pip install Box2D
"""

import logging
import math
import random
from typing import Any

import Box2D
from Box2D import (
    b2_dynamicBody,
    b2BodyDef,
    b2CircleShape,
    b2FixtureDef,
    b2PolygonShape,
    b2Vec2,
    b2World,
)

# Game world constants
GAME_WIDTH = 1280
GAME_HEIGHT = 800

# Physics constants
MAX_ACCELERATION = 5.0
MAX_VELOCITY = 15
PPM = 10  # Pixels Per Meter
DT = 0.1  # Seconds per physics step
ACC_SCALE = (1 / PPM) / (DT**2)
PUSH_RADIUS_UNITS = 50
PUSH_FORCE_MAX = 2000.0
BASE_COLLECTION_RADIUS_UNITS = 100

# Ship and asteroid physical properties
SHIP_RADIUS_UNITS = 15
SHIP_MASS = 10
ASTEROID_RADIUS_UNITS = {"small": 5, "medium": 10, "large": 20}
ASTEROID_MASS = {"small": 15, "medium": 20, "large": 30}
ASTEROID_SCORE = {"small": 5, "medium": 10, "large": 20}

CATEGORY_SHIP = 0x0002
CATEGORY_ASTEROID = 0x0004
CATEGORY_WALL = 0x0001

PRESET_CONFIGS = {
    "Round 1": {"ship_count": 3, "energy_enabled": False, "upgrades_enabled": False},
    "Round 2": {"ship_count": 3, "energy_enabled": True, "upgrades_enabled": False},
    "Final Round": {"ship_count": 3, "energy_enabled": True, "upgrades_enabled": True},
}


def _density(mass: float, radius_units: float) -> float:
    return mass / (math.pi * (radius_units / PPM) ** 2)


class Ship:
    def __init__(self, id: int, body: Box2D.b2Body | None = None):
        self.id = id
        self.body = body
        self.position = b2Vec2(0, 0)
        self.velocity = b2Vec2(0, 0)
        self.energy = 100.0
        self.upgrades = {
            "max_speed": 0,
            "max_accel": 0,
            "push_force": 0,
            "energy_efficiency": 0,
        }

    def get_acceleration_cost(self, accel_mag: float) -> float:
        base_cost = 0.25 * (max(0, accel_mag - 1) ** 1.25)
        eff = self.upgrades["energy_efficiency"]
        return base_cost / (1 + eff * 0.10)

    def get_push_cost(self) -> float:
        return 0.25

    def get_effective_max_speed(self) -> float:
        base = MAX_VELOCITY * (1 + self.upgrades["max_speed"] * 0.10)
        factor = 0.2 + 0.8 * (self.energy / 100.0)
        return base * factor

    def get_effective_max_acceleration(self) -> float:
        base = MAX_ACCELERATION * (1 + self.upgrades["max_accel"] * 0.10)
        factor = 0.2 + 0.8 * (self.energy / 100.0)
        return base * factor

    def get_effective_push_force(self) -> float:
        return PUSH_FORCE_MAX * (1 + self.upgrades["push_force"] * 0.10)

    def regenerate_energy(self) -> None:
        self.energy = min(100.0, self.energy + 4.0)

    def consume_energy(self, amount: float) -> None:
        self.energy = max(0.0, self.energy - amount)

    @property
    def pos_game(self) -> tuple[float, float]:
        """Position in game units."""
        if self.body:
            return self.body.position.x * PPM, self.body.position.y * PPM
        return self.position.x * PPM, self.position.y * PPM

    @property
    def vel_game(self) -> tuple[float, float]:
        """Velocity in game units per tick."""
        if self.body:
            return (
                self.body.linearVelocity.x * PPM * DT,
                self.body.linearVelocity.y * PPM * DT,
            )
        return self.velocity.x * PPM * DT, self.velocity.y * PPM * DT


class Asteroid:
    def __init__(
        self, id: int, body: Box2D.b2Body | None = None, size: str | None = None
    ):
        self.id = id
        self.body = body
        self.size = size
        self.position = b2Vec2(0, 0)
        self.velocity = b2Vec2(0, 0)

    @property
    def pos_game(self) -> tuple[float, float]:
        if self.body:
            return self.body.position.x * PPM, self.body.position.y * PPM
        return self.position.x * PPM, self.position.y * PPM

    @property
    def vel_game(self) -> tuple[float, float]:
        if self.body:
            return (
                self.body.linearVelocity.x * PPM * DT,
                self.body.linearVelocity.y * PPM * DT,
            )
        return self.velocity.x * PPM * DT, self.velocity.y * PPM * DT


class Player:
    def __init__(self, id: int, base_x: float):
        self.id = id
        self.score = 0
        self.ships: list[Ship] = []
        self.base_x = base_x
        self.is_active = True
        self.last_score_change_tick = 0


class SpaceMinersGameState:
    """Game state for Space Miners Hard, adapted for RL use."""

    def __init__(
        self,
        preset: str = "Round 1",
        max_ticks: int = 1000,
        seed: int | None = None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.max_ticks = max_ticks
        self.tick = 0

        preset_config = PRESET_CONFIGS.get(preset, PRESET_CONFIGS["Round 1"])
        self.energy_enabled = preset_config["energy_enabled"]
        self.upgrades_enabled = preset_config["upgrades_enabled"]
        ship_count = preset_config["ship_count"]

        self.width = GAME_WIDTH / PPM
        self.height = GAME_HEIGHT / PPM

        if seed is not None:
            random.seed(seed)

        self.players = [Player(0, 0), Player(1, self.width * PPM)]
        self.asteroids: list[Asteroid] = []
        self._next_asteroid_id = 0
        self.initial_asteroids = random.randint(3, 20)

        self.world = b2World(gravity=(0, 0))
        self._create_walls()
        self._initialize_ships(ship_count)

        while len(self.asteroids) + 1 < self.initial_asteroids:
            self._spawn_asteroid_pair()
        if len(self.asteroids) + 1 == self.initial_asteroids:
            self._spawn_asteroid_middle()

    def _create_walls(self) -> None:
        ground_body_def = b2BodyDef()
        ground_body_def.position = (0, 0)
        ground = self.world.CreateBody(ground_body_def)

        ground_box = b2PolygonShape()
        wall_fixture = b2FixtureDef(
            shape=ground_box,
            density=0.0,
            restitution=0.5,
            categoryBits=CATEGORY_WALL,
            maskBits=CATEGORY_SHIP | CATEGORY_ASTEROID,
        )

        # Bottom
        ground_box.SetAsBox(self.width / 2, 1, b2Vec2(self.width / 2, -1), 0)
        ground.CreateFixture(wall_fixture)
        # Top
        ground_box.SetAsBox(
            self.width / 2, 1, b2Vec2(self.width / 2, self.height + 1), 0
        )
        ground.CreateFixture(wall_fixture)
        # Left
        ground_box.SetAsBox(1, self.height / 2, b2Vec2(-1, self.height / 2), 0)
        ground.CreateFixture(wall_fixture)
        # Right
        ground_box.SetAsBox(
            1, self.height / 2, b2Vec2(self.width + 1, self.height / 2), 0
        )
        ground.CreateFixture(wall_fixture)

    def _initialize_ships(self, ship_count: int) -> None:
        angle_step = math.pi / (ship_count + 1)
        for player in self.players:
            for i in range(ship_count):
                body_def = b2BodyDef()
                body_def.type = b2_dynamicBody
                body_def.bullet = True
                body_def.fixedRotation = True

                dx = self.width / 2 - player.base_x
                angle = math.pi / 2 - dx / abs(dx) * (i + 1) * angle_step
                shift_x = math.cos(angle) * BASE_COLLECTION_RADIUS_UNITS
                shift_y = math.sin(angle) * BASE_COLLECTION_RADIUS_UNITS
                body_def.position = (
                    player.base_x / PPM + shift_x / PPM,
                    self.height / 2 + shift_y / PPM,
                )
                body = self.world.CreateBody(body_def)

                shape = b2CircleShape(radius=SHIP_RADIUS_UNITS / PPM)
                fixture_def = b2FixtureDef(
                    shape=shape,
                    density=_density(SHIP_MASS, SHIP_RADIUS_UNITS),
                    friction=0.0,
                    restitution=0.5,
                    categoryBits=CATEGORY_SHIP,
                    maskBits=CATEGORY_SHIP | CATEGORY_ASTEROID | CATEGORY_WALL,
                )
                body.CreateFixture(fixture_def)
                body.userData = {"type": "ship", "player_id": player.id}
                player.ships.append(Ship(i, body))

    def update(self, actions: list[dict[str, Any] | None]) -> None:
        """Advance game state by one tick.

        actions: list of 2 player action dicts, each with "commands" key.
        Each command: {"ship_id": int, "acceleration": {"x": float, "y": float}, "push": bool}
        """
        self.tick += 1

        energy_costs: dict[int, dict[int, float]] = {}

        for i, player in enumerate(self.players):
            if not player.is_active or actions[i] is None:
                continue

            commands = actions[i]["commands"]
            energy_costs[i] = {}

            for cmd in commands:
                ship_id = cmd["ship_id"]
                ship = player.ships[ship_id]

                acceleration = b2Vec2(
                    cmd["acceleration"]["x"], cmd["acceleration"]["y"]
                )
                accel_mag = acceleration.length

                if self.energy_enabled:
                    max_accel = ship.get_effective_max_acceleration()
                else:
                    max_accel = MAX_ACCELERATION

                if accel_mag > max_accel:
                    acceleration.Normalize()
                    acceleration *= max_accel
                    accel_mag = max_accel

                total_cost = 0.0
                if self.energy_enabled:
                    if accel_mag > 0:
                        total_cost += ship.get_acceleration_cost(accel_mag)
                    if cmd.get("push", False):
                        total_cost += ship.get_push_cost()

                if not self.energy_enabled or ship.energy >= total_cost:
                    energy_costs[i][ship_id] = total_cost

                    a_world = acceleration * ACC_SCALE
                    ship.body.ApplyForceToCenter(a_world * ship.body.mass, True)

                    if cmd.get("push", False):
                        if self.energy_enabled:
                            self._apply_push(ship, ship.get_effective_push_force())
                        else:
                            self._apply_push(ship)
                else:
                    energy_costs[i][ship_id] = 0.0

        # Physics step
        self.world.Step(DT, 10, 10)

        # Cap velocities
        for body in self.world.bodies:
            if not body.userData:
                continue
            if body.userData["type"] == "ship":
                pid = body.userData["player_id"]
                ship = None
                for s in self.players[pid].ships:
                    if s.body == body:
                        ship = s
                        break
                if ship and self.energy_enabled:
                    max_speed = ship.get_effective_max_speed()
                else:
                    max_speed = MAX_VELOCITY
                speed = body.linearVelocity.length
                if speed > max_speed:
                    body.linearVelocity = body.linearVelocity * (max_speed / speed)
            elif body.userData["type"] == "asteroid":
                speed = body.linearVelocity.length
                if speed > MAX_VELOCITY:
                    body.linearVelocity = body.linearVelocity * (MAX_VELOCITY / speed)

        # Deduct energy
        if self.energy_enabled:
            for i, player in enumerate(self.players):
                if i in energy_costs:
                    for ship_id, cost in energy_costs[i].items():
                        player.ships[ship_id].consume_energy(cost)

        # Upgrades & regeneration
        if self.energy_enabled or self.upgrades_enabled:
            for i, player in enumerate(self.players):
                base_pos = b2Vec2(player.base_x / PPM, self.height / 2)

                if (
                    self.upgrades_enabled
                    and i in energy_costs
                    and actions[i] is not None
                ):
                    for cmd in actions[i].get("commands", []):
                        if "upgrade" in cmd:
                            ship = player.ships[cmd["ship_id"]]
                            upgrade_type = cmd["upgrade"]
                            if ship.body:
                                dist = (ship.body.position - base_pos).length
                                if dist <= BASE_COLLECTION_RADIUS_UNITS / PPM:
                                    level = ship.upgrades[upgrade_type]
                                    cost = 2 * (level + 1)
                                    if player.score >= cost:
                                        player.score -= cost
                                        ship.upgrades[upgrade_type] += 1
                                        player.last_score_change_tick = self.tick

                for ship in player.ships:
                    if ship.body:
                        dist = (ship.body.position - base_pos).length
                        if dist <= BASE_COLLECTION_RADIUS_UNITS / PPM:
                            if self.energy_enabled:
                                ship.regenerate_energy()

        # Collection
        self._check_base_collection()

        # Respawn asteroids
        while len(self.asteroids) < self.initial_asteroids:
            self._spawn_asteroid_pair()

    def _apply_push(
        self, ship: Ship, push_force_max: float = PUSH_FORCE_MAX
    ) -> None:
        for asteroid in self.asteroids:
            direction = asteroid.body.position - ship.body.position
            dist = max(
                0,
                direction.length
                - SHIP_RADIUS_UNITS / PPM
                - ASTEROID_RADIUS_UNITS[asteroid.size] / PPM,
            )
            if dist <= PUSH_RADIUS_UNITS / PPM:
                direction.Normalize()
                strength = push_force_max * (1 - dist * PPM / PUSH_RADIUS_UNITS)
                asteroid.body.ApplyForceToCenter(direction * strength, True)

    def _check_base_collection(self) -> None:
        for player in self.players:
            base_pos = b2Vec2(player.base_x / PPM, self.height / 2)
            for asteroid in self.asteroids[:]:
                direction = asteroid.body.position - base_pos
                if direction.length <= BASE_COLLECTION_RADIUS_UNITS / PPM:
                    old_score = player.score
                    player.score += ASTEROID_SCORE[asteroid.size]
                    if player.score != old_score:
                        player.last_score_change_tick = self.tick
                    self.world.DestroyBody(asteroid.body)
                    self.asteroids.remove(asteroid)

    def _is_position_clear(
        self, pos: tuple[float, float], radius: float, margin_units: float = 5
    ) -> bool:
        margin = margin_units / PPM
        check_pos = b2Vec2(pos[0], pos[1])
        for asteroid in self.asteroids:
            ar = ASTEROID_RADIUS_UNITS[asteroid.size] / PPM
            if (check_pos - asteroid.body.position).length < radius + ar + margin:
                return False
        for player in self.players:
            for ship in player.ships:
                sr = SHIP_RADIUS_UNITS / PPM
                if (check_pos - ship.body.position).length < radius + sr + margin:
                    return False
        return True

    def _spawn_asteroid_pair(self) -> None:
        size = random.choice(["small", "medium", "large"])
        radius = ASTEROID_RADIUS_UNITS[size] / PPM
        gap = 10
        for _ in range(150):
            pos = (
                random.uniform(gap, self.width - gap),
                random.uniform(gap, self.height - gap),
            )
            mirror = (self.width - pos[0], self.height - pos[1])
            if self._is_position_clear(pos, radius) and self._is_position_clear(
                mirror, radius
            ):
                self._spawn_asteroid(size, pos)
                self._spawn_asteroid(size, mirror)
                return

    def _spawn_asteroid_middle(self) -> None:
        size = random.choice(["small", "medium", "large"])
        radius = ASTEROID_RADIUS_UNITS[size] / PPM
        pos = (self.width / 2, self.height / 2)
        if self._is_position_clear(pos, radius):
            self._spawn_asteroid(size, pos)

    def _spawn_asteroid(self, size: str, pos: tuple[float, float]) -> None:
        body_def = b2BodyDef()
        body_def.type = b2_dynamicBody
        body_def.position = pos
        body_def.bullet = True
        body_def.fixedRotation = True
        body = self.world.CreateBody(body_def)

        shape = b2CircleShape(radius=ASTEROID_RADIUS_UNITS[size] / PPM)
        fixture_def = b2FixtureDef(
            shape=shape,
            density=_density(ASTEROID_MASS[size], ASTEROID_RADIUS_UNITS[size]),
            friction=0.0,
            restitution=0.5,
            categoryBits=CATEGORY_ASTEROID,
            maskBits=CATEGORY_SHIP | CATEGORY_ASTEROID | CATEGORY_WALL,
        )
        body.CreateFixture(fixture_def)
        body.userData = {"type": "asteroid"}
        self.asteroids.append(Asteroid(self._next_asteroid_id, body, size))
        self._next_asteroid_id += 1

    def is_game_over(self) -> bool:
        return self.tick >= self.max_ticks or all(
            not p.is_active for p in self.players
        )

    def get_winner_index(self) -> int:
        """Return winning player index, or -1 for tie."""
        p0, p1 = self.players
        if p0.score > p1.score:
            return 0
        if p1.score > p0.score:
            return 1
        if p0.last_score_change_tick < p1.last_score_change_tick:
            return 0
        if p1.last_score_change_tick < p0.last_score_change_tick:
            return 1
        return -1
