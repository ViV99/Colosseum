import uuid

from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
from random import randint
from typing import Any

from .base_environment import IEnvironment

@dataclass(unsafe_hash=True)
class Point:
    x: int = field(default=0)
    y: int = field(default=0)
    z: int = field(default=0)

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.z]


@dataclass
class Snake:
    body: list[Point]
    direction: Point = field(default_factory=Point)
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    def next_point(self) -> Point:
        new_point = replace(self.body[-1])
        new_point.x += self.direction.x
        new_point.y += self.direction.y
        new_point.z += self.direction.z
        return new_point

    def move(self, remove_tail: bool = True):
        new_point = self.next_point()
        self.body.append(new_point)

        if remove_tail:
            self.body.pop(0)

    def speed(self) -> int:
        return abs(self.direction.x) + abs(self.direction.y) + abs(self.direction.z)

    def __hash__(self):
        return hash(self.id)


class TangerineType(Enum):
    usual = "USUAL"
    golden = "GOLDEN"
    strange = "STRANGE"


@dataclass(unsafe_hash=True)
class Tangerine:
    point: Point
    type: TangerineType
    score: int

    def get_visible_score(self) -> int:
        if self.type.value == TangerineType.strange:
            return 0
        return self.score


class Snake3DEnvironment(IEnvironment):
    _SNAKES_COUNT_FOR_PLAYER = 3
    _MIN_MAP_SIZE = 20
    _MAX_MAP_SIZE = 500
    _MAX_OBSTACLES_COUNT = 20
    _MAX_TANGERINES_COUNT = 20
    _PLAYERS_COUNT = 1
    _TICS_COUNT = 1000
    _SNAKE_REVIVE_DELAY_TICS = 10

    def __init__(self, id: str, players_ids: list[str]):
        super().__init__(id, players_ids)

        self.occupied_points = set()
        self.map_size = Point(
            randint(self._MIN_MAP_SIZE, self._MAX_MAP_SIZE),
            randint(self._MIN_MAP_SIZE, self._MAX_MAP_SIZE),
            randint(self._MIN_MAP_SIZE, self._MAX_MAP_SIZE)
        )

        self.locked = False
        self.tic = 0

        self.obstacles = {self._generate_point() for _ in range(randint(0, self._MAX_OBSTACLES_COUNT))}
        self.tangerines = {self._generate_tangerine() for _ in range(randint(0, self._MAX_TANGERINES_COUNT))}
        self.players_snakes = {player: [self._generate_snake() for _ in range(self._SNAKES_COUNT_FOR_PLAYER)] for player in players_ids}
        self.scores = {player: 0.0 for player in players_ids}

        self.players_chosen_directions = dict()
        self.snake_death_tic = dict()

    def _turn(self, player_id: str, snakes: list[dict]) -> None:
        self.players_chosen_directions[player_id] = {
            snake["id"]: Point(snake["direction"][0], snake["direction"][1], snake["direction"][2]) for snake in snakes
        }
        if len(self.players_chosen_directions) == self._PLAYERS_COUNT:
            self.locked = True

            for player, snakes in self.players_snakes.items():
                for snake in snakes:
                    snake.direction = self.players_chosen_directions[player][str(snake.id)]

            self._process_tic()
            self.tic += 1
            self.players_chosen_directions.clear()
            self._revive_snakes()

            self.locked = False

    def is_turn_available(self, player_id: str) -> bool:
        return not self.locked and player_id not in self.players_chosen_directions

    def _get_state(self, player_id: str) -> Any:
        obstacles = set()
        tangerines = set()

        for snake in self.players_snakes[player_id]:
            obstacles |= self._get_visible_obstacles(snake)
            tangerines |= self._get_visible_tangerines(snake)

        return {
            "mapSize": [self.map_size.x, self.map_size.y, self.map_size.z],
            "name": self.id,
            "points": self.scores[player_id],
            "fences": [obstacle.to_list() for obstacle in obstacles],
            "snakes": [
                {
                    "id": snake.id,
                    "direction": self.players_chosen_directions.get(player_id).get(snake.id).to_list() if player_id in self.players_chosen_directions else [0, 0, 0],
                    "oldDirection": snake.direction.to_list(),
                    "geometry": [point.to_list() for point in reversed(snake.body)],
                    "deathCount": 0,  # just to keep format
                    "status": "dead" if snake.id in self.snake_death_tic else "alive",
                    "reviveRemainMs": self._SNAKE_REVIVE_DELAY_TICS - (self.tic - self.snake_death_tic[snake.id]) if snake.id in self.snake_death_tic else 0,
                } for snake in self.players_snakes[player_id]
            ],
            "enemies": [
                {
                    "geometry": [point.to_list() for point in reversed(snake.body)],
                    "status": "dead" if snake.id in self.snake_death_tic else "alive",
                    "kills": 0,  # just to keep format
                } for snake in self._get_visible_snakes(player_id)
            ],
            "food": [
                {
                    "c": tangerine.point.to_list(), "points": tangerine.get_visible_score()
                } for tangerine in tangerines
            ],
            "specialFood": {
                "golden": [
                    tangerine.point.to_list() for tangerine in tangerines if tangerine.type.value == TangerineType.golden
                ],
                "suspicious": [
                    tangerine.point.to_list() for tangerine in tangerines if tangerine.type.value == TangerineType.strange
                ]
            },
            "turn": self.tic,
            "reviveTimeoutSec": self._SNAKE_REVIVE_DELAY_TICS,
            "tickRemainMs": 0,  # just to keep format
            "errors": []  # just to keep format
        }

    def get_scores(self) -> dict[str, float]:
        return self.scores

    def is_ended(self) -> bool:
        return self.tic > self._TICS_COUNT

    @classmethod
    def get_players_count(cls) -> int:
        return cls._PLAYERS_COUNT

    def _get_visible_obstacles(self, snake: Snake) -> set[Point]:
        return {obstacle for obstacle in self.obstacles if self._is_point_visible(snake.body[-1], obstacle)}

    def _get_visible_tangerines(self, snake: Snake) -> set[Tangerine]:
        return {tangerine for tangerine in self.tangerines if self._is_point_visible(snake.body[-1], tangerine.point)}

    def _get_visible_snakes(self, player_id: str) -> list[Snake]:
        snakes = []
        for player, snakes in self.players_snakes.items():
            if player == player_id:
                continue
            for enemy_snake in snakes:
                body = []
                for point in reversed(enemy_snake.body):
                    is_point_visible = False
                    for player_snake in self.players_snakes[player_id]:
                        if self._is_point_visible(player_snake.body[-1], point):
                            is_point_visible = True
                    if is_point_visible:
                        body.append(point)
                if len(body) != 0:
                    snakes.append(Snake(body, enemy_snake.direction, enemy_snake.id))

        return snakes

    @staticmethod
    def _is_point_visible(viewer: Point, point_to_view: Point) -> bool:
        return (
            abs(viewer.x // 30 - point_to_view.x // 30) <= 1 and
            abs(viewer.y // 30 - point_to_view.y // 30) <= 1 and
            abs(viewer.z // 30 - point_to_view.z // 30) <= 1
        )

    def _generate_point(self) -> Point:
        while True:
            point = Point(randint(0, self.map_size.x), randint(0, self.map_size.y), randint(0, self.map_size.z))
            if point not in self.occupied_points:
                self.occupied_points.add(point)
                return point

    def _generate_tangerine(self) -> Tangerine:
        point = self._generate_point()
        tangerine_type = TangerineType.usual

        type_int = randint(0, 2)
        if type_int == 1:
            tangerine_type = TangerineType.golden
        elif type_int == 2:
            tangerine_type = TangerineType.strange

        tic_coef = 10
        tic_score = (self.tic * 10 // self._TICS_COUNT + 1) * tic_coef
        map_score = abs(point.x - self.map_size.x // 2) + abs(point.y - self.map_size.y // 2) + abs(point.z - self.map_size.z // 2) // 10

        tangerine_score = tic_score * map_score
        if tangerine_type.value == TangerineType.golden:
            tangerine_score *= 10
        elif tangerine_type.value == TangerineType.strange:
            if randint(0, 1) == 0:
                tangerine_type += randint(2, 5)
            else:
                tangerine_type -= randint(2, 5)

        return Tangerine(point, tangerine_type, tangerine_score)

    def _generate_snake(self) -> Snake:
        return Snake([self._generate_point()])

    def _process_tic(self):
        players_by_tangerine_pos = {tangerine.point: set() for tangerine in self.tangerines}
        score_by_tangerine_pos = {tangerine.point: tangerine.score for tangerine in self.tangerines}

        # move snakes
        for player, snakes in self.players_snakes.items():
            for snake in snakes:
                next_point = snake.next_point()
                if next_point in players_by_tangerine_pos:
                    players_by_tangerine_pos[next_point].add(player)
                else:
                    self.occupied_points.remove(snake.body[0])

                snake.move(next_point not in players_by_tangerine_pos)

        # recalculate scores
        eaten_tangerines_count = 0
        for tangerine_pos, players in players_by_tangerine_pos.items():
            if len(players) > 0:
                eaten_tangerines_count += 1
                for player in players:
                    self.scores[player] += score_by_tangerine_pos[tangerine_pos] / len(players)

        # process collisions
        snakes_heads_by_pos: dict[Point, set[Snake]] = defaultdict(set)
        for snakes in self.players_snakes.values():
            for snake in snakes:
                snakes_heads_by_pos[snake.body[-1]].add(snake)

        for pos, snakes in snakes_heads_by_pos.items():
            # collision with other snakes heads
            if len(snakes) > 1:
                self._kill_snakes(snakes)

            # collision with snake body
            for other_snakes in self.players_snakes.values():
                for other_snake in other_snakes:
                    for other_snake_body_point in other_snake.body[:-1]:
                        if pos == other_snake_body_point:
                            self._kill_snakes(snakes)
                            break

            # collision with obstacle
            if pos in self.obstacles:
                self._kill_snakes(snakes)
                self.obstacles.remove(pos)

            # collision with map end
            if min(pos.x, pos.y, pos.z) < 0 or pos.x > self.map_size.x or pos.y > self.map_size.y or pos.z > self.map_size.z:
                self._kill_snakes(snakes)

        # spawn new tangerines
        for _ in range(eaten_tangerines_count):
            self.tangerines.add(self._generate_tangerine())

    def _kill_snakes(self, snakes: set[Snake]):
        for snake in snakes:
            self.snake_death_tic[snake.id] = self.tic
            for body_point in snake.body:
                if body_point in self.occupied_points:
                    self.occupied_points.remove(body_point)

    def _revive_snakes(self):
        revived_snakes_ids = {dead_snake for dead_snake, death_tic in self.snake_death_tic.items() if death_tic + self._SNAKE_REVIVE_DELAY_TICS == self.tic}
        for snakes in self.players_snakes.values():
            for snake in snakes:
                if snake.id in revived_snakes_ids:
                    snake.direction = Point(0, 0, 0)
                    snake.body = [self._generate_point()]

        self.snake_death_tic = {
            dead_snake: death_tic for dead_snake, death_tic in self.snake_death_tic.items() if
            death_tic + self._SNAKE_REVIVE_DELAY_TICS > self.tic
        }
