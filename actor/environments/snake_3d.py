import uuid
from dataclasses import dataclass, field
from enum import Enum
from random import randint
from typing import Any

from .base_environment import IEnvironment

@dataclass
class Point:
    x: int
    y: int
    z: int

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.z]


@dataclass
class Snake:
    body: list[Point]
    direction: Point = field(default=Point(0, 0, 0))
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    def move(self, remove_tail: bool = True):
        new_point = self.body[-1]
        new_point.x += self.direction.x
        new_point.y += self.direction.y
        new_point.z += self.direction.z
        self.body.append(new_point)

        if remove_tail:
            self.body.pop(0)


class TangerineType(Enum):
    usual = "USUAL"
    golden = "GOLDEN"
    strange = "STRANGE"


@dataclass
class Tangerine:
    point: Point
    type: TangerineType
    score: int

    def get_visible_score(self) -> int:
        if self.type.value == TangerineType.strange:
            return 0
        return self.score


class Snake3DEnvironment(IEnvironment):
    _MIN_MAP_SIZE = 20
    _MAX_MAP_SIZE = 500
    _MAX_OBSTACLES_COUNT = 20
    _MAX_TANGERINES_COUNT = 20
    _PLAYERS_COUNT = 20
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

        self.obstacles = {self._generate_point() for _ in range(randint(0, self._MAX_OBSTACLES_COUNT))}
        self.tangerines = {self._generate_tangerine() for _ in range(randint(0, self._MAX_TANGERINES_COUNT))}
        self.players_snakes = {player: [self._generate_snake() for _ in range(3)] for player in players_ids}
        self.scores = {player: 0.0 for player in players_ids}

        self.tic = 0
        self.players_chosen_directions = dict()
        self.snake_death_tic = dict()

    def _turn(self, player_id: str, snakes: list[dict]) -> None:
        self.players_chosen_directions[player_id] = {
            snake["id"]: Point(snake["direction"][0], snake["direction"][1], snake["direction"][2]) for snake in snakes
        }
        if len(self.players_chosen_directions) == self._PLAYERS_COUNT:
            self._process_tic()
            self.tic += 1
            self.players_chosen_directions.clear()

    def is_turn_available(self, player_id: str) -> bool:
        return player_id not in self.players_chosen_directions

    def _get_state(self, player_id: str) -> Any:
        obstacles = (
            self._get_visible_obstacles(self.players_snakes[player_id][0]) |
            self._get_visible_obstacles(self.players_snakes[player_id][1]) |
            self._get_visible_obstacles(self.players_snakes[player_id][2])
        )
        tangerines = (
            self._get_visible_tangerines(self.players_snakes[player_id][0]) |
            self._get_visible_tangerines(self.players_snakes[player_id][1]) |
            self._get_visible_tangerines(self.players_snakes[player_id][2])
        )

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
            "enemies": [],  # TODO
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
        return self.tic < self._TICS_COUNT

    @classmethod
    def get_players_count(cls) -> int:
        return cls._PLAYERS_COUNT

    def _get_visible_obstacles(self, snake: Snake) -> set[Point]:
        return {obstacle for obstacle in self.obstacles if self._is_point_visible(snake.body[-1], obstacle)}

    def _get_visible_tangerines(self, snake: Snake) -> set[Tangerine]:
        return {tangerine for tangerine in self.tangerines if self._is_point_visible(snake.body[-1], tangerine.point)}

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
        raise NotImplementedError
