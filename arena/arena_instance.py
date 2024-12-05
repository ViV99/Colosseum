from dataclasses import dataclass
from collections import defaultdict

from arena.actor_client import ActorClient
from arena.error import ArenaError


@dataclass
class Player:
    id: str
    learner_id: str


@dataclass
class Environment:
    id: str
    player_ids: list[str]


class Actor:
    def __init__(
        self,
        id: str,
        url: str,
        max_environments: int,
        max_environment_players: int,
    ):
        self.id = id
        self.url = url
        self.max_environments = max_environments
        self.active_environments: dict[str, Environment] = dict()
        self.max_environment_players = max_environment_players
        self.client = ActorClient(url)

    def create_environment(self, player_ids: list[str]):
        environment_id = self.client.create_environment(player_ids)
        environment = Environment(environment_id, player_ids)
        self.active_environments[environment_id] = environment

    @property
    def sorting_key(self) -> tuple:
        return (
            -(self.max_environments - len(self.active_environments)),
        )


class Arena:
    def __init__(self):
        self._actors: dict[str, Actor] = dict()
        self._waiting_players: set[Player] = set()
        self._actor_by_player_id: dict[str, Actor] = dict()

    def register_actor(self, actor_id: str, actor_url: str, max_environments: int, max_environment_players: int):
        self._actors[actor_id] = Actor(actor_id, actor_url, max_environments, max_environment_players)

    def delete_environment(self, actor_id: str, environment_id: str):
        for player_id in self._actors[actor_id].active_environments[environment_id].player_ids:
            self._actor_by_player_id.pop(player_id)
        self._actors[actor_id].active_environments.pop(environment_id)

    def get_environment(self, learner_id: str, player_id: str) -> str:
        if player_id in self._waiting_players:
            raise ArenaError(code=400, type="PLAYER_ALREADY_WAITING")
        if player_id in self._actor_by_player_id:
            return self._actor_by_player_id[player_id].url

        player = Player(player_id, learner_id)
        self._waiting_players.add(player)

        actors_by_priority = list(sorted(self._actors.values(), key=lambda actor: actor.sorting_key))

        for actor in actors_by_priority:
            if actor.max_environment_players <= len(self._waiting_players) and \
                    len(actor.active_environments) < actor.max_environments:
                self._create_environment(actor)
                return actor.url

        raise ArenaError(code=400, type="NO_ENVIRONMENT_YET")

    def _create_environment(self, actor: Actor):
        learner_to_players = defaultdict(list)
        for player in self._waiting_players:
            learner_to_players[player.learner_id].append(player)

        learner_to_players = list(sorted(learner_to_players.items(), key=lambda p: len(p[1]), reverse=True))

        i = 0
        player_ids = []
        while len(player_ids) < actor.max_environment_players:
            learner_id, learner_players = learner_to_players[i]
            i = (i + 1) % len(learner_to_players)
            if len(learner_players) != 0:
                player = learner_players.pop()
                player_ids.append(player.id)
                self._waiting_players.remove(player)

        actor.create_environment(player_ids)

