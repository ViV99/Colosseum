import time
import uuid

from actor.arena_client import get_arena_client
from actor.error import ActorError
from actor.environments import IEnvironment


class Actor:
    def __init__(self, environment_type: type[IEnvironment], player_idleness_timeout: int, max_environments: int):
        self._environment_type = environment_type
        self._player_idleness_timeout = player_idleness_timeout
        self._max_environments = max_environments
        self._running_environments = 0

        self._environment_by_player: dict[str, IEnvironment] = dict()
        self._environment_to_id: dict[str, IEnvironment] = dict()

        self._player_last_request_time = dict()
        self._unimportant_players = set()

    @property
    def max_environments(self):
        return self._max_environments

    @property
    def max_environment_players(self):
        return self._environment_type.get_players_count()

    def create_environment(self, player_ids: list[str]):
        if len(player_ids) != self.max_environment_players:
            raise ActorError(code=400, type="INVALID_PLAYERS_NUMBER")
        if self._running_environments >= self._max_environments:
            raise ActorError(code=400, type="ENVIRONMENTS_LIMIT_REACHED")

        environment = self._environment_type(str(uuid.uuid4()), player_ids)
        for player in player_ids:
            self._environment_by_player[player] = environment

        self._running_environments += 1
        return environment.id

    def get_state(self, player_id: str) -> dict:
        if player_id not in self._environment_by_player:
            raise ActorError(code=400, type="UNKNOWN_PLAYER")

        environment = self._environment_by_player[player_id]
        self._process_environment(environment, player_id)
        if self._is_environment_ended(environment):
            self._unimportant_players.add(player_id)

        return {
            "status": self._get_environment_status(environment),
            "scores": environment.get_scores(),
            "state": environment.get_state(player_id),
        }

    def turn(self, player_id: str, *args, **kwargs):
        if player_id not in self._environment_by_player:
            raise ActorError(code=400, type="UNKNOWN_PLAYER")

        environment = self._environment_by_player[player_id]
        self._process_environment(environment, player_id)
        environment.turn(player_id, *args, **kwargs)

    def _get_environment_status(self, environment: IEnvironment) -> str:
        if environment.is_ended():
            return "ENDED_BY_ENVIRONMENT"
        elif self._is_environment_ended(environment):
            return "ENDED_BY_PLAYER_TIMEOUT"
        else:
            return "ALIVE"

    def _process_environment(self, environment: IEnvironment, player_id: str):
        self._player_last_request_time[player_id] = time.time()

        if self._is_environment_ended(environment):
            self._move_players_to_unimportant(environment)
        if self._is_environment_can_be_deleted(environment):
            self._delete_environment(environment)

    def _is_environment_ended(self, environment: IEnvironment) -> bool:
        if environment.is_ended():
            return True
        for player_id in environment.players:
            if player_id in self._unimportant_players or \
                    time.time() - self._player_last_request_time[player_id] > self._player_idleness_timeout:
                return True
        return False

    def _is_environment_can_be_deleted(self, environment: IEnvironment) -> bool:
        if not self._is_environment_ended(environment):
            return False
        for player_id in environment.players:
            if player_id not in self._unimportant_players:
                return False
        return True

    def _move_players_to_unimportant(self, environment: IEnvironment):
        for player in environment.players:
            if time.time() - self._player_last_request_time[player] > self._player_idleness_timeout:
                self._unimportant_players.add(player)

    def _delete_environment(self, environment: IEnvironment):
        for player in environment.players:
            if player in self._unimportant_players:
                self._unimportant_players.remove(player)
            self._player_last_request_time.pop(player)
            self._environment_by_player.pop(player)
        self._running_environments -= 1
        get_arena_client().delete_environment(environment.id)
