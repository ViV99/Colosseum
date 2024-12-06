from abc import ABC, abstractmethod
from typing import Any

from actor.error import ActorError


class IEnvironment(ABC):
    def __init__(self, id: str, players_ids: list[str]):
        if len(set(players_ids)) != len(players_ids):
            raise ValueError("All players ids should be unique")

        self.id = id
        self._players_ids = players_ids

    @property
    def players(self) -> list[str]:
        return self._players_ids

    @classmethod
    @abstractmethod
    def get_players_count(cls) -> int:
        pass

    @abstractmethod
    def is_turn_available(self, player_id: str) -> bool:
        pass

    def turn(self, player_id: str, *args, **kwargs) -> None:
        if not self.is_turn_available(player_id):
            raise ActorError(type="IT_IS_NOT_YOUR_TURN", code=400)
        if self.is_ended():
            raise ActorError(type="GAME_OVER", code=400)
        self._turn(player_id, *args, **kwargs)

    @abstractmethod
    def _turn(self, player_id: str, *args, **kwargs) -> None:
        pass

    def get_state(self, player_id: str) -> Any:
        if not self.is_turn_available(player_id) and not self.is_ended():
            raise ActorError(type="IT_IS_NOT_YOUR_TURN", code=400)
        return self._get_state(player_id)

    @abstractmethod
    def _get_state(self, player_id) -> Any:
        pass

    @abstractmethod
    def get_scores(self) -> dict[str, float]:
        pass

    @abstractmethod
    def is_ended(self) -> bool:
        pass
