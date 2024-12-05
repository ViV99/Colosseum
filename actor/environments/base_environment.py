from abc import ABC, abstractmethod
from typing import Any


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
    def turn(self, player_id: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def get_state(self, player_id: str) -> Any:
        pass

    @abstractmethod
    def get_scores(self) -> dict[str, float]:
        pass

    @abstractmethod
    def is_ended(self) -> bool:
        pass
