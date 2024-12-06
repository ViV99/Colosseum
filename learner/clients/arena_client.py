import requests
from typing import Optional


class ArenaClient:
    def __init__(self, url: str):
        self._url = url

    def get_actor(self, learner_id: str, player_id: str):
        response = requests.get(
            self._url + "/api/environments", params={"learner_id": learner_id, "player_id": player_id}
        )
        return response.json()["url"]


_arena_client: Optional[ArenaClient] = None


def init_arena_client(url: str):
    global _arena_client
    if _arena_client is None:
        _arena_client = ArenaClient(url)


def get_arena_client() -> ArenaClient:
    if _arena_client is None:
        raise ValueError("Arena client is not initialized")

    return _arena_client
