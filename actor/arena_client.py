import requests
import uuid
from typing import Optional


class ArenaClient:
    def __init__(self, url: str, actor_id: uuid.UUID):
        self._url = url
        self._actor_id = actor_id

    def register_actor(self, max_environments: int):
        requests.post(
            self._url + "/api/actors", data={"id": self._actor_id, "max_environments": max_environments}
        )

    def add_environment(self):
        requests.post(self._url + "/api/environments", data={"id": self._actor_id})

    def delete_game(self):
        requests.delete(self._url + "/api/environments", data={"id": self._actor_id})


_arena_client: Optional[ArenaClient] = None

def init_arena_client(url: str, actor_id: uuid.UUID):
    global _arena_client
    _arena_client = ArenaClient(url, actor_id)


def get_arena_client() -> ArenaClient:
    if _arena_client is None:
        raise ValueError("Arena client is not initialized")

    return _arena_client
