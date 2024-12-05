import requests
from typing import Optional


class ArenaClient:
    def __init__(self, url: str, actor_id: str):
        self._url = url
        self._actor_id = actor_id

    def register_actor(self, max_environments: int, max_environment_players: int):
        requests.post(
            self._url + "/api/actors", json={
                "actor_id": self._actor_id,
                "max_environments": max_environments,
                "max_environment_players": max_environment_players,
            }
        )

    def delete_environment(self, environment_id: str):
        requests.delete(
            self._url + "/api/environments", json={
                "actor_id": self._actor_id, "environment_id": environment_id
            }
        )


_arena_client: Optional[ArenaClient] = None


def init_arena_client(url: str, actor_id: str):
    global _arena_client
    if _arena_client is None:
        _arena_client = ArenaClient(url, actor_id)


def get_arena_client() -> ArenaClient:
    if _arena_client is None:
        raise ValueError("Arena client is not initialized")

    return _arena_client
