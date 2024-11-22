import uuid
from dataclasses import dataclass


@dataclass
class Actor:
    id: uuid.UUID
    url: str
    max_environments: int
    used_environments: int


class Arena:
    def __init__(self):
        self._actors: dict[uuid.UUID, Actor] = dict()

    def register_actor(self, actor_id: uuid.UUID, actor_url: str, max_environments: int):
        self._actors[actor_id] = Actor(actor_id, actor_url, max_environments, 0)

    def add_environment(self, actor_id: uuid.UUID):
        self._actors[actor_id].used_environments += 1

    def delete_environment(self, actor_id: uuid.UUID):
        self._actors[actor_id].used_environments += 1

    def get_actor_url(self, learner_id: str, player_id: str) -> str:
        pass
