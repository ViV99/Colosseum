import requests


class ActorClient:
    def __init__(self, url: str):
        self._url = url

    def create_environment(self, player_ids: list[str]):
        response = requests.post(
            self._url + "/api/environments/create", json={
                "player_ids": player_ids,
            }
        )
        return response.json()["environment_id"]

    def ping(self):
        requests.post(self._url + "/api/ping")
