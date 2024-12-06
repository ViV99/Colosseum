import requests


class ActorClient:
    def __init__(self, url: str):
        self._url = url

    def get_state(self, player_id: str):
        response = requests.get(self._url + "/api/environments/state", params={"player_id": player_id})
        response.raise_for_status()
        return response.json()

    def turn(self, turn_info: dict):
        response = requests.post(self._url + "/api/environments/turn", json={"turn_info": turn_info})
        response.raise_for_status()
