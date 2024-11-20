from abc import ABC, abstractmethod
from uuid import uuid4


class IModel(ABC):
    def __init__(self):
        self.id = uuid4()

    @abstractmethod
    def start_game(self):
        pass

    @abstractmethod
    def end_game(self):
        pass

    @abstractmethod
    def update_game_state(self):
        pass

    @abstractmethod
    def turn(self):
        pass

    @abstractmethod
    def learn(self):
        pass
