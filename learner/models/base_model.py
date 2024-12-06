from abc import ABC, abstractmethod
from uuid import uuid4

import torch
from typing_extensions import NamedTuple

from learner.replay import Replay


class ActResult(NamedTuple):
    action: torch.Tensor
    logprob: torch.Tensor
    state_value: torch.Tensor


class IModel(ABC):
    def __init__(self):
        self.id = uuid4()

    @abstractmethod
    def act(self, state_tensor: torch.Tensor) -> ActResult:
        pass

    @abstractmethod
    def update(self, replays: list[Replay]):
        pass

    @abstractmethod
    def state_to_tensor(self, state: dict):
        pass

    @abstractmethod
    def action_to_dict(self, action: torch.Tensor) -> dict:
        pass
