from uuid import uuid4
from pathlib import Path
from abc import ABC, abstractmethod

import torch
from typing import NamedTuple

from learner.replay import InferenceReplay, TrainReplay, ReplayEndReason


class ActResult(NamedTuple):
    action: torch.Tensor
    logprob: torch.Tensor
    state_value: torch.Tensor


class IModel(ABC):
    def __init__(self):
        self.id = uuid4()

    @abstractmethod
    def act(self, replay: InferenceReplay) -> ActResult:
        pass

    @abstractmethod
    def update(self, replays: list[TrainReplay]):
        pass

    @abstractmethod
    def state_to_tensor(self, state: dict):
        pass

    @abstractmethod
    def action_to_dict(self, action: torch.Tensor) -> dict:
        pass

    @abstractmethod
    def calc_rewards(
        self, states: list[dict], end_reason: ReplayEndReason, final_scores: dict[str, float]
    ) -> list[float]:
        pass

    @abstractmethod
    def save(self, checkpoint_dir: Path):
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, ...]:
        pass

    @abstractmethod
    def load(self, checkpoint_dir: Path):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, ...]):
        pass

    @abstractmethod
    def share_memory(self):
        pass

