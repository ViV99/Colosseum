from uuid import uuid4
from pathlib import Path
from abc import ABC, abstractmethod

import torch
from typing import NamedTuple

from rl import PPO
from learner.replay import Replay, ReplayEndReason
from base_model import IModel, ActResult


class Snake3d(IModel):
    def __init__(self, algorithm: PPO):
        super().__init__()
        self.algorithm = algorithm

    def act(self, states: list[torch.Tensor]) -> ActResult:
        """
        :param states: Sequence of states (N * [<state_dims>,])
        :return: Tuple of
                 chosen action ([,]),
                 action logprob ([,]),
                 state value ([,])
        """
        actions, action_logprobs, state_values = self.algorithm.select_action(states)
        return ActResult(actions[0].squeeze(), action_logprobs[0].squeeze(), state_values.squeeze())

    def update(self, replays: list[Replay]):
        pass

    def state_to_tensor(self, state: dict):
        pass

    def action_to_dict(self, action: torch.Tensor) -> dict:
        pass

    def calc_rewards(self, states: list[dict], end_reason: ReplayEndReason) -> list[float]:
        pass

    def save(self, checkpoint_dir: Path):
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(self.algorithm.state_dict(), checkpoint_dir / "algorithm.pt")

    def load(self, checkpoint_dir: Path):
        self.load_state_dict(
            torch.load(checkpoint_dir / "algorithm.pt",  map_location=lambda storage, loc: storage)
        )

    def load_state_dict(self, state_dict: dict[str, ...]):
        self.algorithm.load_state_dict(state_dict)


if __name__ == "__main__":
    print(torch.tensor([1]).shape)
    print(torch.tensor(1).shape)