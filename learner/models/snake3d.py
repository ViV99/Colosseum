from uuid import uuid4
from pathlib import Path
from abc import ABC, abstractmethod

import torch
from typing import NamedTuple

from rl import PPO
from learner.replay import InferenceReplay, TrainReplay, ReplayEndReason
from base_model import IModel, ActResult


class Snake3d(IModel):
    def __init__(self, algorithm: PPO):
        super().__init__()
        self.algorithm = algorithm

    def act(self, replay: InferenceReplay) -> ActResult:
        """
        :param replay: Sequence of states (N * [<state_dims>,])
        :return: Tuple of
                 chosen action ([,]),
                 action logprob ([,]),
                 state value ([,])
        """
        actions, action_logprobs, state_values = self.algorithm.select_action(replay.states)
        return ActResult(actions[0].squeeze(), action_logprobs[0].squeeze(), state_values.squeeze())

    def update(self, replays: list[TrainReplay]):
        states, actions, rewards, is_terminals, state_values, logprobs, = [], [], [], [], [], []
        for r in replays:
            states.append(r.states)
            actions.append(r.actions)
            rewards.append(r.rewards)
            is_terminal = torch.zeros_like(r.state_values, dtype=torch.int)
            is_terminal[-1] = 1
            is_terminals.append(is_terminal)
            state_values.append(r.state_values)
            logprobs.append(r.logprobs)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        is_terminals = torch.cat(is_terminals)
        state_values = torch.cat(state_values)
        logprobs = torch.cat(logprobs)
        self.algorithm.update(states, actions, rewards, is_terminals, state_values, logprobs)

    def state_to_tensor(self, state: dict):
        pass

    def action_to_dict(self, action: torch.Tensor) -> dict:
        pass

    def calc_rewards(
        self, states: list[dict], end_reason: ReplayEndReason, final_scores: dict[str, float]
    ) -> list[float]:
        pass

    def save(self, checkpoint_dir: Path):
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(self.state_dict(), checkpoint_dir / "algorithm.pt")

    def state_dict(self) -> dict[str, ...]:
        return self.algorithm.state_dict()

    def load(self, checkpoint_dir: Path):
        self.load_state_dict(
            torch.load(checkpoint_dir / "algorithm.pt",  map_location=lambda storage, loc: storage)
        )

    def load_state_dict(self, state_dict: dict[str, ...]):
        self.algorithm.load_state_dict(state_dict)

    def share_memory(self):
        self.algorithm.share_memory()

if __name__ == "__main__":
    a = torch.randn(5)
    b = torch.randn(3)
    c = torch.randn(1)
    print(torch.cat((a, b, c)).shape)