from enum import Enum
from typing import Optional, Callable, Self

import torch
from tensordict import TensorDict


class ReplayEndReason(Enum):
    ORDINARY = "ORDINARY"
    WRONG_MOVE = "WRONG_MOVE"


class TrainReplay:
    def __init__(
        self,
        states: torch.Tensor,
        logprobs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        state_values: torch.Tensor,
    ):
        self.states = states
        self.logprobs = logprobs
        self.actions = actions
        self.rewards = rewards
        self.state_values = state_values

    @staticmethod
    def from_tensordict(replay_dict: TensorDict) -> Self:
        replay = TrainReplay(**replay_dict)
        return replay


class InferenceReplay:
    def __init__(
        self,
        id: str,
        calc_rewards: Callable[[list[dict], ReplayEndReason, Optional[dict[str, float]]], list[float]]
    ):
        self.id = id
        self._calc_rewards = calc_rewards
        self.end_reason = None
        self.state_dicts = list()
        self.states = list()
        self.logprobs = list()
        self.actions = list()
        self.rewards = list()
        self.state_values = list()
        self.final_scores = dict()

    def update_state(
        self,
        state_dict: dict,
        state_tensor: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        state_value: torch.Tensor
    ):
        self.state_dicts.append(state_dict)
        self.states.append(state_tensor)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.state_values.append(state_value)

    def finalize(
        self, state_dict: dict, end_reason: ReplayEndReason, scores: Optional[dict[str, float]] = None
    ) -> TensorDict:
        self.end_reason = end_reason.value
        self.state_dicts.append(state_dict)
        self.rewards = self._calc_rewards(self.state_dicts, end_reason, scores)
        self.final_scores = scores
        final_dict = TensorDict({
            "states": torch.stack(self.states),
            "logprobs": torch.stack(self.logprobs),
            "actions": torch.stack(self.actions),
            "rewards": torch.tensor(self.rewards),
            "state_values": torch.stack(self.state_values),
        }).share_memory_()
        return final_dict
