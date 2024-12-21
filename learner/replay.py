from enum import Enum
from typing import Optional, Callable

import torch
from tensordict import TensorDict


class ReplayEndReason(Enum):
    ORDINARY = "ORDINARY"
    WRONG_MOVE = "WRONG_MOVE"


class Replay:
    def __init__(self, id: str, calc_rewards: Callable[[list, ReplayEndReason], list]):
        self.id = id
        self.calc_rewards = calc_rewards
        self.is_ended = False
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
        self.is_ended = True
        self.end_reason = end_reason.value
        self.state_dicts.append(state_dict)
        self.rewards = self.calc_rewards(self.state_dicts, end_reason)
        self.final_scores = scores
        final_dict = TensorDict({
            "states": torch.stack(self.states),
            "logprobs": torch.stack(self.logprobs),
            "actions": torch.stack(self.actions),
            "rewards": torch.tensor(self.rewards),
            "state_values": torch.stack(self.state_values),
            "final_scores": self.final_scores,
            "end_reason": self.end_reason
        }).share_memory_()
        return final_dict
