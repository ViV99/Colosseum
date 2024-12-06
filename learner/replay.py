from enum import Enum
from typing import Optional


class ReplayEndReason(Enum):
    ORDINARY = "ORDINARY"
    WRONG_MOVE = "WRONG_MOVE"


class Replay:
    def __init__(self, id: str):
        self.id = id
        self.is_ended = False
        self.end_reason = None
        self.states = list()
        self.logprobs = list()
        self.actions = list()
        self.state_values = list()
        self.final_scores = dict()

    def update_state(self, state_tensor, action, logprob, state_value):
        self.states.append(state_tensor)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.state_values.append(state_value)

    def finalize(self, state_tensor, end_reason: ReplayEndReason, scores: Optional[dict[str, float]] = None):
        self.is_ended = True
        self.end_reason = end_reason.value
        self.states.append(state_tensor)
        self.final_scores = scores
