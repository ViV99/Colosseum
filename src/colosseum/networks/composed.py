"""ComposedModel: the default PolicyModel built from encoder, core and heads."""

from __future__ import annotations

from torch import Tensor

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.cores import Core
from colosseum.networks.model import PolicyModel, StepOutput, UnrollOutput
from colosseum.networks.state import State


class ComposedModel(PolicyModel):
    """``encoder(obs) -> core -> policy head / value head``.

    Submodules are named ``encoder``, ``core``, ``policy`` and ``value``
    (``NoCore`` has no parameters, so a stateless model's ``state_dict`` only
    holds ``encoder.*``, ``policy.*`` and ``value.*`` keys).
    """

    def __init__(self, encoder: BaseEncoder, core: Core, policy_head: BasePolicy, value_head: BaseValue) -> None:
        super().__init__()
        self.encoder = encoder
        self.core = core
        self.policy = policy_head
        self.value = value_head

    def initial_state(self, batch_size: int, device="cpu") -> State:
        return self.core.initial_state(batch_size, device)

    def step(self, obs: Tensor, state: State, action_mask: Tensor | None = None) -> StepOutput:
        features, next_state = self.core.step(self.encoder(obs), state)
        dist = self.policy(features)
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        return StepOutput(dist=dist, value=self.value(features), state=next_state)

    def unroll(self, obs: Tensor, state0: State, dones: Tensor,
               action_mask: Tensor | None = None) -> UnrollOutput:
        T, B = obs.shape[0], obs.shape[1]
        self._check_unroll_args(T, state0)
        latent = self.encoder(obs.reshape(T * B, *obs.shape[2:])).reshape(T, B, -1)
        features = self.core.unroll(latent, state0, dones).reshape(T * B, -1)
        dist = self.policy(features)
        if action_mask is not None:
            dist = dist.apply_mask(action_mask.reshape(T * B, *action_mask.shape[2:]))
        return UnrollOutput(dist=dist, value=self.value(features))

    def reset_state(self, state: State, done: Tensor) -> State:
        return self.core.reset_state(state, done)
