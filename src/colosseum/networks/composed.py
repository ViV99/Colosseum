"""``ComposedModel``: encoder -> core -> policy head / value head, with an optional critic encoder."""

from __future__ import annotations

import torch
from torch import Tensor

from colosseum.core.tree import Tree, tree_leaves, tree_map
from colosseum.networks.base import BaseCriticEncoder, BaseEncoder, BasePolicy, BaseValue, EncoderOutput
from colosseum.networks.cores import Core
from colosseum.networks.model import PolicyModel, PolicyStep, UnrollOutput
from colosseum.networks.state import State


def _encode(encoder: BaseEncoder, obs: Tree) -> EncoderOutput:
    out = encoder(obs)
    return out if isinstance(out, EncoderOutput) else EncoderOutput(out, {})


def _flatten_time(tree: Tree, num_slots: int, batch: int) -> Tree:
    return tree_map(lambda x: x.reshape(num_slots * batch, *x.shape[2:]), tree)


class ComposedModel(PolicyModel):
    """``encoder(obs) -> core -> policy(features, aux)``; ``value(features ⊕ critic_encoder(global_state))``.

    Submodules: ``encoder``, ``core``, ``policy``, ``value`` and ``critic_encoder`` (None if unused).
    """

    def __init__(self, encoder: BaseEncoder, core: Core, policy_head: BasePolicy, value_head: BaseValue,
                 critic_encoder: BaseCriticEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.core = core
        self.policy = policy_head
        self.value = value_head
        self.critic_encoder = critic_encoder

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        return self.core.initial_state(batch_size, device)

    def step(self, obs: Tree, state: State, action_mask: Tree | None = None) -> PolicyStep:
        enc = _encode(self.encoder, obs)
        features, next_state = self.core.step(enc.latent, state)
        dist = self.policy(features, enc.aux)
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        return PolicyStep(dist=dist, state=next_state)

    def unroll(self, obs: Tree, state0: State, reset_after: Tensor, action_mask: Tree | None = None,
               global_state: Tree | None = None, with_value: bool = True) -> UnrollOutput:
        S, B = (int(d) for d in tree_leaves(obs)[0].shape[:2])
        self._check_unroll_args(S, state0)
        if with_value and self.critic_encoder is not None and global_state is None:
            raise ValueError("unroll(with_value=True) of a model with a critic_encoder needs global_state")
        enc = _encode(self.encoder, _flatten_time(obs, S, B))
        latent = enc.latent.reshape(S, B, -1)
        features = self.core.unroll(latent, state0, reset_after).reshape(S * B, -1)
        dist = self.policy(features, enc.aux)
        if action_mask is not None:
            dist = dist.apply_mask(_flatten_time(action_mask, S, B))
        if not with_value:
            return UnrollOutput(dist=dist, value=None)
        value_in = features
        if self.critic_encoder is not None:
            value_in = torch.cat([features, self.critic_encoder(_flatten_time(global_state, S, B))], dim=-1)
        return UnrollOutput(dist=dist, value=self.value(value_in))

    def reset_state(self, state: State, done: Tensor) -> State:
        return self.core.reset_state(state, done)
