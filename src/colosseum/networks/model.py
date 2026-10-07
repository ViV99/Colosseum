"""PolicyModel: the stateful actor-critic protocol used by workers, learners and eval.

A model maps ``(obs, state)`` to an action distribution, a value and the next
state. ``state`` is an opaque pytree (see :mod:`colosseum.networks.state`);
stateless models (MLP, CNN) use ``None``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from colosseum.networks.distributions import Distribution
from colosseum.networks.state import State, batch_size_of, tree_leaves, where_done


class StepOutput(NamedTuple):
    dist: Distribution  # batch [B]
    value: Tensor  # [B]
    state: State


class UnrollOutput(NamedTuple):
    dist: Distribution  # batch [T*B], time-major flatten (index t*B + b)
    value: Tensor  # [T*B], time-major flatten


class ActOutput(NamedTuple):
    actions: Tensor  # [B, *action_shape] (flat action layout from ActionSpec)
    log_probs: Tensor  # [B]
    values: Tensor  # [B]
    state: State


class PolicyModel(nn.Module, ABC):
    """Actor-critic with an optional recurrent/memory state."""

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        """State at the start of an episode. Stateless models return None."""
        return None

    @abstractmethod
    def step(self, obs: Tensor, state: State, action_mask: Tensor | None = None) -> StepOutput:
        """One timestep for a batch: obs ``[B, *obs_shape]`` -> (dist [B], value [B], next state).

        ``action_mask`` (``[B, mask_size]`` bool) must be applied to the returned dist.
        """

    def unroll(
        self,
        obs: Tensor,
        state0: State,
        dones: Tensor,
        action_mask: Tensor | None = None,
    ) -> UnrollOutput:
        """Process sequences: obs ``[T, B, ...]``, dones ``[T, B]`` (``dones[t]``: the
        episode ended AFTER step t, so the state is reset before step t+1),
        mask ``[T, B, A]`` or None. Returns time-major flattened ``[T*B]`` outputs.

        Default: a Python loop over :meth:`step` with :meth:`reset_state` after
        every step, which is correct for any model. Stateless models are
        evaluated in one batched ``step`` (equivalent, faster). Distributions
        are concatenated with ``Distribution.cat``.
        """
        T, B = obs.shape[0], obs.shape[1]
        if state0 is None and not self.is_stateful:
            flat_mask = None if action_mask is None else action_mask.reshape(T * B, *action_mask.shape[2:])
            out = self.step(obs.reshape(T * B, *obs.shape[2:]), None, flat_mask)
            return UnrollOutput(dist=out.dist, value=out.value)

        dists: list[Distribution] = []
        values: list[Tensor] = []
        state = state0
        for t in range(T):
            out = self.step(obs[t], state, None if action_mask is None else action_mask[t])
            dists.append(out.dist)
            values.append(out.value)
            state = self.reset_state(out.state, dones[t])
        return UnrollOutput(dist=type(dists[0]).cat(dists), value=torch.cat(values, dim=0))

    def reset_state(self, state: State, done: Tensor) -> State:
        """Replace the rows of ``state`` where ``done`` ([B] bool) with initial-state rows."""
        if state is None:
            return None
        batch = batch_size_of(state)
        device = tree_leaves(state)[0].device
        return where_done(done, self.initial_state(batch, device), state)

    def update_normalizers(self, obs: Tensor) -> None:
        """Update running observation statistics (no-op unless the model has any)."""
        return None

    @property
    def is_stateful(self) -> bool:
        return self.initial_state(1) is not None


@torch.no_grad()
def act(
    model: PolicyModel,
    obs: Tensor,
    state: State,
    action_mask: Tensor | None = None,
    deterministic: bool = False,
) -> ActOutput:
    """Inference helper: step the model, then sample (or take the mode) and score the action."""
    out = model.step(obs, state, action_mask)
    actions = out.dist.mode() if deterministic else out.dist.sample()
    log_probs = out.dist.log_prob(actions)
    return ActOutput(actions=actions, log_probs=log_probs, values=out.value, state=out.state)
