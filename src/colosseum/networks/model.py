"""``PolicyModel``: the model protocol of workers (``step``), learners (``unroll``) and eval (SP2 block 3).

- ``step(obs, state, action_mask)`` -> ``PolicyStep(dist, state)``: the policy path only. Workers
  and eval never compute values.
- ``unroll(obs, state0, reset_after, action_mask, global_state, with_value)`` ->
  ``UnrollOutput(dist, value)``: the learner path over chunk slots ``[S, B, ...]``, time-major
  flattened ``[S*B]`` (index ``s*B + b``). ``reset_after[s, b]`` resets the state AFTER slot s.
  ``with_value=False`` (BC, kickstart teacher) returns ``value=None`` and needs no global state.
- Contract: ``unroll`` gives the same policy distributions as consecutive ``step`` calls with
  the same resets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from colosseum.core.tree import Tree, tree_get
from colosseum.networks.dist import Distribution
from colosseum.networks.normalization import NormalizeObs
from colosseum.networks.state import State, batch_size_of, tree_leaves, where_done


class PolicyStep(NamedTuple):
    dist: Distribution             # batch [B]
    state: State


class UnrollOutput(NamedTuple):
    dist: Distribution             # batch [S*B], time-major (index s*B + b)
    value: Tensor | None           # [S*B]; None when with_value=False


class ActOutput(NamedTuple):
    actions: Tree                  # torch, batch [B]
    log_probs: Tensor              # [B]
    unit_log_probs: Tensor         # [B, K]
    state: State


class PolicyModel(nn.Module, ABC):
    """Actor-critic with an optional recurrent/memory state (an opaque pytree, batch first)."""

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        """State at the start of an episode. Stateless models return None."""
        return None

    @abstractmethod
    def step(self, obs: Tree, state: State, action_mask: Tree | None = None) -> PolicyStep:
        """One decision for a batch: obs leaves ``[B, ...]``; the mask (if any) is applied to the dist."""

    @abstractmethod
    def unroll(self, obs: Tree, state0: State, reset_after: Tensor, action_mask: Tree | None = None,
               global_state: Tree | None = None, with_value: bool = True) -> UnrollOutput:
        """Slots ``[S, B, ...]``; ``reset_after [S, B]`` bool; masks ``[S, B, ...]``."""

    def _check_unroll_args(self, num_slots: int, state0: State) -> None:
        if num_slots < 1:
            raise ValueError(f"unroll needs at least one slot, got S={num_slots}")
        if state0 is None and self.is_stateful:
            raise ValueError(f"unroll: state0 is None but {type(self).__name__} is stateful; pass "
                             f"initial_state(B) or the stored state of the sequence start")

    def reset_state(self, state: State, done: Tensor) -> State:
        """Replace the rows of ``state`` where ``done`` ([B] bool) with initial-state rows."""
        batch = batch_size_of(state)
        if batch is None:
            return state
        device = tree_leaves(state)[0].device
        return where_done(done, self.initial_state(batch, device), state)

    def value_parameters(self) -> list[nn.Parameter]:
        """Parameters only the value path uses (critic warm-up, ``init.critic_warmup_steps``): the value head and
        e.g. a critic encoder, never a parameter the policy uses. Models that support the warm-up override it."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement value_parameters(); return the value head's parameters "
            f"(and a critic encoder's) to use init.critic_warmup_steps"
        )

    @torch.no_grad()
    def update_normalizers(self, obs: Tree | None, global_state: Tree | None = None) -> None:
        """Update every ``NormalizeObs`` submodule from the leaf at its path of its source tree.

        Called once per train step with all fresh observations (leaves ``[N, ...]``). A normalizer
        is skipped when its source tree is None: ``obs=None`` updates only the global-state
        normalizers (the value path; APPO's critic warm-up).
        """
        for module in self.modules():
            if isinstance(module, NormalizeObs):
                source = obs if module.source == "obs" else global_state
                if source is not None:
                    module.update(tree_get(source, module.path))

    @property
    def is_stateful(self) -> bool:
        return self.initial_state(1) is not None


@torch.no_grad()
def act(model: PolicyModel, obs: Tree, state: State, action_mask: Tree | None = None,
        deterministic: bool = False) -> ActOutput:
    """Inference helper: step the model, then sample (or take the mode) and score the action."""
    out = model.step(obs, state, action_mask)
    actions = out.dist.mode() if deterministic else out.dist.sample()
    unit_log_probs = out.dist.unit_log_prob(actions)
    return ActOutput(actions=actions, log_probs=out.dist.log_prob(actions), unit_log_probs=unit_log_probs,
                     state=out.state)
