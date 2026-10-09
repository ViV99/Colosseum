"""The decider-aware distribution protocol (SP2 spec block 2, "Протокол распределения").

An action splits into K *deciders*: every unit of every ``Units`` group is one decider, and
all non-units parts together are one more (decider 0, if any). Per-decider methods return
``[B, K]``; invalid deciders (absent units, units without a valid component) give exactly 0,
selected with ``torch.where`` (never by multiplying with a mask: ``0 * NaN = NaN``).

The recorded action is passed to every per-decider method, because ``only_if`` makes the
validity of a component depend on the chosen value of its parent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from torch import Tensor

from colosseum.core.tree import Tree


class Distribution(ABC):
    """Action distribution over a batch ``[B]`` with ``num_deciders`` deciders."""

    @property
    @abstractmethod
    def batch_size(self) -> int: ...

    @property
    @abstractmethod
    def num_deciders(self) -> int: ...

    @abstractmethod
    def sample(self) -> Tree:
        """A sampled action tree (torch, batch first)."""

    @abstractmethod
    def mode(self) -> Tree:
        """The most likely action tree (argmax / mean)."""

    def log_prob(self, actions: Tree) -> Tensor:
        """Joint log-probability ``[B]``: the sum over valid deciders of :meth:`unit_log_prob`."""
        return self.unit_log_prob(actions).sum(dim=-1)

    @abstractmethod
    def unit_log_prob(self, actions: Tree) -> Tensor:
        """``[B, K]`` log-probability per decider; 0 where invalid."""

    @abstractmethod
    def unit_entropy(self, actions: Tree) -> Tensor:
        """``[B, K]`` entropy per decider (gated by ``only_if`` on ``actions``); 0 where invalid."""

    @abstractmethod
    def unit_valid(self, actions: Tree) -> Tensor:
        """``[B, K]`` bool: the decider counts for ``actions``."""

    @abstractmethod
    def unit_kl(self, other: Distribution, actions: Tree) -> Tensor:
        """``[B, K]`` KL(self || other) per decider; 0 where invalid."""

    @abstractmethod
    def apply_mask(self, mask: Tree | None) -> Distribution:
        """A new distribution with ``mask`` combined into the current one (None: unchanged)."""

    @classmethod
    @abstractmethod
    def cat(cls, dists: Sequence[Distribution]) -> Distribution:
        """Concatenate along the batch dimension."""
