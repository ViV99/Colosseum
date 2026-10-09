"""Base classes for the parts of a ``ComposedModel`` (SP2 spec block 3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple

import torch.nn as nn
from torch import Tensor

from colosseum.core.tree import Tree

if TYPE_CHECKING:
    from colosseum.networks.dist import Distribution


class EncoderOutput(NamedTuple):
    latent: Tensor                 # [B, D]: goes through the core
    aux: dict[str, Tensor]         # bypasses the core, e.g. unit embeddings [B, U, E] or spatial maps


class BaseEncoder(nn.Module, ABC):
    """Observation tree (torch, batch first, env dtypes) -> latent ``[B, D]`` (or an ``EncoderOutput``)."""

    @abstractmethod
    def forward(self, obs: Tree) -> Tensor | EncoderOutput: ...

    @property
    @abstractmethod
    def latent_dim(self) -> int: ...


class BasePolicy(nn.Module, ABC):
    """Core features ``[B, F]`` plus the encoder's ``aux`` -> action distribution (batch ``[B]``)."""

    @abstractmethod
    def forward(self, features: Tensor, aux: dict[str, Tensor]) -> Distribution: ...


class BaseValue(nn.Module, ABC):
    """Value input ``[B, F (+ G)]`` -> values ``[B]``."""

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor: ...


class BaseCriticEncoder(nn.Module, ABC):
    """Global-state tree -> ``[B, G]`` features for the value head only (centralized critic)."""

    @abstractmethod
    def forward(self, global_state: Tree) -> Tensor: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...
