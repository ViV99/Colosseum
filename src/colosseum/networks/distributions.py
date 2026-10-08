from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


class Distribution(ABC):
    """Base class for action distributions."""

    @abstractmethod
    def sample(self) -> torch.Tensor: ...

    @abstractmethod
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def entropy(self) -> torch.Tensor: ...

    @abstractmethod
    def mode(self) -> torch.Tensor:
        """Deterministic action (argmax/mean)."""
        ...

    @property
    def action_dim(self) -> int:
        """Number of values produced by ``sample()`` per batch element.

        Scalar distributions (e.g. Categorical) return 1.
        Vector distributions (e.g. DiagGaussian) return the action dimension.
        """
        raise NotImplementedError

    def kl_divergence(self, other: Distribution) -> torch.Tensor:
        """Compute KL(self || other). Subclasses should override for efficiency."""
        raise NotImplementedError(
            f"kl_divergence not implemented for {type(self).__name__}"
        )

    def apply_mask(self, mask: torch.Tensor) -> Distribution:
        """Return a new distribution with invalid actions masked out.

        Default: no-op (returns self). Override for discrete distributions.
        """
        return self

    @classmethod
    def cat(cls, dists: Sequence[Distribution]) -> Distribution:
        """Concatenate same-type distributions along the batch dimension.

        Used by the default ``PolicyModel.unroll`` (a per-step loop). Custom
        distributions must override it to be used with that default.
        """
        raise NotImplementedError(f"{cls.__name__}.cat is not implemented")


class CategoricalDist(Distribution):
    """For discrete action spaces with optional action masking."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor | None = None):
        self._mask: torch.Tensor | None = None
        if mask is not None:
            self._mask = mask.bool()
            logits = logits.masked_fill(~self._mask, float("-inf"))
        self._dist = torch.distributions.Categorical(logits=logits)

    @property
    def mask(self) -> torch.Tensor | None:
        """Bool legal-action mask (True = legal), or None when unmasked."""
        return self._mask

    @property
    def logits(self) -> torch.Tensor:
        return self._dist.logits

    @property
    def action_dim(self) -> int:
        return 1  # sample() returns [B] — one integer per element

    def sample(self) -> torch.Tensor:
        return self._dist.sample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()

    def mode(self) -> torch.Tensor:
        return self._dist.logits.argmax(dim=-1)

    def kl_divergence(self, other: Distribution) -> torch.Tensor:
        """KL(self || other) over legal actions only.

        The legal set is the intersection of both distributions' masks; both sides are
        renormalized on it. Rows with no common legal action give 0. The result is
        finite even when only one side is masked; it is computed in float32.
        """
        if not isinstance(other, CategoricalDist):
            raise TypeError(f"Cannot compute KL between CategoricalDist and {type(other).__name__}")
        legal = self._mask
        if other.mask is not None:
            legal = other.mask if legal is None else (legal & other.mask)
        p_logits = self.logits.float()
        q_logits = other.logits.float()
        if legal is None:
            log_p = F.log_softmax(p_logits, dim=-1)
            log_q = F.log_softmax(q_logits, dim=-1)
            return (log_p.exp() * (log_p - log_q)).sum(dim=-1)
        neg = torch.finfo(p_logits.dtype).min
        log_p = F.log_softmax(p_logits.masked_fill(~legal, neg), dim=-1)
        log_q = F.log_softmax(q_logits.masked_fill(~legal, neg), dim=-1)
        terms = torch.where(legal, log_p.exp() * (log_p - log_q), torch.zeros_like(log_p))
        return terms.sum(dim=-1)

    def apply_mask(self, mask: torch.Tensor) -> CategoricalDist:
        """Return a new CategoricalDist with invalid actions masked out (masks combine)."""
        mask = mask.bool()
        if self._mask is not None:
            mask = mask & self._mask
        return CategoricalDist(logits=self.logits, mask=mask)

    @classmethod
    def cat(cls, dists: Sequence[CategoricalDist]) -> CategoricalDist:
        """Concatenate along the batch; masks are concatenated (unmasked parts: all legal)."""
        logits = torch.cat([d.logits for d in dists], dim=0)
        if all(d.mask is None for d in dists):
            return CategoricalDist(logits=logits)
        mask = torch.cat([
            d.mask if d.mask is not None else torch.ones_like(d.logits, dtype=torch.bool)
            for d in dists
        ], dim=0)
        return CategoricalDist(logits=logits, mask=mask)


class DiagGaussianDist(Distribution):
    """For continuous action spaces with diagonal covariance."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self._mean = mean
        self._log_std = log_std
        self._dist = torch.distributions.Normal(mean, log_std.exp())

    @property
    def action_dim(self) -> int:
        return self._dist.mean.shape[-1]  # sample() returns [B, D]

    def sample(self) -> torch.Tensor:
        return self._dist.rsample()  # reparameterized

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(actions).sum(dim=-1)  # sum across action dims

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy().sum(dim=-1)

    def mode(self) -> torch.Tensor:
        return self._dist.mean

    def kl_divergence(self, other: Distribution) -> torch.Tensor:
        if not isinstance(other, DiagGaussianDist):
            raise TypeError(f"Cannot compute KL between DiagGaussianDist and {type(other).__name__}")
        return torch.distributions.kl_divergence(self._dist, other._dist).sum(dim=-1)

    @classmethod
    def cat(cls, dists: Sequence[DiagGaussianDist]) -> DiagGaussianDist:
        # log_std may be broadcast (e.g. a [D] parameter): expand it to the mean's shape.
        return DiagGaussianDist(
            torch.cat([d._mean for d in dists], dim=0),
            torch.cat([torch.broadcast_to(d._log_std, d._mean.shape) for d in dists], dim=0),
        )


class CompositeDist(Distribution):
    """Multi-head distribution for composite action spaces (Dict / Tuple / MultiDiscrete).

    Wraps an ordered mapping of sub-distributions. All public methods operate on
    *flat* ``float32`` tensors of shape ``[B, flat_size]``: a discrete component
    takes one column (the category index as a float), a continuous one takes
    ``action_dim`` columns.

    The component order is the insertion order of ``dists`` and defines the flat
    action and mask layout. It must equal the action space's component order
    (``ActionSpec.component_names``): Tuple/MultiDiscrete components are named
    ``"0"``, ``"1"``, ... in index order; Dict components follow
    ``action_space.spaces`` order (gymnasium sorts plain-dict keys, an
    ``OrderedDict`` keeps its order). ``ActionSpec.check_distribution`` verifies it.
    """

    def __init__(self, dists: Mapping[str, Distribution]) -> None:
        if not dists:
            raise ValueError("CompositeDist requires at least one sub-distribution")

        self._keys: list[str] = list(dists.keys())
        self._dists: dict[str, Distribution] = {k: dists[k] for k in self._keys}

        # Action layout: key -> (offset, size, is_discrete)
        offset = 0
        self._layout: dict[str, tuple[int, int, bool]] = {}
        for k in self._keys:
            d = self._dists[k]
            sz = d.action_dim
            self._layout[k] = (offset, sz, isinstance(d, CategoricalDist))
            offset += sz
        self._flat_size: int = offset

        # Mask layout (only discrete components have masks)
        mask_offset = 0
        self._mask_layout: dict[str, tuple[int, int]] = {}
        for k in self._keys:
            d = self._dists[k]
            if isinstance(d, CategoricalDist):
                ms = d.logits.shape[-1]
            else:
                ms = 0
            self._mask_layout[k] = (mask_offset, ms)
            mask_offset += ms
        self._flat_mask_size: int = mask_offset

    # ---- Layout -----------------------------------------------------------

    @property
    def keys(self) -> list[str]:
        """Component names in flat-layout order."""
        return list(self._keys)

    @property
    def components(self) -> list[tuple[str, int, int, bool]]:
        """``(name, flat_offset, size, is_discrete)`` per component, in flat-layout order."""
        return [(k, *self._layout[k]) for k in self._keys]

    @property
    def flat_mask_size(self) -> int:
        """Width of the flat action mask: the summed sizes of the discrete components."""
        return self._flat_mask_size

    # ---- Distribution interface ------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._flat_size

    def sample(self) -> torch.Tensor:
        """Sample from all sub-distributions and return ``[B, flat_size]``."""
        parts: list[torch.Tensor] = []
        for k in self._keys:
            d = self._dists[k]
            s = d.sample()
            _, _, is_disc = self._layout[k]
            if is_disc:
                s = s.float().unsqueeze(-1)  # [B] → [B, 1]
            elif s.dim() == 1:
                s = s.unsqueeze(-1)
            parts.append(s)
        return torch.cat(parts, dim=-1)

    def log_prob(self, flat_actions: torch.Tensor) -> torch.Tensor:
        """Compute log-probability of a flat action tensor ``[B, flat_size]``."""
        total: torch.Tensor | None = None
        for k in self._keys:
            off, sz, is_disc = self._layout[k]
            d = self._dists[k]
            if is_disc:
                sub = flat_actions[:, off].long()
            else:
                sub = flat_actions[:, off:off + sz]
            lp = d.log_prob(sub)
            total = lp if total is None else total + lp
        return total

    def entropy(self) -> torch.Tensor:
        total: torch.Tensor | None = None
        for k in self._keys:
            e = self._dists[k].entropy()
            total = e if total is None else total + e
        return total

    def mode(self) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for k in self._keys:
            d = self._dists[k]
            m = d.mode()
            _, _, is_disc = self._layout[k]
            if is_disc:
                m = m.float().unsqueeze(-1)
            elif m.dim() == 1:
                m = m.unsqueeze(-1)
            parts.append(m)
        return torch.cat(parts, dim=-1)

    def apply_mask(self, flat_mask: torch.Tensor) -> CompositeDist:
        """Apply a flat mask ``[B, flat_mask_size]`` to discrete sub-distributions."""
        new_dists: dict[str, Distribution] = {}
        for k in self._keys:
            d = self._dists[k]
            m_off, m_sz = self._mask_layout[k]
            if m_sz > 0 and flat_mask is not None:
                sub_mask = flat_mask[:, m_off:m_off + m_sz]
                new_dists[k] = d.apply_mask(sub_mask)
            else:
                new_dists[k] = d
        return CompositeDist(new_dists)

    def kl_divergence(self, other: Distribution) -> torch.Tensor:
        if not isinstance(other, CompositeDist):
            raise TypeError(
                f"Cannot compute KL between CompositeDist and {type(other).__name__}"
            )
        if self._keys != other._keys:
            raise ValueError(
                f"Key mismatch: {self._keys} vs {other._keys}"
            )
        total: torch.Tensor | None = None
        for k in self._keys:
            kl = self._dists[k].kl_divergence(other._dists[k])
            total = kl if total is None else total + kl
        return total

    @classmethod
    def cat(cls, dists: Sequence[CompositeDist]) -> CompositeDist:
        keys = dists[0]._keys
        for d in dists:
            if d._keys != keys:
                raise ValueError(f"Key mismatch: {d._keys} vs {keys}")
        return CompositeDist({k: type(dists[0]._dists[k]).cat([d._dists[k] for d in dists]) for k in keys})
