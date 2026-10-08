"""Leaf distributions: categorical, multi-categorical, diagonal Gaussian. Alone, each is ONE decider."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from colosseum.sp2.networks.dist.base import Distribution


def masked_log_softmax(logits: Tensor, mask: Tensor | None) -> Tensor:
    """log_softmax over the last dim with illegal entries at ``finfo.min`` (an all-illegal row stays finite)."""
    if mask is not None:
        logits = torch.where(mask, logits, torch.finfo(logits.dtype).min)
    return torch.log_softmax(logits, dim=-1)


def categorical_log_prob(log_p: Tensor, actions: Tensor) -> Tensor:
    """``log_p[..., n]`` at integer ``actions[...]``."""
    return log_p.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)


def categorical_entropy(log_p: Tensor, mask: Tensor | None) -> Tensor:
    """Entropy over the last dim; illegal entries contribute exactly 0 to the value and the gradient."""
    if mask is not None:
        log_p = torch.where(mask, log_p, torch.zeros_like(log_p))  # exp(0) * 0 = 0, no gradient to illegal logits
    return -(log_p.exp() * log_p).sum(dim=-1)


def categorical_kl(p_logits: Tensor, p_mask: Tensor | None, q_logits: Tensor, q_mask: Tensor | None) -> Tensor:
    """KL(p || q) over the last dim on the intersection of both masks (both renormalized on it), float32."""
    legal = p_mask
    if q_mask is not None:
        legal = q_mask if legal is None else legal & q_mask
    log_p = masked_log_softmax(p_logits.float(), legal)
    log_q = masked_log_softmax(q_logits.float(), legal)
    terms = log_p.exp() * (log_p - log_q)
    if legal is not None:
        terms = torch.where(legal, terms, torch.zeros_like(terms))
    return terms.sum(dim=-1)


def sample_categorical(log_p: Tensor) -> Tensor:
    return torch.distributions.Categorical(logits=log_p, validate_args=False).sample()


def gaussian_log_prob(mean: Tensor, log_std: Tensor, x: Tensor) -> Tensor:
    """Sum over the last dim of the Normal log-density."""
    var = torch.exp(2.0 * log_std)
    return (-((x - mean) ** 2) / (2.0 * var) - log_std - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)


def gaussian_entropy(log_std: Tensor) -> Tensor:
    return (0.5 + 0.5 * math.log(2.0 * math.pi) + log_std).sum(dim=-1)


def gaussian_kl(mean_p: Tensor, log_std_p: Tensor, mean_q: Tensor, log_std_q: Tensor) -> Tensor:
    var_p, var_q = torch.exp(2.0 * log_std_p), torch.exp(2.0 * log_std_q)
    return (log_std_q - log_std_p + (var_p + (mean_p - mean_q) ** 2) / (2.0 * var_q) - 0.5).sum(dim=-1)


def _cat_masks(masks: Sequence[Tensor | None], shapes: Sequence[torch.Size], device: torch.device) -> Tensor | None:
    if all(m is None for m in masks):
        return None
    return torch.cat([m if m is not None else torch.ones(s, dtype=torch.bool, device=device)
                      for m, s in zip(masks, shapes)], dim=0)


class CategoricalDist(Distribution):
    """``logits [B, n]``, optional ``mask [B, n]`` (True = legal). Actions ``int64 [B]``."""

    def __init__(self, logits: Tensor, mask: Tensor | None = None) -> None:
        if logits.dim() != 2:
            raise ValueError(f"CategoricalDist: logits must be [B, n], got shape {tuple(logits.shape)}")
        self._logits = logits
        self._mask = None if mask is None else mask.to(device=logits.device, dtype=torch.bool)
        if self._mask is not None and self._mask.shape != logits.shape:
            raise ValueError(f"CategoricalDist: mask shape {tuple(self._mask.shape)} != logits shape "
                             f"{tuple(logits.shape)}")
        self._log_p = masked_log_softmax(logits, self._mask)

    @property
    def logits(self) -> Tensor:
        return self._logits

    @property
    def mask(self) -> Tensor | None:
        return self._mask

    @property
    def num_categories(self) -> int:
        return int(self._logits.shape[-1])

    @property
    def batch_size(self) -> int:
        return int(self._logits.shape[0])

    @property
    def num_deciders(self) -> int:
        return 1

    def sample(self) -> Tensor:
        return sample_categorical(self._log_p)

    def mode(self) -> Tensor:
        return self._log_p.argmax(dim=-1)

    def unit_log_prob(self, actions: Tensor) -> Tensor:
        return categorical_log_prob(self._log_p, actions).unsqueeze(-1)

    def unit_entropy(self, actions: Tensor) -> Tensor:
        return categorical_entropy(self._log_p, self._mask).unsqueeze(-1)

    def unit_valid(self, actions: Tensor) -> Tensor:
        return torch.ones(self.batch_size, 1, dtype=torch.bool, device=self._logits.device)

    def unit_kl(self, other: Distribution, actions: Tensor) -> Tensor:
        if not isinstance(other, CategoricalDist):
            raise TypeError(f"KL between CategoricalDist and {type(other).__name__} is not defined")
        return categorical_kl(self._logits, self._mask, other._logits, other._mask).unsqueeze(-1)

    def apply_mask(self, mask: Tensor | None) -> CategoricalDist:
        if mask is None:
            return self
        mask = mask.to(device=self._logits.device, dtype=torch.bool)
        return CategoricalDist(self._logits, mask if self._mask is None else mask & self._mask)

    @classmethod
    def cat(cls, dists: Sequence[CategoricalDist]) -> CategoricalDist:
        logits = torch.cat([d._logits for d in dists], dim=0)
        mask = _cat_masks([d._mask for d in dists], [d._logits.shape for d in dists], logits.device)
        return CategoricalDist(logits, mask)


class MultiCategoricalDist(Distribution):
    """Independent categoricals in one decider: ``logits [B, sum(nvec)]``, ``mask [B, sum(nvec)]``.
    Actions ``int64 [B, C]``."""

    def __init__(self, logits: Tensor, nvec: Sequence[int], mask: Tensor | None = None) -> None:
        self.nvec = tuple(int(n) for n in nvec)
        if logits.dim() != 2 or logits.shape[-1] != sum(self.nvec):
            raise ValueError(f"MultiCategoricalDist: logits must be [B, {sum(self.nvec)}] for nvec {self.nvec}, "
                             f"got shape {tuple(logits.shape)}")
        self._logits = logits
        self._mask = None if mask is None else mask.to(device=logits.device, dtype=torch.bool)
        if self._mask is not None and self._mask.shape != logits.shape:
            raise ValueError(f"MultiCategoricalDist: mask shape {tuple(self._mask.shape)} != logits shape "
                             f"{tuple(logits.shape)}")
        self._parts: list[tuple[Tensor, Tensor | None]] = []
        offset = 0
        for n in self.nvec:
            m = None if self._mask is None else self._mask[:, offset:offset + n]
            self._parts.append((masked_log_softmax(logits[:, offset:offset + n], m), m))
            offset += n

    @property
    def logits(self) -> Tensor:
        return self._logits

    @property
    def mask(self) -> Tensor | None:
        return self._mask

    @property
    def batch_size(self) -> int:
        return int(self._logits.shape[0])

    @property
    def num_deciders(self) -> int:
        return 1

    def sample(self) -> Tensor:
        return torch.stack([sample_categorical(log_p) for log_p, _ in self._parts], dim=-1)

    def mode(self) -> Tensor:
        return torch.stack([log_p.argmax(dim=-1) for log_p, _ in self._parts], dim=-1)

    def unit_log_prob(self, actions: Tensor) -> Tensor:
        return sum(categorical_log_prob(log_p, actions[:, i]) for i, (log_p, _) in enumerate(self._parts)).unsqueeze(-1)

    def unit_entropy(self, actions: Tensor) -> Tensor:
        return sum(categorical_entropy(log_p, m) for log_p, m in self._parts).unsqueeze(-1)

    def unit_valid(self, actions: Tensor) -> Tensor:
        return torch.ones(self.batch_size, 1, dtype=torch.bool, device=self._logits.device)

    def unit_kl(self, other: Distribution, actions: Tensor) -> Tensor:
        if not isinstance(other, MultiCategoricalDist) or other.nvec != self.nvec:
            raise TypeError(f"KL between MultiCategoricalDist{self.nvec} and {type(other).__name__} is not defined")
        total, offset = None, 0
        for n in self.nvec:
            sl = slice(offset, offset + n)
            kl = categorical_kl(self._logits[:, sl], None if self._mask is None else self._mask[:, sl],
                                other._logits[:, sl], None if other._mask is None else other._mask[:, sl])
            total = kl if total is None else total + kl
            offset += n
        return total.unsqueeze(-1)

    def apply_mask(self, mask: Tensor | None) -> MultiCategoricalDist:
        if mask is None:
            return self
        mask = mask.to(device=self._logits.device, dtype=torch.bool)
        return MultiCategoricalDist(self._logits, self.nvec, mask if self._mask is None else mask & self._mask)

    @classmethod
    def cat(cls, dists: Sequence[MultiCategoricalDist]) -> MultiCategoricalDist:
        logits = torch.cat([d._logits for d in dists], dim=0)
        mask = _cat_masks([d._mask for d in dists], [d._logits.shape for d in dists], logits.device)
        return MultiCategoricalDist(logits, dists[0].nvec, mask)


class DiagGaussianDist(Distribution):
    """``mean [B, d]``, ``log_std [B, d]`` or ``[d]``. Actions ``float [B, d]``; no mask."""

    def __init__(self, mean: Tensor, log_std: Tensor) -> None:
        if mean.dim() != 2:
            raise ValueError(f"DiagGaussianDist: mean must be [B, d], got shape {tuple(mean.shape)}")
        if tuple(log_std.shape) not in (tuple(mean.shape), tuple(mean.shape[-1:])):
            raise ValueError(f"DiagGaussianDist: log_std must be [B, d] or [d] for mean {tuple(mean.shape)}, "
                             f"got shape {tuple(log_std.shape)}")
        self._mean = mean
        self._log_std = torch.broadcast_to(log_std, mean.shape)

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def log_std(self) -> Tensor:
        return self._log_std

    @property
    def action_dim(self) -> int:
        return int(self._mean.shape[-1])

    @property
    def batch_size(self) -> int:
        return int(self._mean.shape[0])

    @property
    def num_deciders(self) -> int:
        return 1

    def sample(self) -> Tensor:
        with torch.no_grad():
            return self._mean + torch.exp(self._log_std) * torch.randn_like(self._mean)

    def mode(self) -> Tensor:
        return self._mean

    def unit_log_prob(self, actions: Tensor) -> Tensor:
        return gaussian_log_prob(self._mean, self._log_std, actions.to(self._mean.dtype)).unsqueeze(-1)

    def unit_entropy(self, actions: Tensor) -> Tensor:
        return gaussian_entropy(self._log_std).unsqueeze(-1)

    def unit_valid(self, actions: Tensor) -> Tensor:
        return torch.ones(self.batch_size, 1, dtype=torch.bool, device=self._mean.device)

    def unit_kl(self, other: Distribution, actions: Tensor) -> Tensor:
        if not isinstance(other, DiagGaussianDist):
            raise TypeError(f"KL between DiagGaussianDist and {type(other).__name__} is not defined")
        return gaussian_kl(self._mean, self._log_std, other._mean, other._log_std).unsqueeze(-1)

    def apply_mask(self, mask: Tensor | None) -> DiagGaussianDist:
        if mask is not None:
            raise ValueError("DiagGaussianDist (a Box action) takes no mask")
        return self

    @classmethod
    def cat(cls, dists: Sequence[DiagGaussianDist]) -> DiagGaussianDist:
        return DiagGaussianDist(torch.cat([d._mean for d in dists], dim=0),
                                torch.cat([d._log_std for d in dists], dim=0))
