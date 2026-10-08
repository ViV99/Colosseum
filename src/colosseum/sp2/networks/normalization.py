"""Observation normalization (running mean/std) for one leaf of the observation or global-state tree.

``NormalizeObs`` keeps its statistics in registered buffers, so they are part of the model
``state_dict`` (weight sync, checkpoints). They change only in :meth:`NormalizeObs.update`;
``forward`` is a pure function of the current statistics. The learner calls
``PolicyModel.update_normalizers(obs, global_state)`` once per train step; the default
implementation feeds every ``NormalizeObs`` the leaf at its ``path`` of its ``source`` tree.

Usage inside an encoder::

    self.norm = NormalizeObs(shape=(F,), path=("entities",))
    ...
    x = self.norm(obs["entities"])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """Welford-style running mean/variance over a feature shape, in buffers."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update stats from a batch ``x`` of shape ``[N, *shape]``."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m2 = self.var * self.count + batch_var * batch_count + delta.pow(2) * self.count * batch_count / tot
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)


class NormalizeObs(nn.Module):
    """Normalize one tree leaf by running mean/std (any input dtype; the output is float32)."""

    def __init__(self, shape: Sequence[int], path: tuple[str, ...] = (),
                 source: Literal["obs", "global_state"] = "obs", eps: float = 1e-8, clip: float = 10.0) -> None:
        super().__init__()
        if source not in ("obs", "global_state"):
            raise ValueError(f"NormalizeObs: source must be 'obs' or 'global_state', got {source!r}")
        self.shape: tuple[int, ...] = tuple(int(s) for s in shape)
        self.path: tuple[str, ...] = tuple(path)
        self.source = source
        self.rms = RunningMeanStd(self.shape)
        self._clip = clip
        self._eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Add samples ``[..., *shape]`` (any leading dims) to the statistics."""
        n = len(self.shape)
        if x.dim() < n or tuple(x.shape[x.dim() - n:]) != self.shape:
            raise ValueError(
                f"NormalizeObs(shape={self.shape}, path={self.path}) cannot update from a leaf of shape "
                f"{tuple(x.shape)}: the trailing dims must equal {self.shape}. If it normalizes a transformed "
                f"leaf, override PolicyModel.update_normalizers."
            )
        flat = x.reshape(-1, *self.shape).to(self.rms.mean.dtype)
        if flat.shape[0]:
            self.rms.update(flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = (x.to(self.rms.mean.dtype) - self.rms.mean) / torch.sqrt(self.rms.var + self._eps)
        if self._clip > 0:
            normed = torch.clamp(normed, -self._clip, self._clip)
        return normed
