"""Observation normalization utilities (running mean/std).

``NormalizeObs`` keeps its running statistics in registered *buffers*, so they
are part of the network ``state_dict`` and therefore ride along with the normal
weight-sync path (learner -> workers) at zero extra plumbing cost — which is
exactly what an async actor-learner setup needs to keep worker inference and
learner training using the same normalization.

Usage — prepend it inside your encoder::

    class MyEncoder(BaseEncoder):
        def __init__(self, obs_dim=...):
            super().__init__()
            self.norm = NormalizeObs(shape=(obs_dim,))
            self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU())

        def forward(self, obs):
            return self.net(self.norm(obs))

Statistics update only in ``train()`` mode (i.e. on the learner). Workers run
in ``eval()`` mode and just apply the latest synced stats.
"""

from __future__ import annotations

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
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)


class NormalizeObs(nn.Module):
    """Normalize observations by running mean/std; updates stats only in training.

    Args:
        shape: per-observation feature shape (e.g. ``(obs_dim,)`` or ``(C, H, W)``).
        clip: clip normalized values to ``[-clip, clip]`` (0 disables).
        epsilon: numerical floor for the std.
    """

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.rms = RunningMeanStd(shape)
        self._clip = clip
        self._eps = epsilon

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Flatten leading dims to [N, *shape] for the stats update.
            flat = obs.reshape(-1, *self.rms.mean.shape)
            self.rms.update(flat)
        normed = (obs - self.rms.mean) / torch.sqrt(self.rms.var + self._eps)
        if self._clip > 0:
            normed = torch.clamp(normed, -self._clip, self._clip)
        return normed
