"""Observation normalization (running mean/std).

``NormalizeObs`` keeps its running statistics in registered *buffers*, so they
are part of the model ``state_dict`` and ride along with the normal weight sync
(learner -> workers) and checkpoints.

The statistics change only through :meth:`NormalizeObs.update`; ``forward`` is
a pure function of the current statistics, in train and eval mode alike. The
algorithm calls ``PolicyModel.update_normalizers(obs)`` exactly once per train
step with the batch's fresh observations, before any loss forward. So every
sample is counted once, whatever the number of epochs and minibatches, and the
learner's forward uses the same statistics for every minibatch of a step.

Usage: put it inside your encoder::

    class MyEncoder(BaseEncoder):
        def __init__(self, obs_dim=...):
            super().__init__()
            self.norm = NormalizeObs(shape=(obs_dim,))
            self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU())

        def forward(self, obs):
            return self.net(self.norm(obs))

The default ``PolicyModel.update_normalizers`` feeds the raw observations to
every ``NormalizeObs`` submodule. If a normalizer sees something else (a slice
or a transform of the observation), override ``update_normalizers``.
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
    """Normalize observations by running mean/std; statistics change only in ``update``.

    Args:
        shape: per-observation feature shape (e.g. ``(obs_dim,)`` or ``(C, H, W)``).
        clip: clip normalized values to ``[-clip, clip]`` (0 disables).
        epsilon: numerical floor for the variance.
    """

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.shape: tuple[int, ...] = tuple(int(s) for s in shape)
        self.rms = RunningMeanStd(self.shape)
        self._clip = clip
        self._eps = epsilon

    @torch.no_grad()
    def update(self, obs: torch.Tensor) -> None:
        """Add a batch of observations ``[..., *shape]`` (any leading dims) to the statistics."""
        n = len(self.shape)
        if obs.dim() < n or tuple(obs.shape[obs.dim() - n:]) != self.shape:
            raise ValueError(
                f"NormalizeObs(shape={self.shape}) cannot update from observations of shape "
                f"{tuple(obs.shape)}: the trailing dims must equal {self.shape}. If this "
                f"normalizer sees a transformed observation, override "
                f"PolicyModel.update_normalizers for your model."
            )
        flat = obs.reshape(-1, *self.shape).to(self.rms.mean.dtype)
        if flat.shape[0] == 0:
            return
        self.rms.update(flat)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normed = (obs - self.rms.mean) / torch.sqrt(self.rms.var + self._eps)
        if self._clip > 0:
            normed = torch.clamp(normed, -self._clip, self._clip)
        return normed
