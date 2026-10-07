"""Tests for DiagGaussianDist (continuous action distributions)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
import torch

from colosseum.networks.distributions import CategoricalDist, DiagGaussianDist


def test_gaussian_sample_shape():
    """Sample should produce correct shape."""
    mean = torch.randn(8, 3)
    log_std = torch.zeros(8, 3)
    dist = DiagGaussianDist(mean, log_std)
    sample = dist.sample()
    assert sample.shape == (8, 3)


def test_gaussian_log_prob_finite():
    """Log prob should be finite for reasonable inputs."""
    mean = torch.randn(4, 2)
    log_std = torch.zeros(4, 2)
    dist = DiagGaussianDist(mean, log_std)
    actions = torch.randn(4, 2)
    lp = dist.log_prob(actions)
    assert lp.shape == (4,)  # summed over action dims
    assert torch.isfinite(lp).all()


def test_gaussian_entropy_positive():
    """Entropy should be positive and finite."""
    mean = torch.zeros(5, 3)
    log_std = torch.zeros(5, 3)  # std=1.0
    dist = DiagGaussianDist(mean, log_std)
    ent = dist.entropy()
    assert ent.shape == (5,)
    assert (ent > 0).all()
    assert torch.isfinite(ent).all()


def test_gaussian_mode_equals_mean():
    """Mode should return the mean."""
    mean = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    log_std = torch.zeros(2, 3)
    dist = DiagGaussianDist(mean, log_std)
    mode = dist.mode()
    assert torch.allclose(mode, mean)


def test_gaussian_kl_divergence():
    """KL divergence between two Gaussians should be non-negative."""
    mean1 = torch.zeros(4, 2)
    log_std1 = torch.zeros(4, 2)
    mean2 = torch.ones(4, 2)
    log_std2 = torch.zeros(4, 2)

    dist1 = DiagGaussianDist(mean1, log_std1)
    dist2 = DiagGaussianDist(mean2, log_std2)

    kl = dist1.kl_divergence(dist2)
    assert kl.shape == (4,)
    assert (kl >= 0).all()
    assert torch.isfinite(kl).all()

    # KL of identical distributions should be 0
    kl_self = dist1.kl_divergence(dist1)
    assert torch.allclose(kl_self, torch.zeros(4), atol=1e-5)


def test_gaussian_kl_type_mismatch():
    """KL between Gaussian and Categorical should raise TypeError."""
    gaussian = DiagGaussianDist(torch.zeros(2, 3), torch.zeros(2, 3))
    categorical = CategoricalDist(torch.zeros(2, 5))

    with pytest.raises(TypeError):
        gaussian.kl_divergence(categorical)
