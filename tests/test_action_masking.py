"""Tests for action masking support (T1.1)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import numpy as np

from colosseum.networks.distributions import CategoricalDist, DiagGaussianDist
from colosseum.networks.actor_critic import ActorCriticNetwork
from colosseum.core.types import TrajectoryChunk


# ---------------------------------------------------------------
# Distribution-level tests
# ---------------------------------------------------------------

def test_categorical_mask_blocks_invalid_actions():
    """Masked actions should never be sampled."""
    logits = torch.zeros(1, 5)  # uniform over 5 actions
    mask = torch.tensor([[True, True, False, False, False]])  # only actions 0,1 valid
    dist = CategoricalDist(logits, mask=mask)

    samples = torch.stack([dist.sample() for _ in range(200)])
    unique = set(samples.squeeze().tolist())
    assert unique.issubset({0, 1}), f"Invalid actions sampled: {unique}"


def test_categorical_mask_entropy():
    """Entropy should reflect only valid actions."""
    logits = torch.zeros(1, 4)  # uniform
    # All 4 valid
    dist_full = CategoricalDist(logits)
    # Only 2 valid
    mask = torch.tensor([[True, True, False, False]])
    dist_masked = CategoricalDist(logits, mask=mask)

    # Entropy of 2 uniform actions < entropy of 4 uniform actions
    assert dist_masked.entropy().item() < dist_full.entropy().item()


def test_categorical_apply_mask():
    """apply_mask should return a new distribution with mask applied."""
    logits = torch.zeros(2, 3)
    dist = CategoricalDist(logits)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    masked_dist = dist.apply_mask(mask)

    # Original should be unchanged
    assert not torch.isinf(dist.logits).any()
    # Masked dist should have -inf for invalid actions
    assert torch.isinf(masked_dist.logits[0, 1]).item()
    assert torch.isinf(masked_dist.logits[1, 0]).item()


def test_categorical_mode_respects_mask():
    """Mode should only return valid actions."""
    logits = torch.tensor([[10.0, 5.0, 1.0]])  # action 0 is best
    mask = torch.tensor([[False, True, True]])   # but action 0 is invalid
    dist = CategoricalDist(logits, mask=mask)
    assert dist.mode().item() == 1  # should pick action 1 (next best valid)


def test_categorical_log_prob_with_mask():
    """Log prob should work for valid actions with mask."""
    logits = torch.zeros(1, 3)
    mask = torch.tensor([[True, True, False]])
    dist = CategoricalDist(logits, mask=mask)
    lp = dist.log_prob(torch.tensor([0]))
    assert torch.isfinite(lp).all()


def test_gaussian_apply_mask_noop():
    """DiagGaussianDist.apply_mask should return self (no-op for continuous)."""
    mean = torch.zeros(2, 3)
    log_std = torch.zeros(2, 3)
    dist = DiagGaussianDist(mean, log_std)
    masked = dist.apply_mask(torch.ones(2, 3, dtype=torch.bool))
    assert masked is dist  # same object, no-op


# ---------------------------------------------------------------
# ActorCriticNetwork-level tests
# ---------------------------------------------------------------

def _make_network(obs_dim=8, num_actions=4, hidden=32):
    """Create a simple network for testing."""
    from helpers import make_simple_network
    return make_simple_network(obs_dim=obs_dim, hidden_dim=hidden, num_actions=num_actions)


def test_act_with_mask():
    """act() should respect action masks."""
    net = _make_network(obs_dim=4, num_actions=5)
    obs = torch.randn(10, 4)
    mask = torch.zeros(10, 5, dtype=torch.bool)
    mask[:, 2] = True  # only action 2 is valid

    actions, log_probs, values, _ = net.act(obs, action_mask=mask)
    assert (actions == 2).all(), f"Expected all actions=2, got {actions.tolist()}"
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()


def test_act_without_mask():
    """act() without mask should work as before."""
    net = _make_network()
    obs = torch.randn(5, 8)
    actions, log_probs, values, _ = net.act(obs)
    assert actions.shape == (5,)
    assert torch.isfinite(log_probs).all()


def test_evaluate_actions_with_mask():
    """evaluate_actions() should use mask for log_prob and entropy."""
    net = _make_network(obs_dim=4, num_actions=3)
    obs = torch.randn(8, 4)

    # Without mask
    actions = torch.randint(0, 3, (8,))
    lp_no_mask, vals, ent_no_mask = net.evaluate_actions(obs, actions)

    # With mask (allow all) — should give same results
    mask_all = torch.ones(8, 3, dtype=torch.bool)
    lp_mask, vals2, ent_mask = net.evaluate_actions(obs, actions, action_mask=mask_all)

    assert torch.allclose(lp_no_mask, lp_mask, atol=1e-5)
    assert torch.allclose(ent_no_mask, ent_mask, atol=1e-5)


# ---------------------------------------------------------------
# TrajectoryChunk tests
# ---------------------------------------------------------------

def test_trajectory_chunk_with_masks():
    """TrajectoryChunk should support optional action_masks field."""
    chunk = TrajectoryChunk(
        agent_id="test",
        observations=torch.randn(16, 4),
        actions=torch.randint(0, 3, (16,)),
        action_log_probs=torch.randn(16),
        rewards=torch.randn(16),
        dones=torch.zeros(16),
        values=torch.randn(16),
        bootstrap_value=torch.tensor(0.0),
        behavior_policy_version=1,
        action_masks=torch.ones(16, 3, dtype=torch.bool),
    )
    assert chunk.action_masks is not None
    assert chunk.action_masks.shape == (16, 3)

    # to_device should preserve masks
    chunk2 = chunk.to_device("cpu")
    assert chunk2.action_masks is not None
    assert chunk2.action_masks.shape == (16, 3)


def test_trajectory_chunk_without_masks():
    """TrajectoryChunk without masks should work as before."""
    chunk = TrajectoryChunk(
        agent_id="test",
        observations=torch.randn(16, 4),
        actions=torch.randint(0, 3, (16,)),
        action_log_probs=torch.randn(16),
        rewards=torch.randn(16),
        dones=torch.zeros(16),
        values=torch.randn(16),
        bootstrap_value=torch.tensor(0.0),
        behavior_policy_version=1,
    )
    assert chunk.action_masks is None
    chunk2 = chunk.to_device("cpu")
    assert chunk2.action_masks is None


# ---------------------------------------------------------------
# APPO with action masks
# ---------------------------------------------------------------

def test_appo_with_action_masks():
    """APPO should handle chunks with action_masks."""
    from colosseum.algorithms.appo import APPO
    from colosseum.core.config import AlgorithmConfig

    net = _make_network(obs_dim=4, num_actions=3)
    config = AlgorithmConfig(name="appo", num_epochs=1, minibatch_chunks=0)
    appo = APPO(net, config, device="cpu")

    chunks = []
    for _ in range(4):
        masks = torch.ones(8, 3, dtype=torch.bool)
        masks[:, 2] = False  # action 2 always invalid
        chunk = TrajectoryChunk(
            agent_id="test",
            observations=torch.randn(8, 4),
            actions=torch.randint(0, 2, (8,)),  # only actions 0,1
            action_log_probs=torch.randn(8),
            rewards=torch.randn(8),
            dones=torch.zeros(8),
            values=torch.randn(8),
            bootstrap_value=torch.tensor(0.0),
            behavior_policy_version=0,
            action_masks=masks,
        )
        chunks.append(chunk)

    metrics = appo.train_step(chunks)
    assert "total_loss" in metrics
    assert np.isfinite(metrics["total_loss"])
