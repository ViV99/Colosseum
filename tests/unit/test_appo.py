"""Unit tests for APPO algorithm."""

import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.types import TrajectoryChunk
from helpers import make_ttt_model


def _make_network():
    return make_ttt_model()


def _make_chunks(n=4, T=8):
    chunks = []
    for _ in range(n):
        chunks.append(TrajectoryChunk(
            agent_id="test",
            observations=torch.randn(T, 3, 3, 3),
            actions=torch.randint(0, 9, (T,)),
            action_log_probs=torch.randn(T),
            rewards=torch.randn(T),
            dones=torch.zeros(T),
            values=torch.randn(T),
            bootstrap_value=torch.tensor(0.0),
            behavior_policy_version=0,
        ))
    return chunks


def test_compute_loss_returns_expected_keys():
    net = _make_network()
    algo = APPO(net, AlgorithmConfig(), device="cpu")
    chunks = _make_chunks()

    losses = algo.compute_loss(chunks)

    assert "total_loss" in losses
    assert "policy_loss" in losses
    assert "value_loss" in losses
    assert "entropy" in losses
    assert "approx_kl" in losses
    assert "clip_fraction" in losses


def test_train_step_updates_version():
    net = _make_network()
    algo = APPO(net, AlgorithmConfig(), device="cpu")
    assert algo.policy_version == 0

    chunks = _make_chunks()
    metrics = algo.train_step(chunks)

    assert algo.policy_version == 1
    assert "total_loss" in metrics
    assert "learning_rate" in metrics


def test_loss_is_finite():
    net = _make_network()
    algo = APPO(net, AlgorithmConfig(), device="cpu")
    chunks = _make_chunks()

    losses = algo.compute_loss(chunks)
    for key, val in losses.items():
        assert torch.isfinite(val), f"{key} is not finite: {val}"


def test_multiple_train_steps():
    """Verify multiple train steps don't crash and loss changes."""
    net = _make_network()
    algo = APPO(net, AlgorithmConfig(), device="cpu")

    losses = []
    for _ in range(5):
        chunks = _make_chunks()
        metrics = algo.train_step(chunks)
        losses.append(metrics["total_loss"])

    assert algo.policy_version == 5
    # Loss should change across steps (not be identical)
    assert not all(loss == losses[0] for loss in losses)
