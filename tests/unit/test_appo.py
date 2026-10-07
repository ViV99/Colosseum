"""Unit tests for APPO algorithm."""
import sys
import os

import pytest
import torch


from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.types import TrajectoryChunk
from colosseum.networks.actor_critic import ActorCriticNetwork
from examples.tic_tac_toe.networks import TicTacToeEncoder, TicTacToePolicy, TicTacToeValue


def _make_network():
    return ActorCriticNetwork(TicTacToeEncoder(), TicTacToePolicy(), TicTacToeValue())


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
    assert not all(l == losses[0] for l in losses)


# ---------------------------------------------------------------
# T4.5: LR schedule tests
# ---------------------------------------------------------------

def test_linear_lr_decay():
    """LR should decrease linearly over training steps."""
    from colosseum.core.config import LRSchedule

    net = _make_network()
    config = AlgorithmConfig(lr_schedule=LRSchedule.LINEAR, learning_rate=1e-3)
    algo = APPO(net, config, device="cpu")
    algo.setup_lr_schedule(total_steps=20)

    initial_lr = algo._optimizer.param_groups[0]["lr"]
    assert abs(initial_lr - 1e-3) < 1e-8

    for _ in range(10):
        chunks = _make_chunks(n=2, T=4)
        algo.train_step(chunks)

    mid_lr = algo._optimizer.param_groups[0]["lr"]
    assert mid_lr < initial_lr, f"LR should decrease: {mid_lr} >= {initial_lr}"


def test_cosine_lr_decay():
    """LR should follow cosine curve."""
    from colosseum.core.config import LRSchedule

    net = _make_network()
    config = AlgorithmConfig(lr_schedule=LRSchedule.COSINE, learning_rate=1e-3)
    algo = APPO(net, config, device="cpu")
    algo.setup_lr_schedule(total_steps=20)

    for _ in range(10):
        chunks = _make_chunks(n=2, T=4)
        algo.train_step(chunks)

    mid_lr = algo._optimizer.param_groups[0]["lr"]
    assert mid_lr < 1e-3, f"LR should decrease: {mid_lr}"
    assert mid_lr > 0, f"LR should be positive at midpoint: {mid_lr}"


def test_constant_lr():
    """Constant schedule should not change LR."""
    from colosseum.core.config import LRSchedule

    net = _make_network()
    config = AlgorithmConfig(lr_schedule=LRSchedule.CONSTANT, learning_rate=1e-3)
    algo = APPO(net, config, device="cpu")
    algo.setup_lr_schedule(total_steps=20)

    for _ in range(5):
        chunks = _make_chunks(n=2, T=4)
        algo.train_step(chunks)

    lr = algo._optimizer.param_groups[0]["lr"]
    assert abs(lr - 1e-3) < 1e-8, f"LR should stay constant: {lr}"
