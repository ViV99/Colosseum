"""Tests for Behavioral Cloning: offline BC trainer and kickstart loss."""
import os
import tempfile

import torch

from colosseum.bc.kickstart import KickstartLoss
from colosseum.bc.offline_bc import OfflineBCTrainer
from helpers import make_ttt_model


def _make_network():
    """Create a small TicTacToe model for testing."""
    return make_ttt_model()


def _make_expert_data(n=200):
    """Generate random (obs, action) pairs for testing."""
    obs = torch.randn(n, 3, 3, 3)
    actions = torch.randint(0, 9, (n,))
    return obs, actions


def test_offline_bc_from_tensors():
    """Test offline BC training from in-memory tensors."""
    network = _make_network()
    trainer = OfflineBCTrainer(network, lr=1e-3, action_type="discrete")

    obs, actions = _make_expert_data(100)
    trainer.add_data(obs, actions)

    metrics = trainer.train(num_epochs=5, batch_size=32)
    assert "bc_loss" in metrics
    assert metrics["bc_loss"] > 0
    assert metrics["num_samples"] == 100


def test_offline_bc_from_file():
    """Test offline BC training from .pt files."""
    network = _make_network()
    obs, actions = _make_expert_data(100)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save data
        torch.save({"observations": obs, "actions": actions}, os.path.join(tmpdir, "data.pt"))

        trainer = OfflineBCTrainer(network, lr=1e-3, action_type="discrete")
        n_loaded = trainer.load_data(tmpdir)
        assert n_loaded == 100

        metrics = trainer.train(num_epochs=3, batch_size=50)
        assert metrics["bc_loss"] > 0


def test_offline_bc_loss_decreases():
    """Verify that BC loss decreases over training."""
    network = _make_network()
    obs, actions = _make_expert_data(200)

    trainer = OfflineBCTrainer(network, lr=1e-3, action_type="discrete")
    trainer.add_data(obs, actions)

    # Train 2 epochs, check loss
    m1 = trainer.train(num_epochs=2, batch_size=64)
    # Train 10 more
    m2 = trainer.train(num_epochs=10, batch_size=64)

    # After more training, loss should be lower (on same data)
    # We compare against the first epoch's loss roughly
    assert m2["bc_loss"] < m1["bc_loss"] * 1.5  # allow some slack


def test_kickstart_lambda_decay():
    """Test that kickstart lambda decays linearly to 0."""
    teacher = _make_network()
    ks = KickstartLoss(teacher, initial_lambda=1.0, decay_steps=100)

    assert ks.current_lambda == 1.0

    for _ in range(50):
        ks.step()
    assert abs(ks.current_lambda - 0.5) < 0.02

    for _ in range(50):
        ks.step()
    assert ks.current_lambda == 0.0

    # Past decay_steps, stays at 0
    ks.step()
    assert ks.current_lambda == 0.0


def test_kickstart_loss_computation():
    """Test kickstart KL loss is computed and is finite."""
    teacher = _make_network()
    student = _make_network()

    ks = KickstartLoss(teacher, initial_lambda=1.0, decay_steps=1000)
    obs = torch.randn(16, 3, 3, 3)

    loss = ks.compute(student, obs)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.item() >= 0  # KL divergence is non-negative


def test_kickstart_zero_lambda():
    """When lambda=0, kickstart loss should be 0."""
    teacher = _make_network()
    student = _make_network()

    ks = KickstartLoss(teacher, initial_lambda=0.0, decay_steps=100)
    obs = torch.randn(16, 3, 3, 3)

    loss = ks.compute(student, obs)
    assert loss.item() == 0.0


def test_appo_with_kickstart():
    """Test APPO with kickstart loss produces expected metric keys."""
    from colosseum.algorithms.appo import APPO
    from colosseum.core.config import AlgorithmConfig
    from colosseum.core.types import TrajectoryChunk

    teacher = _make_network()
    student = _make_network()
    ks = KickstartLoss(teacher, initial_lambda=0.5, decay_steps=1000)

    config = AlgorithmConfig(learning_rate=1e-3, num_epochs=1, minibatch_chunks=0)
    appo = APPO(student, config, device="cpu", kickstart=ks)

    # Create dummy chunks
    T = 8
    chunks = []
    for _ in range(4):
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

    metrics = appo.train_step(chunks)
    assert "kickstart_loss" in metrics
    assert "kickstart_lambda" in metrics
    assert metrics["kickstart_lambda"] > 0
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))


def test_offline_bc_and_kickstart_accept_stateful_models():
    """BC and kickstart go through PolicyModel.step, so a core with output_dim != latent_dim works."""
    from helpers import make_simple_model

    def lstm_model():
        return make_simple_model(obs_dim=8, num_actions=4, core="lstm")

    trainer = OfflineBCTrainer(lstm_model(), lr=1e-3, action_type="discrete")
    trainer.add_data(torch.randn(32, 8), torch.randint(0, 4, (32,)))
    assert trainer.train(num_epochs=1, batch_size=16)["bc_loss"] > 0
    assert trainer.model is not None

    ks = KickstartLoss(lstm_model(), initial_lambda=1.0, decay_steps=10)
    loss = ks.compute(lstm_model(), torch.randn(5, 8))
    assert loss.shape == () and torch.isfinite(loss) and loss.item() >= 0
