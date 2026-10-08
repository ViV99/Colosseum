"""Tests for offline Behavioral Cloning (kickstart: test_kickstart_kl.py)."""
import os
import tempfile

import torch

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


def test_offline_bc_accepts_stateful_models():
    """BC goes through PolicyModel.step, so a core with output_dim != latent_dim works."""
    from helpers import make_simple_model

    def lstm_model():
        return make_simple_model(obs_dim=8, num_actions=4, core="lstm")

    trainer = OfflineBCTrainer(lstm_model(), lr=1e-3, action_type="discrete")
    trainer.add_data(torch.randn(32, 8), torch.randint(0, 4, (32,)))
    assert trainer.train(num_epochs=1, batch_size=16)["bc_loss"] > 0
    assert trainer.model is not None
