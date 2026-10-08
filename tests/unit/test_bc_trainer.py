"""Offline BC: -log_prob for every distribution, masks, stateful windows, strict types (T4.4)."""
import logging

import pytest
import torch
import torch.nn as nn

from colosseum.bc.offline_bc import OfflineBCTrainer, _window_index
from colosseum.networks.base import BasePolicy
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import NoCore
from colosseum.networks.distributions import CategoricalDist, CompositeDist, DiagGaussianDist
from helpers import SimpleEncoder, SimpleValue, make_simple_model


class GaussPolicy(BasePolicy):
    def __init__(self, in_dim: int = 16, act_dim: int = 2) -> None:
        super().__init__()
        self.mean = nn.Linear(in_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, latent):
        return DiagGaussianDist(self.mean(latent), self.log_std.expand(latent.shape[0], -1))


class DirSpeedPolicy(BasePolicy):
    """Composite: direction Discrete(3) (flat column 0) + speed Box(1) (column 1)."""

    def __init__(self, in_dim: int = 16) -> None:
        super().__init__()
        self.direction = nn.Linear(in_dim, 3)
        self.speed = nn.Linear(in_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, latent):
        return CompositeDist({
            "direction": CategoricalDist(self.direction(latent)),
            "speed": DiagGaussianDist(self.speed(latent), self.log_std.expand(latent.shape[0], -1)),
        })


def _composed(policy):
    torch.manual_seed(0)
    return ComposedModel(SimpleEncoder(4, 16), NoCore(input_dim=16), policy, SimpleValue(16))


def _memory_data(num_episodes: int, seed: int = 0):
    """Episodes of 4 steps: a cue in {0,1,2} is visible only at step 0; the expert repeats it."""
    g = torch.Generator().manual_seed(seed)
    obs, actions, dones = [], [], []
    for _ in range(num_episodes):
        cue = int(torch.randint(0, 3, (1,), generator=g))
        for t in range(4):
            o = torch.zeros(4)
            if t == 0:
                o[cue] = 1.0
            o[3] = t / 3.0
            obs.append(o)
            actions.append(cue)
            dones.append(t == 3)
    return torch.stack(obs), torch.tensor(actions), torch.tensor(dones)


def test_window_index_tiles_data_in_order_with_padding():
    w = _window_index(10, 4, offset=0)
    assert w.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, -1, -1]]
    w = _window_index(10, 4, offset=3)
    assert w.tolist() == [[0, 1, 2, -1], [3, 4, 5, 6], [7, 8, 9, -1]]
    flat = w[w >= 0]
    assert flat.tolist() == list(range(10))


def test_discrete_bc_learns_and_reports_metrics():
    torch.manual_seed(0)
    obs = torch.randn(256, 4)
    actions = obs.argmax(dim=1)
    trainer = OfflineBCTrainer(make_simple_model(obs_dim=4), lr=1e-2)
    trainer.add_data(obs, actions, dones=torch.zeros(256, dtype=torch.bool))
    metrics = trainer.train(num_epochs=30, batch_size=64)
    assert set(metrics) >= {"bc_loss", "bc_loss_first_epoch", "num_epochs", "num_samples", "accuracy"}
    assert metrics["num_samples"] == 256
    assert metrics["bc_loss"] < metrics["bc_loss_first_epoch"]
    assert metrics["accuracy"] > 0.9


def test_masks_are_applied():
    # Exactly one legal action per sample, equal to the expert action -> NLL is 0 from the start.
    n = 64
    actions = torch.randint(0, 4, (n,))
    masks = torch.nn.functional.one_hot(actions, 4).bool()
    trainer = OfflineBCTrainer(make_simple_model(obs_dim=4))
    trainer.add_data(torch.randn(n, 4), actions, action_masks=masks, dones=torch.zeros(n, dtype=torch.bool))
    metrics = trainer.train(num_epochs=1, batch_size=16)
    assert metrics["bc_loss_first_epoch"] < 1e-6
    assert metrics["accuracy"] == 1.0


def test_illegal_expert_action_raises():
    actions = torch.zeros(8, dtype=torch.long)
    masks = torch.ones(8, 4, dtype=torch.bool)
    masks[3, 0] = False
    trainer = OfflineBCTrainer(make_simple_model(obs_dim=4))
    trainer.add_data(torch.randn(8, 4), actions, action_masks=masks, dones=torch.zeros(8, dtype=torch.bool))
    with pytest.raises(ValueError, match="illegal"):
        trainer.train(num_epochs=1, batch_size=8)


def test_float_actions_with_discrete_space_raise():
    trainer = OfflineBCTrainer(make_simple_model(obs_dim=4))
    trainer.add_data(torch.randn(8, 4), torch.rand(8), dones=torch.zeros(8, dtype=torch.bool))
    with pytest.raises(ValueError, match="discrete"):
        trainer.train(num_epochs=1)


def test_continuous_bc_uses_log_prob_and_trains_log_std():
    torch.manual_seed(0)
    obs = torch.randn(512, 4)
    actions = 0.5 * obs[:, :2]
    model = _composed(GaussPolicy(16, 2))
    trainer = OfflineBCTrainer(model, lr=1e-2)
    trainer.add_data(obs, actions, dones=torch.zeros(512, dtype=torch.bool))
    metrics = trainer.train(num_epochs=40, batch_size=128)
    assert "accuracy" not in metrics
    assert metrics["bc_loss"] < metrics["bc_loss_first_epoch"]
    # MSE on the mean would never move log_std; NLL shrinks it on near-deterministic data.
    log_stds = [p for name, p in model.named_parameters() if name.endswith("log_std")]
    assert log_stds and (log_stds[0] < -0.5).all()


def test_composite_bc_and_strict_discrete_columns():
    torch.manual_seed(0)
    obs = torch.randn(512, 4)
    direction = obs[:, :3].argmax(dim=1).float()
    speed = obs[:, 3:4]
    actions = torch.cat([direction[:, None], speed], dim=1)          # flat layout [direction, speed]
    masks = torch.ones(512, 3, dtype=torch.bool)
    trainer = OfflineBCTrainer(_composed(DirSpeedPolicy(16)), lr=1e-2)
    trainer.add_data(obs, actions, action_masks=masks, dones=torch.zeros(512, dtype=torch.bool))
    metrics = trainer.train(num_epochs=30, batch_size=128)
    assert metrics["bc_loss"] < metrics["bc_loss_first_epoch"]
    assert metrics["accuracy"] > 0.85

    bad = actions.clone()
    bad[0, 0] = 1.5
    trainer2 = OfflineBCTrainer(_composed(DirSpeedPolicy(16)))
    trainer2.add_data(obs, bad, action_masks=masks, dones=torch.zeros(512, dtype=torch.bool))
    with pytest.raises(ValueError, match="non-integer"):
        trainer2.train(num_epochs=1)


def test_stateful_bc_uses_unroll_with_resets():
    obs, actions, dones = _memory_data(400)
    lstm = OfflineBCTrainer(make_simple_model(obs_dim=4, hidden_dim=32, num_actions=3, core="lstm"),
                            lr=5e-3, seq_len=16)
    lstm.add_data(obs, actions, dones=dones)
    lstm_metrics = lstm.train(num_epochs=25, batch_size=128)

    flat = OfflineBCTrainer(make_simple_model(obs_dim=4, num_actions=3), lr=5e-3)
    flat.add_data(obs, actions, dones=dones)
    flat_metrics = flat.train(num_epochs=25, batch_size=128)

    # The cue is visible only at step 0: memory is required for steps 1..3.
    assert lstm_metrics["accuracy"] >= 0.95
    assert flat_metrics["accuracy"] < 0.75


def test_missing_dones_warns_and_trains(caplog):
    trainer = OfflineBCTrainer(make_simple_model(obs_dim=4, core="lstm"), seq_len=8)
    with caplog.at_level(logging.WARNING, logger="colosseum.bc.offline_bc"):
        trainer.add_data(torch.randn(20, 4), torch.randint(0, 4, (20,)))
    assert "dones" in caplog.text
    metrics = trainer.train(num_epochs=1, batch_size=16)
    assert torch.isfinite(torch.tensor(metrics["bc_loss"]))


def test_load_data_from_files(tmp_path):
    for i in range(2):
        torch.save({
            "observations": torch.randn(10, 4),
            "actions": torch.randint(0, 4, (10,)),
            "action_masks": torch.ones(10, 4, dtype=torch.bool),
            "dones": torch.zeros(10, dtype=torch.bool),
        }, tmp_path / f"part{i}.pt")
    trainer = OfflineBCTrainer(make_simple_model(obs_dim=4))
    assert trainer.load_data(tmp_path) == 20
    assert trainer.num_samples == 20
    torch.save({"observations": torch.randn(3, 4)}, tmp_path / "broken.pt")
    with pytest.raises(ValueError, match="actions"):
        OfflineBCTrainer(make_simple_model(obs_dim=4)).load_data(tmp_path / "broken.pt")


def test_mixed_presence_of_masks_is_rejected():
    trainer = OfflineBCTrainer(make_simple_model(obs_dim=4))
    trainer.add_data(torch.randn(4, 4), torch.zeros(4, dtype=torch.long),
                     action_masks=torch.ones(4, 4, dtype=torch.bool), dones=torch.zeros(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="action_masks"):
        trainer.add_data(torch.randn(4, 4), torch.zeros(4, dtype=torch.long), dones=torch.zeros(4, dtype=torch.bool))


@pytest.mark.gpu
@pytest.mark.parametrize("core", ["none", "lstm"])
def test_bc_trains_on_cuda(core):
    torch.manual_seed(0)
    n = 64
    obs = torch.randn(n, 4)
    actions = obs.argmax(dim=1)
    masks = torch.ones(n, 4, dtype=torch.bool)
    model = make_simple_model(obs_dim=4, core=core, seed=0)
    trainer = OfflineBCTrainer(model, lr=1e-2, device="cuda", seq_len=8)
    trainer.add_data(obs, actions, action_masks=masks, dones=torch.arange(n) % 8 == 7)
    metrics = trainer.train(num_epochs=3, batch_size=16)
    assert all(p.device.type == "cuda" for p in model.parameters())
    assert torch.isfinite(torch.tensor(metrics["bc_loss"]))
    assert 0.0 <= metrics["accuracy"] <= 1.0
