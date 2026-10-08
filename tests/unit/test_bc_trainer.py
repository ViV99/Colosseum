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


# --- Fix round 1 -----------------------------------------------------------


def test_stateful_windows_reset_state_after_dones():
    model = make_simple_model(obs_dim=4, core="lstm", seed=0)
    trainer = OfflineBCTrainer(model, seq_len=8)
    obs = torch.randn(8, 4, generator=torch.Generator().manual_seed(1))
    index = _window_index(8, 8).t()                                   # [L=8, b=1]
    dones = torch.zeros(8, dtype=torch.bool)
    dones[3] = True
    with torch.no_grad():
        reset_dist, _ = trainer._sequence_forward(index, obs, None, dones)
        carried_dist, _ = trainer._sequence_forward(index, obs, None, torch.zeros(8, dtype=torch.bool))
        fresh = model.step(obs[4:5], model.initial_state(1)).dist
    # After dones[3] the state is reset: step 4 sees exactly what a fresh episode start sees.
    assert torch.allclose(reset_dist.logits[4], fresh.logits[0], atol=1e-6)
    # Sanity: without the reset the carried state changes the output at step 4.
    assert not torch.allclose(carried_dist.logits[4], fresh.logits[0], atol=1e-4)


def test_masks_are_applied_for_stateful_models():
    n = 64
    actions = torch.randint(0, 4, (n,))
    masks = torch.nn.functional.one_hot(actions, 4).bool()
    trainer = OfflineBCTrainer(make_simple_model(obs_dim=4, core="lstm"), seq_len=8)
    trainer.add_data(torch.randn(n, 4), actions, action_masks=masks, dones=torch.arange(n) % 5 == 4)
    metrics = trainer.train(num_epochs=1, batch_size=16)
    assert metrics["bc_loss_first_epoch"] < 1e-6
    assert metrics["accuracy"] == 1.0


def test_bc_updates_obs_normalizers_once_per_sample():
    torch.manual_seed(0)
    n = 5000                                                          # > one 4096-sample update chunk
    obs = torch.randn(n, 4) * torch.tensor([2.0, 0.5, 1.0, 3.0]) + torch.tensor([5.0, -3.0, 1.0, 0.5])
    actions = torch.randint(0, 4, (n,))
    model = make_simple_model(obs_dim=4, normalize=True, seed=0)
    rms = model.encoder.norm.rms
    trainer = OfflineBCTrainer(model)
    trainer.add_data(obs, actions, dones=torch.zeros(n, dtype=torch.bool))
    trainer.train(num_epochs=1, batch_size=1024)
    assert rms.count.item() == pytest.approx(n, abs=1e-2)
    assert torch.allclose(rms.mean, obs.mean(0), atol=1e-4)
    assert torch.allclose(rms.var, obs.var(0, unbiased=False), rtol=1e-3)

    trainer.train(num_epochs=1, batch_size=1024)                     # no new data: no double count
    assert rms.count.item() == pytest.approx(n, abs=1e-2)

    more = torch.randn(1000, 4) - 2.0
    trainer.add_data(more, torch.randint(0, 4, (1000,)), dones=torch.zeros(1000, dtype=torch.bool))
    trainer.train(num_epochs=1, batch_size=1024)
    everything = torch.cat([obs, more])
    assert rms.count.item() == pytest.approx(n + 1000, abs=1e-2)
    assert torch.allclose(rms.mean, everything.mean(0), atol=1e-4)


def test_mask_width_must_match_the_policy():
    trainer = OfflineBCTrainer(make_simple_model(obs_dim=4))         # Categorical over 4 actions
    trainer.add_data(torch.randn(8, 4), torch.zeros(8, dtype=torch.long),
                     action_masks=torch.ones(8, 2, dtype=torch.bool), dones=torch.zeros(8, dtype=torch.bool))
    with pytest.raises(ValueError, match="action_masks have width 2.*expects 4"):
        trainer.train(num_epochs=1)

    obs = torch.randn(8, 4)
    actions = torch.cat([torch.zeros(8, 1), torch.randn(8, 1)], dim=1)
    trainer = OfflineBCTrainer(_composed(DirSpeedPolicy(16)))       # flat mask width 3
    trainer.add_data(obs, actions, action_masks=torch.ones(8, 7, dtype=torch.bool),
                     dones=torch.zeros(8, dtype=torch.bool))
    with pytest.raises(ValueError, match="action_masks have width 7.*expects 3"):
        trainer.train(num_epochs=1)

    trainer = OfflineBCTrainer(_composed(GaussPolicy(16, 2)))
    trainer.add_data(obs, torch.randn(8, 2), action_masks=torch.ones(8, 2, dtype=torch.bool),
                     dones=torch.zeros(8, dtype=torch.bool))
    with pytest.raises(ValueError, match="no discrete"):
        trainer.train(num_epochs=1)


def test_missing_dones_warns_only_for_stateful_models(caplog):
    with caplog.at_level(logging.WARNING, logger="colosseum.bc.offline_bc"):
        OfflineBCTrainer(make_simple_model(obs_dim=4)).add_data(torch.randn(5, 4), torch.zeros(5, dtype=torch.long))
    assert "dones" not in caplog.text


def test_every_add_data_call_ends_an_episode():
    trainer = OfflineBCTrainer(make_simple_model(obs_dim=4, core="lstm"))
    user_dones = torch.zeros(5, dtype=torch.bool)
    trainer.add_data(torch.randn(5, 4), torch.zeros(5, dtype=torch.long), dones=user_dones)
    trainer.add_data(torch.randn(3, 4), torch.zeros(3, dtype=torch.long), dones=torch.zeros(3, dtype=torch.bool))
    assert torch.nonzero(trainer._dataset()[3]).flatten().tolist() == [4, 7]
    assert not user_dones.any()                                        # caller's tensor is not mutated


class _FixedLogProb:
    def __init__(self, values):
        self._values = torch.tensor(values)

    def log_prob(self, actions):
        return self._values


def test_non_finite_nll_errors_name_the_right_cause():
    trainer = OfflineBCTrainer(make_simple_model(obs_dim=4))
    weights = torch.ones(2)
    with pytest.raises(RuntimeError, match="NaN.*diverged"):
        trainer._weighted_nll(_FixedLogProb([float("nan"), 0.0]), None, weights, masked=True)
    with pytest.raises(ValueError, match="illegal under their action_masks"):
        trainer._weighted_nll(_FixedLogProb([float("-inf"), 0.0]), None, weights, masked=True)
    with pytest.raises(ValueError, match="no action_masks") as info:
        trainer._weighted_nll(_FixedLogProb([float("-inf"), 0.0]), None, weights, masked=False)
    assert "illegal" not in str(info.value)
    # Padding rows (weight 0) never raise.
    loss, count = trainer._weighted_nll(_FixedLogProb([float("-inf"), -1.0]), None, torch.tensor([0.0, 1.0]), False)
    assert loss.item() == pytest.approx(1.0) and count.item() == 1.0
