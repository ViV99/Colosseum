"""APPO diagnostic metrics (T4.6)."""
import math

import pytest
import torch

from colosseum.algorithms.appo import APPO, _explained_variance
from colosseum.core.config import AlgorithmConfig
from colosseum.networks.model import PolicyModel
from colosseum.networks.state import tree_leaves
from helpers import MaskedToyEnv, make_simple_model, rollout_chunks

NEW_KEYS = ("explained_variance", "grad_norm", "rho_mean", "rho_clip_frac", "lr")


def tiny_model(core: str = "none", seed: int | None = None) -> PolicyModel:
    """Model matching MaskedToyEnv (4-dim observation, Discrete(4))."""
    return make_simple_model(obs_dim=4, num_actions=4, core=core, seed=seed)


def test_explained_variance_helper():
    target = torch.randn(100)
    assert _explained_variance(target.clone(), target).item() == pytest.approx(1.0)
    assert _explained_variance(torch.full((100,), float(target.mean())), target).item() == pytest.approx(0.0, abs=1e-5)
    assert _explained_variance(torch.randn(10), torch.ones(10)).item() == 0.0   # constant target


def test_appo_reports_diagnostic_metrics_on_policy():
    cfg = AlgorithmConfig(num_epochs=1, minibatch_chunks=0, vtrace_rho_bar=1.5,
                          lr_schedule="constant", learning_rate=3e-4)
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    metrics = APPO(model, cfg, device="cpu").train_step(chunks)
    for key in NEW_KEYS:
        assert key in metrics and math.isfinite(metrics[key]), key
    assert "learning_rate" not in metrics
    assert metrics["lr"] == pytest.approx(3e-4)
    # First update on fresh chunks from identical weights: pi == mu.
    assert metrics["rho_mean"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["rho_clip_frac"] == 0.0
    assert metrics["grad_norm"] > 0.0
    assert metrics["explained_variance"] <= 1.0


def test_grad_norm_is_measured_before_clipping():
    cfg = AlgorithmConfig(num_epochs=1, minibatch_chunks=0, max_grad_norm=1e-6)
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    metrics = APPO(model, cfg, device="cpu").train_step(chunks)
    assert metrics["grad_norm"] > 1e-3


def test_no_skipped_updates_without_overflow():
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    metrics = APPO(model, AlgorithmConfig(num_epochs=2, minibatch_chunks=2), device="cpu").train_step(chunks)
    assert metrics["skipped_updates"] == 0.0
    assert math.isfinite(metrics["grad_norm"]) and metrics["grad_norm"] > 0.0


def test_scaler_skipped_updates_are_counted_and_excluded_from_grad_norm():
    """An infinite loss scale makes every scaled gradient non-finite: GradScaler skips each
    optimizer step, and grad_norm (finite norms only) has nothing to average."""
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    algo = APPO(model, AlgorithmConfig(num_epochs=1, minibatch_chunks=2), device="cpu")
    algo._scaler = torch.amp.GradScaler("cpu", init_scale=float("inf"))
    before = {k: v.detach().clone() for k, v in algo.model.state_dict().items()}
    metrics = algo.train_step(chunks)
    assert metrics["skipped_updates"] == 2.0
    assert math.isnan(metrics["grad_norm"])
    assert math.isfinite(metrics["total_loss"])
    for key, value in algo.model.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_compute_loss_reports_the_new_per_minibatch_metrics():
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=2, chunk_length=8)
    losses = APPO(model, AlgorithmConfig(), device="cpu").compute_loss(chunks)
    for key in ("explained_variance", "rho_mean", "rho_clip_frac"):
        assert key in losses and losses[key].dim() == 0, key


# ---------------------------------------------------------------------------
# CUDA (spec 3.8): AMP float16 / bfloat16 train steps, pinned batch preparation
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("amp_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("core", ["none", "lstm"])
def test_amp_train_step_on_cuda(amp_dtype, core):
    model = tiny_model(core=core)
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    algo = APPO(model, AlgorithmConfig(num_epochs=2, minibatch_chunks=2, use_amp=True, amp_dtype=amp_dtype),
                device="cuda")
    assert algo._use_amp
    if amp_dtype == "float16":
        assert algo._scaler is not None
    before = {k: v.detach().clone() for k, v in algo.model.state_dict().items()}
    metrics = None
    for _ in range(3):
        metrics = algo.train_step(chunks)
    for key in ("total_loss", "policy_loss", "value_loss", "entropy", "rho_mean", "explained_variance", "lr"):
        assert math.isfinite(metrics[key]), key
    assert metrics["policy_version"] == 3.0
    assert 0.0 <= metrics["skipped_updates"] <= 4.0          # 2 epochs x 2 minibatches per step
    if amp_dtype == "float16":
        scale = algo._scaler.get_scale()
        assert math.isfinite(scale) and scale > 0.0
    changed = any(not torch.equal(before[k], v) for k, v in algo.model.state_dict().items())
    assert changed, "three AMP train steps never updated the weights"
    for p in algo.model.parameters():
        assert p.dtype == torch.float32 and torch.isfinite(p).all()


@pytest.mark.gpu
@pytest.mark.parametrize("core", ["none", "lstm"])
def test_pin_memory_batch_preparation_on_cuda(core):
    model = tiny_model(core=core)
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    pinned = APPO(model, AlgorithmConfig(), device="cuda", pin_memory=True)
    plain_batch = APPO(tiny_model(core=core), AlgorithmConfig(), device="cuda")._prepare_batch(chunks)
    batch = pinned._prepare_batch(chunks)
    assert batch.keys() == plain_batch.keys()
    for key, value in batch.items():
        if key == "initial_state":
            assert all(t.device.type == "cuda" for t in tree_leaves(value))
            continue
        assert value.device.type == "cuda", key
        assert torch.equal(value, plain_batch[key]), key
    metrics = pinned.train_step(chunks)
    assert all(math.isfinite(v) for v in metrics.values())
