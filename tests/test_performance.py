"""Tests for Phase 3 performance optimizations."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pytest
import torch

from colosseum.core.types import TrajectoryChunk


def _torch_compile_available() -> bool:
    """Check if torch.compile backend works on this system."""
    try:
        fn = torch.compile(lambda x: x + 1)
        fn(torch.tensor(1.0))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------
# T2.1: Pre-allocated rollout buffers
# ---------------------------------------------------------------

def test_preallocated_buffer_basic():
    """RolloutBuffer pre-allocates and produces correct chunks."""
    from colosseum.worker.rollout_worker import RolloutBuffer, _build_chunk

    buf = RolloutBuffer(
        chunk_length=4,
        obs_shape=(3, 3, 3),
        action_shape=(),
        action_dtype=np.int64,
        num_actions=0,
    )

    for i in range(4):
        buf.append(
            obs=np.full((3, 3, 3), fill_value=float(i), dtype=np.float32),
            action=i,
            log_prob=-0.5,
            reward=1.0,
            done=False,
            value=0.5,
        )
    assert buf.is_full
    assert buf.steps == 4

    chunk = _build_chunk(buf, "test", 0.0, 1)
    assert chunk.observations.shape == (4, 3, 3, 3)
    assert chunk.actions.shape == (4,)
    assert chunk.action_log_probs.shape == (4,)
    assert chunk.rewards.shape == (4,)
    # Verify data integrity
    assert chunk.observations[0, 0, 0, 0].item() == 0.0
    assert chunk.observations[3, 0, 0, 0].item() == 3.0
    assert chunk.actions[2].item() == 2
    assert chunk.action_masks is None


def test_preallocated_buffer_with_masks():
    """Action masks stored correctly in pre-allocated buffer."""
    from colosseum.worker.rollout_worker import RolloutBuffer, _build_chunk

    buf = RolloutBuffer(
        chunk_length=3,
        obs_shape=(4,),
        action_shape=(),
        action_dtype=np.int64,
        num_actions=5,
    )

    for i in range(3):
        mask = np.zeros(5, dtype=bool)
        mask[i] = True
        mask[4] = True
        buf.append(
            obs=np.zeros(4, dtype=np.float32),
            action=i,
            log_prob=-1.0,
            reward=0.0,
            done=False,
            value=0.0,
            action_mask=mask,
        )

    assert buf.is_full
    assert buf.has_masks

    chunk = _build_chunk(buf, "test", 0.0, 0)
    assert chunk.action_masks is not None
    assert chunk.action_masks.shape == (3, 5)
    assert chunk.action_masks[0, 0].item() is True
    assert chunk.action_masks[0, 1].item() is False
    assert chunk.action_masks[1, 1].item() is True


def test_buffer_reset_reuse():
    """Buffer reset + refill produces correct second chunk."""
    from colosseum.worker.rollout_worker import RolloutBuffer, _build_chunk

    buf = RolloutBuffer(
        chunk_length=2,
        obs_shape=(2,),
        action_shape=(),
        action_dtype=np.int64,
    )

    # First fill
    for i in range(2):
        buf.append(obs=np.array([float(i), 0.0]), action=i, log_prob=0.0,
                   reward=0.0, done=False, value=0.0)
    chunk1 = _build_chunk(buf, "test", 0.0, 0)
    buf.reset()

    assert buf.steps == 0
    assert not buf.is_full

    # Second fill with different data
    for i in range(2):
        buf.append(obs=np.array([10.0 + i, 0.0]), action=i + 10, log_prob=0.0,
                   reward=0.0, done=False, value=0.0)
    chunk2 = _build_chunk(buf, "test", 0.0, 1)

    # Verify chunks are independent (no shared memory)
    assert chunk1.observations[0, 0].item() == 0.0
    assert chunk2.observations[0, 0].item() == 10.0
    assert chunk1.actions[0].item() == 0
    assert chunk2.actions[0].item() == 10


# ---------------------------------------------------------------
# T2.3+T2.4: Pinned memory (CPU noop test)
# ---------------------------------------------------------------

def test_pin_memory_cpu_noop():
    """pin_memory=True on CPU doesn't crash, produces valid results."""
    from colosseum.algorithms.appo import APPO
    from colosseum.core.config import AlgorithmConfig

    from tests.test_action_masking import _make_network

    net = _make_network(obs_dim=4, num_actions=3)
    config = AlgorithmConfig(name="appo", num_epochs=1, minibatch_chunks=0)
    appo = APPO(net, config, device="cpu", pin_memory=True)

    chunks = []
    for _ in range(2):
        chunks.append(TrajectoryChunk(
            agent_id="test",
            observations=torch.randn(4, 4),
            actions=torch.randint(0, 3, (4,)),
            action_log_probs=torch.randn(4),
            rewards=torch.randn(4),
            dones=torch.zeros(4),
            values=torch.randn(4),
            bootstrap_value=torch.tensor(0.0),
            behavior_policy_version=0,
        ))

    metrics = appo.train_step(chunks)
    assert "total_loss" in metrics
    assert np.isfinite(metrics["total_loss"])


# ---------------------------------------------------------------
# T2.7: torch.compile for V-trace
# ---------------------------------------------------------------

@pytest.mark.skipif(not _torch_compile_available(), reason="torch.compile backend unavailable")
def test_vtrace_torch_compile():
    """Compiled V-trace should produce same results as eager."""
    from colosseum.algorithms.vtrace import compute_vtrace

    T, B = 8, 4
    torch.manual_seed(42)
    behavior_lp = torch.randn(T, B)
    target_lp = torch.randn(T, B)
    rewards = torch.randn(T, B)
    values = torch.randn(T, B)
    bootstrap = torch.randn(B)
    dones = torch.zeros(T, B)
    dones[3, :] = 1.0  # episode boundary at t=3

    vs_eager, adv_eager = compute_vtrace(
        behavior_lp, target_lp, rewards, values, bootstrap, dones,
    )

    compiled_fn = torch.compile(compute_vtrace)
    vs_compiled, adv_compiled = compiled_fn(
        behavior_lp, target_lp, rewards, values, bootstrap, dones,
    )

    assert torch.allclose(vs_eager, vs_compiled, atol=1e-5)
    assert torch.allclose(adv_eager, adv_compiled, atol=1e-5)


@pytest.mark.skipif(not _torch_compile_available(), reason="torch.compile backend unavailable")
def test_appo_torch_compile():
    """APPO should train successfully with use_torch_compile=True."""
    from colosseum.algorithms.appo import APPO
    from colosseum.core.config import AlgorithmConfig

    from tests.test_action_masking import _make_network

    net = _make_network(obs_dim=4, num_actions=3)
    config = AlgorithmConfig(
        name="appo", num_epochs=1, minibatch_chunks=0, use_torch_compile=True,
    )
    appo = APPO(net, config, device="cpu")

    chunks = []
    for _ in range(2):
        chunks.append(TrajectoryChunk(
            agent_id="test",
            observations=torch.randn(4, 4),
            actions=torch.randint(0, 3, (4,)),
            action_log_probs=torch.randn(4),
            rewards=torch.randn(4),
            dones=torch.zeros(4),
            values=torch.randn(4),
            bootstrap_value=torch.tensor(0.0),
            behavior_policy_version=0,
        ))

    metrics = appo.train_step(chunks)
    assert "total_loss" in metrics
    assert np.isfinite(metrics["total_loss"])
