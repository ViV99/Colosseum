"""Tests for Phase 3 performance optimizations."""



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
# T2.3+T2.4: Pinned memory (CPU noop test)
# ---------------------------------------------------------------

def test_pin_memory_cpu_noop():
    """pin_memory=True on CPU doesn't crash, produces valid results."""
    from colosseum.algorithms.appo import APPO
    from colosseum.core.config import AlgorithmConfig
    from helpers import make_simple_model

    net = make_simple_model(obs_dim=4, hidden_dim=32, num_actions=3)
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

@pytest.mark.slow
def test_vtrace_torch_compile():
    """Compiled V-trace should produce same results as eager."""
    if not _torch_compile_available():
        pytest.skip("torch.compile backend unavailable")
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


@pytest.mark.slow
def test_appo_torch_compile():
    """APPO should train successfully with use_torch_compile=True."""
    if not _torch_compile_available():
        pytest.skip("torch.compile backend unavailable")
    from colosseum.algorithms.appo import APPO
    from colosseum.core.config import AlgorithmConfig
    from helpers import make_simple_model

    net = make_simple_model(obs_dim=4, hidden_dim=32, num_actions=3)
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
