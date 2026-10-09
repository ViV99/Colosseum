"""SP2 performance switches: CPU ``pin_memory`` no-op and ``torch.compile`` of V-trace
(SP1's tests/unit/test_performance.py on the slot V-trace and chunk v2, T7.1)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.specs import ActionSpec
from colosseum.core.types import TrajectoryChunk
from game_helpers import chunk_v2_payload, learner_role, make_test_model


def _torch_compile_available() -> bool:
    """Check if a torch.compile backend works on this system."""
    try:
        fn = torch.compile(lambda x: x + 1)
        fn(torch.tensor(1.0))
        return True
    except Exception:
        return False


def _appo(**kwargs) -> tuple[APPO, list[TrajectoryChunk]]:
    torch.manual_seed(0)
    role = learner_role()
    algo_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in AlgorithmConfig.model_fields}
    appo = APPO(make_test_model(role), AlgorithmConfig(num_epochs=1, minibatch_chunks=0, **algo_kwargs),
                ActionSpec.from_space(role.action_space), device="cpu", **kwargs)
    chunks = [TrajectoryChunk.from_payload(chunk_v2_payload(S=8, version=v, pattern="AATAAARB")) for v in range(2)]
    return appo, chunks


def test_pin_memory_cpu_noop():
    """pin_memory=True on a CPU learner neither crashes nor changes the result."""
    pinned, chunks = _appo(pin_memory=True)
    plain, _ = _appo(pin_memory=False)
    metrics = pinned.train_step(chunks)
    assert "total_loss" in metrics and np.isfinite(metrics["total_loss"])
    assert metrics["total_loss"] == pytest.approx(plain.train_step(chunks)["total_loss"], rel=1e-5)


@pytest.mark.slow
def test_vtrace_slots_torch_compile():
    """Compiled slot V-trace gives the same targets as eager."""
    if not _torch_compile_available():
        pytest.skip("torch.compile backend unavailable")
    from colosseum.algorithms.vtrace import compute_vtrace_slots

    S, B = 8, 4
    torch.manual_seed(42)
    is_act = torch.ones(S, B, dtype=torch.bool)
    is_act[-1] = False          # the chunk-end BOOT
    is_act[5, 1:] = False       # a truncation BOOT / PAD in some columns
    terminal = torch.zeros(S, B, dtype=torch.bool)
    terminal[3, :] = True       # episode boundary at t=3
    kwargs = dict(log_rhos=torch.randn(S, B) * 0.3, rewards=torch.randn(S, B), values=torch.randn(S, B),
                  is_act=is_act, terminal=terminal, gamma=0.99, lam=0.95)

    eager = compute_vtrace_slots(**kwargs)
    compiled = torch.compile(compute_vtrace_slots)(**kwargs)
    for a, b in zip(eager, compiled, strict=True):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.slow
def test_appo_torch_compile():
    """APPO trains with algorithm.use_torch_compile=True."""
    if not _torch_compile_available():
        pytest.skip("torch.compile backend unavailable")
    appo, chunks = _appo(use_torch_compile=True)
    metrics = appo.train_step(chunks)
    assert "total_loss" in metrics and np.isfinite(metrics["total_loss"])
