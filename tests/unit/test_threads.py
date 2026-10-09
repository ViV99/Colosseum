"""Torch thread limits: config fields and the learner auto-thread formula (T2.1)."""
import pytest
import torch
from pydantic import ValidationError

from colosseum.core.config import LearnerConfig, RolloutConfig
from colosseum.core.threads import configure_torch_threads, resolve_learner_threads


def test_config_thread_fields_defaults_and_validation():
    assert RolloutConfig().torch_threads == 1
    assert LearnerConfig().torch_threads is None
    with pytest.raises(ValidationError):
        RolloutConfig(torch_threads=0)
    with pytest.raises(ValidationError):
        LearnerConfig(torch_threads=0)


@pytest.mark.parametrize(
    "torch_threads, device, workers, worker_threads, learners, cpus, expected",
    [
        (None, "cpu", 4, 1, 1, 8, 4),   # (8 - 4*1) // 1
        (None, "cpu", 2, 1, 2, 8, 3),   # (8 - 2*1) // 2
        (None, "cpu", 4, 2, 1, 8, 1),   # max(1, 0)
        (None, "cpu", 8, 1, 3, 4, 1),   # negative remainder -> 1
        (None, "cuda", 4, 1, 1, 8, 2),
        (None, "cuda:1", 4, 1, 1, 8, 2),
        (5, "cpu", 4, 1, 1, 8, 5),      # an explicit value wins
        (3, "cuda", 4, 1, 1, 8, 3),
    ],
)
def test_resolve_learner_threads(torch_threads, device, workers, worker_threads, learners, cpus, expected):
    got = resolve_learner_threads(torch_threads, device, workers, worker_threads, learners, cpu_count=cpus)
    assert got == expected


def test_configure_torch_threads_sets_intra_op_threads():
    before = torch.get_num_threads()
    try:
        configure_torch_threads(2)
        assert torch.get_num_threads() == 2
        configure_torch_threads(1)  # a second call must not raise (inter-op already fixed)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)


def test_configure_torch_threads_warns_when_interop_limit_is_lost(monkeypatch, caplog):
    import colosseum.core.threads as threads_mod

    def too_late(n):
        raise RuntimeError("cannot set number of interop threads after parallel work has started")

    monkeypatch.setattr(threads_mod.torch, "set_num_interop_threads", too_late)
    monkeypatch.setattr(threads_mod.torch, "get_num_interop_threads", lambda: 8)
    before = torch.get_num_threads()
    try:
        with caplog.at_level("WARNING", logger="colosseum.core.threads"):
            configure_torch_threads(1)
    finally:
        torch.set_num_threads(before)
    messages = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(messages) == 1
    assert "8" in messages[0] and "requested 1" in messages[0]
    assert "must run before any torch work" in messages[0]


def test_configure_torch_threads_is_silent_when_interop_matches(monkeypatch, caplog):
    import colosseum.core.threads as threads_mod

    monkeypatch.setattr(threads_mod.torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(threads_mod.torch, "get_num_interop_threads", lambda: 1)
    before = torch.get_num_threads()
    try:
        with caplog.at_level("WARNING", logger="colosseum.core.threads"):
            configure_torch_threads(1)
    finally:
        torch.set_num_threads(before)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


# ---------------------------------------------------------------------------
# Learner device resolution (the learner-entry thread tests are in test_sp2_learner_entry.py)
# ---------------------------------------------------------------------------


def test_resolve_device_auto_without_cuda_is_cpu(monkeypatch):
    from colosseum.learner.learner import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"  # explicit devices pass through unchanged
