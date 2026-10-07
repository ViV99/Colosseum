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


# ---------------------------------------------------------------------------
# Learner process entry: explicit thread call and device resolution
# ---------------------------------------------------------------------------

def _ttt_config(**sections):
    from colosseum.core.config import ColosseumConfig, load_config
    from helpers import example_config

    data = load_config(example_config("tic_tac_toe.yaml")).model_dump()
    data["metrics"]["use_wandb"] = False
    for section, values in sections.items():
        data[section].update(values)
    return ColosseumConfig(**data)


def _run_learner_target(monkeypatch, config, num_learners):
    """Call ``_learner_target`` in-process with ``learner_process`` stubbed out.

    Returns the torch thread count and the algorithm seen by the learner loop.
    """
    import sys

    import colosseum.learner.learner as learner_mod
    from colosseum.launcher import _learner_target

    seen = {}

    def fake_learner_process(**kwargs):
        seen["threads"] = torch.get_num_threads()
        seen["algorithm"] = kwargs["algorithm_factory"]()

    monkeypatch.setattr(learner_mod, "learner_process", fake_learner_process)
    monkeypatch.setattr(sys, "path", list(sys.path))  # _learner_target prepends "."
    before = torch.get_num_threads()
    try:
        _learner_target(
            agent_id="agent_0", config=config, trajectory_queue=None, weight_queues=[],
            stop_event=None, metrics_queue=None, total_train_steps=0, num_learners=num_learners,
        )
    finally:
        torch.set_num_threads(before)
    return seen["threads"], seen["algorithm"]


def test_learner_target_sets_explicit_torch_threads(monkeypatch):
    config = _ttt_config(learner={"device": "cpu", "torch_threads": 3})
    threads, algorithm = _run_learner_target(monkeypatch, config, num_learners=1)
    assert threads == 3
    assert next(algorithm.model.parameters()).device.type == "cpu"


def test_learner_target_auto_threads_and_auto_device_resolve_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = _ttt_config(learner={"device": "auto", "torch_threads": None},
                         rollout={"num_workers": 2, "torch_threads": 1})
    threads, algorithm = _run_learner_target(monkeypatch, config, num_learners=2)
    assert threads == resolve_learner_threads(None, "cpu", 2, 1, 2)
    assert next(algorithm.model.parameters()).device.type == "cpu"


def test_resolve_device_auto_without_cuda_is_cpu(monkeypatch):
    from colosseum.learner.learner import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"  # explicit devices pass through unchanged
