"""SP2 learner entry points: per-agent seed streams and torch threads (SP1 guarantees, T7.1)."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from colosseum.core.config import ColosseumConfig, load_config
from colosseum.utils.seeding import derive_seed, learner_seed
from game_helpers import agent_role_of, make_test_config, make_test_run_dir, write_test_config


def _flat(state: dict) -> np.ndarray:
    return np.concatenate([v.detach().cpu().numpy().ravel() for v in state.values()])


@pytest.fixture
def capture_learner_model(monkeypatch):
    """Replace learner_process with one that builds the algorithm and records its initial weights."""
    import colosseum.core.threads as threads_module
    import colosseum.learner.learner as learner_module

    captured: list[np.ndarray] = []

    def fake_learner_process(*, algorithm_factory, **kwargs):
        captured.append(_flat(algorithm_factory().model.state_dict()))

    monkeypatch.setattr(learner_module, "learner_process", fake_learner_process)
    monkeypatch.setattr(threads_module, "configure_torch_threads", lambda *a, **k: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return captured


def test_learner_seed_streams_are_deterministic_and_distinct():
    assert learner_seed(None, 0) is None
    assert learner_seed(7, 0) == learner_seed(7, 0) == derive_seed(7, "learner", 0)
    assert learner_seed(7, 0) != learner_seed(7, 1)
    assert learner_seed(7, 0) != learner_seed(8, 0)
    worker_streams = {7 + w * 1000 for w in range(64)}
    assert not {learner_seed(7, i) for i in range(16)} & worker_streams
    assert all(0 <= learner_seed(s, i) < 2**32 for s in (-5, 0, 2**40) for i in range(3))


def _learner_main(config: ColosseumConfig, agent_id: str = "agent_0", **kwargs) -> None:
    from colosseum.launcher import _learner_main as main

    _roles, role = agent_role_of(config, agent_id)
    main(agent_id=agent_id, config=config.get_agent_config(agent_id), role_spec=role, trajectory_queue=None,
         weight_queues=[], stop_event=None, metrics_queue=None, **kwargs)


def test_local_learner_weights_follow_training_seed(capture_learner_model, restore_global_rng):
    cfg = make_test_config("turns", training={"seed": 3})

    def run(seed):
        torch.manual_seed(12345 + len(capture_learner_model))  # a different global state each time
        _learner_main(cfg, seed=seed)
        return capture_learner_model[-1]

    a, b, c = run(learner_seed(3, 0)), run(learner_seed(3, 0)), run(learner_seed(4, 0))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert not np.array_equal(a, run(learner_seed(3, 1)))


def test_launcher_gives_each_learner_its_own_seed_stream(tmp_path, monkeypatch, restore_global_rng):
    import colosseum.launcher as launcher_module

    learner_kwargs: list[dict] = []

    class _Stop(Exception):
        pass

    class FakeProcess:
        exitcode = 0

        def __init__(self, target, kwargs, name=None, daemon=None):
            self.target, self.kwargs = target, kwargs

        def start(self):
            if self.target is launcher_module._learner_target:
                learner_kwargs.append(self.kwargs)
            else:
                raise _Stop  # first worker: every learner has been started

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(launcher_module.mp, "Process", FakeProcess)
    cfg = make_test_config("asymmetric", training={"seed": 11})
    run = make_test_run_dir(cfg, tmp_path, name="seeded")
    with pytest.raises(_Stop):
        launcher_module.Launcher(cfg, run).launch()
    assert [k["agent_id"] for k in learner_kwargs] == ["hunter", "prey"]
    assert [k["log_dir"] for k in learner_kwargs] == [str(run.logs)] * 2
    assert [k["seed"] for k in learner_kwargs] == [learner_seed(11, 0), learner_seed(11, 1)]
    assert [k["role_spec"] for k in learner_kwargs] == [agent_role_of(cfg, a)[1] for a in ("hunter", "prey")]
    assert learner_kwargs[0]["checkpoint_interval"] == cfg.checkpoint.interval

    learner_kwargs.clear()
    unseeded = make_test_config("asymmetric", training={"seed": None})
    with pytest.raises(_Stop):
        launcher_module.Launcher(unseeded, make_test_run_dir(unseeded, tmp_path, name="unseeded")).launch()
    assert [k["seed"] for k in learner_kwargs] == [None, None]


def test_run_learner_seeds_before_building_the_model(tmp_path, monkeypatch, capture_learner_model,
                                                      restore_global_rng, restore_root_logging):
    import colosseum.distributed as distributed
    import colosseum.transport.grpc_transport as grpc_transport
    import colosseum.weight_store.grpc_store as grpc_store

    class FakeServer:
        def stop(self, grace):
            pass

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(grpc_transport, "serve_trajectory_receiver", lambda *a, **k: FakeServer())
    monkeypatch.setattr(grpc_store, "GRPCWeightStore", FakeStore)
    monkeypatch.setattr(distributed.ProcessSupervisor, "install_signal_handlers", lambda self: None)
    path = write_test_config(tmp_path / "cfg.yaml", "turns", agents={"alpha": {}, "beta": {}})

    def run(agent_id, seed):
        torch.manual_seed(999 + len(capture_learner_model))
        distributed.run_distributed_learner(str(path), agent_id, 0, "localhost:1",
                                            overrides={"training.seed": seed, "run.dir": str(tmp_path / "runs")})
        return capture_learner_model[-1]

    a, b = run("alpha", 5), run("alpha", 5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, run("alpha", 6))
    assert not np.array_equal(a, run("beta", 5))  # distinct per-agent stream
    # Every auto-named learner role dir is <base run name>-learner-<agent>, and its resolved
    # config records the base name (re-running it reproduces the layout).
    role_dirs = sorted((tmp_path / "runs").iterdir())
    assert len(role_dirs) == 4
    for role_dir in role_dirs:
        resolved = load_config(role_dir / "config.resolved.yaml")
        assert role_dir.name in (f"{resolved.run.name}-learner-alpha", f"{resolved.run.name}-learner-beta")
        assert "learner" not in resolved.run.name
    cfg = make_test_config("turns", agents={"alpha": {}, "beta": {}}, training={"seed": 5})
    _learner_main(cfg, agent_id="alpha", seed=learner_seed(5, 0))  # same stream as a local-mode learner
    assert np.array_equal(a, capture_learner_model[-1])


def _run_learner_target(monkeypatch, config, num_learners, seen=None):
    import colosseum.learner.learner as learner_mod

    seen = {} if seen is None else seen

    def fake_learner_process(**kwargs):
        seen["kwargs"] = kwargs
        seen["threads"] = torch.get_num_threads()
        seen["algorithm"] = kwargs["algorithm_factory"]()

    monkeypatch.setattr(learner_mod, "learner_process", fake_learner_process)
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = torch.get_num_threads()
    try:
        _learner_main(config, num_learners=num_learners)
    finally:
        torch.set_num_threads(before)
    return seen["threads"], seen["algorithm"]


def test_learner_target_sets_explicit_torch_threads(monkeypatch):
    config = make_test_config("turns", learner={"device": "cpu", "torch_threads": 3})
    threads, algorithm = _run_learner_target(monkeypatch, config, num_learners=1)
    assert threads == 3 and next(algorithm.model.parameters()).device.type == "cpu"


def test_learner_target_auto_threads_and_auto_device_resolve_to_cpu(monkeypatch):
    import colosseum.core.threads as threads_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(threads_mod.os, "cpu_count", lambda: 8)
    config = make_test_config("turns", learner={"device": "auto", "torch_threads": None},
                              rollout={"num_workers": 2, "torch_threads": 1})
    threads, algorithm = _run_learner_target(monkeypatch, config, num_learners=2)
    assert threads == 3  # (8 - 2 workers * 1 thread) // 2 learners
    assert next(algorithm.model.parameters()).device.type == "cpu"


def test_learner_target_passes_weight_sync_interval(monkeypatch):
    config = make_test_config("turns", learner={"device": "cpu", "torch_threads": 1},
                              rollout={"weight_sync_interval_sec": 90.0})
    seen = {}
    _run_learner_target(monkeypatch, config, num_learners=1, seen=seen)
    assert seen["kwargs"]["weight_sync_interval"] == 90.0


def test_learner_target_builds_and_uses_the_kickstart_teacher(monkeypatch, tmp_path, restore_global_rng):
    """``training.kickstart_teacher``: the learner loads the teacher for the agent's role and the
    algorithm adds its KL term (the ``_learner_main`` kickstart branch, T5.4 review item)."""
    from colosseum.core.registry import build_model
    from game_helpers import synthetic_chunk

    base = make_test_config("turns", learner={"device": "cpu", "torch_threads": 1})
    _roles, role = agent_role_of(base, "agent_0")
    torch.manual_seed(7)
    source = build_model(base, role)
    with torch.no_grad():
        for p in source.parameters():
            p.add_(0.5)  # the teacher differs from any freshly initialized student
    teacher_path = tmp_path / "teacher.pt"
    torch.save(source.state_dict(), teacher_path)
    config = make_test_config("turns", learner={"device": "cpu", "torch_threads": 1},
                              training={"kickstart_teacher": str(teacher_path), "kickstart_lambda": 0.5,
                                        "kickstart_decay_steps": 10, "kickstart_kl": "reverse"})

    _threads, algorithm = _run_learner_target(monkeypatch, config, num_learners=1)
    kickstart = algorithm._kickstart
    assert kickstart is not None and kickstart.direction == "reverse" and kickstart.current_lambda == 0.5
    assert kickstart.teacher is not algorithm.model
    assert np.array_equal(_flat(kickstart.teacher.state_dict()), _flat(source.state_dict()))
    assert not np.array_equal(_flat(algorithm.model.state_dict()), _flat(source.state_dict()))
    assert not any(p.requires_grad for p in kickstart.teacher.parameters())

    chunks = [synthetic_chunk(algorithm.model, role, "AAAB", seed=s, agent_id="agent_0") for s in (0, 1)]
    metrics = algorithm.train_step(chunks)
    assert metrics["kickstart_lambda"] == 0.5 and metrics["kickstart_loss"] > 0.0
    assert kickstart.step_count == 1

    _threads, plain = _run_learner_target(monkeypatch, make_test_config(
        "turns", learner={"device": "cpu", "torch_threads": 1}), num_learners=1)
    assert plain._kickstart is None and "kickstart_loss" not in plain.train_step(chunks)
