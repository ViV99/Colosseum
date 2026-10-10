"""learner/factory.py (spec block 6): teachers are resolved in the main process into numpy specs and
built in the learner; the launcher and the distributed learner build their algorithms there (T4.1)."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from colosseum.core.errors import ConfigError
from colosseum.core.ipc import assert_no_tensors
from colosseum.core.registry import build_model, env_spec
from colosseum.learner.factory import TeacherSpec, build_algorithm, resolve_teacher
from game_helpers import agent_role_of, make_test_config, make_test_run_dir, write_ckpt_dir, write_test_config

BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}


def _flat(state: dict) -> np.ndarray:
    return np.concatenate([v.detach().cpu().numpy().ravel() for v in state.values()])


def _teacher_pt(tmp_path, config, agent_id: str = "agent_0"):
    """A teacher .pt with the agent's architecture, different from any fresh model."""
    _roles, role = agent_role_of(config, agent_id)
    torch.manual_seed(7)
    source = build_model(config.get_agent_config(agent_id), role)
    with torch.no_grad():
        for p in source.parameters():
            p.add_(0.5)
    path = tmp_path / f"{agent_id}_teacher.pt"
    torch.save(source.state_dict(), path)
    return path, source


def _build(config, agent_id, teacher):
    spec = env_spec(config)
    _roles, role = agent_role_of(config, agent_id)
    return build_algorithm(config.get_agent_config(agent_id), role, spec, device="cpu", teacher=teacher)


def test_no_teacher_builds_plain_appo():
    cfg = make_test_config("turns")
    assert resolve_teacher(cfg, "agent_0", env_spec(cfg)) is None
    algo = _build(cfg, "agent_0", None)
    assert algo._kickstart is None and algo.policy_version == 0


def test_a_pt_teacher_has_the_students_architecture_and_the_kickstart_settings(tmp_path):
    base = make_test_config("turns")
    path, source = _teacher_pt(tmp_path, base)
    cfg = make_test_config("turns", kickstart={"teacher": str(path), "lambda": 0.5, "decay_steps": 10, "kl": "reverse"})
    teacher = resolve_teacher(cfg, "agent_0", env_spec(cfg))
    assert isinstance(teacher, TeacherSpec) and teacher.kind == "neural" and teacher.bot is None
    assert (teacher.lambda_, teacher.decay_steps, teacher.kl, teacher.source) == (0.5, 10, "reverse", str(path))
    assert teacher.frozen.roles == ("player",)
    assert teacher.frozen.networks["model_class"] == "game_helpers.GameTestModel"
    assert_no_tensors(teacher, "teacher spec")                  # crosses into the learner process
    algo = _build(cfg, "agent_0", teacher)
    kick = algo._kickstart
    assert kick.direction == "reverse" and kick.current_lambda == 0.5
    assert kick.teacher is not algo.model and not any(p.requires_grad for p in kick.teacher.parameters())
    assert np.array_equal(_flat(kick.teacher.state_dict()), _flat(source.state_dict()))


def test_a_frozen_agent_is_a_teacher_by_name(tmp_path):
    base = make_test_config("turns")
    ckpt = write_ckpt_dir(tmp_path / "prev", base, seed=3)
    cfg = make_test_config("turns", agents={"agent_0": {"kickstart": {"teacher": "prev"}},
                                            "prev": {"kind": "frozen", "path": str(ckpt)}},
                           matchmaking={"anchors": []})
    teacher = resolve_teacher(cfg, "agent_0", env_spec(cfg))
    assert teacher.kind == "neural" and "prev" in teacher.source
    torch.manual_seed(3)
    expected = build_model(base.get_agent_config("agent_0"), agent_role_of(base, "agent_0")[1])
    assert np.array_equal(_flat(_build(cfg, "agent_0", teacher)._kickstart.teacher.state_dict()),
                          _flat(expected.state_dict()))


def test_a_checkpoint_dir_path_is_a_teacher(tmp_path):
    base = make_test_config("turns")
    ckpt = write_ckpt_dir(tmp_path / "prev", base, seed=4)
    cfg = make_test_config("turns", kickstart={"teacher": str(ckpt)})
    teacher = resolve_teacher(cfg, "agent_0", env_spec(cfg))
    assert teacher.kind == "neural" and teacher.frozen.roles == ("player",)


@pytest.mark.parametrize("teacher, message", [
    ("agent_0", "trainable agent"),
    ("missing_file.pt", "neither an agent"),
])
def test_bad_teacher_references_are_config_errors(teacher, message):
    cfg = make_test_config("turns", kickstart={"teacher": teacher})
    with pytest.raises(ConfigError, match=message):
        resolve_teacher(cfg, "agent_0", env_spec(cfg))


def test_a_scripted_teacher_is_not_supported_yet():
    cfg = make_test_config("turns", agents={"agent_0": {"kickstart": {"teacher": "bot"}}, "bot": BOT})
    with pytest.raises(ConfigError, match="scripted"):
        resolve_teacher(cfg, "agent_0", env_spec(cfg))


def test_teachers_are_per_agent(tmp_path):
    base = make_test_config("turns", agents={"a": {}, "b": {}})
    path, _ = _teacher_pt(tmp_path, base, "a")
    cfg = make_test_config("turns", agents={"a": {"kickstart": {"teacher": str(path), "lambda": 0.3}}, "b": {}})
    spec = env_spec(cfg)
    assert resolve_teacher(cfg, "a", spec).lambda_ == 0.3 and resolve_teacher(cfg, "b", spec) is None


def test_launcher_passes_numpy_teacher_specs_and_the_spec_to_learners(tmp_path, monkeypatch, restore_global_rng):
    import colosseum.launcher as launcher_module

    base = make_test_config("turns")
    path, _ = _teacher_pt(tmp_path, base)
    cfg = make_test_config("turns", kickstart={"teacher": str(path)})
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
                raise _Stop

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(launcher_module.mp, "Process", FakeProcess)
    with pytest.raises(_Stop):
        launcher_module.Launcher(cfg, make_test_run_dir(cfg, tmp_path)).launch()
    (kwargs,) = learner_kwargs
    assert isinstance(kwargs["teacher"], TeacherSpec) and kwargs["teacher"].source == str(path)
    assert kwargs["spec"] == env_spec(cfg)
    assert_no_tensors(kwargs, "learner kwargs")


def test_the_distributed_learner_builds_through_the_factory(tmp_path, monkeypatch, restore_global_rng,
                                                             restore_root_logging):
    import colosseum.distributed as distributed
    import colosseum.learner.learner as learner_module
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

    built = []
    monkeypatch.setattr(grpc_transport, "serve_trajectory_receiver", lambda *a, **k: FakeServer())
    monkeypatch.setattr(grpc_store, "GRPCWeightStore", FakeStore)
    monkeypatch.setattr(distributed.ProcessSupervisor, "install_signal_handlers", lambda self: None)
    monkeypatch.setattr(learner_module, "learner_process", lambda *, algorithm_factory, **kw: built.append(
        algorithm_factory()))
    monkeypatch.setattr(sys, "path", list(sys.path))
    base = make_test_config("turns")
    teacher, _ = _teacher_pt(tmp_path, base)
    config = write_test_config(tmp_path / "cfg.yaml", "turns", kickstart={"teacher": str(teacher), "kl": "reverse"})
    assert distributed.run_distributed_learner(str(config), "agent_0", 0, "localhost:1",
                                               overrides={"run.dir": str(tmp_path / "runs")}) == 0
    (algo,) = built
    assert algo._kickstart is not None and algo._kickstart.direction == "reverse"
