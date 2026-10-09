"""SP2 entry points validate before they build anything; CLI config errors are one line (T7.1)."""
from __future__ import annotations

import logging
import multiprocessing as mp

import pytest
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.errors import ConfigError
from colosseum.networks.model import UnrollOutput
from game_helpers import GameTestModel, make_test_config, write_test_config


def _bad_policy_config(tmp_path):
    """A config whose model fails validation (values shaped [S, B])."""
    return write_test_config(tmp_path / "bad.yaml", "turns",
                             networks={"model_class": "test_sp2_entry_points.MatrixValueModel"})


def _fail(name):
    def _boom(*_args, **_kwargs):
        raise AssertionError(f"{name} ran before validate_config rejected the config")
    return _boom


class MatrixValueModel(GameTestModel):
    """``unroll`` returns values shaped [S, B] instead of [S*B] (validate_config must reject it)."""

    def unroll(self, obs, state0, reset_after, action_mask=None, global_state=None, with_value=True):
        out = super().unroll(obs, state0, reset_after, action_mask, global_state, with_value)
        return out if out.value is None else UnrollOutput(out.dist, out.value.reshape(reset_after.shape))


def test_bc_eval_and_validate_reject_the_config_first(tmp_path):
    cfg = _bad_policy_config(tmp_path)
    data = tmp_path / "data.pt"
    data.write_bytes(b"")  # never read: validation fails first
    for args in (["bc", "-d", str(data), "-o", str(tmp_path / "out.pt")],
                 ["eval", "-a", f"x={tmp_path / 'never.pt'}"],
                 ["validate"]):
        result = CliRunner().invoke(main, [args[0], "-c", str(cfg), *args[1:]])
        assert result.exit_code == 1, (args, result.output)
        assert result.stderr.startswith("Config error:") and "time-major values" in result.stderr
        assert "Traceback" not in result.output
    assert not (tmp_path / "out.pt").exists()


def test_distributed_roles_reject_the_config_before_serving_or_spawning(tmp_path, monkeypatch,
                                                                        restore_root_logging):
    from colosseum import distributed
    from colosseum.transport import grpc_transport

    monkeypatch.setattr(grpc_transport, "serve_trajectory_receiver", _fail("serve_trajectory_receiver"))
    with pytest.raises(ConfigError, match="time-major values"):
        distributed.run_distributed_learner(str(_bad_policy_config(tmp_path)), "agent_0", 0, "localhost:1")
    monkeypatch.setattr(mp, "set_start_method", _fail("mp.set_start_method"))
    monkeypatch.setattr(distributed.mp, "Process", _fail("mp.Process"))
    with pytest.raises(ConfigError, match="time-major values"):
        distributed.run_distributed_workers(str(_bad_policy_config(tmp_path)), "localhost:1",
                                            {"agent_0": "localhost:2"})
    assert not (tmp_path / "runs").exists()


def test_train_rejects_an_invalid_config_without_creating_a_run_dir(tmp_path, monkeypatch, restore_root_logging,
                                                                     restore_global_rng):
    import colosseum.launcher as launcher_module

    monkeypatch.setattr(launcher_module.mp, "set_start_method", lambda *args, **kwargs: None)
    launched = []

    class FakeLauncher:
        def __init__(self, config, run_dir, validated=False):
            assert validated
            launched.append(run_dir)

        def launch(self):
            return 0

    monkeypatch.setattr(launcher_module, "Launcher", FakeLauncher)
    runs = tmp_path / "runs"
    overrides = {"run.dir": str(runs), "run.name": "retry"}
    with pytest.raises(ConfigError, match="time-major values"):
        launcher_module.run_training(str(_bad_policy_config(tmp_path)), overrides)
    assert not runs.exists() and not launched
    fixed = {**overrides, "networks.model_class": "game_helpers.GameTestModel"}
    assert launcher_module.run_training(str(_bad_policy_config(tmp_path)), fixed) == 0
    [run_dir] = launched
    assert run_dir.root == runs / "retry" and run_dir.resolved_config_path.is_file()


def test_train_rejects_a_missing_resume_source_without_creating_a_run_dir(tmp_path, monkeypatch,
                                                                         restore_root_logging):
    import colosseum.launcher as launcher_module

    monkeypatch.setattr(launcher_module.mp, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher_module, "Launcher", _fail("Launcher"))
    cfg = write_test_config(tmp_path / "cfg.yaml", "turns")
    overrides = {"run.dir": str(tmp_path / "runs"), "run.name": "resumed",
                 "training.resume_from": str(tmp_path / "no_such_run")}
    with pytest.raises(ConfigError, match="resume_from"):
        launcher_module.run_training(str(cfg), overrides)
    assert not (tmp_path / "runs").exists()


def test_seed_is_applied_after_overrides(tmp_path, monkeypatch, restore_root_logging, restore_global_rng):
    import random

    import torch

    import colosseum.launcher as launcher_module

    # run_training forces the spawn start method; keep the test free of global side effects.
    monkeypatch.setattr(launcher_module.mp, "set_start_method", lambda *args, **kwargs: None)

    class FakeLauncher:
        def __init__(self, *args, **kwargs):
            pass

        def launch(self):
            return 0

    monkeypatch.setattr(launcher_module, "Launcher", FakeLauncher)
    cfg = write_test_config(tmp_path / "cfg.yaml", "turns", training={"seed": 1})
    launcher_module.run_training(str(cfg), {"training.seed": 123, "run.dir": str(tmp_path / "runs")})
    assert torch.initial_seed() == 123
    assert random.random() == random.Random(123).random()


def test_cli_validate_malformed_set_value_exits_1(tmp_path):
    cfg = write_test_config(tmp_path / "cfg.yaml", "turns")
    result = CliRunner().invoke(main, ["validate", "-c", str(cfg), "--set", "learner.batch_chunks=[1,"])
    assert result.exit_code == 1 and result.stderr.startswith("Config error:")
    assert "learner.batch_chunks=[1," in result.stderr and "Traceback" not in result.output
    assert result.exc_info[0] is SystemExit


@pytest.mark.parametrize(("workers", "envs", "agents", "refresh", "warns"), [
    (1, 3, 2, 0.0, True),    # 3 envs over 2 agents: agent 0 owns 2, agent 1 owns 1 forever
    (1, 1, 2, 0.0, True),    # fewer envs than agents: agent 1 never owns an env
    (2, 2, 2, 0.0, False),   # 4 envs over 2 agents: even split
    (1, 3, 2, 30.0, False),  # rotation advances, so ownership evens out over time
    (1, 3, 1, 0.0, False),   # single agent owns everything
])
def test_static_ownership_skew_warning(workers, envs, agents, refresh, warns, caplog):
    from colosseum.launcher import warn_static_ownership_skew

    cfg = make_test_config("turns", agents={f"a{i}": {} for i in range(agents)},
                           rollout={"num_workers": workers, "envs_per_worker": envs,
                                    "match_refresh_interval_sec": refresh})
    with caplog.at_level(logging.WARNING, logger="colosseum.launcher"):
        warn_static_ownership_skew(cfg)
    assert any("match_refresh_interval_sec" in r.getMessage() for r in caplog.records) == warns
