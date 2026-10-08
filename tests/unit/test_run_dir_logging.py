"""RunDir layout and per-process logging (T6.2)."""
from __future__ import annotations

import logging
import re

import pytest
import yaml
from pydantic import ValidationError

from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.run_dir import RunDir
from colosseum.utils.logging import ENV_LOG_DIR, ENV_PROCESS_NAME, setup_process_logging
from colosseum.utils.process import run_child

TTT = "examples.tic_tac_toe"


def make_config(tmp_path, name=None) -> ColosseumConfig:
    return ColosseumConfig.model_validate({
        "env": {"env_class": f"{TTT}.env.TicTacToeEnv"},
        "networks": {
            "encoder_class": f"{TTT}.networks.TicTacToeEncoder",
            "policy_class": f"{TTT}.networks.TicTacToePolicy",
            "value_class": f"{TTT}.networks.TicTacToeValue",
        },
        "agents": {"alpha": {"algorithm": {"learning_rate": 1e-4}}, "beta": None},
        "run": {"dir": str(tmp_path / "runs"), "name": name},
    })


def test_default_name_uses_config_stem_and_timestamp(tmp_path):
    cfg = make_config(tmp_path)
    run = RunDir.create(cfg, config_path="configs/examples/tic_tac_toe.yaml")
    assert re.fullmatch(r"tic_tac_toe-\d{8}-\d{6}", run.root.name)
    assert run.root.parent == tmp_path / "runs"
    assert run.logs.is_dir() and run.checkpoints.is_dir()
    assert run.metrics_path == run.root / "metrics.jsonl"
    assert run.ratings_path == run.root / "ratings.json"
    again = RunDir.create(cfg, config_path="configs/examples/tic_tac_toe.yaml")
    assert again.root != run.root  # auto names never collide


def test_explicit_name_refuses_existing_run(tmp_path):
    cfg = make_config(tmp_path, name="exp1")
    run = RunDir.create(cfg)
    run.write_resolved_config(cfg)
    with pytest.raises(ConfigError, match="exp1"):
        RunDir.create(cfg)
    assert RunDir.create(cfg, role="learner-alpha").root.name == "exp1-learner-alpha"


@pytest.mark.parametrize("bad", ["..", "a/b", "/abs/path", "", ".hidden", "../escaped"])
def test_run_name_must_be_one_safe_path_component(tmp_path, bad):
    data = make_config(tmp_path).model_dump(mode="json", by_alias=True)
    data["run"]["name"] = bad
    with pytest.raises(ValidationError, match="run name"):
        ColosseumConfig.model_validate(data)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="run name"):
        load_config(path)
    data["run"]["name"] = "ok"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="run name"):
        load_config(path, {"run.name": bad})  # a --set override is checked too
    assert not (tmp_path / "runs").exists()


def test_run_dir_rechecks_a_name_assigned_after_validation(tmp_path):
    cfg = make_config(tmp_path)
    cfg.run.name = "../escaped"  # plain attribute assignment is not validated by pydantic
    with pytest.raises(ConfigError, match="run name"):
        RunDir.create(cfg)
    assert not (tmp_path / "escaped").exists()


def test_auto_name_claims_the_next_free_suffix(tmp_path, monkeypatch):
    from datetime import datetime

    import colosseum.core.run_dir as run_dir_module

    class FrozenClock:
        @staticmethod
        def now():
            return datetime(2026, 1, 2, 3, 4, 5)

    monkeypatch.setattr(run_dir_module, "datetime", FrozenClock)
    cfg = make_config(tmp_path)
    taken = tmp_path / "runs" / "my-cfg-20260102-030405"
    taken.mkdir(parents=True)
    (tmp_path / "runs" / "my-cfg-20260102-030405-2").mkdir()  # e.g. created by a concurrent run
    run = RunDir.create(cfg, config_path="conf/my cfg.yaml")  # the stem is sanitized
    assert run.root == tmp_path / "runs" / "my-cfg-20260102-030405-3"
    assert run.logs.is_dir() and run.checkpoints.is_dir()
    assert not any(taken.iterdir())  # the taken dirs are left alone


def test_explicit_name_refuses_an_existing_even_empty_dir(tmp_path):
    cfg = make_config(tmp_path, name="exp2")
    (tmp_path / "runs" / "exp2").mkdir(parents=True)
    with pytest.raises(ConfigError, match="already exists"):
        RunDir.create(cfg)


def test_resolved_config_records_the_effective_auto_name(tmp_path):
    cfg = make_config(tmp_path)
    run = RunDir.create(cfg, config_path="tic_tac_toe.yaml")
    resolved = load_config(run.write_resolved_config(cfg))
    assert resolved.run.name == run.root.name
    effective = run.with_run_name(cfg)
    assert resolved == effective and effective.run.name == run.root.name and cfg.run.name is None
    assert run.with_run_name(make_config(tmp_path, name="kept")).run.name == "kept"


def test_resolved_config_write_is_atomic(tmp_path, monkeypatch):
    import colosseum.core.run_dir as run_dir_module

    cfg = make_config(tmp_path, name="atomic")
    run = RunDir.create(cfg)
    path = run.write_resolved_config(cfg)
    before = path.read_text()

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(run_dir_module.os, "replace", failing_replace)
    changed = cfg.model_copy(update={"training": cfg.training.model_copy(update={"total_timesteps": 7})})
    with pytest.raises(OSError, match="disk full"):
        run.write_resolved_config(changed)
    assert path.read_text() == before  # the old file is intact ...
    assert sorted(p.name for p in run.root.iterdir()) == ["checkpoints", "config.resolved.yaml", "logs"]  # no tmp


def test_resolved_config_roundtrips(tmp_path):
    cfg = make_config(tmp_path, name="rt")
    run = RunDir.create(cfg)
    path = run.write_resolved_config(cfg)
    assert path == run.root / "config.resolved.yaml"
    assert load_config(path) == cfg
    assert yaml.safe_load(path.read_text())["agents"]["alpha"]["algorithm"] == {"learning_rate": 1e-4}


def test_setup_process_logging_file_info_console_warning(tmp_path, capsys, restore_root_logging):
    setup_process_logging(tmp_path / "logs", "worker-3")
    log = logging.getLogger("colosseum.worker.test")
    log.info("info-line")
    log.warning("warn-line")
    text = (tmp_path / "logs" / "worker-3.log").read_text()
    assert "worker-3 started (pid" in text and "info-line" in text and "warn-line" in text
    err = capsys.readouterr().err
    assert "warn-line" in err and "info-line" not in err


def test_setup_process_logging_main_console_info(tmp_path, capsys, restore_root_logging):
    setup_process_logging(None, "main", console_level=logging.INFO)
    logging.getLogger("colosseum.launcher").info("progress-line")
    assert "progress-line" in capsys.readouterr().err


def test_run_child_logs_finish_and_crash(tmp_path, monkeypatch, restore_root_logging):
    # setenv first so monkeypatch removes the variables run_child exports after the test.
    for var in (ENV_LOG_DIR, ENV_PROCESS_NAME):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    logs = tmp_path / "logs"
    run_child("learner-a", str(logs), lambda: logging.getLogger("colosseum.learner").info("trained"))
    text = (logs / "learner-a.log").read_text()
    assert "trained" in text and "learner-a finished" in text

    def boom():
        raise RuntimeError("bad env")

    with pytest.raises(RuntimeError):
        run_child("worker-1", str(logs), boom)
    text = (logs / "worker-1.log").read_text()
    assert "worker-1 crashed" in text and "RuntimeError: bad env" in text
    import os
    assert os.environ[ENV_LOG_DIR] == str(logs) and os.environ[ENV_PROCESS_NAME] == "worker-1"


@pytest.mark.parametrize("host, role", [
    ("node-1", "workers-node-1"),
    ("my host.local/x", "workers-my-host.local-x"),
    ("-.odd", "workers-odd"),
    ("", "workers-host"),
])
def test_distributed_workers_role_includes_the_sanitized_hostname(monkeypatch, host, role):
    import colosseum.distributed as distributed

    monkeypatch.setattr(distributed.socket, "gethostname", lambda: host)
    assert distributed.workers_role() == role
