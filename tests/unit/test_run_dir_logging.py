"""RunDir layout and per-process logging (T6.2)."""
from __future__ import annotations

import logging
import re

import pytest
import yaml

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
