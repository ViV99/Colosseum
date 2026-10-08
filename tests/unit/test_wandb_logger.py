"""WandB logger: per-agent step axes, no step= argument, optional import (T0.1, T6.4). No network."""

from __future__ import annotations

import logging
import queue
import subprocess
import sys
import types

import pytest

from colosseum.core.config import MetricsConfig, load_config
from colosseum.launcher import Launcher
from colosseum.metrics.hub import MetricsHub
from colosseum.metrics.jsonl import MetricsWriter
from colosseum.metrics.wandb_logger import WandBLogger
from helpers import example_config, make_test_run_dir


class FakeWandb(types.ModuleType):
    """Records calls; mirrors the wandb functions the logger uses."""

    def __init__(self, fail_init: bool = False):
        super().__init__("wandb")
        self.fail_init = fail_init
        self.init_kwargs = None
        self.defined: list[tuple[str, object]] = []
        self.logged: list[dict] = []
        self.finished = False

    def init(self, **kwargs):
        if self.fail_init:
            raise RuntimeError("no network")
        self.init_kwargs = kwargs
        return types.SimpleNamespace(finish=self._finish)

    def _finish(self):
        self.finished = True

    def define_metric(self, name, step_metric=None):
        self.defined.append((name, step_metric))

    def log(self, row, **kwargs):
        assert "step" not in kwargs, "wandb.log must not receive step= (R5-02)"
        self.logged.append(dict(row))


def test_disabled_does_not_import_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # importing would raise
    logger = WandBLogger(MetricsConfig(use_wandb=False))
    assert not logger.enabled
    logger.log_train("a", {"loss": 1.0}, 1)
    logger.log_global({"system/x": 1.0}, 10)
    logger.finish()


def test_missing_wandb_package_disables_with_warning(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "wandb", None)
    with caplog.at_level(logging.WARNING):
        logger = WandBLogger(MetricsConfig(use_wandb=True))
    assert not logger.enabled
    assert "wandb is not installed" in caplog.text


def test_init_failure_disables(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb(fail_init=True))
    with caplog.at_level(logging.WARNING):
        assert not WandBLogger(MetricsConfig(use_wandb=True)).enabled
    assert "no network" in caplog.text


def test_per_agent_step_axes_keep_lagging_agents(monkeypatch):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    logger = WandBLogger(MetricsConfig(use_wandb=True, wandb_project="p"), run_name="r1", run_config={"a": 1})
    assert logger.enabled
    assert fake.init_kwargs["name"] == "r1" and fake.init_kwargs["config"] == {"a": 1}
    logger.log_train("alpha", {"loss": 1.0}, 100)
    logger.log_train("beta", {"loss": 2.0}, 95)   # lags behind alpha
    logger.log_train("alpha", {"loss": 0.5}, 110)
    logger.log_global({"ratings/elo/alpha": 1210.0}, env_steps=5000)
    assert ("alpha/*", "alpha/train_step") in fake.defined
    assert ("beta/*", "beta/train_step") in fake.defined
    assert ("ratings/*", "env_steps") in fake.defined and ("system/*", "env_steps") in fake.defined
    assert sum(1 for name, _ in fake.defined if name == "alpha/*") == 1  # defined once
    assert {"beta/loss": 2.0, "beta/train_step": 95} in fake.logged
    assert {"ratings/elo/alpha": 1210.0, "env_steps": 5000} in fake.logged
    logger.finish()
    assert fake.finished


def test_finish_failure_is_logged_not_raised(monkeypatch, caplog):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    logger = WandBLogger(MetricsConfig(use_wandb=True))

    def broken_finish():
        raise RuntimeError("upload failed")

    logger._run.finish = broken_finish
    with caplog.at_level(logging.WARNING):
        logger.finish()  # must not mask the training outcome
    assert "upload failed" in caplog.text
    logger.finish()  # idempotent


def test_package_imports_without_wandb_installed():
    code = (
        "import sys; sys.modules['wandb'] = None; "
        "import colosseum.cli, colosseum.launcher, colosseum.metrics.wandb_logger"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_launcher_finishes_wandb_when_final_metrics_fail(monkeypatch, tmp_path):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    config = load_config(example_config("tic_tac_toe.yaml"))
    run = make_test_run_dir(config, tmp_path)
    launcher = Launcher(config, run)
    launcher._wandb = WandBLogger(MetricsConfig(use_wandb=True))
    launcher._metrics_writer = MetricsWriter(run.metrics_path)
    launcher._hub = MetricsHub(writer=launcher._metrics_writer, ratings_path=run.ratings_path,
                               agent_ids=["agent_0"], total_timesteps=10, log_interval=1,
                               console_interval_sec=10.0, wandb_logger=launcher._wandb)

    class FailingCoordinator:
        def ratings_snapshot(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        launcher._finish_metrics(queue.Queue(), queue.Queue(), FailingCoordinator())
    assert launcher._metrics_writer.closed
    assert fake.finished
