"""WandB logger: per-agent step axes, no step= argument, optional import (T0.1, T6.4). No network."""

from __future__ import annotations

import logging
import queue
import subprocess
import sys

import pytest
from fake_wandb import FakeWandb

from colosseum.core.config import RESERVED_AGENT_IDS, MetricsConfig, load_config
from colosseum.launcher import Launcher
from colosseum.metrics.hub import MetricsHub
from colosseum.metrics.jsonl import GLOBAL_KINDS, METRIC_KINDS, MetricsWriter
from colosseum.metrics.wandb_logger import WandBLogger
from helpers import example_config, make_test_run_dir

LOGGER = "colosseum.metrics.wandb_logger"


def enabled_logger(monkeypatch, **fake_kwargs) -> tuple[WandBLogger, FakeWandb]:
    fake = FakeWandb(**fake_kwargs)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return WandBLogger(MetricsConfig(use_wandb=True)), fake


def warnings_of(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno >= logging.WARNING]


def test_disabled_does_not_import_wandb(monkeypatch, caplog):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)  # an import would find (and use) this fake
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        logger = WandBLogger(MetricsConfig(use_wandb=False))
        assert not logger.enabled
        logger.log_train("a", {"loss": 1.0}, 1)
        logger.log_global({"system/x": 1.0}, 10)
        logger.finish()
    assert fake.calls == 0 and fake.init_kwargs is None and fake.defined == []
    assert warnings_of(caplog) == []


def test_missing_wandb_package_disables_with_warning(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "wandb", None)  # makes `import wandb` raise ImportError
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        logger = WandBLogger(MetricsConfig(use_wandb=True))
    assert not logger.enabled
    assert "wandb is not installed" in caplog.text
    assert "uv pip install -e '.[wandb]'" in caplog.text


def test_init_failure_disables(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb(fail_init=True))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert not WandBLogger(MetricsConfig(use_wandb=True)).enabled
    assert "no network" in caplog.text
    assert "WANDB_API_KEY" in caplog.text and "WANDB_MODE=offline" in caplog.text


def test_define_metric_failure_at_init_disables_and_finishes_the_run(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        logger, fake = enabled_logger(monkeypatch, fail_define="system/*")
    assert not logger.enabled
    assert fake.finished  # the run wandb.init opened is closed (best effort)
    assert len(warnings_of(caplog)) == 1 and "define_metric(system/*) failed" in caplog.text
    calls = fake.calls
    logger.log_train("a", {"loss": 1.0}, 1)
    logger.log_global({"system/x": 1.0}, 10)
    logger.finish()
    assert fake.calls == calls


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
    assert all(kwargs == {} for kwargs in fake.log_kwargs)  # never step= (R5-02)
    logger.finish()
    assert fake.finished


@pytest.mark.parametrize("first_call", ["train", "global"])
def test_log_failure_warns_once_and_disables_but_finish_still_closes_the_run(monkeypatch, caplog, first_call):
    logger, fake = enabled_logger(monkeypatch, fail_log=True)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        if first_call == "train":
            logger.log_train("a", {"loss": 1.0}, 1)  # raises inside wandb.log: must not propagate
        else:
            logger.log_global({"system/x": 1.0}, 10)
        calls = fake.calls
        logger.log_train("a", {"loss": 1.0}, 2)
        logger.log_train("b", {"loss": 1.0}, 2)
        logger.log_global({"system/x": 1.0}, 20)
    assert fake.calls == calls  # later calls do not touch wandb
    assert not logger.enabled
    assert len(warnings_of(caplog)) == 1 and "log failed" in caplog.text
    logger.finish()
    assert fake.finished


def test_lazy_define_metric_failure_warns_once_and_disables(monkeypatch, caplog):
    logger, fake = enabled_logger(monkeypatch, fail_define="alpha/*")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        logger.log_train("alpha", {"loss": 1.0}, 1)
        calls = fake.calls
        logger.log_train("alpha", {"loss": 1.0}, 2)
        logger.log_global({"system/x": 1.0}, 10)
    assert fake.calls == calls and fake.logged == []
    assert not logger.enabled
    assert len(warnings_of(caplog)) == 1 and "define_metric(alpha/*) failed" in caplog.text
    logger.finish()
    assert fake.finished


def test_finish_failure_is_logged_not_raised(monkeypatch, caplog):
    logger, _ = enabled_logger(monkeypatch)

    def broken_finish():
        raise RuntimeError("upload failed")

    logger._run.finish = broken_finish
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        logger.finish()  # must not mask the training outcome
    assert "upload failed" in caplog.text
    logger.finish()  # idempotent


def test_global_namespaces_match_the_reserved_agent_ids(monkeypatch):
    """One source of truth: the WandB global namespaces are the jsonl global kinds, and the
    reserved agent ids are exactly the jsonl kinds, so an agent can never collide with them."""
    _, fake = enabled_logger(monkeypatch)
    assert {name for name, step in fake.defined if step == "env_steps"} == {f"{k}/*" for k in GLOBAL_KINDS}
    assert set(METRIC_KINDS) == set(RESERVED_AGENT_IDS)


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
