"""WandB is an optional dependency: missing package must only disable logging."""

from __future__ import annotations

import logging
import subprocess
import sys

from colosseum.core.config import MetricsConfig
from colosseum.metrics.wandb_logger import WandBLogger


def test_missing_wandb_package_disables_logging(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "wandb", None)  # makes `import wandb` raise ImportError
    with caplog.at_level(logging.WARNING, logger="colosseum.metrics.wandb_logger"):
        wb = WandBLogger(MetricsConfig(use_wandb=True))
    assert wb._enabled is False
    assert "uv pip install -e '.[wandb]'" in caplog.text
    wb.log_config({"a": 1})
    wb.log_train_step("agent_0", {"loss": 1.0}, step=1)
    wb.finish()


def test_disabled_logger_never_imports_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    wb = WandBLogger(MetricsConfig(use_wandb=False))
    wb.log_train_step("agent_0", {"loss": 1.0}, step=1)
    wb.finish()


def test_package_imports_without_wandb_installed():
    code = (
        "import sys; sys.modules['wandb'] = None; "
        "import colosseum.cli, colosseum.launcher, colosseum.metrics.wandb_logger"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
