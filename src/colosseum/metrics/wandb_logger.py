"""WandB metrics logging."""

from __future__ import annotations

import logging
from typing import Any

from colosseum.core.config import MetricsConfig

logger = logging.getLogger(__name__)


class WandBLogger:
    """Centralized WandB logging for training metrics."""

    def __init__(self, config: MetricsConfig, run_name: str | None = None) -> None:
        self._config = config
        self._enabled = config.use_wandb
        self._run = None

        if self._enabled:
            try:
                import wandb
            except ImportError:
                logger.warning(
                    "metrics.use_wandb is true but the 'wandb' package is not installed "
                    "(it is an optional extra: uv pip install -e '.[wandb]'). WandB logging disabled."
                )
                self._enabled = False
            else:
                try:
                    self._run = wandb.init(
                        project=config.wandb_project,
                        entity=config.wandb_entity,
                        name=run_name,
                        config={},  # will be updated with full config
                    )
                except Exception as e:  # any wandb init failure must not stop training
                    logger.warning(f"Failed to initialize WandB: {e}. Logging disabled.")
                    self._enabled = False

    def log_config(self, config: dict[str, Any]) -> None:
        """Log the full configuration."""
        if self._enabled and self._run is not None:
            self._run.config.update(config)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log training metrics."""
        if self._enabled and self._run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def log_train_step(self, agent_id: str, metrics: dict[str, float], step: int) -> None:
        """Log per-agent training metrics."""
        prefixed = {f"{agent_id}/{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed, step=step)

    def finish(self) -> None:
        """Finish the WandB run."""
        if self._enabled and self._run is not None:
            self._run.finish()
