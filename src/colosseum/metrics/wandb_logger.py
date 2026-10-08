"""Optional WandB viewer for the metrics written to metrics.jsonl.

One run. Every agent gets its own x-axis (``<agent>/train_step``) through
``define_metric``, so agents that train at different speeds are all kept (R5-02).
Ratings, system and episode metrics use ``env_steps``. ``wandb`` is imported only
when ``metrics.use_wandb`` is true (extra ``wandb``). WandB is never allowed to stop
training: any failure logs one warning and disables WandB for the rest of the run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from colosseum.core.config import MetricsConfig
from colosseum.metrics.jsonl import GLOBAL_KINDS

logger = logging.getLogger(__name__)

# The global namespaces are the metrics.jsonl global kinds; agent ids can never equal them
# (core.config.RESERVED_AGENT_IDS).
_GLOBAL_PREFIXES = tuple(f"{kind}/*" for kind in GLOBAL_KINDS)


class WandBLogger:
    def __init__(self, config: MetricsConfig, run_name: str | None = None,
                 run_config: dict[str, Any] | None = None) -> None:
        self._wandb: Any = None
        self._run: Any = None
        self._defined_agents: set[str] = set()
        if not config.use_wandb:
            return
        try:
            import wandb
        except ImportError:
            logger.warning("metrics.use_wandb is true but wandb is not installed "
                           "(optional extra: uv pip install -e '.[wandb]'); WandB logging is disabled")
            return
        run = None
        try:
            run = wandb.init(project=config.wandb_project, entity=config.wandb_entity,
                             name=run_name, config=run_config or {})
            wandb.define_metric("env_steps")
            for pattern in _GLOBAL_PREFIXES:
                wandb.define_metric(pattern, step_metric="env_steps")
        except Exception as e:  # noqa: BLE001 - wandb raises many unrelated error types
            logger.warning(f"Failed to initialize WandB ({e}); WandB logging is disabled. Without "
                           "an interactive login set WANDB_API_KEY, or WANDB_MODE=offline to log locally")
            if run is not None:
                self._run = run
                self.finish()  # best effort: close what wandb.init opened
            return
        self._run = run
        self._wandb = wandb

    @property
    def enabled(self) -> bool:
        return self._wandb is not None

    def log_train(self, agent_id: str, metrics: dict[str, float], train_step: int) -> None:
        """Per-agent training metrics on the axis ``<agent>/train_step``."""
        wandb = self._wandb
        if wandb is None:
            return
        row = {f"{agent_id}/{k}": float(v) for k, v in metrics.items()}
        row[f"{agent_id}/train_step"] = int(train_step)

        def emit() -> None:
            if agent_id not in self._defined_agents:
                wandb.define_metric(f"{agent_id}/train_step")
                wandb.define_metric(f"{agent_id}/*", step_metric=f"{agent_id}/train_step")
                self._defined_agents.add(agent_id)
            wandb.log(row)

        self._guarded(emit)

    def log_global(self, metrics: dict[str, float], env_steps: int) -> None:
        """Ratings / system / episode metrics (already prefixed) on the ``env_steps`` axis."""
        wandb = self._wandb
        if wandb is None:
            return
        row = {**metrics, "env_steps": int(env_steps)}
        self._guarded(lambda: wandb.log(row))

    def _guarded(self, call: Callable[[], None]) -> None:
        """Run a post-init wandb call; on failure warn once and stop calling wandb.

        The run is kept, so :meth:`finish` still closes it.
        """
        try:
            call()
        except Exception as e:  # noqa: BLE001 - wandb raises many unrelated error types
            logger.warning(f"WandB call failed ({e}); WandB logging is disabled for the rest of the run "
                           "(metrics.jsonl is unaffected)")
            self._wandb = None

    def finish(self) -> None:
        """Close the run (idempotent). A failure is logged, never raised: it runs in the
        launcher's ``finally`` and must not mask the training outcome."""
        run, self._run, self._wandb = self._run, None, None
        if run is None:
            return
        try:
            run.finish()
        except Exception as e:  # noqa: BLE001 - wandb.finish raises many unrelated error types
            logger.warning(f"Failed to finish the WandB run ({e})")
