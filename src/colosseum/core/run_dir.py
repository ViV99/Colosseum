"""Run directory: where one training run writes its outputs.

    runs/<name>/
      config.resolved.yaml     after --set overrides and agent merging
      logs/<process>.log       main, learner-<agent>, worker-<i>, worker-<i>-env<k>
      metrics.jsonl            one JSON record per line (T6.3)
      ratings.json             latest ratings snapshot (T6.3)
      checkpoints/<agent>/ckpt_v<N>/{model.pt, trainer_state.pt, meta.json}
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError

RESOLVED_CONFIG_FILE = "config.resolved.yaml"


@dataclass(frozen=True)
class RunDir:
    root: Path

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def ratings_path(self) -> Path:
        return self.root / "ratings.json"

    @property
    def resolved_config_path(self) -> Path:
        return self.root / RESOLVED_CONFIG_FILE

    @classmethod
    def open(cls, root: str | Path) -> RunDir:
        """An existing run dir (no files are created)."""
        return cls(Path(root))

    @classmethod
    def create(cls, config: ColosseumConfig, config_path: str | Path | None = None,
               role: str | None = None) -> RunDir:
        """Create ``<run.dir>/<name>[-<role>]`` with ``logs/`` and ``checkpoints/``.

        The default name is ``<config_stem>-<YYYYmmdd-HHMMSS>`` and gets a ``-2``, ``-3``, ...
        suffix if taken. An explicit ``run.name`` that already holds files is an error,
        so a run never mixes into another run's checkpoints (R4-06, R5-13).
        """
        stem = Path(config_path).stem if config_path is not None else "run"
        explicit = config.run.name is not None
        base = config.run.name if explicit else f"{stem}-{datetime.now():%Y%m%d-%H%M%S}"
        if role:
            base = f"{base}-{role}"
        parent = Path(config.run.dir)
        root = parent / base
        if explicit:
            if root.exists() and any(root.iterdir()):
                raise ConfigError(
                    f"Run directory {root} already exists and is not empty; choose another run.name "
                    f"(--set run.name=...) or delete it"
                )
        else:
            suffix = 2
            while root.exists():
                root = parent / f"{base}-{suffix}"
                suffix += 1
        run = cls(root)
        run.logs.mkdir(parents=True, exist_ok=True)
        run.checkpoints.mkdir(parents=True, exist_ok=True)
        return run

    def write_resolved_config(self, config: ColosseumConfig) -> Path:
        path = self.resolved_config_path
        path.write_text(yaml.safe_dump(config.model_dump(mode="json", by_alias=True), sort_keys=False))
        return path
