"""Run directory: where one training run writes its outputs.

    runs/<name>/
      config.resolved.yaml     after --set overrides and agent merging
      logs/<process>.log       main, learner-<agent>, worker-<i>, worker-<i>-env<k>
      metrics.jsonl            one JSON record per line (T6.3)
      ratings.json             latest ratings snapshot (T6.3)
      checkpoints/<agent>/ckpt_v<N>/{model.pt, trainer_state.pt, meta.json}
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from colosseum.core.config import ColosseumConfig, check_path_component
from colosseum.core.errors import ConfigError

RESOLVED_CONFIG_FILE = "config.resolved.yaml"


def safe_path_component(value: str, fallback: str) -> str:
    """``value`` made safe for :func:`check_path_component`: other characters become ``-``,
    leading ``.``/``-`` are dropped; ``fallback`` if nothing is left."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "-", value).lstrip(".-")
    return check_path_component(cleaned or fallback, "path component")


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
        suffix if taken. An explicit ``run.name`` whose dir already exists is an error,
        so a run never mixes into another run's checkpoints (R4-06, R5-13). The root is
        claimed with an exclusive ``mkdir``, so two processes never get the same dir.
        """
        stem = safe_path_component(Path(config_path).stem, "run") if config_path is not None else "run"
        explicit = config.run.name is not None
        base = config.run.name if explicit else f"{stem}-{datetime.now():%Y%m%d-%H%M%S}"
        if role:
            base = f"{base}-{role}"
        check_path_component(base, "run name")
        parent = Path(config.run.dir)
        parent.mkdir(parents=True, exist_ok=True)
        root = parent / base
        suffix = 2
        while True:
            try:
                root.mkdir()
                break
            except FileExistsError:
                if explicit:
                    raise ConfigError(
                        f"Run directory {root} already exists; choose another run.name "
                        f"(--set run.name=...) or delete it"
                    ) from None
                root = parent / f"{base}-{suffix}"
                suffix += 1
        run = cls(root)
        run.logs.mkdir()
        run.checkpoints.mkdir()
        return run

    def with_run_name(self, config: ColosseumConfig) -> ColosseumConfig:
        """``config`` with ``run.name`` set to this run's effective name.

        An auto-named run (``run.name: null``) records the name it got (``root.name``);
        an explicit name is kept as is.
        """
        if config.run.name is not None:
            return config
        return config.model_copy(update={"run": config.run.model_copy(update={"name": self.root.name})})

    def write_resolved_config(self, config: ColosseumConfig) -> Path:
        """Write ``config`` (with the effective run name) atomically to ``config.resolved.yaml``."""
        data = self.with_run_name(config).model_dump(mode="json", by_alias=True)
        path = self.resolved_config_path
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return path
