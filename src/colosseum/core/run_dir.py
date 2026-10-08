"""Run directory: where one training run writes its outputs.

    runs/<name>/
      config.resolved.yaml     after --set overrides and agent merging
      logs/<process>.log       main, learner-<agent>, worker-<i>, worker-<i>-env<k>
      metrics.jsonl            one JSON record per line (T6.3)
      ratings.json             latest ratings snapshot (T6.3)
      checkpoints/<agent>/ckpt_v<N>/{model.pt, trainer_state.pt, meta.json}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from colosseum.core.config import ColosseumConfig, check_path_component
from colosseum.core.errors import ConfigError
from colosseum.utils.fs import write_text_atomic

RESOLVED_CONFIG_FILE = "config.resolved.yaml"


def safe_path_component(value: str, fallback: str) -> str:
    """``value`` made safe for :func:`check_path_component`: other characters become ``-``,
    leading ``.``/``-`` are dropped; ``fallback`` if nothing is left."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "-", value).lstrip(".-")
    return check_path_component(cleaned or fallback, "path component")


@dataclass(frozen=True)
class RunDir:
    """``root`` is ``<run.dir>/<run_name>`` or, for a distributed role, ``<run.dir>/<run_name>-<role>``.

    ``run_name`` is the base run name shared by all roles of a run (``None`` for
    ``RunDir.open``, which then falls back to ``root.name``).
    """

    root: Path
    run_name: str | None = None

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
        """Create ``<run.dir>/<run_name>[-<role>]`` with ``logs/`` and ``checkpoints/``.

        The default run name is ``<config_stem>-<YYYYmmdd-HHMMSS>``; if its dir is taken
        the run name gets a ``-2``, ``-3``, ... suffix (before the role, so the layout is
        always ``<run_name>-<role>``). An explicit ``run.name`` whose dir already exists is
        an error, so a run never mixes into another run's checkpoints (R4-06, R5-13). The
        root is claimed with an exclusive ``mkdir``, so two processes never get the same dir.
        """
        stem = safe_path_component(Path(config_path).stem, "run") if config_path is not None else "run"
        explicit = config.run.name is not None
        base = config.run.name if explicit else f"{stem}-{datetime.now():%Y%m%d-%H%M%S}"
        check_path_component(base, "run name")
        if role:
            check_path_component(role, "run role")
        parent = Path(config.run.dir)
        parent.mkdir(parents=True, exist_ok=True)
        run_name, suffix = base, 2
        while True:
            root = parent / (f"{run_name}-{role}" if role else run_name)
            try:
                root.mkdir()
                break
            except FileExistsError:
                if explicit:
                    raise ConfigError(
                        f"Run directory {root} already exists; choose another run.name "
                        f"(--set run.name=...) or delete it"
                    ) from None
                run_name = f"{base}-{suffix}"
                suffix += 1
        run = cls(root, run_name)
        run.logs.mkdir()
        run.checkpoints.mkdir()
        return run

    def with_run_name(self, config: ColosseumConfig) -> ColosseumConfig:
        """``config`` with ``run.name`` set to this run's effective base name.

        An auto-named run (``run.name: null``) records the name it got, without any role
        suffix, so re-running the resolved config as the same role reproduces
        ``<run_name>-<role>``; an explicit name is kept as is.
        """
        if config.run.name is not None:
            return config
        name = self.run_name if self.run_name is not None else self.root.name
        return config.model_copy(update={"run": config.run.model_copy(update={"name": name})})

    def write_resolved_config(self, config: ColosseumConfig) -> Path:
        """Write ``config`` (with the effective run name) atomically to ``config.resolved.yaml``."""
        data = self.with_run_name(config).model_dump(mode="json", by_alias=True)
        return write_text_atomic(self.resolved_config_path, yaml.safe_dump(data, sort_keys=False))
