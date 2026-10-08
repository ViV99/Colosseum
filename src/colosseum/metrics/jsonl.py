"""metrics.jsonl: the source of truth for training metrics (WandB is only a viewer)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from colosseum.utils.fs import write_text_atomic

# Kinds not tied to one agent's train step: logged on the global ``env_steps`` axis (and the
# WandB namespaces ``<kind>/*``). Together with "train" they equal
# ``core.config.RESERVED_AGENT_IDS`` (asserted in tests), so no agent id can collide with them.
GLOBAL_KINDS = ("episodes", "ratings", "system")
METRIC_KINDS = ("train", *GLOBAL_KINDS)

REQUIRED_KEYS: dict[str, set[str]] = {
    "train": {"ts", "kind", "agent", "train_step"},
    "episodes": {"ts", "kind", "agent", "env_steps", "episodes", "return_mean", "length_mean", "wdl",
                 "seat_counts"},
    "ratings": {"ts", "kind", "env_steps", "elo", "win_rates", "games", "wr_vs_past"},
    "system": {"ts", "kind", "env_steps", "env_steps_per_sec", "train_steps_per_sec", "queue_depths",
               "parked_buffers", "workers_reporting"},
}


def _sanitize(obj: Any) -> Any:
    """JSON-safe copy: numpy scalars/arrays to python, NaN/inf to None."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


class MetricsWriter:
    """Appends one JSON object per line: ``{"ts", "kind", **fields}``."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def closed(self) -> bool:
        return self._fh.closed

    def write(self, kind: str, **fields: Any) -> None:
        if kind not in METRIC_KINDS:
            raise ValueError(f"Unknown metrics kind {kind!r}; expected one of {METRIC_KINDS}")
        record = _sanitize({"ts": time.time(), "kind": kind, **fields})
        self._fh.write(json.dumps(record, allow_nan=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def write_json_atomic(path: str | Path, data: Any) -> None:
    """Write JSON to ``path`` atomically (readers never see half a file)."""
    write_text_atomic(path, json.dumps(_sanitize(data), indent=2, sort_keys=True, allow_nan=False))
