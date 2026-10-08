"""Filesystem helpers shared by the run dir and the metrics outputs."""

from __future__ import annotations

import os
from pathlib import Path


def write_text_atomic(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` via a temp file and ``os.replace`` (readers never see half a file)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
