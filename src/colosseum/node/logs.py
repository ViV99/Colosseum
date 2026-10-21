"""Node logs (SP5 spec 7.7): a remote node's processes log under ``<cache_home>/colosseum/logs/<ca8>/<run_id>/``
(``train``'s local node logs into the run dir's ``logs/``); ``log_tail`` is what a node sends the hub when a process
dies (T4.4 stores it as ``<proc>.log.tail``)."""

from __future__ import annotations

from pathlib import Path

__all__ = ["LOG_TAIL_BYTES", "log_tail", "node_log_dir"]

LOG_TAIL_BYTES = 64 * 1024


def node_log_dir(ca8: str, run_id: str) -> Path:
    from colosseum.net.security import cache_home

    path = cache_home() / "colosseum" / "logs" / ca8 / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_tail(path: Path, limit: int = LOG_TAIL_BYTES) -> bytes:
    """The last ``limit`` bytes of ``path`` (b"" when it does not exist)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read()
    except OSError:
        return b""
