"""Per-process logging: every process writes its own file under ``<run>/logs/``."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ENV_LOG_DIR = "COLOSSEUM_LOG_DIR"            # inherited by grandchildren (subprocess env workers)
ENV_PROCESS_NAME = "COLOSSEUM_PROCESS_NAME"


def setup_process_logging(log_dir: str | Path | None, process_name: str,
                          console_level: int = logging.WARNING) -> None:
    """Configure the root logger of this process. Call it first in every process target.

    - ``<log_dir>/<process_name>.log`` receives INFO and above (when ``log_dir`` is set);
    - stderr receives ``console_level`` and above: WARNING for children, INFO for main.

    Calling it again replaces the previous handlers.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s [%(levelname)s] {process_name} %(name)s: %(message)s")
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(fmt)
    root.addHandler(console)
    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(directory / f"{process_name}.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        logging.getLogger(__name__).info(f"{process_name} started (pid {os.getpid()})")
    logging.captureWarnings(True)


def inherited_log_dir() -> str | None:
    """Log dir exported by the parent process (see ``utils.process.run_child``)."""
    return os.environ.get(ENV_LOG_DIR)
