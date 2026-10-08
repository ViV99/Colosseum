"""Child-process helpers shared by the launcher and the distributed roles."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from colosseum.utils.logging import ENV_LOG_DIR, ENV_PROCESS_NAME, setup_process_logging

logger = logging.getLogger(__name__)


def run_child(name: str, log_dir: str | None, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Body of every child-process target.

    Sets up logging first, exports the log dir and process name for grandchildren,
    and records a crash traceback in the process's own log before re-raising.
    """
    setup_process_logging(log_dir, name)
    if log_dir is not None:
        os.environ[ENV_LOG_DIR] = str(log_dir)
    os.environ[ENV_PROCESS_NAME] = name
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.exception(f"{name} crashed")
        raise
    logger.info(f"{name} finished")
