"""Child-process helpers: entry wrapper, signal policy, supervision and shutdown."""

from __future__ import annotations

import ctypes
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from colosseum.utils.logging import ENV_LOG_DIR, ENV_PROCESS_NAME, setup_process_logging

logger = logging.getLogger(__name__)

# Grace period between setting stop_event and terminating stragglers. Below the 10 s
# promised for SIGTERM (spec §3.5), so terminate/kill still fit into that window.
# Child-side waits that must finish before the main process gives up derive from it
# (e.g. the learner's final-checkpoint put timeout, ``SHUTDOWN_GRACE_SEC - 2.0``).
SHUTDOWN_GRACE_SEC = 7.0
_PR_SET_PDEATHSIG = 1


def set_parent_death_signal(sig: int = signal.SIGTERM) -> bool:
    """Linux: deliver ``sig`` to this process when its parent dies. False elsewhere."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(_PR_SET_PDEATHSIG, int(sig), 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False


def init_child_process() -> None:
    """Signal policy for every child: ignore Ctrl-C (the main process coordinates the stop)
    and die if the main process disappears (R3-17, R6-07)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    set_parent_death_signal(signal.SIGTERM)


def parent_alive() -> bool:
    """False once the process that started this one has exited (True in the main process)."""
    parent = mp.parent_process()
    return parent is None or parent.is_alive()


def flush_queue(q: Any, timeout: float, poll: float = 0.1) -> bool:
    """Close ``q`` and wait until its feeder thread has written every item into the pipe.

    ``mp.Queue.join_thread`` blocks until a reader takes a payload larger than the pipe
    buffer, i.e. forever once the reader is gone. Here the wait ends after ``timeout``
    seconds or as soon as the parent process has died; the remaining items are then
    abandoned (``cancel_join_thread``) so this process can exit. Returns True if flushed.
    Objects without ``join_thread`` (e.g. ``queue.Queue``) need no flush.
    """
    if not hasattr(q, "join_thread"):
        return True
    q.close()
    joiner = threading.Thread(target=q.join_thread, name="queue-flush", daemon=True)
    joiner.start()
    deadline = time.monotonic() + timeout
    while True:
        joiner.join(poll)
        if not joiner.is_alive():
            return True
        if not parent_alive():
            logger.warning("Parent process is gone; abandoning unread queue items")
            break
        if time.monotonic() >= deadline:
            logger.warning(f"Queue items still unread after {timeout:.0f} s; abandoning them")
            break
    q.cancel_join_thread()
    return False


def run_child(name: str, log_dir: str | None, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Body of every child-process target.

    Sets up logging first, applies the child signal policy, exports the log dir and
    process name for grandchildren, and records a crash traceback in the process's
    own log before re-raising.
    """
    setup_process_logging(log_dir, name)
    init_child_process()
    if log_dir is not None:
        os.environ[ENV_LOG_DIR] = str(log_dir)
    os.environ[ENV_PROCESS_NAME] = name
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.exception(f"{name} crashed")
        raise
    logger.info(f"{name} finished")


@dataclass
class ChildFailure:
    name: str
    exitcode: int | None
    log_path: Path | None

    def message(self) -> str:
        where = f", see {self.log_path}" if self.log_path is not None else ""
        return f"{self.name} died (exit {self.exitcode}){where}"


class ProcessSupervisor:
    """Named child processes plus the main-process signal handling."""

    def __init__(self, stop_event: Any, log_dir: str | Path | None = None) -> None:
        self._stop_event = stop_event
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._procs: dict[str, Any] = {}
        self._old_handlers: dict[int, Any] = {}
        self.received_signal: int | None = None

    def add(self, name: str, proc: Any) -> None:
        self._procs[name] = proc

    @property
    def names(self) -> list[str]:
        return list(self._procs)

    def get(self, name: str) -> Any | None:
        return self._procs.get(name)

    def log_path(self, name: str) -> Path | None:
        return self._log_dir / f"{name}.log" if self._log_dir is not None else None

    def install_signal_handlers(self) -> None:
        """SIGINT/SIGTERM set ``stop_event``; the first signal received is remembered.

        The event is set from a helper thread: the handler runs in the main thread between
        bytecodes, possibly while that thread holds the event's non-reentrant lock (inside
        ``stop_event.set()`` / ``is_set()``); setting it directly could deadlock.
        """
        def _handler(signum, _frame):
            if self.received_signal is None:
                self.received_signal = signum
                logger.warning(f"Received {signal.Signals(signum).name}; stopping")
            threading.Thread(target=self._stop_event.set, name="stop-on-signal", daemon=True).start()

        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old_handlers[sig] = signal.signal(sig, _handler)

    def restore_signal_handlers(self) -> None:
        for sig, handler in self._old_handlers.items():
            signal.signal(sig, handler)
        self._old_handlers.clear()

    def first_failure(self, nonzero_only: bool = False) -> ChildFailure | None:
        """The first child that has exited (with a non-zero code if ``nonzero_only``)."""
        for name, proc in self._procs.items():
            code = proc.exitcode
            if code is None or (nonzero_only and code == 0):
                continue
            return ChildFailure(name, code, self.log_path(name))
        return None

    def failures(self, exclude: set[str] | frozenset[str] = frozenset()) -> list[ChildFailure]:
        """Every exited child with a non-zero exit code, except the names in ``exclude``."""
        return [ChildFailure(name, proc.exitcode, self.log_path(name))
                for name, proc in self._procs.items()
                if name not in exclude and proc.exitcode not in (None, 0)]

    def alive(self) -> list[str]:
        return [name for name, proc in self._procs.items() if proc.is_alive()]

    def wait_all(self, timeout: float, poll: Callable[[], None] | None = None) -> bool:
        """Wait up to ``timeout`` seconds for every child to exit, calling ``poll`` meanwhile."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if poll is not None:
                poll()
            if not self.alive():
                return True
            time.sleep(0.05)
        return not self.alive()

    def kill_remaining(self) -> list[str]:
        """terminate(), then kill() whatever is still alive. Returns the names that had to be stopped."""
        remaining = self.alive()
        for name in remaining:
            self._procs[name].terminate()
        for name in remaining:
            self._procs[name].join(1.0)
        for name in self.alive():
            self._procs[name].kill()
            self._procs[name].join(1.0)
        return remaining
