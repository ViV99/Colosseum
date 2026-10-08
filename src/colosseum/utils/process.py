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
    and die if the main process disappears (R3-17, R6-07).

    ``start_process`` starts children with SIGINT blocked; ignoring it first discards a
    Ctrl-C that is pending since then, and only then is it unblocked (CPython's
    resource_tracker pattern), so this process and its own children have SIGINT ignored
    with a normal signal mask. A parent that died before ``PR_SET_PDEATHSIG`` was set
    sends no signal any more, so that case exits right here.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    set_parent_death_signal(signal.SIGTERM)
    if not parent_alive():
        os._exit(1)


def start_process(proc: Any) -> None:
    """``proc.start()`` with SIGINT blocked in this thread meanwhile.

    The child inherits the blocked mask, so a Ctrl-C cannot interrupt it from its first
    instruction (spawn bootstrap and imports included) until ``init_child_process`` ignores
    SIGINT and unblocks it. The target of every process started this way must therefore
    run ``init_child_process`` (``run_child`` and the subprocess-env loop do). A Ctrl-C
    reaching this process during the start stays pending and is delivered to its handler
    after the unblock, so it is not lost.
    """
    if not hasattr(signal, "pthread_sigmask"):
        proc.start()
        return
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        proc.start()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def parent_alive() -> bool:
    """False once the process that started this one has exited (True in the main process)."""
    parent = mp.parent_process()
    return parent is None or parent.is_alive()


def flush_queue(q: Any, warn_after: float, poll: float = 0.1) -> bool:
    """Close ``q`` and wait until its feeder thread has written every item into the pipe.

    ``mp.Queue.join_thread`` blocks until a reader takes a payload larger than the pipe
    buffer, i.e. forever once the reader is gone. Here the wait runs in ``poll``-second
    slices and gives up only once the parent process has died: then the remaining items
    are abandoned (``cancel_join_thread``) so this process can exit, and False is returned.
    While the parent lives the items are never abandoned (the parent reads them, or
    terminates this process); a warning is logged once after ``warn_after`` seconds.
    Objects without ``join_thread`` (e.g. ``queue.Queue``) need no flush.
    """
    if not hasattr(q, "join_thread"):
        return True
    q.close()
    joiner = threading.Thread(target=q.join_thread, name="queue-flush", daemon=True)
    joiner.start()
    warn_at = time.monotonic() + warn_after
    while True:
        joiner.join(poll)
        if not joiner.is_alive():
            return True
        if not parent_alive():
            logger.warning("Parent process is gone; abandoning unread queue items")
            q.cancel_join_thread()
            return False
        if warn_at is not None and time.monotonic() >= warn_at:
            logger.warning(f"Queue items still unread after {warn_after:.0f} s; waiting for the parent process")
            warn_at = None


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
        self._pipe: tuple[int, int] | None = None
        self._watcher: threading.Thread | None = None
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

        The handler only records the signal and writes a byte to a self-pipe; a watcher
        thread started here sets ``stop_event``. The handler runs in the main thread between
        bytecodes, possibly while that thread holds the event's non-reentrant lock (inside
        ``stop_event.set()`` / ``is_set()``), so setting the event there could deadlock.
        Callers log the signal (``received_signal``). Off the main thread (where Python
        cannot install handlers) this is a no-op.
        """
        if threading.current_thread() is not threading.main_thread():
            logger.debug("Not on the main thread; signal handlers not installed")
            return
        if self._pipe is not None:
            logger.debug("Signal handlers already installed")
            return
        read_fd, write_fd = os.pipe()
        os.set_blocking(write_fd, False)
        self._pipe = (read_fd, write_fd)
        self._watcher = threading.Thread(target=self._watch, args=(read_fd,), name="stop-on-signal", daemon=True)
        self._watcher.start()

        def _handler(signum, _frame):
            if self.received_signal is None:
                self.received_signal = signum
            try:
                os.write(write_fd, b"\0")
            except OSError:  # pipe full: a wake-up is already pending
                pass

        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old_handlers[sig] = signal.signal(sig, _handler)

    def _watch(self, read_fd: int) -> None:
        while True:
            try:
                data = os.read(read_fd, 64)
            except OSError:
                return
            if not data:  # write end closed by restore_signal_handlers
                return
            self._stop_event.set()

    def restore_signal_handlers(self) -> None:
        for sig, handler in self._old_handlers.items():
            signal.signal(sig, handler)
        self._old_handlers.clear()
        if self._pipe is not None:
            read_fd, write_fd = self._pipe
            self._pipe = None
            os.close(write_fd)  # the watcher reads EOF and exits
            watcher, self._watcher = self._watcher, None
            if watcher is not None:
                watcher.join(1.0)
            if watcher is not None and watcher.is_alive():
                # Closing the fd under a still-reading thread could hand its number to an
                # unrelated file the thread would then read from; leave it open instead.
                logger.warning("Signal watcher thread did not exit; leaving its pipe open")
            else:
                os.close(read_fd)

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
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return not self.alive()

    def kill_remaining(self, timeout: float = 1.0) -> list[str]:
        """terminate() every live child, wait up to ``timeout`` for all of them together, then
        kill() the survivors and wait up to ``timeout`` again: about 2 x ``timeout`` at most,
        however many stragglers. Returns the names that had to be stopped."""
        remaining = self.alive()
        for name in remaining:
            self._procs[name].terminate()
        self._join_all(remaining, timeout)
        survivors = self.alive()
        for name in survivors:
            self._procs[name].kill()
        self._join_all(survivors, timeout)
        return remaining

    def _join_all(self, names: list[str], timeout: float) -> None:
        deadline = time.monotonic() + timeout
        for name in names:
            self._procs[name].join(max(0.0, deadline - time.monotonic()))
