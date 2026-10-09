"""Inter-process helpers: numpy <-> torch conversion and payload checks.

Rule (spec block 2): nothing that crosses a process boundary may contain a
``torch.Tensor``. Tensors put on an ``mp.Queue`` (or passed as ``Process``
arguments under spawn) are shared through file descriptors served by the
sending process, so the receiver crashes when the sender has already exited
(R6-02). Everything goes through numpy payloads.

This module is the single place that converts between torch and numpy:
``TrajectoryChunk.to_payload``, ``WeightPayload``, ``state_dict_to_numpy``,
``colosseum.networks.state.state_to_numpy`` and the learner / launcher all use
the helpers below.

Conversion rules:
- torch -> numpy always copies (the payload never aliases live tensors such as
  model parameters, which the optimizer keeps updating while an ``mp.Queue``
  feeder thread may still be pickling the payload). ``bfloat16`` (and other
  reduced-precision float dtypes numpy lacks) is upcast to ``float32``, so a
  round trip returns ``float32``, not the original dtype.
- numpy -> torch shares memory with a writable, C-contiguous array (zero copy:
  received payload arrays are owned by the receiver) and copies otherwise, so
  read-only arrays (e.g. from ``np.frombuffer``) never trigger torch's
  non-writable-array warning.

The module also holds the main process's queue helpers (``QueueReader``,
``release_command_queues``, ``queue_depths``), moved here from ``launcher.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
import multiprocessing.queues
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def put_latest(q: Any, item: Any, timeout: float = 1.0) -> bool:
    """Publish ``item`` into a ``maxsize=1`` queue, replacing a stale item.

    Never blocks for longer than ``timeout`` seconds. Returns ``False`` only if
    the item could not be delivered within ``timeout`` (sustained contention).

    ``mp.Queue.put`` returns before its feeder thread has written the item into
    the pipe, so ``get_nowait`` can miss an item that is still in flight and the
    queue keeps reporting ``Full``. A short blocking ``get`` waits for that flush.

    Only for newest-wins data (weights); worker commands must never go through it.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            pass
        # Check before evicting, so a successful eviction is always followed by a
        # put attempt and the mailbox is never left empty.
        if time.monotonic() >= deadline:
            return False
        try:
            q.get(timeout=0.01)  # evict the stale item (it may still be in flight)
        except queue.Empty:
            pass


def drain_latest(q: Any) -> Any | None:
    """Return the newest item available in ``q`` (draining older ones), or None."""
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            return latest


class SharedCounter:
    """A process-shared 64-bit counter (``mp.Value("q")``).

    Must be handed to child processes as a ``Process`` argument (not through a
    queue), like any ``multiprocessing`` synchronized value.
    """

    def __init__(self, ctx: mp.context.BaseContext | None = None) -> None:
        ctx = ctx if ctx is not None else mp.get_context()
        self._val = ctx.Value("q", 0)

    def add(self, n: int) -> None:
        if n:
            with self._val.get_lock():
                self._val.value += int(n)

    @property
    def value(self) -> int:
        with self._val.get_lock():
            return int(self._val.value)


class BatchedCounter:
    """Accumulates increments locally and flushes them into a SharedCounter.

    Workers call :meth:`add` every env step; the shared (locked) counter is
    touched at most once per ``interval`` seconds. Call :meth:`flush` on exit.
    """

    def __init__(self, counter: SharedCounter, interval: float = 0.5) -> None:
        self._counter = counter
        self._interval = interval
        self._pending = 0
        self._last_flush = time.monotonic()

    def add(self, n: int) -> None:
        self._pending += int(n)
        if time.monotonic() - self._last_flush >= self._interval:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            self._counter.add(self._pending)
            self._pending = 0
        self._last_flush = time.monotonic()


# torch float dtypes without a numpy equivalent; upcast to float32 on export.
_UPCAST_DTYPES = {torch.bfloat16}
for _name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    if hasattr(torch, _name):
        _UPCAST_DTYPES.add(getattr(torch, _name))


def is_namedtuple(x: Any) -> bool:
    """True for instances of ``collections.namedtuple`` / ``typing.NamedTuple`` classes."""
    return isinstance(x, tuple) and hasattr(x, "_fields")


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detached CPU numpy copy of ``t`` (bfloat16/float8 -> float32)."""
    t = t.detach().cpu()
    if t.dtype in _UPCAST_DTYPES:
        t = t.float()
    return t.numpy().copy()


def numpy_to_tensor(a: np.ndarray) -> torch.Tensor:
    """CPU tensor from ``a``: shares memory if writable and C-contiguous, else copies."""
    a = np.asarray(a)
    if not a.flags.writeable or not a.flags.c_contiguous:
        a = np.array(a, copy=True, order="C")
    return torch.from_numpy(a)


def to_numpy_tree(obj: Any) -> Any:
    """Copy of ``obj`` with every ``torch.Tensor`` replaced by a numpy array.

    Recurses into dict / list / tuple (namedtuples keep their type); other
    leaves are returned unchanged.
    """
    if isinstance(obj, torch.Tensor):
        return tensor_to_numpy(obj)
    if isinstance(obj, dict):
        return {k: to_numpy_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_numpy_tree(v) for v in obj]
    if is_namedtuple(obj):
        return type(obj)(*(to_numpy_tree(v) for v in obj))
    if isinstance(obj, tuple):
        return tuple(to_numpy_tree(v) for v in obj)
    return obj


def from_numpy_tree(obj: Any) -> Any:
    """Inverse of :func:`to_numpy_tree`: every ``np.ndarray`` becomes a CPU tensor."""
    if isinstance(obj, np.ndarray):
        return numpy_to_tensor(obj)
    if isinstance(obj, dict):
        return {k: from_numpy_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_numpy_tree(v) for v in obj]
    if is_namedtuple(obj):
        return type(obj)(*(from_numpy_tree(v) for v in obj))
    if isinstance(obj, tuple):
        return tuple(from_numpy_tree(v) for v in obj)
    return obj


def find_tensor(obj: Any, path: str = "item") -> str | None:
    """Return the path of the first ``torch.Tensor`` inside ``obj``, or None.

    Recurses into dict / list / tuple / set and dataclass instances.
    """
    if isinstance(obj, torch.Tensor):
        return path
    if isinstance(obj, dict):
        for k, v in obj.items():
            found = find_tensor(v, f"{path}[{k!r}]")
            if found is not None:
                return found
        return None
    if isinstance(obj, list | tuple | set | frozenset):
        for i, v in enumerate(obj):
            found = find_tensor(v, f"{path}[{i}]")
            if found is not None:
                return found
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            found = find_tensor(getattr(obj, f.name), f"{path}.{f.name}")
            if found is not None:
                return found
    return None


def assert_no_tensors(obj: Any, what: str = "item") -> None:
    """Raise TypeError if ``obj`` contains a ``torch.Tensor`` anywhere."""
    found = find_tensor(obj, what)
    if found is not None:
        raise TypeError(
            f"torch.Tensor at {found}: inter-process data must be numpy/primitives "
            f"(use to_payload() / WeightPayload.from_model() / to_numpy_tree())"
        )


# ---------------------------------------------------------------------------
# Queue helpers of the main process (moved from launcher.py in SP2 T0.1)
# ---------------------------------------------------------------------------


class QueueReader:
    """Drains one ``mp.Queue`` on a helper thread, so the main process never blocks on it.

    ``mp.Queue.get`` reads a whole message in ``recv_bytes`` and ignores any timeout once
    the first bytes are there. A producer killed mid-message (OOM, SIGKILL) therefore
    blocks the reader forever: the main process holds the pipe's write end too, so no EOF
    ever arrives. Here ``drain`` waits at most ``wait`` seconds; a read still in progress
    (a large payload arriving or being unpickled) is collected by a later call. Only the
    shutdown gives up on it (``abandon``, after its deadline), once every producer is gone.

    After ``abandon`` no further item can arrive: the dead producer was killed inside the
    feeder's ``send_bytes``, i.e. while holding the queue's cross-process write lock, so no
    other producer can ever write to that pipe again. Should the stuck read finish anyway,
    the items it got are logged as dropped.
    """

    def __init__(self, q: Any, label: str, dead_producers: Callable[[], list[str]]) -> None:
        self._q = q
        self._label = label
        self._dead_producers = dead_producers
        self._thread: threading.Thread | None = None
        self._items: list = []
        self._error: BaseException | None = None
        self._salvaged: list = []
        self.abandoned = False

    @property
    def busy(self) -> bool:
        """A read is in progress (possibly stuck on an incomplete message)."""
        return self._thread is not None and self._thread.is_alive()

    def _read(self) -> None:
        try:
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                except (EOFError, OSError) as e:
                    if self.abandoned:  # ended by release(): the incomplete item was already reported
                        logger.debug(f"Abandoned read on the {self._label} queue ended: {e!r}")
                    else:
                        logger.warning(f"Dropped an unreadable {self._label} queue item: {e!r}")
                    break
                self._items.append(item)
        except BaseException as e:  # noqa: BLE001 - re-raised by drain() in the main thread
            self._error = e
        if self.abandoned and self._items:
            logger.warning(f"{len(self._items)} item(s) completed on the abandoned {self._label} queue "
                           f"after shutdown gave up on it; dropped")

    def _take(self) -> list:
        items, self._items = self._items, []
        return items

    def drain(self, wait: float = 0.05) -> list:
        """Items read so far; starts a read when the queue has data.

        A new read gets up to ``wait`` seconds to finish, so small items come back from this
        call; a read already in progress is only checked, never waited for, so a stuck
        reader costs nothing per poll. Non-``mp.Queue`` objects (e.g. ``queue.Queue``),
        whose ``get_nowait`` never blocks, are read synchronously.
        """
        if self.abandoned:
            salvaged, self._salvaged = self._salvaged, []
            return salvaged  # complete items read before the stuck message
        if not isinstance(self._q, mp.queues.Queue):
            self._read()
        else:
            if self._thread is None:
                if self._q.empty():
                    return []
                self._thread = threading.Thread(target=self._read, name=f"drain-{self._label}", daemon=True)
                self._thread.start()
                self._thread.join(wait)
            if self._thread.is_alive():
                return []  # still reading: a large item is arriving, or the message is incomplete
            self._thread = None
        error, self._error = self._error, None
        items = self._take()
        if error is not None:
            raise error
        return items

    def abandon(self) -> None:
        """Give up on a read that never completed; the next ``drain`` returns the items read before it."""
        self._salvaged = self._take()
        self.abandoned = True
        dead = self._dead_producers()
        source = f" ({', '.join(dead)} died while sending)" if dead else ""
        logger.error(f"An incomplete item on the {self._label} queue was never completed{source}; "
                     f"it and the rest of that queue are skipped")

    def release(self, timeout: float) -> bool:
        """End an abandoned read; True once no read thread is left (``timeout`` bounds the wait).

        An abandoned read means every producer is gone, so this process holds the pipe's last
        write end (it never writes to the queues it reads): closing it makes the stuck
        ``recv_bytes`` hit end-of-file and the thread exit. The queue can then be closed and
        freed on the calling thread, rather than its semaphores being finalized with this
        daemon thread at interpreter exit. Should a producer still hold a write end, the
        read stays stuck and False is returned.
        """
        if self.busy and self.abandoned:
            try:
                self._q._writer.close()
            except (AttributeError, OSError):
                pass
            self._thread.join(timeout)
        return not self.busy


def _cancel_join(q: Any) -> None:
    try:
        q.cancel_join_thread()
    except (AttributeError, OSError):
        pass


def release_command_queues(queues: list, timeout: float) -> list:
    """Close the queues this process writes to (worker command queues, maxsize 1) and wait,
    bounded by ``timeout`` per step, until their feeder threads have exited.

    A feeder thread still running when its queue is freed, or at interpreter exit, drops the
    last references to the queue's semaphores on that daemon thread, which then unlinks them
    during interpreter shutdown ("leaked semaphore objects" warning). A command its worker
    never took would keep the feeder busy (a large one blocks in the pipe write), so it is
    read back first. A queue still full at the deadline (its reader died holding the read
    lock) or whose feeder does not exit in time is detached instead (``cancel_join_thread``,
    the previous behaviour). Returns the queues that were detached.

    An exception raised meanwhile, including a ``KeyboardInterrupt`` from a second Ctrl+C
    (the launcher has already restored the default signal handlers here), detaches every
    queue whose feeder was not seen to exit and is then re-raised: otherwise interpreter
    exit would wait on the feeder of an unread command larger than the pipe buffer.
    """
    finished: set[int] = set()  # id(q) of queues whose feeder thread was joined
    try:
        return _release_command_queues(queues, timeout, finished)
    except BaseException:
        for q in queues:
            if id(q) not in finished:
                _cancel_join(q)
        raise


def _release_command_queues(queues: list, timeout: float, finished: set[int]) -> list:
    deadline = time.monotonic() + timeout
    detached: list = []
    joiners: list[tuple[Any, threading.Thread]] = []
    for q in queues:
        try:
            while q.full() and time.monotonic() < deadline:
                try:
                    q.get(timeout=0.05)
                except queue.Empty:
                    pass
            if q.full():
                detached.append(q)
                continue
            q.close()
        except Exception as e:  # noqa: BLE001 - best effort: a release never replaces the run's outcome
            logger.debug(f"Could not drain and close a command queue: {e!r}")
            detached.append(q)
            continue
        joiner = threading.Thread(target=q.join_thread, name="queue-release", daemon=True)
        joiner.start()
        joiners.append((q, joiner))
    join_deadline = time.monotonic() + timeout
    for q, joiner in joiners:
        joiner.join(max(0.0, join_deadline - time.monotonic()))
        if joiner.is_alive():
            detached.append(q)
        else:
            finished.add(id(q))
    for q in detached:
        _cancel_join(q)
    if detached:
        logger.debug(f"{len(detached)} command queue(s) could not be drained and closed in time; "
                     f"their feeder threads are detached")
    return detached


def queue_depths(queues: dict[str, Any]) -> dict[str, int]:
    """Approximate items waiting per queue (-1 where the platform cannot tell)."""
    depths = {}
    for name, q in queues.items():
        try:
            depths[name] = int(q.qsize())
        except (NotImplementedError, OSError):
            depths[name] = -1
    return depths
