"""Queue helpers live in core/ipc.py; a second Ctrl+C during the release detaches feeders (SP2 T0.1)."""
import multiprocessing as mp
import queue

import numpy as np
import pytest

import colosseum.launcher as launcher_module
from colosseum.core import ipc
from colosseum.core.ipc import QueueReader, queue_depths, release_command_queues


class _FakeCommandQueue:
    """Records the release calls; ``interrupt_in`` names the method that raises KeyboardInterrupt."""

    def __init__(self, name: str, interrupt_in: str | None = None) -> None:
        self.name = name
        self.interrupt_in = interrupt_in
        self.calls: list[str] = []

    def _call(self, what: str) -> None:
        self.calls.append(what)
        if self.interrupt_in == what:
            raise KeyboardInterrupt

    def full(self) -> bool:
        self._call("full")
        return False

    def get(self, timeout=None):
        raise queue.Empty

    def close(self) -> None:
        self._call("close")

    def join_thread(self) -> None:
        self._call("join_thread")

    def cancel_join_thread(self) -> None:
        self.calls.append("cancel_join_thread")


class _SyncJoiner:
    """Stand-in for threading.Thread: runs the target on start(); join() of queue ``b`` is interrupted."""

    def __init__(self, target, name=None, daemon=None) -> None:
        self._target = target

    def start(self) -> None:
        self._target()

    def join(self, timeout=None) -> None:
        if self._target.__self__.name == "b":
            raise KeyboardInterrupt

    def is_alive(self) -> bool:
        return False


def test_helpers_moved_out_of_the_launcher():
    for name in ("_QueueReader", "_release_command_queues", "_queue_depths"):
        assert not hasattr(launcher_module, name)
    assert launcher_module.QueueReader is QueueReader
    assert launcher_module.release_command_queues is release_command_queues


def test_interrupt_while_draining_detaches_every_queue_and_reraises():
    queues = [_FakeCommandQueue("a"), _FakeCommandQueue("b", interrupt_in="full"), _FakeCommandQueue("c")]
    with pytest.raises(KeyboardInterrupt):
        release_command_queues(queues, timeout=0.5)
    # a was closed but its feeder was never seen to exit; b and c were not reached: all are detached.
    assert queues[0].calls[:3] == ["full", "full", "close"] and "cancel_join_thread" in queues[0].calls
    assert queues[1].calls == ["full", "cancel_join_thread"]
    assert queues[2].calls == ["cancel_join_thread"]


def test_interrupt_while_joining_keeps_joined_queues_and_detaches_the_rest(monkeypatch):
    monkeypatch.setattr(ipc.threading, "Thread", _SyncJoiner)
    queues = [_FakeCommandQueue("a"), _FakeCommandQueue("b"), _FakeCommandQueue("c")]
    with pytest.raises(KeyboardInterrupt):
        release_command_queues(queues, timeout=0.5)
    drained = ["full", "full", "close", "join_thread"]
    assert queues[0].calls == drained                                 # joined: left alone
    assert queues[1].calls == [*drained, "cancel_join_thread"]
    assert queues[2].calls == [*drained, "cancel_join_thread"]


def test_release_reads_back_an_unread_large_command_and_joins_the_feeder():
    q = mp.Queue(maxsize=1)
    q.put({"payload": np.zeros(200_000, dtype=np.uint8)})  # > 64 KiB: the feeder blocks in the pipe write
    assert release_command_queues([q], timeout=2.0) == []
    assert q._closed
    assert q._thread is None or not q._thread.is_alive()


def test_queue_reader_drains_a_plain_queue_synchronously():
    q: queue.Queue = queue.Queue()
    for i in range(3):
        q.put(i)
    reader = QueueReader(q, "results", dead_producers=lambda: [])
    assert reader.drain() == [0, 1, 2]
    assert reader.drain() == []
    assert not reader.busy


def test_queue_depths_reports_sizes_and_unknown():
    class _NoSize:
        def qsize(self):
            raise NotImplementedError

    q: queue.Queue = queue.Queue()
    q.put(1)
    assert queue_depths({"a": q, "b": _NoSize()}) == {"a": 1, "b": -1}
