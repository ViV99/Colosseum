"""Newest-wins weight mailboxes between the SP2 learner and workers (SP1 guarantees, T7.1)."""
import multiprocessing as mp
import queue
import time

import numpy as np

from colosseum.core.ipc import drain_latest, put_latest
from colosseum.sp2.core.types import WeightPayload
from colosseum.sp2.learner.learner import _push_weights
from game_helpers import learner_role, make_test_model


def publish_versions(q, n: int, done) -> None:
    """Spawn target: publish WeightPayload v1..vn into a size-1 mailbox as fast as possible."""
    for version in range(1, n + 1):
        assert put_latest(q, WeightPayload("a", version, {"w": np.full(64, version, dtype=np.float32)}))
    done.set()


def test_put_latest_replaces_the_stale_item():
    q = queue.Queue(maxsize=1)
    for version in range(5):
        assert put_latest(q, version)
    assert drain_latest(q) == 4
    assert drain_latest(q) is None


def test_put_latest_follows_every_eviction_with_a_put():
    """An eviction that ends past the deadline must still be followed by a put (mailbox never left empty)."""
    class _SlowEvictQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            item = super().get(block, timeout)
            time.sleep(0.05)  # the eviction finishes after put_latest's deadline
            return item

    q = _SlowEvictQueue(maxsize=1)
    q.put("stale")
    assert put_latest(q, "new", timeout=0.02)
    assert q.get_nowait() == "new"


def test_put_latest_returns_false_after_timeout_when_jammed():
    """A mailbox that stays Full and never yields its item: give up after about ``timeout``."""
    class _JammedQueue:
        def put_nowait(self, item):
            raise queue.Full

        def get(self, block=True, timeout=None):
            time.sleep(timeout or 0)
            raise queue.Empty

    start = time.monotonic()
    assert put_latest(_JammedQueue(), "x", timeout=0.2) is False
    assert 0.2 <= time.monotonic() - start < 0.7


def test_drain_latest_returns_newest_of_many():
    q = queue.Queue()
    for version in range(3):
        q.put(version)
    assert drain_latest(q) == 2
    assert q.empty()


def test_put_latest_never_blocks_without_a_consumer():
    q = mp.get_context("spawn").Queue(maxsize=1)
    start = time.monotonic()
    for version in range(200):
        put_latest(q, version)
    assert time.monotonic() - start < 5.0
    got, deadline = None, time.monotonic() + 5.0
    while got != 199 and time.monotonic() < deadline:
        item = drain_latest(q)   # the last put may still be in the feeder thread
        if item is not None:
            got = item
        time.sleep(0.01)
    assert got == 199


def test_push_weights_leaves_only_the_newest_payload():
    class _Algo:
        def __init__(self):
            self.model = make_test_model(learner_role())
            self.policy_version = 0

    algo, mailbox = _Algo(), queue.Queue(maxsize=1)
    for version in range(1, 6):
        algo.policy_version = version
        _push_weights(algo, "a", [mailbox])
    assert mailbox.get_nowait().policy_version == 5


def test_worker_gets_newest_weights_after_a_burst_of_publishes():
    """A learner process publishes v1..v50 between two syncs; the worker drains v50."""
    ctx = mp.get_context("spawn")
    mailbox, done = ctx.Queue(maxsize=1), ctx.Event()
    publisher = ctx.Process(target=publish_versions, args=(mailbox, 50, done))
    publisher.start()
    try:
        assert done.wait(60)
        publisher.join(timeout=10)  # exit flushes the feeder: v50 is in the pipe
        assert publisher.exitcode == 0  # every put_latest returned True
        last = drain_latest(mailbox)
    finally:
        if publisher.is_alive():
            publisher.kill()
            publisher.join()
    assert last is not None and last.policy_version == 50
    assert np.all(last.state_dict["w"] == 50)
