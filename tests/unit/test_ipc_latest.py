"""Newest-wins weight mailboxes (T2.3, regression for R2-05 / R3-10)."""
import multiprocessing as mp
import queue
import time

import numpy as np

from colosseum.core.ipc import drain_latest, put_latest
from colosseum.learner.learner import _push_weights
from dataflow_helpers import TinyModel, publish_versions


def test_put_latest_replaces_the_stale_item():
    q = queue.Queue(maxsize=1)
    for version in range(5):
        assert put_latest(q, version)
    assert drain_latest(q) == 4
    assert drain_latest(q) is None


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
            self.model = TinyModel()
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
    last = None
    try:
        assert done.wait(60)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            item = drain_latest(mailbox)
            if item is not None:
                last = item
            if last is not None and last.policy_version == 50:
                break
            time.sleep(0.01)
    finally:
        publisher.join(timeout=10)
    assert last is not None and last.policy_version == 50
    assert np.all(last.state_dict["w"] == 50)
