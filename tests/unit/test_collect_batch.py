"""collect_batch blocks for a full batch; the learner never trains on partial batches (T2.4)."""
import queue
import threading
import time

import pytest

from colosseum.core.config import LearnerConfig
from colosseum.learner.learner import collect_batch, learner_process
from dataflow_helpers import CheckedQueue, RecordingAlgorithm, chunk_payload


def _collect_in_thread(q, batch_size, stop):
    out = {}
    thread = threading.Thread(
        target=lambda: out.setdefault("batch", collect_batch(q, batch_size, stop, poll_interval=0.05))
    )
    thread.start()
    return thread, out


def test_collect_batch_blocks_until_the_batch_is_full():
    q, stop = queue.Queue(), threading.Event()
    for version in range(2):
        q.put(chunk_payload(version=version))
    thread, out = _collect_in_thread(q, 3, stop)
    time.sleep(0.3)
    assert thread.is_alive()                       # 2 of 3 chunks: still waiting
    q.put(chunk_payload(version=2))
    thread.join(timeout=5)
    assert [c.behavior_policy_version for c in out["batch"]] == [0, 1, 2]


def test_collect_batch_returns_none_on_stop():
    q, stop = queue.Queue(), threading.Event()
    q.put(chunk_payload())
    thread, out = _collect_in_thread(q, 3, stop)
    time.sleep(0.2)
    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive() and out["batch"] is None


def test_collect_batch_rejects_non_payload_items():
    q = queue.Queue()
    q.put(object())
    with pytest.raises(TypeError):
        collect_batch(q, 1, threading.Event(), poll_interval=0.05)


def test_learner_trains_only_on_full_batches():
    algo = RecordingAlgorithm()
    traj, stop = CheckedQueue(), threading.Event()
    for version in range(7):
        traj.put(chunk_payload(version=version))
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, trajectory_queue=traj,
        weight_queues=[CheckedQueue(maxsize=1)], config=LearnerConfig(batch_chunks=3),
        stop_event=stop,
    ))
    thread.start()
    deadline = time.monotonic() + 10
    while len(algo.batches) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.3)        # the 7th chunk alone must not trigger a train step
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert algo.batches == [3, 3]
