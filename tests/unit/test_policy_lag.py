"""The learner reports policy lag = learner version - chunk behavior version (T2.6)."""
import threading
import time

import pytest

from colosseum.core.config import LearnerConfig
from colosseum.learner.learner import learner_process
from dataflow_helpers import CheckedQueue, RecordingAlgorithm, chunk_payload


def test_learner_emits_policy_lag_mean_and_max():
    algo = RecordingAlgorithm(start_version=10)
    traj, metrics_q, stop = CheckedQueue(), CheckedQueue(), threading.Event()
    for version in (7, 10, 9, 10, 10, 8):
        traj.put(chunk_payload(version=version))
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, trajectory_queue=traj,
        weight_queues=[CheckedQueue(maxsize=1)], config=LearnerConfig(batch_chunks=3),
        stop_event=stop, metrics_queue=metrics_q,
    ))
    thread.start()
    deadline = time.monotonic() + 10
    while len(algo.batches) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=5)
    metrics = [metrics_q.get_nowait() for _ in range(metrics_q.qsize())]
    assert metrics[0]["policy_lag_mean"] == pytest.approx((3 + 0 + 1) / 3)   # learner at v10
    assert metrics[0]["policy_lag_max"] == 3.0
    assert metrics[1]["policy_lag_mean"] == pytest.approx((1 + 1 + 3) / 3)   # learner at v11
    assert metrics[1]["policy_lag_max"] == 3.0
