"""Policy lag on the learner (SP5 T3.3, spec 4.7, 8): lag percentiles of every trained batch, max_policy_lag drops
too-old chunks before decoding (counted stale), the hub keeps learner-side drops monotonic, a starving
configuration is refused."""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import secrets
import threading
import time

import pytest

from colosseum.core.config import LearnerConfig
from colosseum.core.errors import ConfigError
from colosseum.core.types import TrajectoryChunk
from colosseum.learner.learner import lag_metrics, learner_process
from colosseum.metrics.counters import CounterBook
from colosseum.metrics.hub import MetricsHub
from colosseum.metrics.jsonl import MetricsWriter
from colosseum.net.chunks import ChunkValidator, encode_chunk
from colosseum.net.local import (
    ChildHello,
    ConnectionIntake,
    ControlChannel,
    QueueIntake,
    QueueOutputs,
    address,
    connect,
    listener,
)
from game_helpers import NumpyOnlyQueue, RecordingLearnerAlgorithm, chunk_v2_payload, learner_role

KEY = b"p" * 32


def test_lag_metrics_are_percentiles_of_the_batch():
    m = lag_metrics(10, [7, 10, 9])                      # lags 3, 0, 1
    assert m == pytest.approx({"policy_lag_mean": 4 / 3, "policy_lag_p50": 1.0, "policy_lag_p95": 2.8,
                               "policy_lag_max": 3.0})


def _run(algo, payloads, until, *, batch_chunks, max_policy_lag=None, timeout=10.0):
    """``learner_process`` in a thread over a ``QueueIntake``; stops once ``until(intake)`` holds (PF3-21: no
    wall-clock settle wait)."""
    traj, metrics, stop = NumpyOnlyQueue(), NumpyOnlyQueue(), threading.Event()
    for payload in payloads:
        traj.put(payload)
    intake = QueueIntake(traj, agent_id="a", max_policy_lag=max_policy_lag)
    outputs = QueueOutputs(weight_queues=[NumpyOnlyQueue(maxsize=1)], metrics_queue=metrics)
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, intake=intake, outputs=outputs,
        config=LearnerConfig(batch_chunks=batch_chunks, device="cpu", max_policy_lag=max_policy_lag,
                             weight_push_interval=1),
        stop_event=stop))
    thread.start()
    deadline = time.monotonic() + timeout
    while not until(intake) and time.monotonic() < deadline and thread.is_alive():
        time.sleep(0.02)
    assert until(intake)
    stop.set()
    thread.join(10)
    assert not thread.is_alive()
    return intake, [metrics.get_nowait() for _ in range(metrics.qsize())]


def test_the_learner_reports_lag_percentiles_and_keeps_the_old_keys():
    algo = RecordingLearnerAlgorithm(start_version=10)
    _, metrics = _run(algo, [chunk_v2_payload(version=v) for v in (7, 10, 9)], lambda _: len(algo.batches) >= 1,
                      batch_chunks=3)
    m = metrics[0]
    assert (m["policy_lag_mean"], m["policy_lag_p50"], m["policy_lag_p95"], m["policy_lag_max"]) == \
        pytest.approx((4 / 3, 1.0, 2.8, 3.0))
    assert m["dropped_stale_chunks"] == 0


def test_max_policy_lag_drops_old_chunks_and_counts_them_stale():
    algo = RecordingLearnerAlgorithm(start_version=10)
    intake, metrics = _run(algo, [chunk_v2_payload(version=v) for v in (7, 10, 9, 10, 10, 8)],
                           lambda i: len(algo.batches) >= 2 and i.dropped.get("stale", 0) == 2,
                           batch_chunks=2, max_policy_lag=1)
    assert algo.behavior_versions[:2] == [[10, 9], [10, 10]]     # 7 is older than 10 - 1; 8 older than 12 - 1
    assert metrics[0]["dropped_stale_chunks"] == 1
    assert metrics[0]["policy_lag_max"] == 1.0 and metrics[1]["policy_lag_max"] == 1.0
    assert intake.dropped["stale"] == 2                          # 8 arrived after the second batch


def test_without_max_policy_lag_nothing_is_dropped():
    algo = RecordingLearnerAlgorithm(start_version=50)
    intake, metrics = _run(algo, [chunk_v2_payload(version=v) for v in (0, 1)], lambda _: len(algo.batches) >= 1,
                           batch_chunks=2)
    assert algo.behavior_versions[0] == [0, 1] and intake.dropped.get("stale", 0) == 0
    assert metrics[0]["policy_lag_max"] == 50.0


def test_a_lag_bound_below_the_push_interval_minus_one_is_refused():
    with pytest.raises((ConfigError, ValueError), match="max_policy_lag"):
        LearnerConfig(max_policy_lag=2, weight_push_interval=5)
    assert LearnerConfig(max_policy_lag=4, weight_push_interval=5).max_policy_lag == 4
    assert LearnerConfig(max_policy_lag=0, weight_push_interval=1).max_policy_lag == 0


def _lag_frame(version):
    chunk = TrajectoryChunk.from_payload(chunk_v2_payload(version=version))
    return encode_chunk(chunk, generation=0, node=1, worker=0)


def test_connection_intake_drops_stale_frames_before_decoding():
    name = address("abcdef12", secrets.token_hex(4), 1, "learner-a-i1")
    node_end, learner_end = mp.Pipe()
    intake = ConnectionIntake(listener(name, KEY), ControlChannel(learner_end),
                              ChunkValidator(learner_role(), agent_id="a", teacher=False), authkey=KEY,
                              agent_id="a", generation=1, version_floor=0, capacity=16, cap_bytes=1 << 26,
                              max_policy_lag=1)
    try:
        sender = connect(name, KEY, timeout=5.0)
        sender.send(ChildHello("worker", "w", 1, os.getpid(), "chunks"))
        truncated = _lag_frame(2)
        truncated = truncated[:len(truncated) - 64]      # the header says version 2; the data region is cut
        intake.note_policy_version(10)                   # admissible: version >= 10 - 1
        for frame in (_lag_frame(3), _lag_frame(9), truncated, _lag_frame(10)):
            sender.send_bytes(frame)
        assert [intake.get(timeout=2.0).policy_version for _ in range(2)] == [9, 10]
        # A stale frame is never decoded, so its body is not checked: the truncated one counts stale, not invalid.
        assert intake.dropped["stale"] == 2 and intake.dropped.get("invalid", 0) == 0
        intake.note_policy_version(20)                   # the bound follows the learner
        sender.send_bytes(_lag_frame(9))
        with pytest.raises(queue.Empty):
            intake.get(timeout=0.5)
        deadline = time.monotonic() + 10.0              # the reader may not have taken the frame yet under load
        while intake.dropped.get("stale", 0) < 3 and time.monotonic() < deadline:
            with pytest.raises(queue.Empty):
                intake.get(timeout=0.1)
        assert intake.dropped == {"stale": 3}
        sender.close()
    finally:
        intake.close()
        node_end.close()


def test_metrics_hub_sums_learner_side_chunk_drops_over_incarnations(tmp_path):
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    try:
        hub = MetricsHub(writer=writer, ratings_path=tmp_path / "ratings.json", agent_ids=["main"],
                         total_timesteps=1000, log_interval=1, console_interval_sec=60.0, clock=lambda: 0.0)
        hub.on_intake_stats("main", 1, 3, {"stale": 3})
        hub.on_intake_stats("main", 1, 0, {"stale": 5, "invalid": 1})     # cumulative per incarnation
        hub.on_intake_stats("main", 2, 0, {"stale": 1})                   # a new generation counts from zero
        assert hub.chunk_drops() == {"stale": 6, "invalid": 1}
        back = CounterBook.from_state(hub.intake_book.state())
        assert back.totals() == hub.intake_book.totals()
        assert hub.intake_queue_depths() == {"main": 0}
    finally:
        writer.close()                       # never left to the GC (a ResourceWarning in a later test under -W error)
