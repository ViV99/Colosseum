"""SP2 learner process on chunk v2 payloads (T4.4; ported from SP1's learner tests)."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest
import torch

from colosseum.core.ipc import SharedCounter, assert_no_tensors
from colosseum.sp2.core.config import LearnerConfig
from colosseum.sp2.core.types import TrajectoryChunk, WeightPayload
from colosseum.sp2.learner.learner import (
    FINAL_CHECKPOINT_TIMEOUT_SEC,
    _weight_flush_timeout,
    collect_batch,
    learner_process,
    make_checkpoint_payload,
    resolve_device,
)
from game_helpers import NumpyOnlyQueue, RecordingLearnerAlgorithm, chunk_v2_payload, learner_appo


def _collect_in_thread(q, batch_size, stop):
    out = {}
    thread = threading.Thread(
        target=lambda: out.setdefault("batch", collect_batch(q, batch_size, stop, poll_interval=0.05))
    )
    thread.start()
    return thread, out


def _run_learner(algo, payloads, until, *, batch_chunks, timeout=10.0, **kwargs):
    traj, stop = NumpyOnlyQueue(), threading.Event()
    for payload in payloads:
        traj.put(payload)
    kwargs.setdefault("weight_queues", [NumpyOnlyQueue(maxsize=1)])
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, trajectory_queue=traj,
        config=LearnerConfig(batch_chunks=batch_chunks, device="cpu"), stop_event=stop, **kwargs,
    ))
    thread.start()
    deadline = time.monotonic() + timeout
    while not until() and time.monotonic() < deadline and thread.is_alive():
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_collect_batch_blocks_until_full_and_decodes_chunk_v2():
    q, stop = queue.Queue(), threading.Event()
    for version in range(2):
        q.put(chunk_v2_payload(version=version))
    thread, out = _collect_in_thread(q, 3, stop)
    time.sleep(0.3)
    assert thread.is_alive()                       # 2 of 3 chunks: still waiting
    q.put(chunk_v2_payload(version=2, pattern="ARAR"))
    thread.join(timeout=5)
    batch = out["batch"]
    assert [c.policy_version for c in batch] == [0, 1, 2]
    assert all(isinstance(c, TrajectoryChunk) for c in batch)
    assert [c.num_acts for c in batch] == [3, 3, 2]


def test_collect_batch_returns_none_on_stop_and_rejects_non_payloads():
    q, stop = queue.Queue(), threading.Event()
    q.put(chunk_v2_payload())
    thread, out = _collect_in_thread(q, 3, stop)
    time.sleep(0.2)
    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive() and out["batch"] is None
    bad = queue.Queue()
    bad.put(object())
    with pytest.raises(TypeError):
        collect_batch(bad, 1, threading.Event(), poll_interval=0.05)


def test_collect_batch_rejects_a_chunk_with_a_broken_slot_structure():
    q = queue.Queue()
    q.put(chunk_v2_payload(version=0))
    q.put(chunk_v2_payload(version=1, agent_id="hero", pattern="AAPB"))
    with pytest.raises(ValueError, match=r"agent 'hero'.*policy_version 1.*slot 1: an open ACT followed by a PAD"):
        collect_batch(q, 2, threading.Event(), poll_interval=0.05)


def test_learner_trains_only_on_full_batches():
    algo = RecordingLearnerAlgorithm()
    _run_learner(algo, [chunk_v2_payload(version=v) for v in range(7)], lambda: len(algo.batches) >= 2,
                 batch_chunks=3)
    time.sleep(0.1)
    assert algo.batches == [3, 3]


def test_learner_reports_policy_lag_from_chunk_versions():
    algo, metrics_q = RecordingLearnerAlgorithm(start_version=10), NumpyOnlyQueue()
    _run_learner(algo, [chunk_v2_payload(version=v) for v in (7, 10, 9, 10, 10, 8)],
                 lambda: len(algo.batches) >= 2, batch_chunks=3, metrics_queue=metrics_q)
    metrics = [metrics_q.get_nowait() for _ in range(metrics_q.qsize())]
    assert metrics[0]["policy_lag_mean"] == pytest.approx((3 + 0 + 1) / 3)
    assert metrics[0]["policy_lag_max"] == 3.0
    assert metrics[1]["policy_lag_mean"] == pytest.approx((1 + 1 + 3) / 3)


@pytest.mark.parametrize("counted,expected", [(50, 0.5), (300, 1.0)])
def test_progress_comes_from_the_shared_counter_and_is_capped(counted, expected):
    algo, counter = RecordingLearnerAlgorithm(), SharedCounter()
    counter.add(counted)
    _run_learner(algo, [chunk_v2_payload()], lambda: len(algo.batches) >= 1, batch_chunks=1,
                 progress_counter=counter, total_timesteps=100)
    assert algo.progress_at_train == [expected]


def test_without_a_counter_progress_counts_act_slots_and_the_learner_stops_itself():
    algo, traj = RecordingLearnerAlgorithm(), NumpyOnlyQueue()
    for _ in range(6):
        traj.put(chunk_v2_payload(S=4))           # "AAAB": 3 ACT slots per chunk
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, trajectory_queue=traj,
        weight_queues=[NumpyOnlyQueue(maxsize=1)], config=LearnerConfig(batch_chunks=2, device="cpu"),
        stop_event=threading.Event(), total_timesteps=12,
    ))
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()                  # stopped by itself at 12 consumed ACT slots
    assert algo.progress_at_train == [0.5, 1.0]


def test_learner_queues_carry_only_numpy_with_a_real_appo():
    wq, mq, ckq = NumpyOnlyQueue(maxsize=1), NumpyOnlyQueue(), NumpyOnlyQueue()
    algo = learner_appo()
    _run_learner(algo, [chunk_v2_payload(version=v, pattern="ARTP") for v in range(4)],
                 lambda: ckq.qsize() >= 2, batch_chunks=2, timeout=60, weight_queues=[wq], metrics_queue=mq,
                 checkpoint_queue=ckq, checkpoint_interval=1)
    ckpt = ckq.get_nowait()
    assert all(isinstance(v, np.ndarray) for v in ckpt["model_state"].values())
    assert isinstance(ckpt["trainer_state_bytes"], bytes)
    assert isinstance(wq.get_nowait(), WeightPayload)
    metrics = mq.get_nowait()
    assert metrics["consumed_samples"] == 4 and "ess" in metrics and "pad_frac" in metrics


def test_final_checkpoint_is_sent_on_stop():
    stop, ckq = threading.Event(), NumpyOnlyQueue()
    stop.set()
    learner_process(
        agent_id="a", algorithm_factory=learner_appo, trajectory_queue=NumpyOnlyQueue(),
        weight_queues=[NumpyOnlyQueue(maxsize=1)], config=LearnerConfig(batch_chunks=2, device="cpu"),
        stop_event=stop, checkpoint_queue=ckq, checkpoint_interval=100,
    )
    final = ckq.get_nowait()
    assert final["final"] is True and final["agent_id"] == "a" and final["policy_version"] == 0
    assert ckq.empty()
    assert FINAL_CHECKPOINT_TIMEOUT_SEC == pytest.approx(5.0)


def _resume_state(source) -> dict:
    payload = make_checkpoint_payload("a", source)
    state = {"model_state": payload["model_state"], "trainer_state": payload["trainer_state_bytes"],
             "policy_version": payload["policy_version"], "env_steps": 0, "source": "test"}
    assert_no_tensors(state)
    return state


def test_resume_continues_train_step_consumed_samples_and_lr_progress():
    source = learner_appo(lr_schedule="linear")
    for step in range(2):
        source.train_step([TrajectoryChunk.from_payload(chunk_v2_payload(version=step + v)) for v in range(2)])
    assert source.policy_version == 2 and source.consumed_samples == 12
    built = []

    def factory():
        built.append(learner_appo(lr_schedule="linear"))
        return built[-1]

    counter, mq = SharedCounter(), NumpyOnlyQueue()
    counter.add(400)
    traj = NumpyOnlyQueue()
    for version in range(2):
        traj.put(chunk_v2_payload(version=2 + version))
    stop = threading.Event()
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=factory, trajectory_queue=traj, weight_queues=[NumpyOnlyQueue(maxsize=1)],
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"), stop_event=stop,
        metrics_queue=mq, resume_state=_resume_state(source), progress_counter=counter, total_timesteps=1000,
    ))
    thread.start()
    deadline = time.monotonic() + 60
    while mq.qsize() < 1 and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=10)
    metrics = mq.get_nowait()
    assert metrics["train_step"] == 3 and metrics["consumed_samples"] == 18
    assert built[0].policy_version == 3 and built[0].consumed_samples == 18
    assert metrics["progress"] == pytest.approx(0.4)
    for key, value in source.model.state_dict().items():
        assert value.shape == built[0].model.state_dict()[key].shape


def test_resume_pushes_the_restored_weights():
    source = learner_appo()
    source.train_step([TrajectoryChunk.from_payload(chunk_v2_payload(version=v)) for v in range(2)])
    state = _resume_state(source)
    built, wq, stop = [], NumpyOnlyQueue(maxsize=1), threading.Event()
    stop.set()
    learner_process(
        agent_id="a", algorithm_factory=lambda: built.append(learner_appo()) or built[-1],
        trajectory_queue=NumpyOnlyQueue(), weight_queues=[wq], config=LearnerConfig(batch_chunks=2, device="cpu"),
        stop_event=stop, resume_state=state,
    )
    assert built[0].policy_version == 1
    for key, value in source.model.state_dict().items():
        assert torch.equal(built[0].model.state_dict()[key], value), key
    pushed = wq.get_nowait()
    assert pushed.policy_version == 1
    for key, value in state["model_state"].items():
        assert np.array_equal(pushed.state_dict[key], value), key


def test_weight_flush_timeout_and_device_resolution(monkeypatch):
    assert _weight_flush_timeout(5.0) == 60.0 and _weight_flush_timeout(90.0) == 270.0
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu" and resolve_device("cuda:1") == "cuda:1"
