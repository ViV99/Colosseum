"""Global env-step budget: shared counter, worker reporting, learner progress (T2.5)."""
import multiprocessing as mp
import threading
import time

import pytest
import torch

from colosseum.core.config import LearnerConfig
from colosseum.core.ipc import BatchedCounter, SharedCounter
from colosseum.learner.learner import learner_process
from colosseum.worker.rollout_worker import rollout_worker_process
from dataflow_helpers import (
    CheckedQueue,
    EnvFactory,
    GridStepEnv,
    RecordingAlgorithm,
    add_to_counter,
    chunk_payload,
    make_loop,
    make_tiny_model,
)


@pytest.fixture
def restore_torch_threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def test_shared_counter_sums_across_spawned_processes():
    ctx = mp.get_context("spawn")
    counter = SharedCounter(ctx)
    procs = [ctx.Process(target=add_to_counter, args=(counter, 500)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    assert counter.value == 1000


def test_batched_counter_flushes_on_interval_and_on_demand():
    counter = SharedCounter()
    lazy = BatchedCounter(counter, interval=3600.0)
    lazy.add(5)
    lazy.add(7)
    assert counter.value == 0
    lazy.flush()
    assert counter.value == 12
    eager = BatchedCounter(counter, interval=0.0)
    eager.add(3)
    assert counter.value == 15


def test_rollout_loop_reports_env_steps_every_step():
    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(3,)), make_tiny_model, num_envs=2)
    for _ in range(5):
        loop.step()
    loop.close()
    assert col.env_steps == [2] * 5


def test_worker_adds_its_env_steps_to_the_shared_counter(restore_torch_threads):
    counter = SharedCounter()
    rollout_worker_process(
        worker_id=0, env_fn=EnvFactory(GridStepEnv, lengths=(3,)), num_envs=2, chunk_length=4,
        agent_ids=["a"], model_factories={"a": make_tiny_model},
        trajectory_queues={"a": CheckedQueue()}, weight_queues={"a": CheckedQueue(maxsize=1)},
        stop_event=threading.Event(), max_env_steps=20, env_step_counter=counter,
    )
    assert counter.value == 20


def _run_learner_until(algo, payloads, until, batch_chunks, **kwargs):
    traj, stop = CheckedQueue(), threading.Event()
    for payload in payloads:
        traj.put(payload)
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, trajectory_queue=traj,
        weight_queues=[CheckedQueue(maxsize=1)], config=LearnerConfig(batch_chunks=batch_chunks),
        stop_event=stop, **kwargs,
    ))
    thread.start()
    deadline = time.monotonic() + 10
    while not until() and time.monotonic() < deadline:
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_learner_progress_comes_from_the_shared_counter():
    algo, counter = RecordingAlgorithm(), SharedCounter()
    counter.add(50)
    _run_learner_until(algo, [chunk_payload()], lambda: len(algo.batches) >= 1, batch_chunks=1,
                       progress_counter=counter, total_timesteps=100)
    assert algo.progress_at_train == [0.5]


def test_learner_progress_is_capped_at_one():
    algo, counter = RecordingAlgorithm(), SharedCounter()
    counter.add(300)
    _run_learner_until(algo, [chunk_payload()], lambda: len(algo.batches) >= 1, batch_chunks=1,
                       progress_counter=counter, total_timesteps=100)
    assert algo.progress_at_train == [1.0]


def test_without_a_counter_progress_is_consumed_samples_and_the_learner_stops_itself():
    algo, traj = RecordingAlgorithm(), CheckedQueue()
    for _ in range(6):
        traj.put(chunk_payload(T=4))
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, trajectory_queue=traj,
        weight_queues=[CheckedQueue(maxsize=1)], config=LearnerConfig(batch_chunks=2),
        stop_event=threading.Event(), total_timesteps=16,
    ))
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()              # stopped by itself at 16 consumed samples
    assert algo.progress_at_train == [0.5, 1.0]
