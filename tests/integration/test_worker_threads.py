"""Spawned workers and SubprocessVectorEnv children run with the configured torch threads (T2.1).

``conftest.py`` exports ``OMP_NUM_THREADS=1`` to every child, which would make
these tests pass without any explicit thread call. Each test therefore sets
``OMP_NUM_THREADS=3`` before spawning, so only the explicit
``torch.set_num_threads`` calls in the worker / env-child code can produce the
asserted values.
"""
import multiprocessing as mp
import os

import numpy as np
import pytest

from colosseum.envs.subproc_vec_env import SubprocessVectorEnv
from colosseum.worker.rollout_worker import rollout_worker_process
from dataflow_helpers import ThreadProbeEnv, make_tiny_model

_PARENT_OMP = "3"


@pytest.fixture(autouse=True)
def _omp_three(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", _PARENT_OMP)
    yield
    assert os.environ.get("OMP_NUM_THREADS") == _PARENT_OMP  # the parent's value is untouched


def _observations(item) -> np.ndarray:
    """Chunk observations from a queue item (a TrajectoryChunk before T2.2, a payload dict after)."""
    if isinstance(item, dict):
        return np.asarray(item["observations"])
    return item.observations.numpy()


def _run_probe_worker(torch_threads: int, vec_env_kind: str) -> np.ndarray:
    ctx = mp.get_context("spawn")
    tq = ctx.Queue()
    wq = ctx.Queue(maxsize=1)
    stop = ctx.Event()
    proc = ctx.Process(
        target=rollout_worker_process,
        kwargs=dict(
            worker_id=0, env_fn=ThreadProbeEnv, num_envs=1, chunk_length=4,
            agent_ids=["a"], model_factories={"a": make_tiny_model},
            trajectory_queues={"a": tq}, weight_queues={"a": wq}, stop_event=stop,
            torch_threads=torch_threads, vec_env_kind=vec_env_kind, subproc_workers=1,
        ),
        daemon=False,  # a subprocess vec env spawns its own children
    )
    proc.start()
    try:
        item = tq.get(timeout=120)
    finally:
        stop.set()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
            proc.join()
    return _observations(item)


def test_spawned_worker_uses_rollout_torch_threads():
    obs = _run_probe_worker(torch_threads=2, vec_env_kind="sync")
    assert set(obs[:, 0].tolist()) == {2.0}   # intra-op threads in the worker
    assert set(obs[:, 2].tolist()) == {1.0}   # inter-op threads in the worker


def test_subprocess_env_children_use_one_thread():
    obs = _run_probe_worker(torch_threads=2, vec_env_kind="subprocess")
    assert set(obs[:, 0].tolist()) == {1.0}   # torch.get_num_threads() in the env child
    assert set(obs[:, 1].tolist()) == {1.0}   # OMP_NUM_THREADS in the env child


def test_subproc_vec_env_children_threads_and_parent_env_restored():
    before = os.environ.get("OMP_NUM_THREADS")
    vec = SubprocessVectorEnv(ThreadProbeEnv, num_envs=2, num_workers=2)
    try:
        obs, _ = vec.reset_all()
        assert obs[:, 0, 0].tolist() == [1.0, 1.0]
        assert obs[:, 0, 1].tolist() == [1.0, 1.0]
    finally:
        vec.close()
    assert os.environ.get("OMP_NUM_THREADS") == before
