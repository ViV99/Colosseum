"""Spawned SP2 workers and SubprocessVectorEnv children run with the configured torch threads (T7.1).

``conftest.py`` exports ``OMP_NUM_THREADS=1`` to every child, so each test sets ``OMP_NUM_THREADS=3``
before spawning: only the explicit thread calls in the worker / env-child code can give the
asserted values.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.sp2.core.types import Lineup, SeatAssignment
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, StepResult
from colosseum.sp2.envs.vector import SubprocessVectorEnv
from colosseum.sp2.worker.rollout_worker import rollout_worker_process
from game_helpers import GameTestModel

_PARENT_OMP = "3"
OBS = gymnasium.spaces.Box(0.0, 64.0, (3,), np.float32)
ACT = gymnasium.spaces.Discrete(2)


class ThreadProbeGame(MultiAgentEnv):
    """Solo; the observation reports [torch intra-op threads, OMP_NUM_THREADS, torch inter-op threads]."""

    spec = GameSpec.solo(OBS, ACT)

    def _obs(self) -> np.ndarray:
        omp = float(os.environ.get("OMP_NUM_THREADS", "0"))
        return np.array([torch.get_num_threads(), omp, torch.get_num_interop_threads()], np.float32)

    def reset(self, seed, layout) -> StepResult:
        self._t = 0
        return StepResult(acting={0}, obs={0: self._obs()})

    def step(self, actions) -> StepResult:
        self._t += 1
        if self._t >= 3:
            return StepResult(acting=set(), obs={}, episode_over=True)
        return StepResult(acting={0}, obs={0: self._obs()})


def probe_model() -> GameTestModel:
    return GameTestModel(OBS, ACT)


@pytest.fixture(autouse=True)
def _omp_three(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", _PARENT_OMP)
    yield
    assert os.environ.get("OMP_NUM_THREADS") == _PARENT_OMP


def _run_probe_worker(torch_threads: int, vec_env_kind: str) -> np.ndarray:
    ctx = mp.get_context("spawn")
    tq, wq, stop = ctx.Queue(), ctx.Queue(maxsize=1), ctx.Event()
    proc = ctx.Process(
        target=rollout_worker_process,
        kwargs=dict(
            worker_id=0, env_fn=ThreadProbeGame, num_envs=1, chunk_length=4, agent_ids=["a"],
            agent_roles={"a": ["player"]}, model_factories={"a": probe_model},
            trajectory_queues={"a": tq}, weight_queues={"a": wq}, stop_event=stop,
            lineups=[Lineup("solo", [SeatAssignment("a")])], torch_threads=torch_threads,
            vec_env_kind=vec_env_kind, subproc_workers=1,
        ),
        daemon=False,  # a subprocess vector env spawns its own children
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
    return np.asarray(item["obs"])


def test_spawned_worker_uses_rollout_torch_threads():
    obs = _run_probe_worker(torch_threads=2, vec_env_kind="sync")
    assert set(obs[:, 0].tolist()) == {2.0} and set(obs[:, 2].tolist()) == {1.0}


def test_subprocess_env_children_use_one_thread():
    obs = _run_probe_worker(torch_threads=2, vec_env_kind="subprocess")
    assert set(obs[:, 0].tolist()) == {1.0} and set(obs[:, 1].tolist()) == {1.0}


def test_subprocess_vector_env_children_threads_and_parent_env_restored():
    before = os.environ.get("OMP_NUM_THREADS")
    vec = SubprocessVectorEnv(ThreadProbeGame, num_envs=2, num_workers=2)
    try:
        results = vec.reset({0: (None, "solo"), 1: (None, "solo")})
        assert [results[e].obs[0][0] for e in (0, 1)] == [1.0, 1.0]
        assert [results[e].obs[0][1] for e in (0, 1)] == [1.0, 1.0]
    finally:
        vec.close()
    assert os.environ.get("OMP_NUM_THREADS") == before
