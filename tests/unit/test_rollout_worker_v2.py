"""SP2 rollout worker process: queues, numpy payloads, budget, commands (T3.4)."""

from __future__ import annotations

import queue
import threading
import time

import pytest
import torch

from colosseum.core.ipc import SharedCounter
from colosseum.sp2.core.types import Lineup, SeatAssignment, TrajectoryChunk, WorkerCommand
from colosseum.sp2.worker.rollout_worker import _drain_commands, rollout_worker_process
from game_helpers import NumpyOnlyQueue, Tick, TickGame, make_test_model

SCRIPT = [Tick(acting={0}), Tick(acting={0}, rewards={0: 1.0}), Tick(over=True, rewards={0: 1.0})]
ROLE = TickGame(SCRIPT, 1).spec.roles["player"]


@pytest.fixture
def restore_torch_threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def _env_fn():
    return TickGame(SCRIPT, 1)


def _model_factory():
    return make_test_model(ROLE)


def test_worker_runs_to_its_step_limit_and_sends_numpy_payloads(restore_torch_threads):
    traj, weights, results, stats = NumpyOnlyQueue(), NumpyOnlyQueue(), NumpyOnlyQueue(), NumpyOnlyQueue()
    counter = SharedCounter()
    rollout_worker_process(
        worker_id=3, env_fn=_env_fn, num_envs=2, chunk_length=3, agent_ids=["a"],
        agent_roles={"a": ["player"]}, model_factories={"a": _model_factory},
        trajectory_queues={"a": traj}, weight_queues={"a": weights}, stop_event=threading.Event(),
        max_env_steps=20, env_step_counter=counter, lineups=[Lineup("solo", [SeatAssignment("a")])] * 2,
        results_queue=results, stats_queue=stats, stats_interval_sec=0.0, seed=1,
    )
    assert counter.value == 20
    payloads = [traj.get_nowait() for _ in range(traj.qsize())]
    assert payloads and all(isinstance(p, dict) for p in payloads)
    chunk = TrajectoryChunk.from_payload(payloads[0])
    assert chunk.agent_id == "a" and chunk.num_slots == 3
    match_ids = [results.get_nowait().match_id for _ in range(results.qsize())]
    assert match_ids[:2] == ["w3_e0_ep0", "w3_e1_ep0"]
    worker_stats = stats.get_nowait()
    assert worker_stats["kind"] == "worker_stats" and worker_stats["worker_id"] == 3


def test_worker_stops_on_the_stop_event(restore_torch_threads):
    stop = threading.Event()
    traj = NumpyOnlyQueue(maxsize=1)               # fills up: send_chunk must still see the stop
    thread = threading.Thread(target=rollout_worker_process, kwargs=dict(
        worker_id=0, env_fn=_env_fn, num_envs=1, chunk_length=2, agent_ids=["a"],
        agent_roles={"a": ["player"]}, model_factories={"a": _model_factory},
        trajectory_queues={"a": traj}, weight_queues={"a": NumpyOnlyQueue()}, stop_event=stop,
        lineups=[Lineup("solo", [SeatAssignment("a")])],
    ))
    thread.start()
    deadline = time.monotonic() + 30
    while traj.qsize() < 1 and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert traj.qsize() == 1, "the worker never sent a chunk"
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_drain_commands_merges_checkpoints_and_lineups_per_env():
    q = queue.Queue()
    a = Lineup("solo", [SeatAssignment("a")])
    b = Lineup("solo", [SeatAssignment("b")])
    q.put(WorkerCommand(lineups=[a, a, None], new_checkpoints={"a": {"ckpt_v1": {}}}))
    q.put(WorkerCommand(lineups=[None, b], new_checkpoints={"a": {"ckpt_v2": {}}, "b": {"ckpt_v3": {}}}))
    cmd = _drain_commands(q)
    assert cmd.lineups == [a, b, None]
    assert cmd.new_checkpoints == {"a": {"ckpt_v1": {}, "ckpt_v2": {}}, "b": {"ckpt_v3": {}}}
    assert _drain_commands(q) is None
