"""Rollout worker process: a thin process wrapper around :class:`RolloutLoop`.

The wrapper limits torch threads (R2-04), turns queues into :class:`LoopIO`
callbacks and runs the loop until ``stop_event`` is set (or ``max_env_steps``
env steps were taken), adding its env steps to the global budget counter. The
loop itself (envs, inference, chunking) lives in ``colosseum.worker.rollout_loop``
and is testable in-process.
"""

from __future__ import annotations

import logging
import queue
from collections.abc import Callable
from typing import Any

import numpy as np

from colosseum.core.ipc import BatchedCounter, SharedCounter, drain_latest
from colosseum.core.threads import configure_torch_threads
from colosseum.core.types import MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand
from colosseum.envs.base_env import BaseEnv
from colosseum.networks.model import PolicyModel
from colosseum.worker.rollout_loop import LATEST_NETWORK_ID, LoopIO, RolloutLoop

__all__ = ["LATEST_NETWORK_ID", "rollout_worker_process"]

logger = logging.getLogger(__name__)


def _drain_commands(command_queue: Any) -> WorkerCommand | None:
    """Newest WorkerCommand, with ``new_checkpoints`` merged over all drained ones.

    Checkpoints are sent to a worker only once (as deltas), so a coalesced
    command must keep the checkpoints of the commands it replaces (R2-06).
    """
    latest: WorkerCommand | None = None
    merged: dict[str, dict[str, Any]] = {}
    while True:
        try:
            cmd = command_queue.get_nowait()
        except queue.Empty:
            break
        for aid, ckpts in cmd.new_checkpoints.items():
            merged.setdefault(aid, {}).update(ckpts)
        latest = cmd
    if latest is not None:
        latest.new_checkpoints = merged
    return latest


def rollout_worker_process(
    *,
    worker_id: int,
    env_fn: Callable[[], BaseEnv],
    num_envs: int,
    chunk_length: int,
    agent_ids: list[str],
    model_factories: dict[str, Callable[[], PolicyModel]],
    trajectory_queues: dict[str, Any],
    weight_queues: dict[str, Any],
    stop_event: Any,
    gamma: float | dict[str, float] = 0.99,
    weight_sync_interval: float = 5.0,
    torch_threads: int = 1,
    max_env_steps: int = 0,
    env_step_counter: SharedCounter | None = None,
    checkpoint_state_dicts_by_agent: dict[str, dict[str, dict[str, np.ndarray]]] | None = None,
    slot_agent_map: list[list[str]] | None = None,
    slot_network_map: list[list[str]] | None = None,
    collect_mask: list[list[bool]] | None = None,
    results_queue: Any = None,
    command_queue: Any = None,
    seed: int | None = None,
    vec_env_kind: str = "sync",
    subproc_workers: int | None = None,
) -> None:
    """Worker process entry point (see module docstring).

    Queues are adapted to ``LoopIO`` callbacks: chunk payloads
    (``TrajectoryChunk.to_payload()``) go to ``trajectory_queues[chunk.agent_id]``
    (blocking put that gives up when ``stop_event`` is set), weights are
    drained newest-wins from ``weight_queues[agent_id]``, results are put non-blocking on
    ``results_queue`` and commands are drained with merged checkpoints from
    ``command_queue``. ``max_env_steps`` limits this worker's env steps
    (0 = until stopped; local workers run until ``stop_event``, distributed
    workers get a per-worker share of the budget). Env steps are added to
    ``env_step_counter`` (the global budget counter, if given) about every 0.5 s
    and once more on exit.
    """
    configure_torch_threads(torch_threads)

    def send_chunk(chunk: TrajectoryChunk) -> None:
        payload = chunk.to_payload()  # numpy only across processes (R6-02)
        q = trajectory_queues[chunk.agent_id]
        while not stop_event.is_set():
            try:
                q.put(payload, timeout=1.0)
                return
            except queue.Full:
                continue

    def poll_weights(agent_id: str) -> WeightPayload | None:
        return drain_latest(weight_queues[agent_id])

    def report_result(result: MatchResult) -> None:
        try:
            results_queue.put_nowait(result)
        except queue.Full:
            logger.debug(f"Worker {worker_id}: results queue full, dropping {result.match_id}")

    def poll_command() -> WorkerCommand | None:
        return _drain_commands(command_queue)

    counter = BatchedCounter(env_step_counter) if env_step_counter is not None else None
    io = LoopIO(
        send_chunk=send_chunk,
        poll_weights=poll_weights,
        report_result=report_result if results_queue is not None else None,
        poll_command=poll_command if command_queue is not None else None,
        add_env_steps=counter.add if counter is not None else None,
    )
    logger.info(
        f"Worker {worker_id}: starting with {num_envs} envs ({vec_env_kind}), "
        f"chunk_length={chunk_length}, agents={agent_ids}, torch_threads={torch_threads}"
    )
    loop = RolloutLoop(
        worker_id=worker_id, env_fn=env_fn, num_envs=num_envs, chunk_length=chunk_length,
        agent_ids=agent_ids, model_factories=model_factories, io=io, gamma=gamma,
        weight_sync_interval=weight_sync_interval, slot_agent_map=slot_agent_map,
        slot_network_map=slot_network_map, collect_mask=collect_mask,
        checkpoint_state_dicts_by_agent=checkpoint_state_dicts_by_agent, seed=seed,
        vec_env_kind=vec_env_kind, subproc_workers=subproc_workers,
    )
    try:
        loop.run(should_stop=stop_event.is_set, max_env_steps=max_env_steps)
    finally:
        if counter is not None:
            counter.flush()
        try:
            loop.close()
        finally:
            # Detach feeder threads of queues this worker produced to, so undrained
            # items (e.g. chunks a stopped learner never consumed) cannot block exit.
            for q in [*trajectory_queues.values(), results_queue]:
                if q is not None and hasattr(q, "cancel_join_thread"):
                    q.cancel_join_thread()
    logger.info(f"Worker {worker_id}: finished. {loop.stats}")
