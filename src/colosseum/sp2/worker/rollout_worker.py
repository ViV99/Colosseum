"""Rollout worker process: a thin process wrapper around :class:`RolloutLoop`.

The wrapper limits torch threads, turns queues into :class:`LoopIO` callbacks and runs the
loop until ``stop_event`` is set (or ``max_env_steps`` env steps were taken), adding its env
steps to the global budget counter. The loop itself (envs, inference, chunking) lives in
``colosseum.sp2.worker.rollout_loop`` and is testable in-process.
"""

from __future__ import annotations

import logging
import queue
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from colosseum.core.ipc import BatchedCounter, SharedCounter, drain_latest
from colosseum.core.threads import configure_torch_threads
from colosseum.metrics.aggregator import WORKER_STATS_INTERVAL_SEC
from colosseum.sp2.core.types import Lineup, MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand
from colosseum.sp2.envs.game import MultiAgentEnv
from colosseum.sp2.networks.model import PolicyModel
from colosseum.sp2.worker.rollout_loop import LATEST_NETWORK_ID, LoopIO, RolloutLoop

__all__ = ["LATEST_NETWORK_ID", "report_worker_stats", "rollout_worker_process"]

logger = logging.getLogger(__name__)


def _drain_commands(command_queue: Any) -> WorkerCommand | None:
    """Newest WorkerCommand, merged over all drained ones.

    Checkpoints are sent to a worker only once (as deltas), so ``new_checkpoints`` of all
    drained commands are merged. Lineups are merged per env: a later command's lineup wins,
    and a ``None`` entry keeps the earlier command's lineup for that env.
    """
    latest: WorkerCommand | None = None
    merged: dict[str, dict[str, Any]] = {}
    lineups: list[Lineup | None] = []
    while True:
        try:
            cmd = command_queue.get_nowait()
        except queue.Empty:
            break
        for aid, ckpts in cmd.new_checkpoints.items():
            merged.setdefault(aid, {}).update(ckpts)
        if len(cmd.lineups) > len(lineups):
            lineups.extend([None] * (len(cmd.lineups) - len(lineups)))
        for e, lineup in enumerate(cmd.lineups):
            if lineup is not None:
                lineups[e] = lineup
        latest = cmd
    if latest is None:
        return None
    return WorkerCommand(lineups=lineups, new_checkpoints=merged)


def report_worker_stats(q, worker_id: int, stats: dict) -> None:
    """Best-effort worker stats for the main process (system metrics). Never blocks."""
    try:
        q.put_nowait({"kind": "worker_stats", "worker_id": int(worker_id),
                      **{k: int(v) for k, v in stats.items()}})
    except queue.Full:
        pass


def rollout_worker_process(
    *,
    worker_id: int,
    env_fn: Callable[[], MultiAgentEnv],
    num_envs: int,
    chunk_length: int,
    agent_ids: list[str],
    agent_roles: Mapping[str, Sequence[str]],
    model_factories: Mapping[str, Callable[[], PolicyModel]],
    trajectory_queues: Mapping[str, Any],
    weight_queues: Mapping[str, Any],
    stop_event: Any,
    weight_sync_interval: float = 5.0,
    torch_threads: int = 1,
    max_env_steps: int = 0,
    env_step_counter: SharedCounter | None = None,
    checkpoint_state_dicts_by_agent: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]] | None = None,
    lineups: Sequence[Lineup],
    results_queue: Any = None,
    command_queue: Any = None,
    seed: int | None = None,
    vec_env_kind: Literal["sync", "subprocess"] = "sync",
    subproc_workers: int | None = None,
    stats_queue: Any = None,
    stats_interval_sec: float = WORKER_STATS_INTERVAL_SEC,
    max_idle_steps: int = 1000,
) -> None:
    """Worker process entry point (see module docstring).

    Chunk payloads (``TrajectoryChunk.to_payload()``) go to ``trajectory_queues[agent_id]``
    (blocking put that gives up when ``stop_event`` is set); weights are drained newest-wins
    from ``weight_queues[agent_id]``; results are put non-blocking on ``results_queue``;
    commands are drained (merged) from ``command_queue``. ``max_env_steps`` limits this
    worker's env steps (0 = until stopped). Env steps are added to ``env_step_counter``
    about every 0.5 s and once more on exit.
    """
    configure_torch_threads(torch_threads)

    def send_chunk(chunk: TrajectoryChunk) -> None:
        payload = chunk.to_payload()  # numpy only across processes
        q = trajectory_queues[chunk.agent_id]
        while not stop_event.is_set():  # waits in short slices so a stop is seen quickly
            try:
                q.put(payload, timeout=0.5)
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
        agent_ids=agent_ids, agent_roles=agent_roles, model_factories=model_factories, io=io,
        lineups=lineups, weight_sync_interval=weight_sync_interval,
        checkpoint_state_dicts_by_agent=checkpoint_state_dicts_by_agent, seed=seed,
        vec_env_kind=vec_env_kind, subproc_workers=subproc_workers, max_idle_steps=max_idle_steps,
    )
    last_stats = [time.monotonic()]

    def should_stop() -> bool:
        now = time.monotonic()
        if stats_queue is not None and now - last_stats[0] >= stats_interval_sec:
            last_stats[0] = now
            report_worker_stats(stats_queue, worker_id, loop.stats)
        return stop_event.is_set()

    try:
        loop.run(should_stop=should_stop, max_env_steps=max_env_steps)
    finally:
        if counter is not None:
            counter.flush()
        try:
            loop.close()
        finally:
            # Detach feeder threads of queues this worker produced to, so undrained
            # items (e.g. chunks a stopped learner never consumed) cannot block exit.
            for q in [*trajectory_queues.values(), results_queue, stats_queue]:
                if q is not None and hasattr(q, "cancel_join_thread"):
                    q.cancel_join_thread()
    logger.info(f"Worker {worker_id}: finished. {loop.stats}")
