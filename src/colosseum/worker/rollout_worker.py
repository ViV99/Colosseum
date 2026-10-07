"""Rollout worker process: wires multiprocessing queues to a ``RolloutLoop``.

The loop itself (envs, inference, chunking) lives in
``colosseum.worker.rollout_loop`` and is testable in-process; this module only
adapts queues to the loop's ``LoopIO`` callbacks.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
from collections.abc import Callable
from typing import Any

from colosseum.core.types import MatchResult, TrajectoryChunk
from colosseum.envs.base_env import BaseEnv
from colosseum.networks.model import PolicyModel
from colosseum.worker.rollout_loop import LATEST_NETWORK_ID, LoopIO, RolloutLoop

__all__ = ["LATEST_NETWORK_ID", "rollout_worker_process"]

logger = logging.getLogger(__name__)


def _drain_latest(q: Any) -> Any | None:
    """Return the newest item available on ``q`` (dropping older ones), or None."""
    if q is None:
        return None
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            break
    return latest


def rollout_worker_process(
    worker_id: int,
    env_fn: Callable[[], BaseEnv],
    num_envs: int,
    chunk_length: int,
    agent_ids: list[str],
    model_factories: dict[str, Callable[[], PolicyModel]],
    trajectory_queues: dict[str, mp.Queue],
    weight_queues: dict[str, mp.Queue],
    stop_event: mp.Event,
    weight_sync_interval: float = 5.0,
    total_timesteps: int = 0,
    checkpoint_state_dicts_by_agent: dict[str, dict[str, dict]] | None = None,
    slot_network_map: list[list[str]] | None = None,
    collect_mask: list[list[bool]] | None = None,
    slot_agent_map: list[list[str]] | None = None,
    results_queue: mp.Queue | None = None,
    seed: int | None = None,
    command_queue: mp.Queue | None = None,
    vec_env_kind: str = "sync",
    subproc_workers: int | None = None,
) -> None:
    """Worker process entry: build a ``RolloutLoop`` and run it until stopped.

    Args mirror ``RolloutLoop``; queues are adapted to ``LoopIO`` callbacks:
    chunks go to ``trajectory_queues[chunk.agent_id]`` (blocking put that gives
    up when ``stop_event`` is set), weights are drained newest-wins from
    ``weight_queues[agent_id]``, results are put non-blocking on
    ``results_queue`` and commands are drained newest-wins from ``command_queue``.
    ``total_timesteps`` limits this worker's env steps (0 = until stopped).
    """

    def send_chunk(chunk: TrajectoryChunk) -> None:
        tq = trajectory_queues[chunk.agent_id]
        while not stop_event.is_set():
            try:
                tq.put(chunk, timeout=1.0)
                return
            except queue.Full:
                continue

    def report_result(result: MatchResult) -> None:
        try:
            results_queue.put_nowait(result)
        except queue.Full:
            pass  # non-critical, never block the rollout

    io = LoopIO(
        send_chunk=send_chunk,
        poll_weights=lambda aid: _drain_latest(weight_queues[aid]),
        report_result=report_result if results_queue is not None else None,
        poll_command=(lambda: _drain_latest(command_queue)) if command_queue is not None else None,
    )

    loop = RolloutLoop(
        worker_id=worker_id,
        env_fn=env_fn,
        num_envs=num_envs,
        chunk_length=chunk_length,
        agent_ids=agent_ids,
        model_factories=model_factories,
        io=io,
        weight_sync_interval=weight_sync_interval,
        slot_agent_map=slot_agent_map,
        slot_network_map=slot_network_map,
        collect_mask=collect_mask,
        checkpoint_state_dicts_by_agent=checkpoint_state_dicts_by_agent,
        seed=seed,
        vec_env_kind=vec_env_kind,
        subproc_workers=subproc_workers,
    )
    try:
        loop.run(should_stop=stop_event.is_set, max_env_steps=total_timesteps)
    finally:
        try:
            loop.close()
        finally:
            # Detach feeder threads for queues this worker produced to, so undrained
            # data (e.g. chunks a stopped learner never consumed) can't block exit.
            for tq in trajectory_queues.values():
                try:
                    tq.cancel_join_thread()
                except AttributeError:
                    pass
            if results_queue is not None:
                results_queue.cancel_join_thread()
    stats = loop.stats
    logger.info(
        f"Worker {worker_id}: finished. Steps={stats['env_steps']}, chunks_sent={stats['chunks_sent']}"
    )
