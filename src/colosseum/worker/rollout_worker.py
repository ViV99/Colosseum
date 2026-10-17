"""Rollout worker process: a thin process wrapper around :class:`RolloutLoop`.

The wrapper limits torch threads, turns queues into :class:`LoopIO` callbacks and runs the
loop until ``stop_event`` is set (or ``max_env_steps`` env steps were taken), adding its env
steps to the global budget counter. The loop itself (envs, inference, chunking) lives in
``colosseum.worker.rollout_loop`` and is testable in-process.
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
from colosseum.core.types import Lineup, MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand
from colosseum.envs.game import MultiAgentEnv
from colosseum.metrics.aggregator import WORKER_STATS_INTERVAL_SEC
from colosseum.networks.model import PolicyModel
from colosseum.players.registry import BotSpec, FixedPlayers
from colosseum.utils.process import SHUTDOWN_GRACE_SEC, flush_queue
from colosseum.worker.rollout_loop import LATEST_NETWORK_ID, LoopIO, RolloutLoop

__all__ = ["LATEST_NETWORK_ID", "RESULTS_BATCH_MAX", "RESULTS_BATCH_SEC", "ResultBatcher", "report_worker_stats",
           "rollout_worker_process"]

logger = logging.getLogger(__name__)

# Match results travel to the main process in batches (lists), one per RESULTS_BATCH_SEC or RESULTS_BATCH_MAX
# results, whichever comes first (SP4 FX1). Why: the main process reads the results queue in one short burst per
# monitor pass (0.2 s), and the queue's pipe (64 KiB) holds only about 100-140 results; results put one by one pile
# up in this process's feeder buffer, which the queue's semaphore counts, until the queue reports Full (the drops
# diagnostic measured 73-79 % lost on the default tic-tac-toe config). A batch is one queue item, and the feeder
# thread wakes (taking this process's GIL) once per batch, not once per match. Cost: results arrive up to 0.1 s
# later.
RESULTS_BATCH_SEC = 0.1
RESULTS_BATCH_MAX = 256


def _drain_commands(command_queue: Any) -> WorkerCommand | None:
    """Newest WorkerCommand, merged over all drained ones.

    Checkpoints are sent to a worker only once (as deltas), so ``new_checkpoints`` of all
    drained commands are merged, and ``evict`` lists of all drained commands are merged (union,
    first-seen order). Lineups are merged per env: a later command's lineup wins, and a ``None``
    entry keeps the earlier command's lineup for that env.
    """
    latest: WorkerCommand | None = None
    merged: dict[str, dict[str, Any]] = {}
    evict: dict[str, list[str]] = {}
    lineups: list[Lineup | None] = []
    while True:
        try:
            cmd = command_queue.get_nowait()
        except queue.Empty:
            break
        for aid, ckpts in cmd.new_checkpoints.items():
            merged.setdefault(aid, {}).update(ckpts)
        for aid, ckpt_ids in cmd.evict.items():
            bucket = evict.setdefault(aid, [])
            bucket.extend(c for c in ckpt_ids if c not in bucket)
        if len(cmd.lineups) > len(lineups):
            lineups.extend([None] * (len(cmd.lineups) - len(lineups)))
        for e, lineup in enumerate(cmd.lineups):
            if lineup is not None:
                lineups[e] = lineup
        latest = cmd
    if latest is None:
        return None
    return WorkerCommand(lineups=lineups, new_checkpoints=merged, evict=evict)


class ResultBatcher:
    """One worker's match results, put on the results queue as lists (see ``RESULTS_BATCH_SEC``).

    ``add`` keeps a result and puts the batch as soon as ``max_results`` are pending; ``flush_if_due`` puts the
    pending results once ``max_age_sec`` has passed since the last put (the caller calls it regularly, also when no
    match finishes: the worker does so once per loop step); ``flush`` puts whatever is pending (on exit). Never
    blocks: a full queue drops the whole batch, and every result in it is counted in ``dropped`` (the first drop is
    a WARNING, later ones are only counted). ``clock`` (default ``time.monotonic``) is injectable for tests.
    """

    def __init__(self, q: Any, worker_id: int, *, max_results: int = RESULTS_BATCH_MAX,
                 max_age_sec: float = RESULTS_BATCH_SEC, clock: Callable[[], float] = time.monotonic) -> None:
        self._q = q
        self._worker_id = worker_id
        self._max_results = max_results
        self._max_age_sec = max_age_sec
        self._clock = clock
        self._pending: list[MatchResult] = []
        self._last_put = clock()
        self._warned = False
        self.dropped = 0

    def add(self, result: MatchResult) -> None:
        self._pending.append(result)
        if len(self._pending) >= self._max_results:
            self.flush()

    def flush_if_due(self, now: float | None = None) -> None:
        """Put the pending results if ``max_age_sec`` has passed since the last put; ``now``: the caller's reading of
        the same clock (saves a call)."""
        if self._pending and (self._clock() if now is None else now) - self._last_put >= self._max_age_sec:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        batch, self._pending = self._pending, []
        self._last_put = self._clock()
        try:
            self._q.put_nowait(batch)
        except queue.Full:
            self.dropped += len(batch)
            if not self._warned:
                self._warned = True
                logger.warning(f"Worker {self._worker_id}: the results queue is full; dropped {len(batch)} match "
                               f"results (counted in dropped_match_results of the system metrics; further drops "
                               f"are only counted)")


def report_worker_stats(q, worker_id: int, stats: dict, timeout: float | None = None) -> None:
    """Best-effort worker stats for the main process (system metrics). Never blocks, unless ``timeout`` (seconds)
    is given: the final report on exit waits that long for room in the queue."""
    item = {"kind": "worker_stats", "worker_id": int(worker_id), **{k: int(v) for k, v in stats.items()}}
    try:
        if timeout is None:
            q.put_nowait(item)
        else:
            q.put(item, timeout=timeout)
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
    fixed_players: FixedPlayers | None = None,
    teachers: Mapping[str, BotSpec] | None = None,
    run_id: str | None = None,
) -> None:
    """Worker process entry point (see module docstring).

    Chunk payloads (``TrajectoryChunk.to_payload()``) go to ``trajectory_queues[agent_id]``
    (blocking put that gives up when ``stop_event`` is set); weights are drained newest-wins
    from ``weight_queues[agent_id]``; results are put non-blocking on ``results_queue`` in batches (lists of
    ``MatchResult``, ``ResultBatcher``; a full queue drops the batch: its results are counted in the stats as
    ``dropped_results``, the first drop is a WARNING);
    commands are drained (merged) from ``command_queue``. ``max_env_steps`` limits this
    worker's env steps (0 = until stopped). Env steps are added to ``env_step_counter``
    about every 0.5 s and once more on exit. ``fixed_players``: scripted and frozen agents, built
    inside this process (served under ``FIXED_NETWORK_ID``; they never collect). ``teachers``: scripted
    kickstart teachers by student agent (``BotSpec``, never instances; DAgger labels in the chunks).
    ``run_id`` prefixes the match uids (``<run_id>/train/...``). On exit the pending results are put, then a final
    stats report is sent (it counts the drops of that last put), and the results and stats queues are flushed
    (``flush_queue``) so that neither the last results nor the final counters are lost.
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

    batcher = ResultBatcher(results_queue, worker_id) if results_queue is not None else None

    def poll_command() -> WorkerCommand | None:
        return _drain_commands(command_queue)

    counter = BatchedCounter(env_step_counter) if env_step_counter is not None else None
    io = LoopIO(
        send_chunk=send_chunk,
        poll_weights=poll_weights,
        report_result=batcher.add if batcher is not None else None,   # drops are counted by the batcher
        poll_command=poll_command if command_queue is not None else None,
        add_env_steps=counter.add if counter is not None else None,
    )
    logger.info(
        f"Worker {worker_id}: starting with {num_envs} envs ({vec_env_kind}), "
        f"chunk_length={chunk_length}, agents={agent_ids}, torch_threads={torch_threads}, "
        f"fixed players={sorted((fixed_players.bots | fixed_players.frozen) if fixed_players else {})}"
    )
    loop = RolloutLoop(
        worker_id=worker_id, env_fn=env_fn, num_envs=num_envs, chunk_length=chunk_length,
        agent_ids=agent_ids, agent_roles=agent_roles, model_factories=model_factories, io=io,
        lineups=lineups, weight_sync_interval=weight_sync_interval,
        checkpoint_state_dicts_by_agent=checkpoint_state_dicts_by_agent, seed=seed,
        vec_env_kind=vec_env_kind, subproc_workers=subproc_workers, max_idle_steps=max_idle_steps,
        fixed_players=fixed_players, teachers=teachers, run_id=run_id,
    )
    last_stats = [time.monotonic()]

    def worker_stats() -> dict[str, int]:
        stats = loop.stats
        if batcher is not None:
            stats["dropped_results"] += batcher.dropped
        return stats

    def should_stop() -> bool:
        now = time.monotonic()
        if batcher is not None:
            batcher.flush_if_due(now)   # once per loop step, also when no match finished
        if stats_queue is not None and now - last_stats[0] >= stats_interval_sec:
            last_stats[0] = now
            report_worker_stats(stats_queue, worker_id, worker_stats())
        return stop_event.is_set()

    try:
        loop.run(should_stop=should_stop, max_env_steps=max_env_steps)
    finally:
        if counter is not None:
            counter.flush()
        try:
            if batcher is not None:
                batcher.flush()           # the last results, before the final report counts their drops
            if stats_queue is not None:   # the final counters (dropped results, dropped reward episodes)
                report_worker_stats(stats_queue, worker_id, worker_stats(), timeout=1.0)
            loop.close()
        finally:
            # The results and the final stats must reach the main process (it drains them until this process
            # exits); then detach every feeder thread, so undrained items (e.g. chunks a stopped learner never
            # consumed) cannot block exit.
            for q in (results_queue, stats_queue):
                if q is not None:
                    flush_queue(q, warn_after=SHUTDOWN_GRACE_SEC)
            for q in [*trajectory_queues.values(), results_queue, stats_queue]:
                if q is not None and hasattr(q, "cancel_join_thread"):
                    q.cancel_join_thread()
    logger.info(f"Worker {worker_id}: finished. {worker_stats()}")
