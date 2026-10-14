"""Learner process: receives trajectory chunks, trains the model, pushes weights.

Each learner owns one trainable agent. It:
1. collects exactly ``batch_chunks`` chunk payloads (numpy,
   ``TrajectoryChunk.to_payload()``) from its queue and decodes them;
2. sets the algorithm's progress (share of the env-step budget) and trains;
3. publishes new weights (numpy ``WeightPayload``, with the scripted teacher's
   ``teacher_active`` flag) to every worker's size-1 mailbox (newest wins);
4. sends checkpoint payloads (numpy weights + trainer-state bytes, see
   ``make_checkpoint_payload``) and metrics to the main process, and a final
   snapshot when it stops.

Nothing this process puts on a queue contains a torch tensor (R6-02).

Progress and stopping: in local mode ``progress = min(1, env_step_counter /
total_timesteps)`` from the shared counter that workers increment; the main
process is the only budget authority and sets ``stop_event`` at the budget, so
the learner runs until ``stop_event``. In distributed mode there is no shared
counter: ``progress = consumed_samples / total_timesteps``, where
``consumed_samples`` counts this learner's ACT slots (decisions, not env steps;
one budget semantics for distributed mode is SP5), and the learner stops by
itself once ``consumed_samples >= total_timesteps``.
"""

from __future__ import annotations

import io
import logging
import multiprocessing as mp
import threading
import time
from collections.abc import Callable
from queue import Empty, Full
from typing import Any

import numpy as np
import torch

from colosseum.algorithms.base import BaseAlgorithm
from colosseum.core.config import LearnerConfig
from colosseum.core.ipc import SharedCounter, put_latest
from colosseum.core.types import (
    TrajectoryChunk,
    WeightPayload,
    state_dict_from_numpy,
    state_dict_to_numpy,
    validate_slot_structure,
)
from colosseum.utils.process import SHUTDOWN_GRACE_SEC, flush_queue, parent_alive

logger = logging.getLogger(__name__)

# The final snapshot must be on the queue before the main process stops waiting for the
# children (SHUTDOWN_GRACE_SEC after stop_event) and starts terminating them.
FINAL_CHECKPOINT_TIMEOUT_SEC = SHUTDOWN_GRACE_SEC - 2.0


def make_checkpoint_payload(agent_id: str, algorithm: BaseAlgorithm, final: bool = False) -> dict:
    """Checkpoint snapshot that may cross a process boundary: numpy weights + bytes.

    The trainer state (optimizer, LR progress, scaler, kickstart, policy_version,
    consumed_samples) is serialized with ``torch.save`` into bytes, because it
    contains tensors (R6-02).
    """
    buf = io.BytesIO()
    torch.save(algorithm.state_dict(), buf)
    return {
        "agent_id": agent_id,
        "policy_version": int(algorithm.policy_version),
        "model_state": state_dict_to_numpy(algorithm.model.state_dict()),
        "trainer_state_bytes": buf.getvalue(),
        "final": bool(final),
    }


def send_checkpoint(q, payload: dict, block: bool, timeout: float = FINAL_CHECKPOINT_TIMEOUT_SEC) -> bool:
    """Put a checkpoint payload on ``q``. Returns False (and logs) if the queue stays full."""
    try:
        if block:
            q.put(payload, timeout=timeout)
        else:
            q.put_nowait(payload)
        return True
    except Full:
        level = logging.ERROR if block else logging.WARNING
        logger.log(level, f"Checkpoint queue full; dropped snapshot v{payload.get('policy_version')}")
        return False


def apply_resume_state(algorithm: BaseAlgorithm, resume_state: dict) -> None:
    """Load weights and trainer state produced by ``resolve_resume`` into ``algorithm``.

    Without a trainer state, only ``policy_version`` is restored (from meta.json), so
    checkpoint ids keep increasing after a resume.
    """
    model = algorithm.model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.load_state_dict(state_dict_from_numpy(resume_state["model_state"]))
    blob = resume_state.get("trainer_state")
    if blob is not None:
        algorithm.load_state_dict(torch.load(io.BytesIO(blob), map_location=device, weights_only=True))
    else:
        state = algorithm.state_dict()
        state["policy_version"] = int(resume_state.get("policy_version", 0))
        algorithm.load_state_dict(state)
    logger.info(f"Resumed from {resume_state.get('source')} at policy_version {algorithm.policy_version}")


def learner_process(
    *,
    agent_id: str,
    algorithm_factory: Callable[[], BaseAlgorithm],
    trajectory_queue: mp.Queue,
    weight_queues: list[mp.Queue],
    config: LearnerConfig,
    stop_event: mp.Event,
    metrics_queue: mp.Queue | None = None,
    checkpoint_queue: mp.Queue | None = None,
    checkpoint_interval: int = 0,
    resume_state: dict | None = None,
    progress_counter: SharedCounter | None = None,
    total_timesteps: int = 0,
    weight_sync_interval: float = 5.0,
) -> None:
    """Main learner process function (see module docstring).

    Args:
        agent_id: the trainable agent this learner owns
        algorithm_factory: factory to create the algorithm (includes network)
        trajectory_queue: queue to receive chunk payloads (``TrajectoryChunk.to_payload()``)
        weight_queues: list of queues to push weights to workers
        config: learner configuration
        stop_event: set to signal the learner to stop
        metrics_queue: optional queue to send metrics to the main process
        checkpoint_queue: optional queue for checkpoint payloads (``make_checkpoint_payload``)
            to the main process; a final snapshot is always sent when the loop ends
        checkpoint_interval: send a checkpoint every N policy versions (0 = only the final one)
        resume_state: optional ``checkpoint_manager.resolve_resume`` result (numpy
            weights, trainer-state bytes or None, policy_version) to resume from
        progress_counter: the global env-step counter (local mode); None in
            distributed mode, where progress is ``consumed_samples / total_timesteps``
        total_timesteps: ``training.total_timesteps`` (0 = no budget: progress stays 0)
        weight_sync_interval: the workers' ``rollout.weight_sync_interval_sec``; sizes
            the exit wait for pending weight payloads (see ``_weight_flush_timeout``)
    """
    logger.info(f"Learner [{agent_id}]: starting on device={resolve_device(config.device)}")

    algorithm = algorithm_factory()

    # On resume both counters continue: train_step from the restored policy_version,
    # consumed_samples (the distributed budget) from the trainer state.
    consumed_samples = 0
    if resume_state is not None:
        apply_resume_state(algorithm, resume_state)
        consumed_samples = int(algorithm.state_dict().get("consumed_samples", 0))
    train_step = int(algorithm.policy_version)

    total_chunks_received = 0
    last_ckpt_version = -1

    # Off-policy: create replay buffer if algorithm requires it
    replay_buffer = algorithm.create_replay_buffer(config.queue_size * 4)

    try:
        # Push initial weights to workers
        _push_weights(algorithm, agent_id, weight_queues)

        while not stop_event.is_set():
            if (progress_counter is None and total_timesteps > 0
                    and consumed_samples >= total_timesteps):
                logger.info(f"Learner [{agent_id}]: consumed {consumed_samples} samples; budget reached")
                break

            # Block until exactly batch_chunks chunks arrived (None: stop requested).
            chunks = collect_batch(trajectory_queue, config.batch_chunks, stop_event, agent_id=agent_id)
            if chunks is None:
                break
            total_chunks_received += len(chunks)
            consumed_samples += sum(c.num_acts for c in chunks)
            # Policy lag of the batch, measured before this train step bumps the version.
            lags = [algorithm.policy_version - c.policy_version for c in chunks]

            progress = _progress(progress_counter, consumed_samples, total_timesteps)
            algorithm.set_progress(progress)

            # Train step: off-policy adds to buffer, on-policy trains directly
            if replay_buffer is not None:
                for chunk in chunks:
                    replay_buffer.add(chunk)
                if len(replay_buffer) < config.batch_chunks:
                    continue
                metrics = algorithm.train_step(replay_buffer.sample(config.batch_chunks))
            else:
                metrics = algorithm.train_step(chunks)
            train_step += 1
            metrics["progress"] = float(progress)
            metrics["policy_lag_mean"] = float(np.mean(lags))
            metrics["policy_lag_max"] = float(np.max(lags))

            # Push updated weights to all workers at configured interval
            if train_step % config.weight_push_interval == 0:
                _push_weights(algorithm, agent_id, weight_queues)

            if checkpoint_queue is not None and checkpoint_interval > 0:
                pv = algorithm.policy_version
                if pv > 0 and pv % checkpoint_interval == 0 and pv != last_ckpt_version:
                    if send_checkpoint(checkpoint_queue, make_checkpoint_payload(agent_id, algorithm), block=False):
                        last_ckpt_version = pv

            if metrics_queue is not None:
                metrics["agent_id"] = agent_id
                metrics["train_step"] = train_step
                metrics["chunks_received"] = total_chunks_received
                metrics["consumed_samples"] = consumed_samples
                try:
                    metrics_queue.put_nowait(metrics)
                except Full:
                    pass  # Non-critical, don't block on metrics

            if train_step % 10 == 0:
                logger.info(
                    f"Learner [{agent_id}]: step={train_step}, chunks={total_chunks_received}, "
                    f"progress={progress:.3f}, loss={metrics.get('total_loss', 0.0):.4f}"
                )
    finally:
        _release_weight_queues(
            weight_queues, trajectory_queue, stop_event, timeout=_weight_flush_timeout(weight_sync_interval),
        )

    if checkpoint_queue is not None:
        # Final snapshot on every stop. The main process saves it before tearing children down (R3-07).
        if send_checkpoint(checkpoint_queue, make_checkpoint_payload(agent_id, algorithm, final=True),
                           block=True, timeout=FINAL_CHECKPOINT_TIMEOUT_SEC):
            logger.info(f"Learner [{agent_id}]: sent final checkpoint v{algorithm.policy_version}")
        # Wait until the snapshot is flushed into the pipe while the main process reads it.
        # Given up only once the main process is gone (it terminates us after its grace
        # period otherwise), so this process never hangs.
        flush_queue(checkpoint_queue, warn_after=SHUTDOWN_GRACE_SEC)

    logger.info(f"Learner [{agent_id}]: finished. Total train_steps={train_step}")


def _progress(
    progress_counter: SharedCounter | None,
    consumed_samples: int,
    total_timesteps: int,
) -> float:
    """Share of the budget used: the global counter if given, else ``consumed_samples``."""
    if total_timesteps <= 0:
        return 0.0
    done = progress_counter.value if progress_counter is not None else consumed_samples
    return min(1.0, done / total_timesteps)


# Shortest exit wait for workers to read pending weight payloads (see
# _weight_flush_timeout). A worker silent for longer is assumed gone (e.g.
# crashed), so its queue is abandoned.
_MIN_WEIGHT_FLUSH_TIMEOUT_SEC = 60.0


def _weight_flush_timeout(weight_sync_interval: float) -> float:
    """Exit wait for pending weight payloads: a live worker syncs weights every
    ``weight_sync_interval`` seconds (while this learner drains its chunks), so
    wait for several intervals, and never less than a minute."""
    return max(_MIN_WEIGHT_FLUSH_TIMEOUT_SEC, 3.0 * float(weight_sync_interval))


def _release_weight_queues(
    weight_queues: list,
    trajectory_queue,
    stop_event,
    *,
    timeout: float,
    poll: float = 0.1,
) -> None:
    """Let pending weight payloads reach the workers before this process exits.

    Numpy weight payloads are pickled in full, so a payload larger than the pipe
    buffer keeps an ``mp.Queue`` feeder thread blocked until a worker reads it.

    - ``stop_event`` set: workers are stopping and will never read it, so do not
      wait for the feeders (``cancel_join_thread``). This is the normal local
      exit: the main process sets ``stop_event`` at the global budget.
    - Otherwise (the learner raised while workers still run, or it stopped by
      itself at a ``consumed_samples`` budget with ``mp.Queue`` mailboxes; the
      distributed weight sinks have no feeder): wait until every feeder has flushed.
      Cancelling would kill a feeder mid-message, and a live worker would then
      block forever on the truncated payload. While waiting, keep discarding
      chunks: a worker blocked on this learner's full trajectory queue only
      syncs weights once its put succeeds. The wait ends early if
      ``stop_event`` gets set, and after ``timeout`` seconds the remaining
      feeders are abandoned (a reader silent that long is assumed dead).
    """
    mp_queues = [wq for wq in weight_queues if hasattr(wq, "join_thread")]
    if not stop_event.is_set():
        joiners = []
        for wq in mp_queues:
            wq.close()  # no more puts; join_thread() waits for the feeder to flush
            joiner = threading.Thread(target=wq.join_thread, daemon=True)
            joiner.start()
            joiners.append(joiner)
        deadline = time.monotonic() + timeout
        while any(j.is_alive() for j in joiners) and not stop_event.is_set():
            if not parent_alive():
                logger.warning("Learner: main process is gone; abandoning unread weight payloads")
                break
            if time.monotonic() > deadline:
                logger.warning(f"Learner: weight payloads unread after {timeout:.0f} s; abandoning them")
                break
            try:
                while True:
                    trajectory_queue.get_nowait()
            except Empty:
                pass
            stop_event.wait(poll)
    for wq in mp_queues:
        wq.cancel_join_thread()  # no-op for the feeders that already flushed


def resolve_device(device_str: str) -> str:
    """Resolve ``learner.device``: ``"auto"`` -> ``"cuda"`` if available else ``"cpu"``; others unchanged."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def collect_batch(
    q: Any,
    batch_size: int,
    stop_event: Any,
    poll_interval: float = 0.5,
    agent_id: str | None = None,
) -> list[TrajectoryChunk] | None:
    """Block until exactly ``batch_size`` chunk v2 payloads arrived; decode them.

    Waits in ``poll_interval`` slices and returns None as soon as ``stop_event``
    is set (a partial batch is dropped: no training on incomplete batches).
    Every decoded chunk's slot structure is checked (``validate_slot_structure``):
    a broken chunk raises ``ValueError`` naming its agent instead of silently
    corrupting the V-trace targets. With ``agent_id``, a chunk of another agent (a
    misrouted worker in distributed mode) raises ``ValueError``.
    """
    chunks: list[TrajectoryChunk] = []
    while len(chunks) < batch_size:
        if stop_event.is_set():
            return None
        try:
            payload = q.get(timeout=poll_interval)
        except Empty:
            continue
        if not isinstance(payload, dict):
            raise TypeError(
                f"trajectory queue item must be a chunk payload dict, got {type(payload).__name__}"
            )
        chunk = TrajectoryChunk.from_payload(payload)
        if agent_id is not None and chunk.agent_id != agent_id:
            raise ValueError(f"learner of agent {agent_id!r} received a chunk of agent {chunk.agent_id!r}; "
                             f"check the workers' learner addresses (run-workers -l AGENT=HOST:PORT)")
        validate_slot_structure(chunk)
        chunks.append(chunk)
    return chunks


def _push_weights(
    algorithm: BaseAlgorithm,
    agent_id: str,
    weight_queues: list,
) -> None:
    """Publish the current weights to every worker mailbox (newest wins)."""
    payload = WeightPayload.from_model(agent_id, algorithm.policy_version, algorithm.model,
                                       teacher_active=bool(getattr(algorithm, "teacher_active", False)))
    for wq in weight_queues:
        if not put_latest(wq, payload):
            logger.debug(f"Learner [{agent_id}]: weight mailbox busy, v{payload.policy_version} not delivered")
