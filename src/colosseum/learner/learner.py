"""Learner process: receives trajectory chunks, trains the model, pushes weights.

Each Learner runs in a separate process and owns one trainable agent's
training loop. It:
1. Receives chunk payloads (numpy, ``TrajectoryChunk.to_payload()``) from
   workers via a mp.Queue and decodes them
2. Batches them
3. Calls algorithm.train_step() to compute loss and update weights
4. Pushes new weights (numpy ``WeightPayload``) to worker processes via weight queues
5. Sends metrics and numpy checkpoint snapshots to the main process

Nothing this process puts on a queue contains a torch tensor (R6-02).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Callable
from queue import Empty, Full

import torch

from colosseum.algorithms.base import BaseAlgorithm
from colosseum.core.config import LearnerConfig
from colosseum.core.ipc import from_numpy_tree, to_numpy_tree
from colosseum.core.types import TrajectoryChunk, WeightPayload, state_dict_from_numpy, state_dict_to_numpy

logger = logging.getLogger(__name__)


def learner_process(
    agent_id: str,
    algorithm_factory: Callable[[], BaseAlgorithm],
    trajectory_queue: mp.Queue,
    weight_queues: list[mp.Queue],
    config: LearnerConfig,
    stop_event: mp.Event,
    metrics_queue: mp.Queue | None = None,
    total_train_steps: int = 0,
    checkpoint_queue: mp.Queue | None = None,
    checkpoint_interval: int = 0,
    resume_state: dict | None = None,
) -> None:
    """Main learner process function.

    Args:
        agent_id: the trainable agent this learner owns
        algorithm_factory: factory to create the algorithm (includes network)
        trajectory_queue: queue to receive chunk payloads (``TrajectoryChunk.to_payload()``)
        weight_queues: list of queues to push weights to workers
        config: learner configuration
        stop_event: set to signal the learner to stop
        metrics_queue: optional queue to send metrics to the main process
        total_train_steps: stop after this many training steps (0 = run until stop_event)
        checkpoint_queue: optional queue to send checkpoint snapshots to main process
        checkpoint_interval: save checkpoint every N train steps (0 = disabled)
        resume_state: optional dict with 'state_dict' (numpy), 'optimizer_state'
            (numpy tree or None) and 'policy_version' to resume training from a checkpoint
    """
    logger.info(f"Learner [{agent_id}]: starting on device={resolve_device(config.device)}")

    # Create algorithm and network
    algorithm = algorithm_factory()

    # Resume from checkpoint if provided
    train_step = 0
    if resume_state is not None:
        # resume_state is numpy (it crossed the process boundary as a Process argument).
        algorithm.model.load_state_dict(state_dict_from_numpy(resume_state["state_dict"]))
        if resume_state.get("optimizer_state") is not None and hasattr(algorithm, "_optimizer"):
            algorithm._optimizer.load_state_dict(from_numpy_tree(resume_state["optimizer_state"]))
        train_step = resume_state.get("policy_version", 0)
        if hasattr(algorithm, "_policy_version"):
            algorithm._policy_version = train_step
        logger.info(f"Learner [{agent_id}]: resumed from checkpoint at step {train_step}")

    # Set up LR schedule if the algorithm supports it
    if hasattr(algorithm, "setup_lr_schedule") and total_train_steps > 0:
        algorithm.setup_lr_schedule(total_train_steps)

    total_chunks_received = 0

    # Off-policy: create replay buffer if algorithm requires it
    replay_buffer = algorithm.create_replay_buffer(config.queue_size * 4)

    try:
        # Push initial weights to workers
        _push_weights(algorithm, agent_id, weight_queues)

        while not stop_event.is_set():
            if total_train_steps > 0 and train_step >= total_train_steps:
                break

            # Collect a batch of chunks
            chunks = _collect_chunks(
                trajectory_queue,
                batch_size=config.batch_chunks,
                timeout=1.0,
            )

            if not chunks:
                continue

            total_chunks_received += len(chunks)

            # Train step: off-policy adds to buffer, on-policy trains directly
            if replay_buffer is not None:
                for chunk in chunks:
                    replay_buffer.add(chunk)
                if len(replay_buffer) < config.batch_chunks:
                    continue
                batch = replay_buffer.sample(config.batch_chunks)
                metrics = algorithm.train_step(batch)
            else:
                metrics = algorithm.train_step(chunks)
            train_step += 1

            # Push updated weights to all workers at configured interval
            if train_step % config.weight_push_interval == 0:
                _push_weights(algorithm, agent_id, weight_queues)

            # Checkpoint: send state_dict to main process at checkpoint intervals
            pv = algorithm.policy_version
            if (checkpoint_queue is not None
                    and checkpoint_interval > 0
                    and pv > 0
                    and pv % checkpoint_interval == 0):
                optimizer_state = getattr(algorithm, "optimizer_state_dict", None)
                try:
                    checkpoint_queue.put_nowait({
                        "policy_version": pv,
                        "state_dict": state_dict_to_numpy(algorithm.model.state_dict()),
                        "optimizer_state": (
                            None if optimizer_state is None else to_numpy_tree(optimizer_state)
                        ),
                    })
                    logger.info(f"Learner [{agent_id}]: sent checkpoint at version {pv}")
                except Full:
                    logger.warning(f"Learner [{agent_id}]: checkpoint queue full, skipping v{pv}")

            # Log metrics
            if metrics_queue is not None:
                metrics["agent_id"] = agent_id
                metrics["train_step"] = train_step
                metrics["chunks_received"] = total_chunks_received
                try:
                    metrics_queue.put_nowait(metrics)
                except Full:
                    pass  # Non-critical, don't block on metrics

            if train_step % 10 == 0:
                logger.info(
                    f"Learner [{agent_id}]: step={train_step}, "
                    f"chunks={total_chunks_received}, "
                    f"loss={metrics.get('total_loss', 0):.4f}"
                )
    finally:
        # Weight payloads are pickled in full (numpy), so an unread one can keep a
        # weight queue's feeder thread blocked on a full pipe; workers that already
        # stopped never read it. Weights are replaceable: do not wait at exit.
        for wq in weight_queues:
            if hasattr(wq, "cancel_join_thread"):
                wq.cancel_join_thread()

    logger.info(f"Learner [{agent_id}]: finished. Total train_steps={train_step}")


def resolve_device(device_str: str) -> str:
    """Resolve ``learner.device``: ``"auto"`` -> ``"cuda"`` if available else ``"cpu"``; others unchanged."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _collect_chunks(
    queue: mp.Queue,
    batch_size: int,
    timeout: float = 1.0,
) -> list[TrajectoryChunk]:
    """Collect up to batch_size chunk payloads from the queue and decode them.

    Waits up to `timeout` seconds for the first chunk, then collects
    remaining chunks non-blocking up to batch_size.
    """
    chunks: list[TrajectoryChunk] = []
    try:
        chunks.append(TrajectoryChunk.from_payload(queue.get(timeout=timeout)))
    except Empty:
        return chunks
    while len(chunks) < batch_size:
        try:
            chunks.append(TrajectoryChunk.from_payload(queue.get_nowait()))
        except Empty:
            break
    return chunks


def _push_weights(
    algorithm: BaseAlgorithm,
    agent_id: str,
    weight_queues: list,
) -> None:
    """Push current model weights (a numpy WeightPayload) to all worker weight queues."""
    payload = WeightPayload.from_model(agent_id, algorithm.policy_version, algorithm.model)
    for wq in weight_queues:
        try:
            wq.put_nowait(payload)
        except Full:
            pass  # workers will get the next weight update
