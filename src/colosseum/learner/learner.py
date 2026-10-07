"""Learner process: receives trajectory chunks, trains the model, pushes weights.

Each Learner runs in a separate process and owns one trainable agent's
training loop. It:
1. Receives TrajectoryChunks from workers via a mp.Queue
2. Batches them
3. Calls algorithm.train_step() to compute loss and update weights
4. Pushes new weights to worker processes via weight queues
5. Logs metrics
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Callable
from queue import Empty, Full

import torch

from colosseum.algorithms.base import BaseAlgorithm
from colosseum.core.config import LearnerConfig
from colosseum.core.types import TrajectoryChunk, WeightPayload

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
        trajectory_queue: queue to receive TrajectoryChunks from workers
        weight_queues: list of queues to push weights to workers
        config: learner configuration
        stop_event: set to signal the learner to stop
        metrics_queue: optional queue to send metrics to the main process
        total_train_steps: stop after this many training steps (0 = run until stop_event)
        checkpoint_queue: optional queue to send checkpoint snapshots to main process
        checkpoint_interval: save checkpoint every N train steps (0 = disabled)
        resume_state: optional dict with 'state_dict', 'optimizer_state', 'policy_version'
            to resume training from a checkpoint
    """
    logger.info(f"Learner [{agent_id}]: starting on device={resolve_device(config.device)}")

    # Create algorithm and network
    algorithm = algorithm_factory()

    # Resume from checkpoint if provided
    train_step = 0
    if resume_state is not None:
        algorithm.model.load_state_dict(resume_state["state_dict"])
        if "optimizer_state" in resume_state and hasattr(algorithm, "_optimizer"):
            algorithm._optimizer.load_state_dict(resume_state["optimizer_state"])
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
            state_dict_cpu = {
                k: v.cpu().clone()
                for k, v in algorithm.model.state_dict().items()
            }
            optimizer_state = algorithm.optimizer_state_dict
            try:
                checkpoint_queue.put_nowait({
                    "policy_version": pv,
                    "state_dict": state_dict_cpu,
                    "optimizer_state": optimizer_state,
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
    """Collect up to batch_size chunks from the queue.

    Waits up to `timeout` seconds for the first chunk, then collects
    remaining chunks non-blocking up to batch_size.
    """
    chunks: list[TrajectoryChunk] = []

    # Wait for the first chunk with timeout
    try:
        chunk = queue.get(timeout=timeout)
        chunks.append(chunk)
    except Empty:
        return chunks

    # Collect remaining chunks non-blocking
    while len(chunks) < batch_size:
        try:
            chunk = queue.get_nowait()
            chunks.append(chunk)
        except Empty:
            break

    return chunks


def _push_weights(
    algorithm: BaseAlgorithm,
    agent_id: str,
    weight_queues: list[mp.Queue],
) -> None:
    """Push current model weights to all worker weight queues."""
    state_dict = {k: v.cpu().clone() for k, v in algorithm.model.state_dict().items()}
    payload = WeightPayload(
        agent_id=agent_id,
        policy_version=algorithm.policy_version,
        state_dict=state_dict,
    )

    for wq in weight_queues:
        try:
            # Non-blocking put; if queue is full, skip
            # (workers will get the next weight update)
            wq.put_nowait(payload)
        except Full:
            pass
