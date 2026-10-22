"""Learner process: receives trajectory chunks, trains the model, publishes weights.

Each learner owns one trainable agent. It:
1. collects exactly ``batch_chunks`` chunks from its intake (``ChunkIntake``: on a node the
   ``net.local.ConnectionIntake``, which decodes and checks frames; SP4's queue path ``QueueIntake``, which
   decodes ``TrajectoryChunk.to_payload()`` dicts);
2. sets the algorithm's progress (share of the env-step budget) and trains;
3. publishes new weights (numpy ``WeightPayload``, with the scripted teacher's
   ``teacher_active`` flag) through its outputs (``LearnerOutputs``): the node's shared-memory
   weight cache (``NodeOutputs``), or SP4's size-1 worker mailboxes, newest wins (``QueueOutputs``);
4. sends checkpoint payloads (numpy weights + trainer-state bytes, see
   ``make_checkpoint_payload``) and metrics through the outputs, and the final
   snapshot through the outputs when it stops.

Nothing this process sends contains a torch tensor (R6-02).

Progress and stopping: ``progress = min(1, env_step_counter / total_timesteps)`` from the global env-step
counter ``progress_counter.value`` (on a node: the hub's counter, through ``ConnectionIntake.value``; SP4's
local mode: the shared counter that workers increment); the hub (SP4: the main process) is the only budget
authority and sets ``stop_event`` at the budget, so the learner runs until ``stop_event``. Without a counter
(``None``, SP2's distributed roles until T7.2): ``progress = consumed_samples / total_timesteps``, where
``consumed_samples`` counts this learner's ACT slots (decisions, not env steps), and the learner stops by
itself once ``consumed_samples >= total_timesteps``.

With a counter, the env-step value used for progress never goes backward, also across a resume (R29): the learner
uses the largest value it has seen, records it in its trainer state (``progress_env_steps``, see
``make_checkpoint_payload``), and a resumed learner starts from the recorded value. The hub stamps a checkpoint with
its budget counter, which may lag a node's estimate (R28), so a resumed counter can start below the value the learner
last trained with. The value is in env steps, so a resume with another ``total_timesteps`` follows the new budget.
Trainer states without the key (SP2-SP4) resume from the counter alone.

``version_floor`` (SP5, R9): a learner generation starts at the hub's version floor, so versions never go back.

Policy lag (SP5 spec 4.7, 8): ``train`` records carry the lag mean / p50 / p95 / max of the batch and
``dropped_stale_chunks``; with ``learner.max_policy_lag`` the intake drops chunks older than
``current - max_policy_lag`` before decoding them (spec 4.7).
"""

from __future__ import annotations

import io
import logging
import threading
import time
from collections.abc import Callable, Sequence
from queue import Empty, Full
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from colosseum.algorithms.base import BaseAlgorithm
from colosseum.core.config import LearnerConfig
from colosseum.core.types import (
    TrajectoryChunk,
    WeightPayload,
    state_dict_from_numpy,
    state_dict_to_numpy,
    validate_slot_structure,
)
from colosseum.utils.process import SHUTDOWN_GRACE_SEC, parent_alive

if TYPE_CHECKING:
    from colosseum.net.local import ChunkIntake, LearnerOutputs

logger = logging.getLogger(__name__)

# The final snapshot must be on the queue before the main process stops waiting for the
# children (SHUTDOWN_GRACE_SEC after stop_event) and starts terminating them.
FINAL_CHECKPOINT_TIMEOUT_SEC = SHUTDOWN_GRACE_SEC - 2.0

# The learner's own trainer-state key (R29): the env-step value it last used for progress.
PROGRESS_ENV_STEPS_KEY = "progress_env_steps"


def make_checkpoint_payload(agent_id: str, algorithm: BaseAlgorithm, final: bool = False, *,
                            progress_env_steps: int | None = None) -> dict:
    """Checkpoint snapshot that may cross a process boundary: numpy weights + bytes.

    The trainer state (optimizer, LR progress, scaler, kickstart, policy_version,
    consumed_samples) is serialized with ``torch.save`` into bytes, because it
    contains tensors (R6-02). ``progress_env_steps`` (R29): the env-step value the learner last used for progress,
    stored under that key next to the algorithm's state (``apply_resume_state`` takes it out again).
    """
    state = algorithm.state_dict()
    if progress_env_steps is not None:
        state[PROGRESS_ENV_STEPS_KEY] = int(progress_env_steps)
    buf = io.BytesIO()
    torch.save(state, buf)
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


def apply_resume_state(algorithm: BaseAlgorithm, resume_state: dict) -> int:
    """Load weights and trainer state produced by ``resolve_resume`` into ``algorithm``.

    Without a trainer state, only ``policy_version`` is restored (from meta.json), so
    checkpoint ids keep increasing after a resume.

    Returns the trainer state's recorded ``progress_env_steps`` (R29; taken out before the algorithm loads the
    state), 0 without a trainer state or without the key (SP2-SP4).
    """
    model = algorithm.model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.load_state_dict(state_dict_from_numpy(resume_state["model_state"]))
    blob = resume_state.get("trainer_state")
    recorded = 0
    if blob is not None:
        state = torch.load(io.BytesIO(blob), map_location=device, weights_only=True)
        recorded = int(state.pop(PROGRESS_ENV_STEPS_KEY, 0))
        algorithm.load_state_dict(state)
    else:
        state = algorithm.state_dict()
        state["policy_version"] = int(resume_state.get("policy_version", 0))
        algorithm.load_state_dict(state)
    logger.info(f"Resumed from {resume_state.get('source')} at policy_version {algorithm.policy_version}")
    return recorded


def set_policy_version(algorithm: BaseAlgorithm, version: int) -> None:
    """Set ``algorithm``'s policy version (R9: a new learner generation starts at the hub's version floor); the
    rest of its state is unchanged (a ``state_dict`` round trip, as ``apply_resume_state`` does without a trainer
    state)."""
    state = algorithm.state_dict()
    state["policy_version"] = int(version)
    algorithm.load_state_dict(state)


def learner_process(
    *,
    agent_id: str,
    algorithm_factory: Callable[[], BaseAlgorithm],
    intake: ChunkIntake,
    outputs: LearnerOutputs,
    config: LearnerConfig,
    stop_event: Any,
    checkpoint_interval: int = 0,
    resume_state: dict | None = None,
    progress_counter: Any | None = None,
    total_timesteps: int = 0,
    weight_sync_interval: float = 5.0,
    version_floor: int = 0,
    recovery_interval_sec: float | None = None,
) -> None:
    """Main learner process function (see module docstring).

    Args:
        agent_id: the trainable agent this learner owns
        algorithm_factory: factory to create the algorithm (includes network)
        intake: where decoded, checked chunks come from (``net.local.ConnectionIntake`` on a node,
            ``QueueIntake`` on SP4's queue path); ``collect_batch`` takes exactly ``batch_chunks`` of them
        outputs: where weights, checkpoints and metrics go (``NodeOutputs`` / ``QueueOutputs``)
        config: learner configuration
        stop_event: ``.is_set()`` / ``.wait(t)``: the run stops (``ConnectionIntake.stop_flag`` on a node)
        checkpoint_interval: send a checkpoint every N policy versions (0 = only the final one)
        resume_state: optional ``checkpoint_manager.resolve_resume`` result (numpy
            weights, trainer-state bytes or None, policy_version) to resume from
        progress_counter: ``.value`` = the global env-step counter (the hub's, through ``ConnectionIntake``;
            SP4's ``SharedCounter``); None: progress is ``consumed_samples / total_timesteps`` (SP2 distributed)
        total_timesteps: ``training.total_timesteps`` (0 = no budget: progress stays 0)
        weight_sync_interval: the workers' ``rollout.weight_sync_interval_sec`` (``QueueOutputs.finish``)
        version_floor: the first version this learner may publish (R9: the hub's max announced + 1); the
            policy version is raised to it before the first publish
        recovery_interval_sec: recovery points (T4.2); None: none
    """
    logger.info(f"Learner [{agent_id}]: starting on device={resolve_device(config.device)}")

    algorithm = algorithm_factory()

    # On resume both counters continue: train_step from the restored policy_version,
    # consumed_samples (the distributed budget) from the trainer state.
    # progress_env_steps (R29): the largest env-step value used for progress, carried over a resume.
    consumed_samples = 0
    progress_env_steps = 0
    if resume_state is not None:
        progress_env_steps = apply_resume_state(algorithm, resume_state)
        consumed_samples = int(algorithm.state_dict().get("consumed_samples", 0))
    if version_floor > algorithm.policy_version:
        logger.info(f"Learner [{agent_id}]: policy version {algorithm.policy_version} raised to the hub's floor "
                    f"{version_floor}")
        set_policy_version(algorithm, version_floor)
    train_step = int(algorithm.policy_version)

    def recorded() -> int | None:
        """The trainer state's ``progress_env_steps`` (None without a counter: SP2's distributed progress)."""
        return progress_env_steps if progress_counter is not None else None

    total_chunks_received = 0
    last_ckpt_version = -1

    # Off-policy: create replay buffer if algorithm requires it
    replay_buffer = algorithm.create_replay_buffer(config.queue_size * 4)

    try:
        _publish_weights(algorithm, agent_id, outputs)    # the first weights (the workers' first-weights wait)

        while not stop_event.is_set():
            if (progress_counter is None and total_timesteps > 0
                    and consumed_samples >= total_timesteps):
                logger.info(f"Learner [{agent_id}]: consumed {consumed_samples} samples; budget reached")
                break

            # Block until exactly batch_chunks chunks arrived (None: stop requested). The version changes only in a
            # train step, so the intake's max_policy_lag bound is exact for the whole batch.
            intake.note_policy_version(algorithm.policy_version)
            chunks = collect_batch(intake, config.batch_chunks, stop_event, agent_id=agent_id)
            if chunks is None:
                break
            total_chunks_received += len(chunks)
            consumed_samples += sum(c.num_acts for c in chunks)
            # Policy lag of the batch, measured before this train step bumps the version.
            lag = lag_metrics(algorithm.policy_version, [c.policy_version for c in chunks])

            if progress_counter is not None:
                progress_env_steps = max(progress_env_steps, int(progress_counter.value))
                progress = _progress(progress_env_steps, total_timesteps)
            else:
                progress = _progress(consumed_samples, total_timesteps)
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
            metrics.update(lag)
            metrics["dropped_stale_chunks"] = int(intake.dropped.get("stale", 0))   # this learner incarnation

            # Publish updated weights at the configured interval
            if train_step % config.weight_push_interval == 0:
                _publish_weights(algorithm, agent_id, outputs)

            if outputs.checkpoints and checkpoint_interval > 0:
                pv = algorithm.policy_version
                if pv > 0 and pv % checkpoint_interval == 0 and pv != last_ckpt_version:
                    payload = make_checkpoint_payload(agent_id, algorithm, progress_env_steps=recorded())
                    if outputs.send_checkpoint(payload, kind="periodic", block=False):
                        last_ckpt_version = pv

            metrics["agent_id"] = agent_id
            metrics["train_step"] = train_step
            metrics["chunks_received"] = total_chunks_received
            metrics["consumed_samples"] = consumed_samples
            outputs.send_metrics(metrics)       # never blocks on a full sink

            if train_step % 10 == 0:
                logger.info(
                    f"Learner [{agent_id}]: step={train_step}, chunks={total_chunks_received}, "
                    f"progress={progress:.3f}, loss={metrics.get('total_loss', 0.0):.4f}"
                )
    finally:
        outputs.finish(stop_event)

    # Final snapshot on every stop (R3-07); the outputs deliver it (a node: header + frame on the control
    # connection; SP4: a blocking put, flushed while the main process reads it).
    if outputs.checkpoints and outputs.send_checkpoint(
            make_checkpoint_payload(agent_id, algorithm, final=True, progress_env_steps=recorded()),
            kind="final", block=True):
        logger.info(f"Learner [{agent_id}]: sent final checkpoint v{algorithm.policy_version}")

    logger.info(f"Learner [{agent_id}]: finished. Total train_steps={train_step}")


def _progress(done: int, total_timesteps: int) -> float:
    """Share of the budget used: ``done`` (env steps with a counter, else ``consumed_samples``) of
    ``total_timesteps``; 0 without a budget."""
    if total_timesteps <= 0:
        return 0.0
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


def lag_metrics(current: int, versions: Sequence[int]) -> dict[str, float]:
    """Policy lag (``current - policy_version``) of the chunks of one trained batch (spec 4.7, 8): mean, p50, p95
    (``numpy.percentile``, linear interpolation) and max. Written to every ``train`` record."""
    lags = np.asarray([int(current) - int(v) for v in versions], dtype=np.float64)
    return {"policy_lag_mean": float(lags.mean()), "policy_lag_p50": float(np.percentile(lags, 50)),
            "policy_lag_p95": float(np.percentile(lags, 95)), "policy_lag_max": float(lags.max())}


def collect_batch(
    q: Any,
    batch_size: int,
    stop_event: Any,
    poll_interval: float = 0.5,
    agent_id: str | None = None,
) -> list[TrajectoryChunk] | None:
    """Block until exactly ``batch_size`` chunks arrived; decode payloads.

    ``q`` is a queue of chunk v2 payload dicts (SP4) or a chunk intake (``ChunkIntake``) whose items are
    ``TrajectoryChunk`` objects, decoded and checked already. Waits in ``poll_interval`` slices and returns None as
    soon as ``stop_event`` is set (a partial batch is dropped: no training on incomplete batches).
    Every decoded payload's slot structure is checked (``validate_slot_structure``):
    a broken chunk raises ``ValueError`` naming its agent instead of silently
    corrupting the V-trace targets. With ``agent_id``, a payload of another agent (a
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
        if isinstance(payload, TrajectoryChunk):     # from an intake: decoded and checked already
            chunks.append(payload)
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


def _publish_weights(algorithm: BaseAlgorithm, agent_id: str, outputs: LearnerOutputs) -> None:
    """Publish the current weights (numpy, with the scripted teacher's ``teacher_active``) through ``outputs``."""
    outputs.publish_weights(WeightPayload.from_model(
        agent_id, algorithm.policy_version, algorithm.model,
        teacher_active=bool(getattr(algorithm, "teacher_active", False))))
