"""A node's learner process (SP5 spec 4.4, 4.5, 4.7; rulings R4, R9, R10, R16) and the node-side preparation of
its start state.

The SP4 ``learner_process`` over ``net.local``: chunk frames arrive on the learner's own listener (one connection per
sender, read by ``ConnectionIntake`` with the node's key; validated with the agent's spaces, ``chunk_length`` and the
model's state template, A17); ``LearnerEnvSteps`` (the hub's global counter, LR progress) and ``Stop`` arrive on the
control connection to the node; weights go into the shared-memory cache with a ``VersionPublished`` to the node,
checkpoints as ``CheckpointHeader`` + frame, metrics as ``LearnerMetrics`` (``NodeOutputs``). The listener and the
intake exist before the first ``VersionPublished``: the node routes workers to the learner only at that message.

Start state (built by the node, numpy only; no file is read on the node): a ``resume_state`` from the assignment's
start point (A28: one checkpoint frame, ``resume_state_from_frame``; its trainer state carries R29's
``progress_env_steps``), a teacher (a ``teacher`` blob of
``Welcome.fixed`` -> ``teacher_from_frozen``; a scripted teacher from the config), ``init`` weights (an ``init``
blob -> ``init_from_blob``)."""

from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["CONNECT_TIMEOUT_SEC", "NO_FINAL_REASONS", "init_from_cached", "learner_target", "resume_state_from_frame",
           "teacher_for"]

CONNECT_TIMEOUT_SEC = 10.0
NO_FINAL_REASONS = ("superseded", "hub_lost")   # Stop reasons without a final snapshot (T4.1, P-PF3-19)


def learner_target(*, name: str, log_dir: str, incarnation: int, **kwargs: Any) -> None:
    """Process entry point: process logging and the child signal policy, then the learner body."""
    from colosseum.utils.process import run_child

    run_child(name, log_dir, partial(_learner_main, name=name), incarnation=incarnation, **kwargs)


def _learner_main(*, name: str, incarnation: int, node: int, agent_id: str, generation: int, listener_name: str,
                  node_address: str, authkey: bytes, config: Any, role_spec: Any, spec: Any, seed: int | None,
                  teacher: Any, init_state: Any, resume_state: dict | None, version_floor: int, env_steps: int,
                  num_learners: int, cache_root: str, cache_prefix: str, checkpoint_interval: int,
                  total_timesteps: int, cap_bytes: int, start_point: str = "factory") -> None:
    """The assignment (SP5 T4.1): ``config.learner.device`` is the assigned device (the node replaced it),
    ``generation`` / ``version_floor`` / ``seed`` (``learner_seed(..., generation)``) are this incarnation's;
    ``start_point`` names the start for the log. A ``Stop`` with reason ``superseded`` or ``hub_lost`` ends the
    learner without a final snapshot (P-PF3-19)."""
    from colosseum.core.threads import configure_torch_threads, resolve_learner_threads
    from colosseum.learner.factory import apply_init, build_algorithm
    from colosseum.learner.learner import learner_process, resolve_device
    from colosseum.net.chunks import ChunkValidator
    from colosseum.net.local import ChildError, ChildHello, ConnectionIntake, ControlChannel, NodeOutputs, connect
    from colosseum.net.local import listener as make_listener
    from colosseum.networks.state import state_to_numpy
    from colosseum.node.weights import WeightCacheWriter
    from colosseum.utils.seeding import apply_global_seed

    apply_global_seed(seed)
    device = resolve_device(config.learner.device)
    logger.info(f"learner-{agent_id}: generation {generation}, device {device}, versions from {version_floor}, "
                f"start {start_point}")
    configure_torch_threads(resolve_learner_threads(config.learner.torch_threads, device, config.rollout.num_workers,
                                                    config.rollout.torch_threads, num_learners))
    channel = None
    intake = None
    lst = None

    def open_control() -> Any:
        conn = connect(node_address, authkey, timeout=CONNECT_TIMEOUT_SEC)
        conn.send(ChildHello("learner", name, incarnation, os.getpid(), "control"))
        return ControlChannel(conn)

    try:
        # Everything slow first (model, optimizer, a CUDA context); then the listener, the control hello (the node's
        # acceptor wants it within its handshake bound) and the intake's acceptor, one right after the other. The node
        # routes workers to this listener only at the first VersionPublished, which learner_process sends once the
        # intake runs; a worker's connection is then accepted at once (its sender counts as broken after
        # OPEN_TIMEOUT_SEC).
        algorithm = build_algorithm(config, role_spec, spec, device=device, teacher=teacher)
        if init_state is not None:
            apply_init(algorithm.model, init_state)          # weights only: version 0, fresh optimizer
        validator = ChunkValidator(role_spec, agent_id=agent_id,
                                   teacher=teacher is not None and teacher.kind == "scripted",
                                   chunk_length=config.rollout.chunk_length,
                                   state_template=state_to_numpy(algorithm.model.initial_state(1)))
        lst = make_listener(listener_name, authkey)
        channel = open_control()
        intake = ConnectionIntake(lst, channel, validator, authkey=authkey, agent_id=agent_id, generation=generation,
                                  version_floor=version_floor, capacity=config.learner.queue_size,
                                  cap_bytes=cap_bytes, max_policy_lag=config.learner.max_policy_lag)
        outputs = NodeOutputs(channel, WeightCacheWriter(Path(cache_root), cache_prefix), agent_id=agent_id,
                              generation=generation, cap_bytes=cap_bytes)
        learner_process(agent_id=agent_id, algorithm_factory=lambda: algorithm, intake=intake, outputs=outputs,
                        config=config.learner, stop_event=intake.stop_flag, checkpoint_interval=checkpoint_interval,
                        resume_state=resume_state, progress_counter=_ProgressFloor(intake, env_steps),
                        total_timesteps=total_timesteps,
                        weight_sync_interval=config.rollout.weight_sync_interval_sec, version_floor=version_floor,
                        send_final=lambda: intake.stop_reason not in NO_FINAL_REASONS)
    except Exception as e:
        error = ChildError("crash", f"{type(e).__name__}: {e}", f"learner {agent_id} (generation {generation})")
        try:
            if channel is None:
                channel = open_control()                    # the node learns the reason, not only the death
            channel.send(error)
        except Exception:  # noqa: BLE001 - best effort: the node reports the death without a reason
            logger.exception("cannot report the error to the node")
        raise
    finally:
        if intake is not None:
            intake.close()
        elif lst is not None:
            lst.close()
        if channel is not None:
            channel.close()


class _ProgressFloor:
    """``learner_process``'s ``progress_counter``: the count the node forwards (``ConnectionIntake.value``), at least
    the spawn-time ``env_steps``. The intake's reader thread runs from its construction and may already hold a newer
    count (the node sends one at the control hello), so the spawn value is a floor, never written into the intake
    (FX3)."""

    def __init__(self, intake: Any, floor: int) -> None:
        self._intake = intake
        self._floor = int(floor)

    @property
    def value(self) -> int:
        return max(self._floor, int(self._intake.value))


# ---------------------------------------------------------------------------------------------------------------
# The node side: start state from hub blobs
# ---------------------------------------------------------------------------------------------------------------


def resume_state_from_frame(frame: bytes | bytearray, *, agent_id: str, cap: int, env_steps: int, source: str) -> dict:
    """``learner_process``'s ``resume_state`` from a learner start point (A28: one checkpoint frame of the hub's
    ``Download(kind="start")``, weights plus trainer-state bytes; ``trainer_state`` None = weights only).
    ``FrameError`` (a ValueError) for a frame that is not one, ValueError for another agent's."""
    from colosseum.net.chunks import decode_checkpoint

    payload, _meta = decode_checkpoint(bytearray(frame), cap=cap)
    if payload["agent_id"] != agent_id:
        raise ValueError(f"the start point of {agent_id!r} is a frame of {payload['agent_id']!r}")
    return {"model_state": {k: np.array(v) for k, v in payload["model_state"].items()},
            "trainer_state": payload["trainer_state_bytes"], "policy_version": int(payload["policy_version"]),
            "env_steps": int(env_steps), "source": source}


def teacher_for(config: Any, agent_id: str, *, cached: Any = None, bot: Any = None) -> Any:
    """The kickstart teacher of ``agent_id``: a neural one from a ``teacher`` blob read from the cache (``cached``:
    ``CachedWeights``, its meta naming the architecture and roles), a scripted one (``bot``: ``BotSpec``), or None."""
    from colosseum.learner.factory import TeacherSpec, teacher_from_frozen
    from colosseum.players.registry import FrozenSpec

    if cached is not None:
        meta = cached.meta
        frozen = FrozenSpec(agent_id=meta.agent_id, roles=tuple(meta.roles or ()),
                            networks=dict(meta.architecture or {}),
                            model_state={k: np.array(v) for k, v in cached.state_dict.items()},
                            source=f"hub blob {cached.key.ident}")
        return teacher_from_frozen(config, agent_id, frozen)
    if bot is not None:
        ks = config.get_agent_config(agent_id).kickstart
        return TeacherSpec(kind="scripted", source=f"scripted agent {ks.teacher!r}", lambda_=float(ks.lambda_),
                           decay_steps=int(ks.decay_steps), kl=ks.kl, bot=bot)
    return None


def init_from_cached(config: Any, agent_id: str, cached: Any) -> Any:
    """``init`` weights of ``agent_id`` from an ``init`` blob read from the cache (``CachedWeights``)."""
    from colosseum.learner.factory import init_from_blob

    return init_from_blob(config, agent_id, {k: np.array(v) for k, v in cached.state_dict.items()},
                          f"hub blob {cached.key.ident}")
