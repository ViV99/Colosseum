"""Distributed (multi-machine) training over gRPC.

This wires the previously-unused gRPC transport and weight store into real
training by providing thin *queue-like adapters* so the existing
``learner_process`` and ``rollout_worker_process`` run unchanged:

- Workers send trajectory chunks to a learner's gRPC TrajectoryService and pull
  fresh weights from a gRPC WeightStore.
- Learners receive chunks via a local queue filled by their TrajectoryService,
  train, and push weights to the WeightStore.

Roles (separate processes / machines):
- ``serve-weight-store`` : a WeightStore gRPC server (see cli.py).
- ``run-learner``        : one trainable agent — TrajectoryService + training +
                           weight push to the store.
- ``run-workers``        : N rollout workers feeding the learner(s).

Scope: distributed mode currently runs self-play with the latest policy of each
agent (opponents = latest weights pulled from the store). The dynamic
coordinator-driven matchmaking (PFSP / historical-checkpoint opponents, C1) is a
single-machine feature; closing that loop across machines would require running
the coordinator as its own service and is left as future work.

Budget and progress: there is no shared env-step counter across machines. Each
distributed learner uses progress = consumed_samples / training.total_timesteps
(its own transitions, which drives the LR schedule) and stops by itself once
consumed_samples >= total_timesteps. Each worker stops after
total_timesteps / num_workers env steps. These numbers differ from the local
mode budget (env steps summed over workers); a single semantics comes with the
hub in SP5.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import signal
import threading
import time
from functools import partial

import torch

from colosseum.core.config import ColosseumConfig, config_hash, load_config
from colosseum.core.run_dir import RunDir
from colosseum.core.types import WeightPayload
from colosseum.utils.logging import setup_process_logging
from colosseum.utils.process import run_child
from colosseum.utils.seeding import apply_global_seed, learner_seed

logger = logging.getLogger(__name__)


# =====================================================================
# Queue-like gRPC adapters (so learner_process / rollout_worker_process
# work unchanged whether the backend is mp.Queue or gRPC)
# =====================================================================


class GRPCTrajectorySink:
    """``.put``-compatible sink that ships chunk payloads to a learner over gRPC.

    Transient RPC failures (e.g. the learner restarting or shutting down) drop
    the chunk rather than crashing the worker — trajectory data is replaceable,
    and a worker should survive a learner blip.
    """

    def __init__(self, transport, agent_id: str) -> None:
        self._transport = transport
        self._agent_id = agent_id

    def _send(self, chunk) -> None:
        import grpc
        try:
            self._transport.send_chunk(self._agent_id, chunk)
        except grpc.RpcError as e:
            logger.debug(f"Dropping chunk for {self._agent_id}: {e.code()}")

    def put(self, chunk, timeout: float | None = None) -> None:  # noqa: ARG002
        self._send(chunk)

    def put_nowait(self, chunk) -> None:
        self._send(chunk)

    def cancel_join_thread(self) -> None:  # parity with mp.Queue shutdown path
        pass


class GRPCWeightSource:
    """``.get_nowait``-compatible source that polls the weight store.

    Returns a :class:`WeightPayload` only when a *newer* policy version is
    available, otherwise raises ``queue.Empty`` — matching how the worker drains
    an mp weight queue (take the latest, stop on Empty).
    """

    def __init__(self, store, agent_id: str) -> None:
        self._store = store
        self._agent_id = agent_id
        self._last_version = -1

    def get_nowait(self) -> WeightPayload:
        version = self._store.get_version(self._agent_id)
        if version <= self._last_version:
            raise queue.Empty
        payload = self._store.get(self._agent_id)
        if payload is None:
            raise queue.Empty
        self._last_version = payload.policy_version
        return payload


class GRPCWeightSink:
    """``.put_nowait``-compatible sink that publishes weights to the store."""

    def __init__(self, store, agent_id: str) -> None:
        self._store = store
        self._agent_id = agent_id

    def put_nowait(self, payload: WeightPayload) -> None:
        self._store.put(self._agent_id, payload)


# =====================================================================
# Learner role
# =====================================================================


def run_distributed_learner(
    config_path: str,
    agent_id: str,
    traj_port: int,
    weight_store_address: str,
    overrides: dict | None = None,
) -> None:
    """Run one trainable agent's learner as a standalone gRPC service.

    Starts a TrajectoryService on ``traj_port`` (workers send chunks here),
    trains with the configured algorithm, and pushes weights to the WeightStore
    at ``weight_store_address``.
    """
    from colosseum.core.registry import build_model, import_class, validate_config
    from colosseum.core.threads import configure_torch_threads, resolve_learner_threads
    from colosseum.learner.learner import learner_process, resolve_device
    from colosseum.transport.grpc_transport import serve_trajectory_receiver
    from colosseum.weight_store.grpc_store import GRPCWeightStore

    setup_process_logging(None, f"learner-{agent_id}", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    run_dir = RunDir.create(config, config_path, role=f"learner-{agent_id}")
    setup_process_logging(run_dir.logs, f"learner-{agent_id}", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
    acfg = config.get_agent_config(agent_id)
    validate_config(acfg)
    max_mb = config.transport.grpc_max_message_mb

    device = resolve_device(acfg.learner.device)
    # The learner role does not know which workers share its machine, so with
    # learner.torch_threads unset it assumes none (num_workers=0).
    configure_torch_threads(resolve_learner_threads(
        acfg.learner.torch_threads, device, num_workers=0,
        worker_threads=acfg.rollout.torch_threads, num_learners=1,
    ))

    # Trajectory inbox filled by the gRPC server, drained by learner_process.
    chunk_queue: queue.Queue = queue.Queue(maxsize=acfg.learner.queue_size)
    traj_server = serve_trajectory_receiver(
        chunk_queue, port=traj_port, max_message_mb=max_mb,
    )

    store = GRPCWeightStore(weight_store_address, max_message_mb=max_mb)
    weight_sink = [GRPCWeightSink(store, agent_id)]

    # Checkpoint persistence (drained off-thread so training never blocks). Always on:
    # the learner's final snapshot is saved even without periodic checkpoints.
    from colosseum.coordinator.checkpoint_manager import CheckpointManager
    checkpoint_queue: queue.Queue = queue.Queue(maxsize=16)
    coordinator_ckpt = CheckpointManager(
        base_dir=run_dir.checkpoints,
        pool_size=config.self_play.pool_size,
    )

    algo_class_path = acfg.algorithm.algorithm_class
    algo_cls = import_class(algo_class_path)
    teacher_path = acfg.training.kickstart_teacher

    def algorithm_factory():
        model = build_model(acfg)
        kickstart = None
        if teacher_path:
            from colosseum.bc.kickstart import KickstartLoss
            teacher = build_model(acfg)
            teacher.load_state_dict(torch.load(teacher_path, weights_only=True, map_location=device))
            teacher.to(device)
            kickstart = KickstartLoss(
                teacher,
                initial_lambda=acfg.training.kickstart_lambda,
                decay_steps=acfg.training.kickstart_decay_steps,
                direction=acfg.training.kickstart_kl,
            )
        kwargs = {"device": device, "pin_memory": acfg.learner.pin_memory}
        if kickstart is not None:
            kwargs["kickstart"] = kickstart
        return algo_cls(model, acfg.algorithm, **kwargs)

    stop_event = threading.Event()
    _install_stop_signal_handlers(stop_event)

    # Seed right before learner_process builds the model (validate_config above also draws
    # from the RNGs). Same per-agent stream as a local-mode learner.
    agent_index = config.get_trainable_agent_ids().index(agent_id)
    apply_global_seed(learner_seed(config.training.seed, agent_index))

    # Drain checkpoint payloads (learner.make_checkpoint_payload) to disk in the background.
    cfg_hash = config_hash(config)

    def _save(data: dict) -> None:
        """Persist one payload; a failure is logged and never kills the caller."""
        trainer_state = data.get("trainer_state_bytes") if config.checkpoint.save_optimizer else None
        try:
            coordinator_ckpt.save(
                agent_id=agent_id,
                policy_version=int(data["policy_version"]),
                model_state=data["model_state"],
                trainer_state=trainer_state,
                # env_steps is null: a distributed learner has no global env-step count
                # (its consumed_samples counts transitions of its own seats, a different
                # quantity), so a resume from it does not seed the env-step budget.
                meta_extra={"final": bool(data.get("final", False)),
                            "networks": acfg.networks.model_dump(mode="json", by_alias=True),
                            "config_hash": cfg_hash,
                            "env_steps": None},
            )
        except Exception:  # noqa: BLE001 - one failed save must not stop checkpointing
            logger.exception(f"Distributed learner [{agent_id}]: failed to save checkpoint "
                             f"v{data.get('policy_version')}")

    def _drain_checkpoints():
        while not stop_event.is_set():
            try:
                data = checkpoint_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            _save(data)

    drainer = threading.Thread(target=_drain_checkpoints, daemon=True)
    drainer.start()

    logger.info(
        f"Distributed learner [{agent_id}] serving trajectories on :{traj_port}, "
        f"weights -> {weight_store_address}"
    )
    try:
        learner_process(
            agent_id=agent_id,
            algorithm_factory=algorithm_factory,
            trajectory_queue=chunk_queue,
            weight_queues=weight_sink,
            config=acfg.learner,
            stop_event=stop_event,
            metrics_queue=None,
            progress_counter=None,
            total_timesteps=config.training.total_timesteps,
            checkpoint_queue=checkpoint_queue,
            checkpoint_interval=config.self_play.checkpoint_interval,
            weight_sync_interval=acfg.rollout.weight_sync_interval_sec,
        )
    finally:
        stop_event.set()
        drainer.join()  # it finishes the save in progress, then sees stop_event
        while True:  # the final snapshot arrives after stop_event is set
            try:
                _save(checkpoint_queue.get_nowait())
            except queue.Empty:
                break
        traj_server.stop(0)
        store.close()
        logger.info(f"Distributed learner [{agent_id}] stopped.")


# =====================================================================
# Worker role
# =====================================================================


def _dist_worker_target(*, worker_id: int, log_dir: str | None = None, **kwargs) -> None:
    """Distributed worker process entry point: process logging first."""
    run_child(f"worker-{worker_id}", log_dir, _dist_worker_main, worker_id=worker_id, **kwargs)


def _dist_worker_main(
    *,
    worker_id: int,
    config: ColosseumConfig,
    agent_ids: list[str],
    agent_configs: dict[str, ColosseumConfig],
    weight_store_address: str,
    learner_addresses: dict[str, str],
    stop_event,
    total_timesteps: int,
    slot_agent_map: list[list[str]],
) -> None:
    """Worker process body: gRPC clients in, rollout_worker_process unchanged."""
    import sys
    sys.path.insert(0, ".")

    from colosseum.core.registry import build_model
    from colosseum.launcher import _create_env
    from colosseum.transport.grpc_transport import GRPCTransport
    from colosseum.weight_store.grpc_store import GRPCWeightStore
    from colosseum.worker.rollout_worker import rollout_worker_process

    max_mb = config.transport.grpc_max_message_mb
    store = GRPCWeightStore(weight_store_address, max_message_mb=max_mb)
    transports = {
        aid: GRPCTransport(learner_addresses[aid], max_message_mb=max_mb)
        for aid in agent_ids
    }

    worker_seed = None
    if config.training.seed is not None:
        worker_seed = config.training.seed + worker_id * 1000

    rollout_worker_process(
        worker_id=worker_id,
        env_fn=partial(_create_env, config.env.env_class, config.env.kwargs),
        num_envs=config.rollout.envs_per_worker,
        chunk_length=config.rollout.chunk_length,
        agent_ids=agent_ids,
        model_factories={aid: partial(build_model, agent_configs[aid]) for aid in agent_ids},
        trajectory_queues={aid: GRPCTrajectorySink(transports[aid], aid) for aid in agent_ids},
        weight_queues={aid: GRPCWeightSource(store, aid) for aid in agent_ids},
        stop_event=stop_event,
        gamma={aid: agent_configs[aid].algorithm.gamma for aid in agent_ids},
        weight_sync_interval=config.rollout.weight_sync_interval_sec,
        torch_threads=config.rollout.torch_threads,
        max_env_steps=total_timesteps,
        slot_agent_map=slot_agent_map,
        seed=worker_seed,
        vec_env_kind=config.rollout.vec_env,
        subproc_workers=config.rollout.subproc_workers,
    )


def run_distributed_workers(
    config_path: str,
    weight_store_address: str,
    learner_addresses: dict[str, str],
    overrides: dict | None = None,
) -> None:
    """Launch rollout workers that feed remote learners over gRPC.

    Args:
        config_path: path to the YAML config.
        weight_store_address: ``host:port`` of the WeightStore service.
        learner_addresses: ``{agent_id: host:port}`` of each agent's
            TrajectoryService.
    """
    from colosseum.core.registry import validate_config

    setup_process_logging(None, "workers-main", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    run_dir = RunDir.create(config, config_path, role="workers")
    setup_process_logging(run_dir.logs, "workers-main", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
    agent_ids = list(learner_addresses.keys()) or config.get_trainable_agent_ids()
    agent_configs = {aid: config.get_agent_config(aid) for aid in agent_ids}
    for aid in agent_ids:
        validate_config(agent_configs[aid])
    mp.set_start_method("spawn", force=True)

    num_players = config.env.num_players
    num_envs = config.rollout.envs_per_worker
    # Round-robin agents across slots so every agent's learner receives data.
    slot_agent_map = [
        [agent_ids[(e * num_players + p) % len(agent_ids)] for p in range(num_players)]
        for e in range(num_envs)
    ]

    stop_event = mp.Event()
    procs: list[mp.Process] = []
    worker_daemon = config.rollout.vec_env != "subprocess"
    per_worker_steps = config.training.total_timesteps // config.rollout.num_workers

    for worker_id in range(config.rollout.num_workers):
        p = mp.Process(
            target=_dist_worker_target,
            name=f"worker-{worker_id}",
            kwargs=dict(
                worker_id=worker_id,
                log_dir=str(run_dir.logs),
                config=config,
                agent_ids=agent_ids,
                agent_configs=agent_configs,
                weight_store_address=weight_store_address,
                learner_addresses=learner_addresses,
                stop_event=stop_event,
                total_timesteps=per_worker_steps,
                slot_agent_map=slot_agent_map,
            ),
            daemon=worker_daemon,
        )
        p.start()
        procs.append(p)

    logger.info(
        f"Started {len(procs)} distributed workers -> learners {learner_addresses}, "
        f"weights <- {weight_store_address}"
    )
    try:
        while any(p.is_alive() for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Interrupt — stopping workers")
    finally:
        stop_event.set()
        time.sleep(1.0)
        for p in procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        logger.info("Distributed workers stopped.")


# =====================================================================
# Helpers
# =====================================================================


def _install_stop_signal_handlers(stop_event: threading.Event) -> None:
    """SIGINT / SIGTERM set ``stop_event``.

    The handler sets the event from a helper thread: Python runs signal handlers
    in the main thread between bytecodes, possibly while the main thread holds
    the event's non-reentrant lock (e.g. inside ``stop_event.set()``); setting
    it directly in the handler would then deadlock the process forever.
    """
    def _handler(*_: object) -> None:
        threading.Thread(target=stop_event.set, daemon=True).start()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
