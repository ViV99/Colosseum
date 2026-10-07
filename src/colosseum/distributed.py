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

from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.types import WeightPayload

logger = logging.getLogger(__name__)


# =====================================================================
# Queue-like gRPC adapters (so learner_process / rollout_worker_process
# work unchanged whether the backend is mp.Queue or gRPC)
# =====================================================================


class GRPCTrajectorySink:
    """``.put``-compatible sink that ships chunks to a learner over gRPC.

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
    from colosseum.core.registry import build_model, import_class
    from colosseum.core.threads import configure_torch_threads, resolve_learner_threads
    from colosseum.learner.learner import learner_process, resolve_device
    from colosseum.transport.grpc_transport import serve_trajectory_receiver
    from colosseum.weight_store.grpc_store import GRPCWeightStore

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = _load(config_path, overrides)
    acfg = config.get_agent_config(agent_id)
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

    # Optional checkpoint persistence (drained off-thread so training never blocks).
    checkpoint_queue: queue.Queue = queue.Queue(maxsize=16)
    coordinator_ckpt = None
    if config.self_play.checkpoint_interval > 0:
        from colosseum.coordinator.checkpoint_manager import CheckpointManager
        coordinator_ckpt = CheckpointManager(
            base_dir=config.checkpoint.dir,
            pool_size=config.self_play.pool_size,
            save_optimizer=config.checkpoint.save_optimizer,
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
            )
        kwargs = {"device": device, "pin_memory": acfg.learner.pin_memory}
        if kickstart is not None:
            kwargs["kickstart"] = kickstart
        return algo_cls(model, acfg.algorithm, **kwargs)

    env_steps_per_train_step = config.rollout.chunk_length * acfg.learner.batch_chunks
    total_train_steps = max(1, config.training.total_timesteps // env_steps_per_train_step)

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    # Drain checkpoint snapshots to disk in the background.
    def _drain_checkpoints():
        while not stop_event.is_set():
            try:
                data = checkpoint_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if coordinator_ckpt is not None:
                coordinator_ckpt.save(
                    agent_id=agent_id,
                    policy_version=data["policy_version"],
                    state_dict=data["state_dict"],
                    optimizer_state=data.get("optimizer_state"),
                )

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
            total_train_steps=total_train_steps,
            checkpoint_queue=checkpoint_queue if coordinator_ckpt is not None else None,
            checkpoint_interval=config.self_play.checkpoint_interval,
        )
    finally:
        stop_event.set()
        traj_server.stop(0)
        store.close()
        logger.info(f"Distributed learner [{agent_id}] stopped.")


# =====================================================================
# Worker role
# =====================================================================


def _dist_worker_target(
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
    """Worker process: gRPC clients in, rollout_worker_process unchanged."""
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
        gamma=config.algorithm.gamma,
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
    mp.set_start_method("spawn", force=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = _load(config_path, overrides)
    agent_ids = list(learner_addresses.keys()) or config.get_trainable_agent_ids()
    agent_configs = {aid: config.get_agent_config(aid) for aid in agent_ids}

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
            args=(
                worker_id, config, agent_ids, agent_configs,
                weight_store_address, learner_addresses, stop_event,
                per_worker_steps, slot_agent_map,
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


def _load(config_path: str, overrides: dict | None) -> ColosseumConfig:
    config = load_config(config_path)
    if overrides:
        data = config.model_dump()
        for key, value in overrides.items():
            parts = key.split(".")
            d = data
            for part in parts[:-1]:
                d = d[part]
            d[parts[-1]] = value
        config = ColosseumConfig(**data)
    return config
