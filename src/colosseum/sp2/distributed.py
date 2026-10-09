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

Scope (SP2, spec block 10): self-play on the latest weights, without a coordinator, for games
where every agent of the run plays every role of the enabled layouts. Each worker env gets a
fixed lineup: a layout drawn by ``matchmaking.layouts`` and every seat the latest weights of one
agent (agents in rotation over the envs). Anything else (asymmetric agents, leagues across
machines) is a ConfigError pointing to SP5. Layouts are drawn once per worker env (ruling
PR-3) from an RNG seeded by ``training.seed + worker_id``, and the agent rotation starts at env 0
of worker 0, so every worker machine of a run starts with the same layout mix and rotation.

Budget and progress: there is no shared env-step counter across machines. Each
distributed learner uses progress = consumed_samples / training.total_timesteps
(its own ACT slots, which drives the LR schedule) and stops by itself once
consumed_samples >= total_timesteps. Each worker stops after
total_timesteps / num_workers env steps. These numbers differ from the local
mode budget (env steps summed over workers); a single semantics comes with the
hub in SP5.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import random
import signal
import socket
import threading
import time
from dataclasses import dataclass
from functools import partial

import torch

from colosseum.core.errors import ConfigError
from colosseum.sp2.core.config import ColosseumConfig, config_hash, load_config
from colosseum.sp2.core.run_dir import RunDir, safe_path_component
from colosseum.sp2.core.types import LATEST_NETWORK_ID, Lineup, SeatAssignment, WeightPayload
from colosseum.sp2.envs.game import GameSpec, RoleSpec
from colosseum.utils.logging import setup_process_logging
from colosseum.utils.process import SHUTDOWN_GRACE_SEC, ProcessSupervisor, run_child, start_process
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
    and a worker should survive a learner blip. A chunk the learner refuses
    (e.g. ``INVALID_ARGUMENT``) is dropped too, with a WARNING.
    """

    def __init__(self, transport, agent_id: str) -> None:
        self._transport = transport
        self._agent_id = agent_id

    def _send(self, chunk) -> None:
        import grpc
        try:
            self._transport.send_chunk(self._agent_id, chunk)
        except grpc.RpcError as e:
            transient = (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.CANCELLED)
            if e.code() in transient:  # the learner is down or restarting
                logger.debug(f"Dropping chunk for {self._agent_id}: {e.code()}")
            else:  # e.g. INVALID_ARGUMENT: the learner refused the chunk itself
                logger.warning(f"Chunk for {self._agent_id} rejected by the learner: {e.code()}: {e.details()}")

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
# Scope check and fixed lineups (spec block 10, ruling PR-3)
# =====================================================================


@dataclass(frozen=True)
class DistributedSetup:
    """Spec, layouts and per-agent roles / role specs of a distributed role."""

    spec: GameSpec
    layouts: dict[str, float]
    agent_roles: dict[str, list[str]]
    role_specs: dict[str, RoleSpec]


def distributed_setup(config: ColosseumConfig, agent_ids: list[str]) -> DistributedSetup:
    """Validate the config and check the distributed scope (module docstring); ConfigError otherwise."""
    from colosseum.sp2.coordinator.matchmaker import enabled_layouts
    from colosseum.sp2.core.registry import env_spec, validate_config
    from colosseum.sp2.core.roles import agent_role_spec, resolve_agent_roles

    validate_config(config)
    spec = env_spec(config)
    all_roles = resolve_agent_roles(config, spec)
    unknown = sorted(set(agent_ids) - set(all_roles))
    if unknown:
        raise ConfigError(f"agents {unknown} are not trainable agents of the config ({sorted(all_roles)})")
    layouts = enabled_layouts(spec, config.matchmaking)
    needed = {seat.role for name in layouts for seat in spec.layouts[name]}
    for aid in agent_ids:
        missing = sorted(needed - set(all_roles[aid]))
        if missing:
            raise ConfigError(
                f"distributed mode supports only games where every agent plays every role of the enabled "
                f"layouts; agent {aid!r} does not play {missing}. Asymmetric agents and leagues across "
                f"machines come with SP5 (train such configs with 'train' on one machine)"
            )
    return DistributedSetup(
        spec=spec, layouts=layouts,
        agent_roles={aid: list(all_roles[aid]) for aid in agent_ids},
        role_specs={aid: agent_role_spec(spec, all_roles[aid]) for aid in agent_ids},
    )


def distributed_lineups(setup: DistributedSetup, agent_ids: list[str], num_envs: int,
                        rng: random.Random) -> list[Lineup]:
    """Fixed lineups of one worker: env ``e`` plays a layout drawn by the layout weights, every
    seat the latest weights of ``agent_ids[e % n]`` (collecting)."""
    names = list(setup.layouts)
    weights = [setup.layouts[name] for name in names]
    lineups = []
    for e in range(num_envs):
        layout = rng.choices(names, weights=weights, k=1)[0]
        agent_id = agent_ids[e % len(agent_ids)]
        seats = [SeatAssignment(agent_id, LATEST_NETWORK_ID, True) for _ in setup.spec.layouts[layout]]
        lineups.append(Lineup(layout=layout, seats=seats))
    return lineups


# =====================================================================
# Learner role
# =====================================================================


def run_distributed_learner(
    config_path: str,
    agent_id: str,
    traj_port: int,
    weight_store_address: str,
    overrides: dict | None = None,
) -> int:
    """Run one trainable agent's learner as a standalone gRPC service; returns the exit code
    (0 when it stopped at its budget, 128 + signum after SIGINT / SIGTERM).

    Starts a TrajectoryService on ``traj_port`` (workers send chunks here), trains with the
    configured algorithm, and pushes weights to the WeightStore at ``weight_store_address``.
    """
    from colosseum.core.threads import configure_torch_threads, resolve_learner_threads
    from colosseum.sp2.core.registry import build_model, import_class
    from colosseum.sp2.core.roles import role_signature
    from colosseum.sp2.core.specs import ActionSpec
    from colosseum.sp2.learner.learner import learner_process, resolve_device

    setup_process_logging(None, f"learner-{agent_id}", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    setup = distributed_setup(config, [agent_id])  # before the run dir exists: a bad config leaves nothing
    # gRPC (an optional extra) is imported only after the scope check: a bad config is a
    # ConfigError even where grpc is not installed.
    from colosseum.sp2.transport.grpc_transport import serve_trajectory_receiver
    from colosseum.sp2.weight_store.grpc_store import GRPCWeightStore

    acfg = config.get_agent_config(agent_id)
    role_spec = setup.role_specs[agent_id]
    run_dir = RunDir.create(config, config_path, role=f"learner-{agent_id}")
    config = run_dir.with_run_name(config)
    setup_process_logging(run_dir.logs, f"learner-{agent_id}", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
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
    traj_server = serve_trajectory_receiver(chunk_queue, port=traj_port, max_message_mb=max_mb)

    store = GRPCWeightStore(weight_store_address, max_message_mb=max_mb)
    weight_sink = [GRPCWeightSink(store, agent_id)]

    # Checkpoint persistence (drained off-thread so training never blocks). Always on:
    # the learner's final snapshot is saved even without periodic checkpoints.
    from colosseum.sp2.coordinator.checkpoint_manager import CheckpointManager
    checkpoint_queue: queue.Queue = queue.Queue(maxsize=16)
    coordinator_ckpt = CheckpointManager(base_dir=run_dir.checkpoints, pool_size=config.checkpoint.pool_size)

    algo_cls = import_class(acfg.algorithm.algorithm_class)
    action_spec = ActionSpec.from_space(role_spec.action_space)
    teacher_path = acfg.training.kickstart_teacher

    def algorithm_factory():
        model = build_model(acfg, role_spec)
        kickstart = None
        if teacher_path:
            from colosseum.sp2.bc.kickstart import KickstartLoss
            teacher = build_model(acfg, role_spec)
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
        return algo_cls(model, acfg.algorithm, action_spec, **kwargs)

    stop_event = threading.Event()
    supervisor = ProcessSupervisor(stop_event)

    # Seed right before learner_process builds the model (validation above also draws from the
    # RNGs). Same per-agent stream as a local-mode learner.
    agent_index = config.get_trainable_agent_ids().index(agent_id)
    apply_global_seed(learner_seed(config.training.seed, agent_index))

    cfg_hash = config_hash(config)
    roles_meta = {"roles": setup.agent_roles[agent_id], "role_signature": role_signature(role_spec)}

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
                # (consumed_samples counts ACT slots of its own seats, a different quantity),
                # so a resume from it does not seed the env-step budget.
                meta_extra={"final": bool(data.get("final", False)),
                            "networks": acfg.networks.model_dump(mode="json", by_alias=True),
                            "config_hash": cfg_hash, "env_steps": None, **roles_meta},
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

    logger.info(f"Distributed learner [{agent_id}] serving trajectories on :{traj_port}, "
                f"weights -> {weight_store_address}")
    try:
        # SIGINT / SIGTERM set stop_event (and are remembered for the exit code); installed
        # inside the try so the finally always restores them.
        supervisor.install_signal_handlers()
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
            checkpoint_interval=config.checkpoint.interval,
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
        supervisor.restore_signal_handlers()
        logger.info(f"Distributed learner [{agent_id}] stopped.")
    if supervisor.received_signal is not None:
        signum = int(supervisor.received_signal)
        logger.warning(f"Distributed learner [{agent_id}] stopped by {signal.Signals(signum).name}")
        return 128 + signum
    return 0


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
    agent_roles: dict[str, list[str]],
    agent_configs: dict[str, ColosseumConfig],
    role_specs: dict[str, RoleSpec],
    weight_store_address: str,
    learner_addresses: dict[str, str],
    stop_event,
    total_timesteps: int,
    lineups: list[Lineup],
) -> None:
    """Worker process body: gRPC clients in, rollout_worker_process unchanged."""
    from colosseum.sp2.core.registry import build_model
    from colosseum.sp2.launcher import _create_env
    from colosseum.sp2.transport.grpc_transport import GRPCTransport
    from colosseum.sp2.weight_store.grpc_store import GRPCWeightStore
    from colosseum.sp2.worker.rollout_worker import rollout_worker_process

    max_mb = config.transport.grpc_max_message_mb
    store = GRPCWeightStore(weight_store_address, max_message_mb=max_mb)
    transports = {aid: GRPCTransport(learner_addresses[aid], max_message_mb=max_mb) for aid in agent_ids}

    worker_seed = None
    if config.training.seed is not None:
        worker_seed = config.training.seed + worker_id * 1000

    rollout_worker_process(
        worker_id=worker_id,
        env_fn=partial(_create_env, config.env.env_class, config.env.kwargs),
        num_envs=config.rollout.envs_per_worker,
        chunk_length=config.rollout.chunk_length,
        agent_ids=agent_ids,
        agent_roles=agent_roles,
        model_factories={aid: partial(build_model, agent_configs[aid], role_specs[aid]) for aid in agent_ids},
        trajectory_queues={aid: GRPCTrajectorySink(transports[aid], aid) for aid in agent_ids},
        weight_queues={aid: GRPCWeightSource(store, aid) for aid in agent_ids},
        stop_event=stop_event,
        weight_sync_interval=config.rollout.weight_sync_interval_sec,
        torch_threads=config.rollout.torch_threads,
        max_env_steps=total_timesteps,
        lineups=lineups,
        seed=worker_seed,
        vec_env_kind=config.rollout.vec_env,
        subproc_workers=config.rollout.subproc_workers,
        max_idle_steps=config.env.max_idle_steps,
    )


def run_distributed_workers(
    config_path: str,
    weight_store_address: str,
    learner_addresses: dict[str, str],
    overrides: dict | None = None,
) -> int:
    """Launch rollout workers that feed remote learners over gRPC; returns the exit code.

    Workers stop by themselves after their share of the budget (exit 0). A worker exiting
    non-zero stops the others (exit code 1); SIGINT / SIGTERM stop all of them (128 + signum).

    Args:
        config_path: path to the YAML config.
        weight_store_address: ``host:port`` of the WeightStore service.
        learner_addresses: ``{agent_id: host:port}`` of each agent's TrajectoryService.
    """
    setup_process_logging(None, "workers-main", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    agent_ids = list(learner_addresses.keys()) or config.get_trainable_agent_ids()
    setup = distributed_setup(config, agent_ids)  # before the run dir exists
    agent_configs = {aid: config.get_agent_config(aid) for aid in agent_ids}
    # One dir per machine: worker hosts sharing a run.name on a shared filesystem never collide.
    run_dir = RunDir.create(config, config_path, role=workers_role())
    config = run_dir.with_run_name(config)
    setup_process_logging(run_dir.logs, "workers-main", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
    mp.set_start_method("spawn", force=True)

    stop_event = mp.Event()
    supervisor = ProcessSupervisor(stop_event, log_dir=run_dir.logs)
    supervisor.install_signal_handlers()
    worker_daemon = config.rollout.vec_env != "subprocess"
    per_worker_steps = config.training.total_timesteps // config.rollout.num_workers

    code = 0
    try:
        for worker_id in range(config.rollout.num_workers):
            seed = None if config.training.seed is None else config.training.seed + worker_id
            # Rotation continues across the workers of this machine, so every agent gets envs
            # even when envs_per_worker < number of agents.
            shift = worker_id * config.rollout.envs_per_worker % len(agent_ids)
            lineups = distributed_lineups(setup, agent_ids[shift:] + agent_ids[:shift],
                                          config.rollout.envs_per_worker, random.Random(seed))
            proc = mp.Process(
                target=_dist_worker_target,
                name=f"worker-{worker_id}",
                kwargs=dict(
                    worker_id=worker_id,
                    log_dir=str(run_dir.logs),
                    config=config,
                    agent_ids=agent_ids,
                    agent_roles=setup.agent_roles,
                    agent_configs=agent_configs,
                    role_specs=setup.role_specs,
                    weight_store_address=weight_store_address,
                    learner_addresses=learner_addresses,
                    stop_event=stop_event,
                    total_timesteps=per_worker_steps,
                    lineups=lineups,
                ),
                daemon=worker_daemon,
            )
            start_process(proc)
            supervisor.add(f"worker-{worker_id}", proc)

        logger.info(f"Started {config.rollout.num_workers} distributed workers -> learners "
                    f"{learner_addresses}, weights <- {weight_store_address}")
        while not stop_event.is_set() and supervisor.alive():
            failure = supervisor.first_failure(nonzero_only=True)
            if failure is not None:
                logger.error(failure.message())
                code = 1
                break
            time.sleep(0.5)
    finally:
        stop_event.set()
        supervisor.wait_all(SHUTDOWN_GRACE_SEC)
        killed = supervisor.kill_remaining()
        supervisor.restore_signal_handlers()
        logger.info("Distributed workers stopped.")
    if supervisor.received_signal is not None:
        signum = int(supervisor.received_signal)
        logger.warning(f"Received {signal.Signals(signum).name}; distributed workers stopped")
        return 128 + signum
    if code == 0:
        # Exits after the loop ended (e.g. one worker finished, another crashed meanwhile).
        failures = supervisor.failures(exclude=set(killed))
        for failure in failures:
            logger.error(failure.message())
        code = 1 if failures else 0
    return code


# =====================================================================
# Helpers
# =====================================================================


def workers_role() -> str:
    """Run-dir role of ``run-workers`` on this machine: ``workers-<host>``."""
    return f"workers-{safe_path_component(socket.gethostname(), 'host')}"
