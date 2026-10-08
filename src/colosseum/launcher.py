"""Launcher: reads config, instantiates all components, starts processes.

This is the top-level entry point that wires together:
- Coordinator (matchmaking, checkpoint management)
- Workers (env + inference)
- Learners (one per trainable agent)
- Weight synchronization (via mp.Queue)
- Metrics logging

Everything that crosses a process boundary (queue items and Process
arguments) is numpy + primitives, never torch tensors (R6-02).

With mp.set_start_method("spawn"), all arguments to Process targets must be
picklable. We pass config objects (pydantic models) and string class paths
instead of closures/lambdas, and let each process instantiate its own objects.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.queues
import queue
import signal
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.ipc import SharedCounter
from colosseum.core.run_dir import RunDir
from colosseum.core.types import LATEST_NETWORK_ID, MatchConfig
from colosseum.metrics.wandb_logger import WandBLogger
from colosseum.utils.logging import setup_process_logging
from colosseum.utils.process import SHUTDOWN_GRACE_SEC, ProcessSupervisor, run_child, start_process

logger = logging.getLogger(__name__)


# Queue size constants
_WEIGHT_QUEUE_SIZE = 1  # newest-wins mailbox per (agent, worker); see core.ipc.put_latest
_CHECKPOINT_QUEUE_SIZE = 16
_METRICS_QUEUE_SIZE = 100
_RESULTS_QUEUE_SIZE = 1000
# One slot: a command that was put has been taken by the worker before the next one
# can be put, so no command (and none of its new_checkpoints) is replaced unseen (D9).
_COMMAND_QUEUE_SIZE = 1


# =====================================================================
# Top-level process targets (must be picklable for spawn)
# =====================================================================

def _create_env(env_class_path: str, kwargs: dict):
    """Create env instance inside a worker process."""
    from colosseum.core.registry import import_class
    cls = import_class(env_class_path)
    return cls(**kwargs)


def warn_static_ownership_skew(config: ColosseumConfig) -> None:
    """Warn when static matchmaking splits envs unevenly between trainable agents.

    Env ``g`` is owned by ``agents[(g + refresh_round) % n]``. With
    ``match_refresh_interval_sec == 0`` the round never advances, so if the env count
    is not a multiple of the agent count (or smaller than it) some agents get
    permanently more training data than others, or none at all.
    """
    n = len(config.get_trainable_agent_ids())
    total_envs = config.rollout.num_workers * config.rollout.envs_per_worker
    if config.rollout.match_refresh_interval_sec == 0 and n > 1 and total_envs % n != 0:
        logger.warning(
            f"rollout.match_refresh_interval_sec=0 with {total_envs} envs "
            f"(num_workers * envs_per_worker) and {n} trainable agents: env ownership never "
            f"rotates, so data per agent is skewed"
            + (" and some agents get no data" if total_envs < n else "")
            + ". Use a multiple of the agent count or enable match refresh."
        )


def trainable_agent_configs(config: ColosseumConfig) -> dict[str, ColosseumConfig]:
    """Effective config (overrides merged) of every trainable agent, without validation."""
    return {aid: config.get_agent_config(aid) for aid in config.get_trainable_agent_ids()}


def validate_agent_configs(config: ColosseumConfig) -> dict[str, ColosseumConfig]:
    """Effective config of every trainable agent, each checked by ``validate_config``.

    Fails fast (ConfigError) on structural errors, e.g. a head/encoder dim mismatch.
    """
    from colosseum.core.registry import validate_config

    agent_configs = trainable_agent_configs(config)
    for acfg in agent_configs.values():
        validate_config(acfg)
    return agent_configs


def _create_model(config: ColosseumConfig):
    """Create the agent's PolicyModel inside a worker/learner process."""
    from colosseum.core.registry import build_model
    return build_model(config)


def _worker_target(*, worker_id: int, log_dir: str | None = None, **kwargs) -> None:
    """Worker process entry point: process logging first, then the worker body."""
    run_child(f"worker-{worker_id}", log_dir, _worker_main, worker_id=worker_id, **kwargs)


def _worker_main(
    *,
    worker_id: int,
    config: ColosseumConfig,
    agent_ids: list[str],
    agent_configs: dict[str, ColosseumConfig],
    trajectory_queues: dict[str, mp.Queue],
    weight_queues: dict[str, mp.Queue],
    stop_event: mp.Event,
    env_step_counter: SharedCounter | None = None,
    checkpoint_state_dicts_by_agent: dict[str, dict[str, dict]] | None = None,
    slot_network_map: list[list[str]] | None = None,
    collect_mask: list[list[bool]] | None = None,
    slot_agent_map: list[list[str]] | None = None,
    results_queue: mp.Queue | None = None,
    command_queue: mp.Queue | None = None,
    metrics_queue: mp.Queue | None = None,
) -> None:
    """Worker process body (see ``_worker_target``).

    Env and model factories are built INSIDE the process (``functools.partial``
    over top-level functions is picklable under spawn, which the subprocess
    vec_env needs to ship ``env_fn`` to its children).
    """
    from functools import partial

    from colosseum.core.registry import build_model
    from colosseum.worker.rollout_worker import rollout_worker_process

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
        trajectory_queues=trajectory_queues,
        weight_queues=weight_queues,
        stop_event=stop_event,
        gamma={aid: agent_configs[aid].algorithm.gamma for aid in agent_ids},
        weight_sync_interval=config.rollout.weight_sync_interval_sec,
        torch_threads=config.rollout.torch_threads,
        env_step_counter=env_step_counter,
        checkpoint_state_dicts_by_agent=checkpoint_state_dicts_by_agent,
        slot_agent_map=slot_agent_map,
        slot_network_map=slot_network_map,
        collect_mask=collect_mask,
        results_queue=results_queue,
        command_queue=command_queue,
        stats_queue=metrics_queue,
        seed=worker_seed,
        vec_env_kind=config.rollout.vec_env,
        subproc_workers=config.rollout.subproc_workers,
    )


def _learner_target(*, agent_id: str, log_dir: str | None = None, **kwargs) -> None:
    """Learner process entry point: process logging first, then the learner body."""
    run_child(f"learner-{agent_id}", log_dir, _learner_main, agent_id=agent_id, **kwargs)


def _learner_main(
    *,
    agent_id: str,
    config: ColosseumConfig,
    trajectory_queue: mp.Queue,
    weight_queues: list[mp.Queue],
    stop_event: mp.Event,
    metrics_queue: mp.Queue,
    checkpoint_queue: mp.Queue | None = None,
    checkpoint_interval: int = 0,
    resume_state: dict | None = None,
    progress_counter: SharedCounter | None = None,
    total_timesteps: int = 0,
    num_learners: int = 1,
    seed: int | None = None,
) -> None:
    """Learner process body (see ``_learner_target``).

    ``num_learners`` (learner processes on this machine) feeds the automatic
    torch thread count when ``learner.torch_threads`` is unset. ``seed`` (from
    ``utils.seeding.learner_seed``) seeds this process before the model is built.
    """
    from colosseum.core.registry import import_class
    from colosseum.core.threads import configure_torch_threads, resolve_learner_threads
    from colosseum.learner.learner import learner_process, resolve_device
    from colosseum.utils.seeding import apply_global_seed

    apply_global_seed(seed)

    device = resolve_device(config.learner.device)
    configure_torch_threads(resolve_learner_threads(
        config.learner.torch_threads, device, config.rollout.num_workers,
        config.rollout.torch_threads, num_learners,
    ))

    # Resolve algorithm class from config
    algo_class_path = config.algorithm.algorithm_class
    if not algo_class_path:
        raise ValueError(
            "algorithm.algorithm_class is not set. "
            "Provide a dotted import path (e.g. 'colosseum.algorithms.appo.APPO')."
        )
    algo_cls = import_class(algo_class_path)

    teacher_path = config.training.kickstart_teacher

    def algorithm_factory():
        model = _create_model(config)
        kickstart = None
        if teacher_path:
            from colosseum.bc.kickstart import KickstartLoss
            teacher = _create_model(config)
            teacher_sd = torch.load(teacher_path, weights_only=True, map_location=device)
            teacher.load_state_dict(teacher_sd)
            teacher.to(device)
            kickstart = KickstartLoss(
                teacher,
                initial_lambda=config.training.kickstart_lambda,
                decay_steps=config.training.kickstart_decay_steps,
                direction=config.training.kickstart_kl,
            )
            logger.info(f"Kickstart enabled for {agent_id} from {teacher_path}")
        kwargs = {"device": device, "pin_memory": config.learner.pin_memory}
        if kickstart is not None:
            kwargs["kickstart"] = kickstart
        return algo_cls(model, config.algorithm, **kwargs)

    learner_process(
        agent_id=agent_id,
        algorithm_factory=algorithm_factory,
        trajectory_queue=trajectory_queue,
        weight_queues=weight_queues,
        config=config.learner,
        stop_event=stop_event,
        metrics_queue=metrics_queue,
        checkpoint_queue=checkpoint_queue,
        checkpoint_interval=checkpoint_interval,
        resume_state=resume_state,
        progress_counter=progress_counter,
        total_timesteps=total_timesteps,
        weight_sync_interval=config.rollout.weight_sync_interval_sec,
    )


# =====================================================================
# Match config helpers
# =====================================================================

def _derive_worker_configs(
    match_configs: list[MatchConfig],
    coordinator: Coordinator,
    agent_ids: list[str],
    already_sent: dict[str, set[str]] | None = None,
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],  # new checkpoints by agent: {agent_id: {ckpt_id: numpy state_dict}}
    list[list[str]],             # slot_network_map
    list[list[bool]],            # collect_mask
    list[list[str]],             # slot_agent_map
]:
    """Turn match configs into worker slot maps.

    Checkpoint weights cross the process boundary (worker Process arguments and
    ``WorkerCommand.new_checkpoints``), so they are numpy, never torch.
    Checkpoints listed in ``already_sent[agent]`` are referenced but not reloaded.
    A checkpoint that cannot be loaded (evicted or missing) is replaced by the
    latest weights with ``collect=True`` and a warning (R4-19).
    """
    already_sent = already_sent or {}
    new_ckpts: dict[str, dict[str, dict[str, np.ndarray]]] = {aid: {} for aid in agent_ids}
    missing: set[tuple[str, str]] = set()
    collect_mask: list[list[bool]] = []
    slot_network_map: list[list[str]] = []
    slot_agent_map: list[list[str]] = []

    for mc in match_configs:
        env_collect: list[bool] = []
        env_nets: list[str] = []
        env_agents: list[str] = []
        for slot in mc.player_slots:
            net_id = LATEST_NETWORK_ID
            collect = slot.collect_trajectories
            ckpt_id = slot.checkpoint_id
            if ckpt_id is not None:
                agent_new = new_ckpts.setdefault(slot.agent_id, {})
                available = ckpt_id in already_sent.get(slot.agent_id, set()) or ckpt_id in agent_new
                if not available and (slot.agent_id, ckpt_id) not in missing:
                    try:
                        agent_new[ckpt_id] = coordinator.checkpoint_manager.load_model(slot.agent_id, ckpt_id)
                        available = True
                    except FileNotFoundError:
                        missing.add((slot.agent_id, ckpt_id))
                        logger.warning(
                            f"Checkpoint {ckpt_id} of {slot.agent_id} is missing; that slot plays "
                            f"the latest weights and collects trajectories"
                        )
                if available:
                    net_id = ckpt_id
                else:
                    collect = True
            env_collect.append(collect)
            env_nets.append(net_id)
            env_agents.append(slot.agent_id)
        collect_mask.append(env_collect)
        slot_network_map.append(env_nets)
        slot_agent_map.append(env_agents)

    return new_ckpts, slot_network_map, collect_mask, slot_agent_map


class _QueueReader:
    """Drains one ``mp.Queue`` on a helper thread, so the main process never blocks on it.

    ``mp.Queue.get`` reads a whole message in ``recv_bytes`` and ignores any timeout once
    the first bytes are there. A producer killed mid-message (OOM, SIGKILL) therefore
    blocks the reader forever: the main process holds the pipe's write end too, so no EOF
    ever arrives. Here ``drain`` waits at most ``wait`` seconds; a read still in progress
    (a large payload arriving or being unpickled) is collected by a later call. Only the
    shutdown gives up on it (``abandon``, after its deadline), once every producer is gone.

    After ``abandon`` no further item can arrive: the dead producer was killed inside the
    feeder's ``send_bytes``, i.e. while holding the queue's cross-process write lock, so no
    other producer can ever write to that pipe again. Should the stuck read finish anyway,
    the items it got are logged as dropped.
    """

    def __init__(self, q: Any, label: str, dead_producers: Callable[[], list[str]]) -> None:
        self._q = q
        self._label = label
        self._dead_producers = dead_producers
        self._thread: threading.Thread | None = None
        self._items: list = []
        self._error: BaseException | None = None
        self._salvaged: list = []
        self.abandoned = False

    @property
    def busy(self) -> bool:
        """A read is in progress (possibly stuck on an incomplete message)."""
        return self._thread is not None and self._thread.is_alive()

    def _read(self) -> None:
        try:
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                except (EOFError, OSError) as e:
                    logger.warning(f"Dropped an unreadable {self._label} queue item: {e!r}")
                    break
                self._items.append(item)
        except BaseException as e:  # noqa: BLE001 - re-raised by drain() in the main thread
            self._error = e
        if self.abandoned:
            logger.warning(f"{len(self._items)} item(s) completed on the abandoned {self._label} queue "
                           f"after shutdown gave up on it; dropped")

    def _take(self) -> list:
        items, self._items = self._items, []
        return items

    def drain(self, wait: float = 0.05) -> list:
        """Items read so far; starts a read when the queue has data.

        A new read gets up to ``wait`` seconds to finish, so small items come back from this
        call; a read already in progress is only checked, never waited for, so a stuck
        reader costs nothing per poll. Non-``mp.Queue`` objects (e.g. ``queue.Queue``),
        whose ``get_nowait`` never blocks, are read synchronously.
        """
        if self.abandoned:
            salvaged, self._salvaged = self._salvaged, []
            return salvaged  # complete items read before the stuck message
        if not isinstance(self._q, mp.queues.Queue):
            self._read()
        else:
            if self._thread is None:
                if self._q.empty():
                    return []
                self._thread = threading.Thread(target=self._read, name=f"drain-{self._label}", daemon=True)
                self._thread.start()
                self._thread.join(wait)
            if self._thread.is_alive():
                return []  # still reading: a large item is arriving, or the message is incomplete
            self._thread = None
        error, self._error = self._error, None
        items = self._take()
        if error is not None:
            raise error
        return items

    def abandon(self) -> None:
        """Give up on a read that never completed; the next ``drain`` returns the items read before it."""
        self._salvaged = self._take()
        self.abandoned = True
        dead = self._dead_producers()
        source = f" ({', '.join(dead)} died while sending)" if dead else ""
        logger.error(f"An incomplete item on the {self._label} queue was never completed{source}; "
                     f"it and the rest of that queue are skipped")


def _queue_depths(queues: dict[str, Any]) -> dict[str, int]:
    """Approximate items waiting per queue (-1 where the platform cannot tell)."""
    depths = {}
    for name, q in queues.items():
        try:
            depths[name] = int(q.qsize())
        except (NotImplementedError, OSError):
            depths[name] = -1
    return depths


# =====================================================================
# Launcher
# =====================================================================

class Launcher:
    """Reads config, instantiates components, starts training processes.

    Spawns one learner per trainable agent, shared workers that route
    trajectory chunks to the correct agent's learner.
    """

    def __init__(self, config: ColosseumConfig, run_dir: RunDir, validated: bool = False) -> None:
        """``validated``: the trainable agent configs were already checked by
        ``validate_agent_configs`` (``run_training`` does it before the run dir exists),
        so ``launch`` does not repeat the dummy forward passes."""
        self._config = config
        self._run_dir = run_dir
        self._validated = validated
        self._stop_event = mp.Event()
        # Named children (learner-<agent>, worker-<id>), signals and teardown (T6.5).
        self._supervisor = ProcessSupervisor(self._stop_event, log_dir=run_dir.logs)
        self._readers: dict[int, _QueueReader] = {}
        self._reported_failures: set[str] = set()
        self._signal_logged = False
        self._all_queues: list[mp.Queue] = []
        # Env steps taken by all workers together (spec block 2: global budget).
        self._env_step_counter = SharedCounter()
        # Set by launch(); used by the monitor loop and _shutdown to save checkpoints.
        self._coordinator: Coordinator | None = None
        self._agent_ids: list[str] = []
        self._checkpoint_queues: dict[str, mp.Queue] = {}
        self._checkpoint_meta: dict[str, dict] = {}
        # Set by launch(): the main process's single metrics sink (T6.3) and its queue-depth source.
        self._hub = None
        self._metrics_writer = None
        self._wandb: WandBLogger | None = None
        self._trajectory_queues: dict[str, mp.Queue] = {}
        self._command_queues: list[mp.Queue] = []
        self._results_queue: mp.Queue | None = None
        self._metrics_queue: mp.Queue | None = None

    @property
    def env_steps_done(self) -> int:
        """Env steps taken by all workers so far (the global budget counter)."""
        return self._env_step_counter.value

    def launch(self) -> int:
        """Run the full training pipeline; returns the process exit code.

        0: the env-step budget was reached and every child stopped cleanly; 1: a child
        died (or exited non-zero on its own); 128 + signum after SIGINT / SIGTERM.
        """
        cfg = self._config
        trainable_agents = cfg.get_trainable_agent_ids()

        logger.info("Colosseum: Starting training pipeline")
        logger.info(f"  Agents: {trainable_agents}")
        logger.info(f"  Algorithm: {cfg.algorithm.name}")
        logger.info(f"  Workers: {cfg.rollout.num_workers}")
        logger.info(f"  Envs per worker: {cfg.rollout.envs_per_worker}")
        logger.info(f"  Chunk length: {cfg.rollout.chunk_length}")

        # Per-agent effective configs (overrides merged), validated before any process starts
        # (run_training validates them before the run dir exists and passes validated=True).
        agent_configs = trainable_agent_configs(cfg) if self._validated else validate_agent_configs(cfg)
        warn_static_ownership_skew(cfg)

        # Initialize coordinator
        coordinator = Coordinator(cfg, checkpoint_dir=self._run_dir.checkpoints)

        # Resume (before any process starts: a bad resume source fails fast).
        resume_states = self._resolve_resume(agent_configs)

        from colosseum.core.config import config_hash
        cfg_hash = config_hash(cfg)
        self._checkpoint_meta = {
            aid: {"networks": agent_configs[aid].networks.model_dump(mode="json", by_alias=True),
                  "config_hash": cfg_hash}
            for aid in trainable_agents
        }

        # Per-agent queues
        trajectory_queues: dict[str, mp.Queue] = {}
        checkpoint_queues: dict[str, mp.Queue] = {}
        weight_queues_per_agent: dict[str, list[mp.Queue]] = {}

        for aid in trainable_agents:
            acfg = agent_configs[aid]
            trajectory_queues[aid] = mp.Queue(maxsize=acfg.learner.queue_size)
            checkpoint_queues[aid] = mp.Queue(maxsize=_CHECKPOINT_QUEUE_SIZE)
            weight_queues_per_agent[aid] = [
                mp.Queue(maxsize=_WEIGHT_QUEUE_SIZE) for _ in range(cfg.rollout.num_workers)
            ]

        # Shared queues
        metrics_queue = mp.Queue(maxsize=_METRICS_QUEUE_SIZE)
        results_queue = mp.Queue(maxsize=_RESULTS_QUEUE_SIZE)

        # Per-worker command queues (runtime match re-assignment, C1) and the
        # set of checkpoint ids already shipped to each worker (to send deltas).
        command_queues: list[mp.Queue] = [
            mp.Queue(maxsize=_COMMAND_QUEUE_SIZE) for _ in range(cfg.rollout.num_workers)
        ]
        worker_sent_ckpts: list[dict[str, set]] = [
            {aid: set() for aid in trainable_agents} for _ in range(cfg.rollout.num_workers)
        ]

        # Track every queue so shutdown can cancel feeder threads and avoid the
        # interpreter blocking on unflushed data when consumers have exited.
        self._all_queues = [metrics_queue, results_queue, *command_queues]
        for aid in trainable_agents:
            self._all_queues.append(trajectory_queues[aid])
            self._all_queues.append(checkpoint_queues[aid])
            self._all_queues.extend(weight_queues_per_agent[aid])

        self._coordinator = coordinator
        self._agent_ids = list(trainable_agents)
        self._checkpoint_queues = checkpoint_queues
        self._trajectory_queues = trajectory_queues
        self._command_queues = command_queues
        self._results_queue = results_queue
        self._metrics_queue = metrics_queue

        code = 1
        try:
            # SIGINT / SIGTERM only set stop_event from here on: children ignore SIGINT and the
            # shutdown below saves their final checkpoints (T6.5).
            self._supervisor.install_signal_handlers()
            # The metrics hub (and its file) exists before any child starts: an open failure
            # here cannot orphan children (T6.3). _finish_metrics closes whatever was opened.
            self._open_metrics(resume_states)
            started = False
            try:
                self._start_children(
                    agent_configs, resume_states, trajectory_queues, checkpoint_queues,
                    weight_queues_per_agent, metrics_queue, results_queue, command_queues,
                    worker_sent_ckpts,
                )
                started = True
                logger.info("Training started. Press Ctrl+C to stop.")
                code = self._monitor_loop(coordinator, trainable_agents, command_queues, worker_sent_ckpts)
            except BaseException as error:
                # Also after a failed start or a monitor error the started children are
                # stopped (final checkpoints saved); then the original error propagates.
                if not started:
                    logger.error(f"Starting the child processes failed; stopping the "
                                 f"{len(self._supervisor.names)} already started")
                self._shutdown_after_error(error)
                raise
            killed = self._shutdown()
            code = self._exit_code_after_shutdown(code, killed)
        finally:
            try:
                self._finish_metrics()
            finally:
                try:
                    self._supervisor.restore_signal_handlers()
                finally:
                    self._release_queues()
        return code

    def _open_metrics(self, resume_states: dict[str, dict | None]) -> None:
        """metrics.jsonl writer, optional WandB viewer and the hub that feeds both (T6.3/T6.4).

        Resumed runs start the rate baselines at the resumed counters, so the first system
        record has no resume spike.
        """
        from colosseum.metrics.hub import MetricsHub
        from colosseum.metrics.jsonl import MetricsWriter

        cfg = self._config
        self._metrics_writer = MetricsWriter(self._run_dir.metrics_path)
        # Optional viewer of the same records (metrics.jsonl stays the source of truth, T6.4).
        self._wandb = WandBLogger(
            cfg.metrics,
            run_name=self._run_dir.run_name or self._run_dir.root.name,
            run_config=cfg.model_dump(mode="json", by_alias=True),
        )
        self._hub = MetricsHub(
            writer=self._metrics_writer,
            ratings_path=self._run_dir.ratings_path,
            agent_ids=self._agent_ids,
            total_timesteps=cfg.training.total_timesteps,
            log_interval=cfg.metrics.log_interval,
            console_interval_sec=cfg.metrics.console_interval_sec,
            initial_env_steps=int(self._env_step_counter.value),
            initial_train_steps={aid: int(s["policy_version"]) if s else 0 for aid, s in resume_states.items()},
            wandb_logger=self._wandb,
        )

    def _start_children(
        self,
        agent_configs: dict[str, ColosseumConfig],
        resume_states: dict[str, dict | None],
        trajectory_queues: dict[str, mp.Queue],
        checkpoint_queues: dict[str, mp.Queue],
        weight_queues_per_agent: dict[str, list[mp.Queue]],
        metrics_queue: mp.Queue,
        results_queue: mp.Queue,
        command_queues: list[mp.Queue],
        worker_sent_ckpts: list[dict[str, set]],
    ) -> None:
        """Start one learner per trainable agent, then the workers shared by all agents.

        Every started child is registered with the supervisor right away, so a failure
        part-way through tears down exactly the children that exist.
        """
        from colosseum.utils.seeding import learner_seed

        cfg = self._config
        trainable_agents = self._agent_ids
        coordinator = self._coordinator
        for agent_index, aid in enumerate(trainable_agents):
            acfg = agent_configs[aid]
            # numpy + bytes only: Process arguments cross the process boundary (R6-02).
            learner_proc = mp.Process(
                target=_learner_target,
                name=f"learner-{aid}",
                kwargs=dict(
                    agent_id=aid,
                    log_dir=str(self._run_dir.logs),
                    config=acfg,
                    trajectory_queue=trajectory_queues[aid],
                    weight_queues=weight_queues_per_agent[aid],
                    stop_event=self._stop_event,
                    metrics_queue=metrics_queue,
                    checkpoint_queue=checkpoint_queues[aid],
                    checkpoint_interval=cfg.self_play.checkpoint_interval,
                    resume_state=resume_states[aid],
                    progress_counter=self._env_step_counter,
                    total_timesteps=cfg.training.total_timesteps,
                    num_learners=len(trainable_agents),
                    seed=learner_seed(cfg.training.seed, agent_index),
                ),
                daemon=True,
            )
            start_process(learner_proc)  # SIGINT ignored from the child's first instruction
            self._supervisor.add(f"learner-{aid}", learner_proc)
            logger.info(f"Learner started for agent {aid}")

        # Workers (shared across all agents)
        for worker_id in range(cfg.rollout.num_workers):
            match_configs = coordinator.generate_match_configs(
                cfg.rollout.envs_per_worker,
                env_offset=worker_id * cfg.rollout.envs_per_worker,
            )

            (
                ckpt_dicts_by_agent,
                slot_network_map,
                collect_mask,
                slot_agent_map,
            ) = _derive_worker_configs(match_configs, coordinator, trainable_agents)
            # Initial checkpoints travel as process arguments: delivered by construction.
            for aid, ckpts in ckpt_dicts_by_agent.items():
                worker_sent_ckpts[worker_id].setdefault(aid, set()).update(ckpts)

            worker_weight_queues: dict[str, mp.Queue] = {
                aid: weight_queues_per_agent[aid][worker_id]
                for aid in trainable_agents
            }

            # Subprocess vec_env makes the worker spawn its own children, which
            # daemonic processes are not allowed to do — so run such workers
            # non-daemon (the shutdown path joins/terminates them explicitly).
            worker_daemon = cfg.rollout.vec_env != "subprocess"
            worker_proc = mp.Process(
                target=_worker_target,
                name=f"worker-{worker_id}",
                kwargs=dict(
                    worker_id=worker_id,
                    log_dir=str(self._run_dir.logs),
                    config=cfg,
                    agent_ids=trainable_agents,
                    agent_configs=agent_configs,
                    trajectory_queues=trajectory_queues,
                    weight_queues=worker_weight_queues,
                    stop_event=self._stop_event,
                    env_step_counter=self._env_step_counter,
                    checkpoint_state_dicts_by_agent=ckpt_dicts_by_agent,
                    slot_network_map=slot_network_map,
                    collect_mask=collect_mask,
                    slot_agent_map=slot_agent_map,
                    results_queue=results_queue,
                    command_queue=command_queues[worker_id],
                    metrics_queue=metrics_queue,
                ),
                daemon=worker_daemon,
            )
            start_process(worker_proc)
            self._supervisor.add(f"worker-{worker_id}", worker_proc)

    def _monitor_loop(
        self,
        coordinator: Coordinator,
        agent_ids: list[str],
        command_queues: list[mp.Queue],
        worker_sent_ckpts: list[dict[str, set]],
    ) -> int:
        """Run until the budget is reached, a signal arrives, or a child dies.

        Saves checkpoints, feeds results and metrics to the coordinator and the hub, and
        periodically re-generates match assignments for the workers (C1), so
        self-play-vs-history and PFSP opponent selection evolve during training.

        The main process is the only budget authority: learners and workers run until
        ``stop_event``, so any child exiting earlier is a failure (a dead learner would
        also leave the workers blocked on its full trajectory queue).

        Returns the exit code: 0 budget reached (or ``stop_event`` set by the caller), 1 child
        failure, 128+signum on SIGINT/SIGTERM.
        """
        total = self._config.training.total_timesteps
        refresh_interval = self._config.rollout.match_refresh_interval_sec
        last_refresh = time.monotonic()
        while True:
            self._drain_all_checkpoints()
            self._drain_results()
            self._drain_metrics()
            env_steps = int(self._env_step_counter.value)
            self._hub.maybe_tick(env_steps=env_steps, ratings=coordinator.ratings_snapshot(),
                                 queue_depths=_queue_depths(self._trajectory_queues))
            if self._supervisor.received_signal is not None:
                return self._signal_exit_code()
            if self._stop_event.is_set():
                if self._supervisor.received_signal is not None:  # arrived since the check above
                    return self._signal_exit_code()
                # Set by the caller (e.g. scripts/bench_throughput.py stops after a time window).
                logger.info(f"Stop requested after {env_steps} env steps (budget {total})")
                return 0
            if env_steps >= total:
                logger.info(f"Training budget reached: {env_steps} env steps (budget {total})")
                return 0
            failure = self._supervisor.first_failure()
            if failure is not None:
                logger.error(failure.message())
                self._reported_failures.add(failure.name)
                return 1
            if refresh_interval > 0 and time.monotonic() - last_refresh >= refresh_interval:
                self._refresh_worker_matches(coordinator, agent_ids, command_queues, worker_sent_ckpts)
                last_refresh = time.monotonic()
            time.sleep(0.2)

    def _shutdown_after_error(self, error: BaseException) -> None:
        """Tear down after ``error``. A teardown failure is logged and noted on ``error``,
        which stays the exception the caller raises."""
        try:
            self._shutdown()
        except Exception as teardown_error:  # noqa: BLE001 - must not replace ``error``
            logger.exception(f"Shutdown after {type(error).__name__} failed too")
            error.add_note(f"Shutdown afterwards also failed: {teardown_error!r}")

    def _exit_code_after_shutdown(self, code: int, killed: list[str]) -> int:
        """A child that exited non-zero on its own (not terminated by the shutdown) fails the
        run, even if the budget was reached (R3-08). After a signal the signal's code stays
        and such exits are only warned about. A signal that arrived while the run was
        finishing (e.g. right after the budget was reached) still gives 128 + signum."""
        if code == 0 and self._supervisor.received_signal is not None:
            code = self._signal_exit_code()
        for failure in self._supervisor.failures(exclude=set(killed) | self._reported_failures):
            self._reported_failures.add(failure.name)
            if code in (0, 1):
                logger.error(failure.message())
                code = 1
            else:
                logger.warning(failure.message())
        return code

    def _signal_exit_code(self) -> int:
        """128 + signum of the first SIGINT/SIGTERM; logged once (the handler itself only records it)."""
        signum = int(self._supervisor.received_signal)
        if not self._signal_logged:
            self._signal_logged = True
            logger.warning(f"Received {signal.Signals(signum).name}, shutting down")
        return 128 + signum

    def _dead_producers(self, names: list[str]) -> list[str]:
        dead = []
        for name in names:
            proc = self._supervisor.get(name)
            if proc is not None and proc.exitcode is not None:
                dead.append(name)
        return dead

    def _drain(self, q: Any, label: str, producers: Callable[[], list[str]]) -> list:
        """Everything currently in ``q``, read without ever blocking on a dead producer."""
        reader = self._readers.get(id(q))
        if reader is None:
            reader = self._readers[id(q)] = _QueueReader(q, label, lambda: self._dead_producers(producers()))
        return reader.drain()

    def _drain_results(self) -> None:
        """Feed queued match results to the coordinator (ratings, PFSP) and the hub."""
        if self._results_queue is None:
            return
        workers = lambda: [n for n in self._supervisor.names if n.startswith("worker-")]  # noqa: E731
        for result in self._drain(self._results_queue, "results", workers):
            self._coordinator.report_match_result(result)
            self._hub.on_match_result(result)

    def _drain_metrics(self) -> None:
        """Learner train metrics and worker stats share the metrics queue."""
        if self._metrics_queue is None:
            return
        for item in self._drain(self._metrics_queue, "metrics", lambda: self._supervisor.names):
            if item.get("kind") == "worker_stats":
                self._hub.on_worker_stats(item)
            else:
                self._hub.on_train_metrics(item)

    def _finish_metrics(self) -> None:
        """Record what arrived during shutdown (final train metrics, last results), write the
        final records and ratings.json; the metrics file is closed and the WandB run finished
        even if any of that fails (or if the hub was never built)."""
        try:
            if self._hub is not None:
                self._drain_results()
                self._drain_metrics()
                self._hub.close(env_steps=int(self._env_step_counter.value),
                                ratings=self._coordinator.ratings_snapshot(),
                                queue_depths=_queue_depths(self._trajectory_queues))
        finally:
            try:
                if self._metrics_writer is not None:
                    self._metrics_writer.close()
            finally:
                if self._wandb is not None:
                    self._wandb.finish()

    def _resolve_resume(self, agent_configs: dict[str, ColosseumConfig]) -> dict[str, dict | None]:
        """Resolve ``training.resume_from`` for every trainable agent (see ``resolve_resume``).

        Checks each resumed agent's weights against its architecture (ConfigError
        before any process starts) and continues the global env-step counter from
        the largest resumed ``env_steps``, so the budget and LR progress continue (D3).
        """
        from colosseum.coordinator.checkpoint_manager import check_model_state, resolve_resume
        from colosseum.core.registry import build_model

        resume_from = self._config.training.resume_from
        resume_states: dict[str, dict | None] = {aid: None for aid in agent_configs}
        if not resume_from:
            return resume_states
        for aid, acfg in agent_configs.items():
            state = resolve_resume(resume_from, aid)
            if state is not None:
                check_model_state(build_model(acfg), state["model_state"], state["source"])
                logger.info(f"Resume [{aid}]: {state['source']} (policy_version {state['policy_version']})")
            resume_states[aid] = state
        start_env_steps = max((s["env_steps"] for s in resume_states.values() if s), default=0)
        if start_env_steps > 0:
            self._env_step_counter.add(start_env_steps)
            logger.info(f"Resume: env-step counter continues from {start_env_steps}")
        return resume_states

    def _save_checkpoint(self, payload: dict) -> None:
        aid = payload["agent_id"]
        meta = {**self._checkpoint_meta.get(aid, {}), "env_steps": int(self._env_step_counter.value)}
        self._coordinator.save_checkpoint_payload(payload, meta_extra=meta)

    def _drain_all_checkpoints(self) -> None:
        """Save every queued checkpoint payload.

        A payload that fails to save is logged with its traceback and the remaining
        ones are still saved; the first error is re-raised at the end.
        """
        first_error: Exception | None = None
        for aid, cq in self._checkpoint_queues.items():
            # A learner killed mid-flush (OOM, SIGKILL) leaves a partial payload: skipped (T5.3).
            for payload in self._drain(cq, f"checkpoint-{aid}", lambda aid=aid: [f"learner-{aid}"]):
                try:
                    self._save_checkpoint(payload)
                except Exception as e:  # noqa: BLE001 - keep saving the others, re-raise below
                    logger.exception(
                        f"Failed to save checkpoint v{payload.get('policy_version')} of {payload.get('agent_id')}"
                    )
                    first_error = first_error or e
        if first_error is not None:
            raise first_error

    def _refresh_worker_matches(
        self,
        coordinator: Coordinator,
        agent_ids: list[str],
        command_queues: list[mp.Queue],
        worker_sent_ckpts: list[dict[str, set]],
    ) -> None:
        """Advance the owner rotation and send every worker fresh slot maps.

        Matches for worker ``w`` are generated at global env offset
        ``w * envs_per_worker``. Each command carries only checkpoints that the
        worker does not have yet. A checkpoint counts as delivered only after its
        command was put successfully; a full command queue means the worker skips
        this round and gets the deltas with the next refresh (D9).
        """
        from colosseum.core.types import WorkerCommand

        num_envs = self._config.rollout.envs_per_worker
        coordinator.next_round()
        for worker_id, cq in enumerate(command_queues):
            sent = worker_sent_ckpts[worker_id]
            match_configs = coordinator.generate_match_configs(num_envs, env_offset=worker_id * num_envs)
            new_ckpts, slot_network_map, collect_mask, slot_agent_map = _derive_worker_configs(
                match_configs, coordinator, agent_ids, already_sent=sent,
            )
            cmd = WorkerCommand(
                slot_agent_map=slot_agent_map,
                slot_network_map=slot_network_map,
                collect_mask=collect_mask,
                new_checkpoints={aid: c for aid, c in new_ckpts.items() if c},
            )
            try:
                cq.put_nowait(cmd)
            except queue.Full:
                logger.debug(f"worker-{worker_id} has not consumed its previous command; skipping this refresh")
                continue
            for aid, ckpts in new_ckpts.items():
                sent.setdefault(aid, set()).update(ckpts)

    def _shutdown(self) -> list[str]:
        """Stop every child within SHUTDOWN_GRACE_SEC; returns the names that had to be terminated.

        While the children wind down, their final checkpoints are saved and results and
        metrics drained (so no child blocks on a full queue). A failed save neither ends the
        grace window (other learners' final snapshots still arrive) nor skips the teardown;
        the first error is raised after both. Stragglers are then terminated and killed, and
        reads still in progress get until the grace deadline (at least 1 s after the kill)
        to complete; a read stuck on an incomplete message is then abandoned.
        """
        deadline = time.monotonic() + SHUTDOWN_GRACE_SEC
        self._stop_event.set()
        first_error: Exception | None = None

        def poll() -> None:
            nonlocal first_error
            for drain in (self._drain_all_checkpoints, self._drain_results, self._drain_metrics):
                try:
                    drain()
                except Exception as e:  # noqa: BLE001 - re-raised after the teardown
                    if first_error is None and drain is not self._drain_all_checkpoints:
                        logger.exception("Draining queues during shutdown failed")  # checkpoints log their own
                    first_error = first_error or e

        killed: list[str] = []
        try:
            self._supervisor.wait_all(SHUTDOWN_GRACE_SEC, poll=poll)
            poll()
        finally:
            # Teardown runs even if the wait was interrupted (the error propagates after it).
            killed = self._supervisor.kill_remaining()
            if killed:
                logger.warning(f"Terminated processes that did not stop within {SHUTDOWN_GRACE_SEC:.0f}s: "
                               f"{', '.join(killed)}")
            poll()
            self._finish_reads(max(deadline, time.monotonic() + 1.0), poll)
            # Detach queue feeder threads so a queue still holding undrained data
            # (e.g. trajectory chunks a now-dead learner never consumed, or large
            # WorkerCommands) cannot block this process from exiting.
            for q in self._all_queues:
                try:
                    q.cancel_join_thread()
                except (AttributeError, OSError):
                    pass
            logger.info("All processes stopped")
        if first_error is not None:
            raise first_error
        return killed

    def _release_queues(self) -> None:
        """After the teardown and the final drains: close the queues and drop the references.

        Without this, the queues (and their semaphores) outlive ``launch`` until the cyclic
        GC frees them, e.g. along with a traceback; a GC that runs inside the
        multiprocessing resource tracker then warns that they "might leak" (gh-109629).
        - A queue still being read by a helper thread (a read abandoned on an incomplete
          message) stays open: closing its pipe under that thread is unsafe.
        - Command queues (the main process is their producer) are not closed: closing the
          read end would turn a feeder blocked on an unread command into a BrokenPipeError
          traceback. Their feeders exit once the queue objects are freed.
        """
        busy = {key: reader for key, reader in self._readers.items() if reader.busy}
        produced_here = {id(q) for q in self._command_queues}
        for q in self._all_queues:
            try:
                q.cancel_join_thread()  # idempotent; also when _shutdown never got this far
                if id(q) not in busy and id(q) not in produced_here:
                    q.close()
            except (AttributeError, OSError):
                pass
        self._readers = busy
        self._all_queues = []
        self._command_queues = []
        self._trajectory_queues = {}
        self._checkpoint_queues = {}
        self._results_queue = self._metrics_queue = None

    def _finish_reads(self, deadline: float, poll: Callable[[], None]) -> None:
        """All children are gone: let in-progress reads complete until ``deadline``, then
        abandon the stuck ones (a producer died mid-message) and take what they read before."""
        while any(r.busy for r in self._readers.values()) and time.monotonic() < deadline:
            poll()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        stuck = [r for r in self._readers.values() if r.busy and not r.abandoned]
        for reader in stuck:
            reader.abandon()
        poll()  # what the finished reads got (and the abandoned ones' complete items)


def run_training(config_path: str, overrides: dict | None = None) -> int:
    """Entry point of ``colosseum train``. Returns the process exit code (D10).

    The run directory is printed (and logged) once it exists.
    """
    from colosseum.utils.seeding import apply_global_seed

    mp.set_start_method("spawn", force=True)
    setup_process_logging(None, "main", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    # Before the run dir exists: an invalid config leaves nothing behind, so a corrected
    # retry with the same run.name works. Launcher.launch does not validate again.
    validate_agent_configs(config)
    if config.training.resume_from:
        from colosseum.coordinator.checkpoint_manager import classify_resume_source
        classify_resume_source(config.training.resume_from)  # exists and has a known layout (nothing loaded)
    apply_global_seed(config.training.seed)  # after overrides: --set training.seed works (R3-27)
    run_dir = RunDir.create(config, config_path)
    config = run_dir.with_run_name(config)  # the resolved config and checkpoint hashes agree
    setup_process_logging(run_dir.logs, "main", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
    code = Launcher(config, run_dir, validated=True).launch()
    if code == 0:
        logger.info(f"Training finished; outputs in {run_dir.root}")
    return code
