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
import queue
import time

import numpy as np
import torch

from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.ipc import SharedCounter
from colosseum.core.run_dir import RunDir
from colosseum.core.types import LATEST_NETWORK_ID, MatchConfig
from colosseum.metrics.wandb_logger import WandBLogger
from colosseum.utils.logging import setup_process_logging
from colosseum.utils.process import run_child

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


def validate_agent_configs(config: ColosseumConfig) -> dict[str, ColosseumConfig]:
    """Effective config of every trainable agent, each checked by ``validate_config``.

    Fails fast (ConfigError) on structural errors, e.g. a head/encoder dim mismatch.
    """
    from colosseum.core.registry import validate_config

    agent_configs = {aid: config.get_agent_config(aid) for aid in config.get_trainable_agent_ids()}
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
) -> None:
    """Worker process body (see ``_worker_target``).

    Env and model factories are built INSIDE the process (``functools.partial``
    over top-level functions is picklable under spawn, which the subprocess
    vec_env needs to ship ``env_fn`` to its children).
    """
    import sys
    from functools import partial
    sys.path.insert(0, ".")

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
    import sys
    sys.path.insert(0, ".")

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


def _drain_queue(q) -> list:
    """Everything currently in ``q``. A broken item (dead producer) ends the drain with a warning."""
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items
        except (EOFError, OSError) as e:
            logger.warning(f"Dropped an unreadable queue item: {e!r}")
            return items


# =====================================================================
# Launcher
# =====================================================================

class Launcher:
    """Reads config, instantiates components, starts training processes.

    Spawns one learner per trainable agent, shared workers that route
    trajectory chunks to the correct agent's learner.
    """

    def __init__(self, config: ColosseumConfig, run_dir: RunDir) -> None:
        self._config = config
        self._run_dir = run_dir
        self._processes: list[mp.Process] = []
        self._stop_event = mp.Event()
        self._all_queues: list[mp.Queue] = []
        # Env steps taken by all workers together (spec block 2: global budget).
        self._env_step_counter = SharedCounter()
        # Set by launch(); used by the monitor loop and _shutdown to save checkpoints.
        self._coordinator: Coordinator | None = None
        self._agent_ids: list[str] = []
        self._checkpoint_queues: dict[str, mp.Queue] = {}
        self._checkpoint_meta: dict[str, dict] = {}

    @property
    def env_steps_done(self) -> int:
        """Env steps taken by all workers so far (the global budget counter)."""
        return self._env_step_counter.value

    def launch(self) -> None:
        """Launch the full training pipeline."""
        cfg = self._config
        trainable_agents = cfg.get_trainable_agent_ids()

        logger.info("Colosseum: Starting training pipeline")
        logger.info(f"  Agents: {trainable_agents}")
        logger.info(f"  Algorithm: {cfg.algorithm.name}")
        logger.info(f"  Workers: {cfg.rollout.num_workers}")
        logger.info(f"  Envs per worker: {cfg.rollout.envs_per_worker}")
        logger.info(f"  Chunk length: {cfg.rollout.chunk_length}")

        # Per-agent effective configs (overrides merged), validated before any process starts.
        agent_configs = validate_agent_configs(cfg)
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

        checkpoint_interval = cfg.self_play.checkpoint_interval

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

        # Start one learner per agent
        from colosseum.utils.seeding import learner_seed
        for agent_index, aid in enumerate(trainable_agents):
            acfg = agent_configs[aid]
            # numpy + bytes only: Process arguments cross the process boundary (R6-02).
            resume_state = resume_states[aid]
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
                    checkpoint_interval=checkpoint_interval,
                    resume_state=resume_state,
                    progress_counter=self._env_step_counter,
                    total_timesteps=cfg.training.total_timesteps,
                    num_learners=len(trainable_agents),
                    seed=learner_seed(cfg.training.seed, agent_index),
                ),
                daemon=True,
            )
            learner_proc.start()
            self._processes.append(learner_proc)
            logger.info(f"Learner started for agent {aid}")

        # Start workers (shared across all agents)
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
                ),
                daemon=worker_daemon,
            )
            worker_proc.start()
            self._processes.append(worker_proc)

        # Initialize WandB
        wandb_logger = WandBLogger(
            cfg.metrics,
            run_name=f"colosseum_{'_'.join(trainable_agents)}",
        )
        wandb_logger.log_config(cfg.model_dump())

        # Monitor loop
        logger.info("Training started. Press Ctrl+C to stop.")
        try:
            self._monitor_loop(
                metrics_queue,
                results_queue,
                wandb_logger,
                cfg.metrics.log_interval,
                coordinator,
                trainable_agents,
                command_queues,
                worker_sent_ckpts,
            )
        except KeyboardInterrupt:
            logger.info("Received interrupt, stopping...")
        finally:
            try:
                self._shutdown()
            finally:
                wandb_logger.finish()

    def _monitor_loop(
        self,
        metrics_queue: mp.Queue,
        results_queue: mp.Queue,
        wandb_logger: WandBLogger,
        log_interval: int,
        coordinator: Coordinator,
        agent_ids: list[str],
        command_queues: list[mp.Queue] | None = None,
        worker_sent_ckpts: list[dict[str, set]] | None = None,
    ) -> None:
        """Main process monitors metrics, saves checkpoints, checks for completion.

        Also periodically re-generates match assignments and pushes them (plus
        any newly-saved checkpoints) to workers (C1), so self-play-vs-history
        and PFSP opponent selection evolve during training.

        The main process is the only budget authority: it sets ``stop_event``
        once all workers together have taken ``training.total_timesteps`` env
        steps. Learners and workers run until ``stop_event``, so any of them
        exiting earlier is unexpected. A dead learner stops the run: workers
        would block on its full trajectory queue and starve the other agents.
        (Learners are the first ``len(agent_ids)`` entries of ``self._processes``.)
        """
        num_learners = len(agent_ids)
        learner_procs = dict(zip(agent_ids, self._processes[:num_learners], strict=True))
        last_steps: dict[str, int] = {aid: 0 for aid in agent_ids}

        refresh_interval = self._config.rollout.match_refresh_interval_sec
        last_refresh = time.time()

        while not self._stop_event.is_set():
            dead_learners = [aid for aid, p in learner_procs.items() if not p.is_alive()]
            if dead_learners and not self._stop_event.is_set():
                logger.error(
                    f"Learner process(es) for {dead_learners} exited unexpectedly "
                    f"(exit codes {[learner_procs[a].exitcode for a in dead_learners]}); stopping."
                )
                self._stop_event.set()
                break

            # If every worker has died, learners would starve forever — stop.
            worker_procs = self._processes[num_learners:]
            if worker_procs and not any(p.is_alive() for p in worker_procs):
                logger.error("All worker processes exited unexpectedly; stopping.")
                self._stop_event.set()
                break

            # Process per-agent checkpoint saves
            self._drain_all_checkpoints()

            # Process episode results from workers
            while True:
                try:
                    result = results_queue.get_nowait()
                    coordinator.report_match_result(result)
                except queue.Empty:
                    break

            # Drain metrics queue
            while True:
                try:
                    metrics = metrics_queue.get_nowait()
                    step = int(metrics.get("train_step", 0))
                    agent_id_m = metrics.pop("agent_id", "agent_0")

                    last = last_steps.get(agent_id_m, 0)
                    if step - last >= log_interval or step == 1:
                        wandb_logger.log_train_step(agent_id_m, metrics, step)
                        last_steps[agent_id_m] = step
                except queue.Empty:
                    break

            # Periodically refresh worker match assignments (C1).
            if (command_queues is not None
                    and refresh_interval > 0
                    and time.time() - last_refresh >= refresh_interval):
                self._refresh_worker_matches(
                    coordinator, agent_ids, command_queues, worker_sent_ckpts,
                )
                last_refresh = time.time()

            # Global env-step budget (spec block 2): all workers together have
            # taken training.total_timesteps env steps -> stop everything.
            if self.env_steps_done >= self._config.training.total_timesteps:
                logger.info(
                    f"Env-step budget reached ({self.env_steps_done} >= "
                    f"{self._config.training.total_timesteps}); stopping."
                )
                self._stop_event.set()
                break

            time.sleep(0.5)

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
        for cq in self._checkpoint_queues.values():
            for payload in _drain_queue(cq):
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

    def _shutdown(self) -> None:
        """Stop children; save every learner's final checkpoint before tearing them down."""
        self._stop_event.set()
        learner_procs = self._processes[: len(self._agent_ids)]
        # A failed save neither ends the grace window (other learners' final snapshots
        # still arrive) nor skips teardown; the first error is raised after both.
        first_error: Exception | None = None

        def drain() -> None:
            nonlocal first_error
            try:
                self._drain_all_checkpoints()
            except Exception as e:  # noqa: BLE001 - logged by _drain_all_checkpoints, re-raised below
                first_error = first_error or e

        try:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and any(p.is_alive() for p in learner_procs):
                drain()
                time.sleep(0.1)
            drain()
        finally:
            # Teardown runs even if a checkpoint save failed (the error propagates after it).
            # Detach queue feeder threads so a queue still holding undrained data
            # (e.g. trajectory chunks a now-dead learner never consumed, or large
            # WorkerCommands) cannot block this process from exiting on join.
            for q in self._all_queues:
                try:
                    q.cancel_join_thread()
                except (AttributeError, OSError):
                    pass
            for proc in self._processes:
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2)
            logger.info("All processes stopped")
        if first_error is not None:
            raise first_error


def run_training(config_path: str, overrides: dict | None = None) -> RunDir:
    """Entry point: load config, create the run dir, log to it, launch training."""
    from colosseum.utils.seeding import apply_global_seed

    mp.set_start_method("spawn", force=True)
    setup_process_logging(None, "main", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    # Before the run dir exists: an invalid config leaves nothing behind, so a corrected
    # retry with the same run.name works.
    validate_agent_configs(config)
    apply_global_seed(config.training.seed)  # after overrides: --set training.seed works (R3-27)
    run_dir = RunDir.create(config, config_path)
    config = run_dir.with_run_name(config)  # the resolved config and checkpoint hashes agree
    setup_process_logging(run_dir.logs, "main", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
    Launcher(config, run_dir).launch()
    return run_dir
