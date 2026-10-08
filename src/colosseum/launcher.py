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
from colosseum.core.ipc import SharedCounter, from_numpy_tree, to_numpy_tree
from colosseum.core.types import MatchConfig, state_dict_from_numpy, state_dict_to_numpy
from colosseum.metrics.wandb_logger import WandBLogger

logger = logging.getLogger(__name__)


# Queue size constants
_WEIGHT_QUEUE_SIZE = 1  # newest-wins mailbox per (agent, worker); see core.ipc.put_latest
_CHECKPOINT_QUEUE_SIZE = 16
_METRICS_QUEUE_SIZE = 100
_RESULTS_QUEUE_SIZE = 1000
_COMMAND_QUEUE_SIZE = 2


# =====================================================================
# Top-level process targets (must be picklable for spawn)
# =====================================================================

def _create_env(env_class_path: str, kwargs: dict):
    """Create env instance inside a worker process."""
    from colosseum.core.registry import import_class
    cls = import_class(env_class_path)
    return cls(**kwargs)


def _create_model(config: ColosseumConfig):
    """Create the agent's PolicyModel inside a worker/learner process."""
    from colosseum.core.registry import build_model
    return build_model(config)


def _worker_target(
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
    """Worker process entry point.

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


def _learner_target(
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
) -> None:
    """Learner process entry point.

    ``num_learners`` (learner processes on this machine) feeds the automatic
    torch thread count when ``learner.torch_threads`` is unset.
    """
    import sys
    sys.path.insert(0, ".")

    from colosseum.core.registry import import_class
    from colosseum.core.threads import configure_torch_threads, resolve_learner_threads
    from colosseum.learner.learner import learner_process, resolve_device

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
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],  # checkpoint_state_dicts_by_agent
    list[list[str]],             # slot_network_map
    list[list[bool]],            # collect_mask
    list[list[str]],             # slot_agent_map
]:
    """Derive network pool config from match configs.

    Returns:
        (checkpoint_state_dicts_by_agent, slot_network_map, collect_mask, slot_agent_map)
        checkpoint_state_dicts_by_agent: {agent_id: {ckpt_id: numpy state_dict}};
            these cross the process boundary (worker Process arguments and
            ``WorkerCommand.new_checkpoints``), so they are numpy, never torch.
        slot_agent_map[env_idx][player_idx] -> agent_id
    """
    from colosseum.worker.rollout_loop import LATEST_NETWORK_ID

    collect_mask: list[list[bool]] = []
    slot_network_map: list[list[str]] = []
    slot_agent_map: list[list[str]] = []
    checkpoint_state_dicts_by_agent: dict[str, dict[str, dict[str, np.ndarray]]] = {
        aid: {} for aid in agent_ids
    }

    for mc in match_configs:
        env_collect = []
        env_nets = []
        env_agents = []
        for slot in mc.player_slots:
            env_collect.append(slot.collect_trajectories)
            env_agents.append(slot.agent_id)

            if slot.checkpoint_id is not None:
                ckpt_id = slot.checkpoint_id
                agent_ckpts = checkpoint_state_dicts_by_agent.get(
                    slot.agent_id, {},
                )
                if ckpt_id not in agent_ckpts:
                    try:
                        sd = coordinator.checkpoint_manager.load(
                            slot.agent_id, ckpt_id,
                        )
                        agent_ckpts[ckpt_id] = state_dict_to_numpy(sd)
                    except FileNotFoundError:
                        logger.warning(
                            f"Checkpoint {ckpt_id} for {slot.agent_id} "
                            f"not found, using latest"
                        )
                        ckpt_id = LATEST_NETWORK_ID
                env_nets.append(ckpt_id)
            else:
                env_nets.append(LATEST_NETWORK_ID)

        collect_mask.append(env_collect)
        slot_network_map.append(env_nets)
        slot_agent_map.append(env_agents)

    return (
        checkpoint_state_dicts_by_agent,
        slot_network_map,
        collect_mask,
        slot_agent_map,
    )


def _resolve_resume_state(
    config: ColosseumConfig,
    agent_id: str,
    coordinator: Coordinator,
) -> dict | None:
    """Build a learner resume_state from ``training.resume_from``.

    Accepts either a path to a .pt state_dict (e.g. a BC output) or a checkpoint
    id resolvable by the checkpoint manager. Returns None if unset/not found.
    """
    import os

    rf = config.training.resume_from
    if not rf:
        return None

    for path in (rf, rf + ".pt"):
        if os.path.isfile(path):
            sd = torch.load(path, weights_only=True)
            logger.info(f"Resume [{agent_id}]: loaded weights from {path}")
            return {"state_dict": sd, "policy_version": 0}

    try:
        sd = coordinator.checkpoint_manager.load(agent_id, rf)
    except FileNotFoundError:
        logger.warning(
            f"Resume target {rf!r} not found for {agent_id}; starting from scratch"
        )
        return None

    state = {"state_dict": sd, "policy_version": 0}
    opt = coordinator.checkpoint_manager.load_optimizer(agent_id, rf)
    if opt is not None:
        state["optimizer_state"] = opt
    if rf.startswith("ckpt_v"):
        try:
            state["policy_version"] = int(rf[len("ckpt_v"):])
        except ValueError:
            pass
    logger.info(f"Resume [{agent_id}]: loaded checkpoint {rf}")
    return state


# =====================================================================
# Launcher
# =====================================================================

class Launcher:
    """Reads config, instantiates components, starts training processes.

    Spawns one learner per trainable agent, shared workers that route
    trajectory chunks to the correct agent's learner.
    """

    def __init__(self, config: ColosseumConfig) -> None:
        self._config = config
        self._processes: list[mp.Process] = []
        self._stop_event = mp.Event()
        self._all_queues: list[mp.Queue] = []
        # Env steps taken by all workers together (spec block 2: global budget).
        self._env_step_counter = SharedCounter()

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

        # Build per-agent effective configs (with overrides merged)
        agent_configs: dict[str, ColosseumConfig] = {}
        for aid in trainable_agents:
            agent_configs[aid] = cfg.get_agent_config(aid)

        # Fail fast on structural config errors (e.g. head/encoder dim mismatch)
        # before spawning any processes.
        from colosseum.core.registry import validate_config
        for aid in trainable_agents:
            validate_config(agent_configs[aid])

        # Initialize coordinator
        coordinator = Coordinator(cfg)

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
        worker_broadcast_ckpts: list[dict[str, set]] = [
            {aid: set() for aid in trainable_agents} for _ in range(cfg.rollout.num_workers)
        ]

        # Track every queue so shutdown can cancel feeder threads and avoid the
        # interpreter blocking on unflushed data when consumers have exited.
        self._all_queues = [metrics_queue, results_queue, *command_queues]
        for aid in trainable_agents:
            self._all_queues.append(trajectory_queues[aid])
            self._all_queues.append(checkpoint_queues[aid])
            self._all_queues.extend(weight_queues_per_agent[aid])

        # Start one learner per agent
        for aid in trainable_agents:
            acfg = agent_configs[aid]
            # numpy only: Process arguments cross the process boundary (R6-02).
            resume_state = to_numpy_tree(_resolve_resume_state(cfg, aid, coordinator))
            learner_proc = mp.Process(
                target=_learner_target,
                kwargs=dict(
                    agent_id=aid,
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
            ) = _derive_worker_configs(
                match_configs, coordinator, trainable_agents,
            )

            # Record which checkpoints this worker already has (sent at launch).
            for aid in trainable_agents:
                worker_broadcast_ckpts[worker_id][aid] |= set(
                    ckpt_dicts_by_agent.get(aid, {}).keys()
                )

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
                kwargs=dict(
                    worker_id=worker_id,
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
                checkpoint_queues,
                results_queue,
                wandb_logger,
                cfg.metrics.log_interval,
                coordinator,
                trainable_agents,
                command_queues,
                worker_broadcast_ckpts,
            )
        except KeyboardInterrupt:
            logger.info("Received interrupt, stopping...")
        finally:
            self._shutdown()
            wandb_logger.finish()

    def _monitor_loop(
        self,
        metrics_queue: mp.Queue,
        checkpoint_queues: dict[str, mp.Queue],
        results_queue: mp.Queue,
        wandb_logger: WandBLogger,
        log_interval: int,
        coordinator: Coordinator,
        agent_ids: list[str],
        command_queues: list[mp.Queue] | None = None,
        worker_broadcast_ckpts: list[dict[str, set]] | None = None,
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
            for aid in agent_ids:
                cq = checkpoint_queues[aid]
                while True:
                    try:
                        ckpt_data = cq.get_nowait()
                        ckpt_id = coordinator.maybe_save_checkpoint(
                            agent_id=aid,
                            policy_version=ckpt_data["policy_version"],
                            state_dict=state_dict_from_numpy(ckpt_data["state_dict"]),
                            optimizer_state=from_numpy_tree(ckpt_data.get("optimizer_state")),
                        )
                        if ckpt_id is not None:
                            logger.info(f"Saved checkpoint {ckpt_id} for {aid}")
                    except queue.Empty:
                        break

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
                    coordinator, agent_ids, command_queues, worker_broadcast_ckpts,
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

    def _refresh_worker_matches(
        self,
        coordinator: Coordinator,
        agent_ids: list[str],
        command_queues: list[mp.Queue],
        worker_broadcast_ckpts: list[dict[str, set]],
    ) -> None:
        """Re-generate per-worker match assignments and push them to workers.

        Advances the coordinator's owner rotation (``next_round``) first, then
        generates matches for worker ``w`` at global env offset
        ``w * envs_per_worker``. Each worker gets a fresh set of slot
        assignments (reflecting current checkpoints / PFSP win rates) plus any
        checkpoint state_dicts it does not yet have (deltas only, to avoid
        resending large payloads).
        """
        from colosseum.core.types import WorkerCommand

        num_envs = self._config.rollout.envs_per_worker

        coordinator.next_round()
        for worker_id, cq in enumerate(command_queues):
            match_configs = coordinator.generate_match_configs(num_envs, env_offset=worker_id * num_envs)
            (
                ckpt_dicts_by_agent,
                slot_network_map,
                collect_mask,
                slot_agent_map,
            ) = _derive_worker_configs(match_configs, coordinator, agent_ids)

            # Checkpoint deltas not yet sent to this worker.
            new_ckpts: dict[str, dict[str, dict[str, np.ndarray]]] = {}
            sent_sets = worker_broadcast_ckpts[worker_id]
            for aid in agent_ids:
                for ckpt_id, sd in ckpt_dicts_by_agent.get(aid, {}).items():
                    if ckpt_id not in sent_sets[aid]:
                        new_ckpts.setdefault(aid, {})[ckpt_id] = sd
                        sent_sets[aid].add(ckpt_id)

            cmd = WorkerCommand(
                slot_agent_map=slot_agent_map,
                slot_network_map=slot_network_map,
                collect_mask=collect_mask,
                new_checkpoints=new_ckpts,
            )
            try:
                cq.put_nowait(cmd)
            except queue.Full:
                pass  # worker will get the next refresh

    def _shutdown(self) -> None:
        """Signal all processes to stop and wait for them."""
        self._stop_event.set()

        # Detach queue feeder threads so a queue still holding undrained data
        # (e.g. trajectory chunks a now-dead learner never consumed, or large
        # WorkerCommands) cannot block this process — or the workers — from
        # exiting on join. Without this, a non-daemon worker / the main process
        # can hang in the queue's join_thread at interpreter shutdown.
        for q in self._all_queues:
            try:
                q.cancel_join_thread()
            except Exception:
                pass

        # Give processes a moment to notice the stop event
        time.sleep(1.0)

        for proc in self._processes:
            proc.join(timeout=3)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)

        logger.info("All processes stopped")


def run_training(config_path: str, overrides: dict | None = None) -> None:
    """Entry point: load config and launch training."""
    mp.set_start_method("spawn", force=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(config_path)

    # Set global seeds for reproducibility
    if config.training.seed is not None:
        import random
        seed = config.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f"Global seed set to {seed}")

    if overrides:
        config_dict = config.model_dump()
        for key, value in overrides.items():
            parts = key.split(".")
            d = config_dict
            for part in parts[:-1]:
                d = d[part]
            d[parts[-1]] = value
        config = ColosseumConfig(**config_dict)

    launcher = Launcher(config)
    launcher.launch()
