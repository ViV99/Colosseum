"""Launcher: reads config, instantiates all components, starts processes.

This is the top-level entry point that wires together:
- Coordinator (matchmaking, checkpoint management)
- Workers (env + inference)
- Learners (one per trainable agent)
- Weight synchronization (via mp.Queue)
- Metrics logging

With mp.set_start_method("spawn"), all arguments to Process targets must be
picklable. We pass config objects (pydantic models) and string class paths
instead of closures/lambdas, and let each process instantiate its own objects.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time

import torch

from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.types import MatchConfig
from colosseum.metrics.wandb_logger import WandBLogger

logger = logging.getLogger(__name__)


# Queue size constants
_WEIGHT_QUEUE_SIZE = 2
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


def _create_network(config: ColosseumConfig):
    """Create ActorCriticNetwork inside a worker/learner process."""
    from colosseum.core.registry import build_network
    return build_network(config)


def _worker_target(
    worker_id: int,
    config: ColosseumConfig,
    agent_ids: list[str],
    agent_configs: dict[str, ColosseumConfig],
    trajectory_queues: dict[str, mp.Queue],
    weight_queues: dict[str, mp.Queue],
    stop_event: mp.Event,
    total_timesteps: int,
    checkpoint_state_dicts_by_agent: dict[str, dict[str, dict]] | None = None,
    slot_network_map: list[list[str]] | None = None,
    collect_mask: list[list[bool]] | None = None,
    slot_agent_map: list[list[str]] | None = None,
    results_queue: mp.Queue | None = None,
    command_queue: mp.Queue | None = None,
) -> None:
    """Worker process entry point.

    Builds per-agent network factories INSIDE the process to avoid
    pickling lambdas across the spawn boundary.
    """
    import sys
    from functools import partial
    sys.path.insert(0, ".")

    from colosseum.worker.rollout_worker import rollout_worker_process

    env_class_path = config.env.env_class
    env_kwargs = config.env.kwargs

    worker_seed = None
    if config.training.seed is not None:
        worker_seed = config.training.seed + worker_id * 1000

    # Build per-agent network factories inside this process (pickling safe)
    network_factories = {}
    for aid in agent_ids:
        acfg = agent_configs[aid]
        # Default-arg capture ensures each lambda gets its own config
        network_factories[aid] = lambda _cfg=acfg: _create_network(_cfg)

    rollout_worker_process(
        worker_id=worker_id,
        # functools.partial over a top-level function is picklable under spawn,
        # which the subprocess vec_env requires (it ships env_fn to children).
        env_fn=partial(_create_env, env_class_path, env_kwargs),
        num_envs=config.rollout.envs_per_worker,
        chunk_length=config.rollout.chunk_length,
        weight_sync_interval=config.rollout.weight_sync_interval_sec,
        stop_event=stop_event,
        total_timesteps=total_timesteps,
        results_queue=results_queue,
        seed=worker_seed,
        # Multi-agent params
        agent_ids=agent_ids,
        network_factories=network_factories,
        trajectory_queues=trajectory_queues,
        weight_queues=weight_queues,
        checkpoint_state_dicts_by_agent=checkpoint_state_dicts_by_agent,
        slot_agent_map=slot_agent_map,
        slot_network_map=slot_network_map,
        collect_mask=collect_mask,
        # Runtime match refresh + vec-env backend
        command_queue=command_queue,
        vec_env_kind=config.rollout.vec_env,
        subproc_workers=config.rollout.subproc_workers,
    )


def _learner_target(
    agent_id: str,
    config: ColosseumConfig,
    trajectory_queue: mp.Queue,
    weight_queues: list[mp.Queue],
    stop_event: mp.Event,
    metrics_queue: mp.Queue,
    total_train_steps: int,
    checkpoint_queue: mp.Queue | None = None,
    checkpoint_interval: int = 0,
    resume_state: dict | None = None,
) -> None:
    """Learner process entry point."""
    import sys
    sys.path.insert(0, ".")

    from colosseum.core.registry import import_class
    from colosseum.learner.learner import learner_process

    device = config.learner.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
        net = _create_network(config)
        kickstart = None
        if teacher_path:
            from colosseum.bc.kickstart import KickstartLoss
            teacher = _create_network(config)
            teacher_sd = torch.load(teacher_path, weights_only=True, map_location=device)
            teacher.load_state_dict(teacher_sd)
            teacher.to(device)
            kickstart = KickstartLoss(
                teacher,
                initial_lambda=config.training.kickstart_lambda,
                decay_steps=config.training.kickstart_decay_steps,
            )
            logger.info(f"Kickstart enabled for {agent_id} from {teacher_path}")
        kwargs = {"device": device, "pin_memory": config.learner.pin_memory}
        if kickstart is not None:
            kwargs["kickstart"] = kickstart
        return algo_cls(net, config.algorithm, **kwargs)

    learner_process(
        agent_id=agent_id,
        algorithm_factory=algorithm_factory,
        trajectory_queue=trajectory_queue,
        weight_queues=weight_queues,
        config=config.learner,
        stop_event=stop_event,
        metrics_queue=metrics_queue,
        total_train_steps=total_train_steps,
        checkpoint_queue=checkpoint_queue,
        checkpoint_interval=checkpoint_interval,
        resume_state=resume_state,
    )


# =====================================================================
# Match config helpers
# =====================================================================

def _derive_worker_configs(
    match_configs: list[MatchConfig],
    coordinator: Coordinator,
    agent_ids: list[str],
) -> tuple[
    dict[str, dict[str, dict]],  # checkpoint_state_dicts_by_agent
    list[list[str]],             # slot_network_map
    list[list[bool]],            # collect_mask
    list[list[str]],             # slot_agent_map
]:
    """Derive network pool config from match configs.

    Returns:
        (checkpoint_state_dicts_by_agent, slot_network_map, collect_mask, slot_agent_map)
        checkpoint_state_dicts_by_agent: {agent_id: {ckpt_id: state_dict}}
        slot_agent_map[env_idx][player_idx] -> agent_id
    """
    from colosseum.worker.rollout_loop import LATEST_NETWORK_ID

    collect_mask: list[list[bool]] = []
    slot_network_map: list[list[str]] = []
    slot_agent_map: list[list[str]] = []
    checkpoint_state_dicts_by_agent: dict[str, dict[str, dict]] = {
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
                        agent_ckpts[ckpt_id] = sd
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
        for aid in trainable_agents:
            coordinator.agent_pool.register_trainable(aid)

        # Compute total train steps (same for all agents)
        steps_per_chunk = cfg.rollout.chunk_length
        chunks_per_step = cfg.learner.batch_chunks
        env_steps_per_train_step = steps_per_chunk * chunks_per_step
        total_train_steps = max(
            1, cfg.training.total_timesteps // env_steps_per_train_step,
        )

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
            resume_state = _resolve_resume_state(cfg, aid, coordinator)
            learner_proc = mp.Process(
                target=_learner_target,
                args=(
                    aid, acfg, trajectory_queues[aid],
                    weight_queues_per_agent[aid],
                    self._stop_event, metrics_queue, total_train_steps,
                    checkpoint_queues[aid], checkpoint_interval,
                    resume_state,
                ),
                daemon=True,
            )
            learner_proc.start()
            self._processes.append(learner_proc)
            logger.info(f"Learner started for agent {aid}")

        # Start workers (shared across all agents)
        for worker_id in range(cfg.rollout.num_workers):
            match_configs = coordinator.generate_match_configs(
                trainable_agents[0], cfg.rollout.envs_per_worker,
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
                args=(
                    worker_id, cfg, trainable_agents, agent_configs,
                    trajectory_queues, worker_weight_queues,
                    self._stop_event,
                    cfg.training.total_timesteps // cfg.rollout.num_workers,
                    ckpt_dicts_by_agent,
                    slot_network_map, collect_mask, slot_agent_map,
                    results_queue,
                    command_queues[worker_id],
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

        Stops when ALL learner processes have exited (they are the first
        N processes in self._processes where N = len(agent_ids)).
        """
        num_learners = len(agent_ids)
        learner_procs = self._processes[:num_learners]
        last_steps: dict[str, int] = {aid: 0 for aid in agent_ids}

        refresh_interval = self._config.rollout.match_refresh_interval_sec
        last_refresh = time.time()

        while not self._stop_event.is_set():
            # Stop when ALL learner processes have exited
            learner_alive = [p.is_alive() for p in learner_procs]
            if not any(learner_alive):
                logger.info("All learner processes exited, stopping workers...")
                self._stop_event.set()
                break

            # Check if all processes are dead (unexpected)
            alive = [p.is_alive() for p in self._processes]
            if not any(alive):
                logger.info("All processes have exited")
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
                            state_dict=ckpt_data["state_dict"],
                            optimizer_state=ckpt_data.get("optimizer_state"),
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

            time.sleep(0.5)

    def _refresh_worker_matches(
        self,
        coordinator: Coordinator,
        agent_ids: list[str],
        command_queues: list[mp.Queue],
        worker_broadcast_ckpts: list[dict[str, set]],
    ) -> None:
        """Re-generate per-worker match assignments and push them to workers.

        Each worker gets a fresh set of slot assignments (reflecting current
        checkpoints / PFSP win rates) plus any checkpoint state_dicts it does
        not yet have (deltas only, to avoid resending large payloads).
        """
        from colosseum.core.types import WorkerCommand

        num_envs = self._config.rollout.envs_per_worker
        primary = agent_ids[0]

        for worker_id, cq in enumerate(command_queues):
            match_configs = coordinator.generate_match_configs(primary, num_envs)
            (
                ckpt_dicts_by_agent,
                slot_network_map,
                collect_mask,
                slot_agent_map,
            ) = _derive_worker_configs(match_configs, coordinator, agent_ids)

            # Checkpoint deltas not yet sent to this worker.
            new_ckpts: dict[str, dict[str, dict]] = {}
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
        import numpy as np
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
