"""A node's worker process (SP5 spec 4.2, 4.7, 7.1; rulings R3, R4, R10, R11).

The SP4 ``RolloutLoop`` over ``net.local.WorkerIO``: lineups, routes, evictions, ``WeightsReady`` and ``Stop`` come
on the control connection to the node; weights (latest, snapshots, frozen agents' blobs) are read from the
shared-memory cache; chunks go to the local learners' listeners (or are dropped and counted); results and cumulative
progress go back on the control connection. The worker waits for the first lineup of every env before it builds the
loop; it plays no match before ``WeightsReady`` (A-M3). Match uids are ``<run_id>/train/n<node>/w<W>_e<E>_g<epoch>_
ep<K>`` and episode seeds come from cell seeds (spec 7.6).

Errors: an ``EnvContractError`` / ``PlayerError`` / any other exception is sent to the node as a ``ChildError``
(``contract`` / ``player`` / ``crash``) with its location text, then re-raised (the process exits 1; the node does not
restart a worker that reported its error: the hub stops the run, A12)."""

from __future__ import annotations

import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["CONNECT_TIMEOUT_SEC", "FIRST_LINEUPS_POLL_SEC", "create_env", "worker_target"]

CONNECT_TIMEOUT_SEC = 10.0
FIRST_LINEUPS_POLL_SEC = 0.05


def create_env(env_class_path: str, kwargs: dict):
    """Create an env instance inside a worker process (was ``launcher._create_env``)."""
    from colosseum.core.registry import import_class

    cls = import_class(env_class_path)
    return cls(**kwargs)


def worker_target(*, name: str, log_dir: str, incarnation: int, **kwargs: Any) -> None:
    """Process entry point: process logging and the child signal policy, then the worker body."""
    from colosseum.utils.process import run_child

    run_child(name, log_dir, partial(_worker_main, name=name), incarnation=incarnation, **kwargs)


def _context(error: BaseException) -> str:
    """The location prefix of an error text ("worker W, env E, seat P, episode step K, layout L"), else ""."""
    text = str(error)
    head, sep, _ = text.partition(": ")
    return head if sep and head.startswith("worker ") else ""


def _worker_main(*, name: str, incarnation: int, node: int, worker: int, node_address: str, authkey: bytes,
                 config: Any, agent_ids: list[str], agent_roles: dict[str, list[str]], agent_configs: dict[str, Any],
                 role_specs: dict[str, Any], bots: dict[str, Any], frozen_keys: dict[str, Any],
                 fixed_roles: dict[str, tuple[str, ...]], teachers: dict[str, Any], cache_root: str,
                 cache_prefix: str, run_id: str) -> None:
    from colosseum.core.errors import EnvContractError, PlayerError
    from colosseum.core.threads import configure_torch_threads
    from colosseum.net.local import ChildError, ChildHello, WorkerIO, connect
    from colosseum.node.weights import WeightCacheReader
    from colosseum.worker.rollout_worker import RESULTS_BATCH_MAX, RESULTS_BATCH_SEC

    configure_torch_threads(config.rollout.torch_threads)
    control = connect(node_address, authkey, timeout=CONNECT_TIMEOUT_SEC)
    control.send(ChildHello("worker", name, incarnation, os.getpid(), "control"))
    route = connect(node_address, authkey, timeout=CONNECT_TIMEOUT_SEC)
    route.send(ChildHello("worker", name, incarnation, os.getpid(), "route"))
    cache = WeightCacheReader(Path(cache_root), cache_prefix)
    io = WorkerIO(node=node, worker=worker, incarnation=incarnation, control=control, route=route, cache=cache,
                  routes={}, authkey=authkey, results_batch=RESULTS_BATCH_MAX, results_flush_sec=RESULTS_BATCH_SEC)
    try:
        _play(io, cache, node=node, worker=worker, incarnation=incarnation, config=config, agent_ids=agent_ids,
              agent_roles=agent_roles, agent_configs=agent_configs, role_specs=role_specs, bots=bots,
              frozen_keys=frozen_keys, fixed_roles=fixed_roles, teachers=teachers, run_id=run_id)
    except EnvContractError as e:
        io.send_error(ChildError("contract", str(e), _context(e)))
        raise
    except PlayerError as e:
        io.send_error(ChildError("player", str(e), _context(e)))
        raise
    except Exception as e:
        io.send_error(ChildError("crash", f"{type(e).__name__}: {e}", _context(e)))
        raise
    finally:
        for conn in (route, control):
            try:
                conn.close()
            except OSError:
                pass


def _play(io: Any, cache: Any, *, node: int, worker: int, incarnation: int, config: Any, agent_ids: list[str],
          agent_roles: dict, agent_configs: dict, role_specs: dict, bots: dict, frozen_keys: dict, fixed_roles: dict,
          teachers: dict, run_id: str) -> None:
    from colosseum.core.registry import build_model
    from colosseum.players.registry import FixedPlayers, FrozenSpec
    from colosseum.utils.seeding import cell_seed, worker_seed
    from colosseum.worker.rollout_loop import RolloutLoop

    frozen = {}
    for agent, key in frozen_keys.items():
        cached = cache.get(key)
        if cached is None:   # the node fetches every fixed blob before it starts a worker
            raise RuntimeError(f"worker {worker}: the weights of frozen agent {agent!r} ({key.ident}) are not in the "
                               f"weight cache")
        frozen[agent] = FrozenSpec(agent_id=agent, roles=tuple(cached.meta.roles or fixed_roles.get(agent, ())),
                                   networks=dict(cached.meta.architecture or {}), model_state=cached.state_dict,
                                   source=f"hub blob {key.ident}")
    fixed = FixedPlayers(bots=dict(bots), frozen=frozen, roles=dict(fixed_roles))
    num_envs = int(config.rollout.envs_per_worker)
    first: dict = {}
    while len([e for e in first if e < num_envs]) < num_envs:      # the first lineup of every env
        if io.stopped():
            logger.info(f"Worker {worker}: stopped before its first lineups")
            return
        first.update(io.take_lineups())
        if len([e for e in first if e < num_envs]) < num_envs:
            time.sleep(FIRST_LINEUPS_POLL_SEC)
    checkpoints: dict[str, dict[str, Any]] = {}
    for e in range(num_envs):
        for key in first[e].keys.values():
            if key.kind != "snapshot" or key.ident in checkpoints.get(key.agent_id, {}):
                continue
            cached = cache.get(key)
            if cached is None:   # R11: a lineup is sent only when its weights are cached
                raise RuntimeError(f"worker {worker}: snapshot {key.agent_id}/{key.ident} of a granted lineup is not "
                                   f"in the weight cache")
            checkpoints.setdefault(key.agent_id, {})[key.ident] = cached.state_dict
    seed = config.training.seed
    logger.info(f"Worker {worker}: starting (node {node}, incarnation {incarnation}) with {num_envs} envs "
                f"({config.rollout.vec_env}), chunk_length={config.rollout.chunk_length}, agents={agent_ids}, "
                f"fixed players={sorted(set(fixed.bots) | set(fixed.frozen))}")
    loop = RolloutLoop(
        worker_id=worker, env_fn=partial(create_env, config.env.env_class, config.env.kwargs), num_envs=num_envs,
        chunk_length=config.rollout.chunk_length, agent_ids=agent_ids, agent_roles=agent_roles,
        model_factories={a: partial(build_model, agent_configs[a], role_specs[a]) for a in agent_ids},
        io=io.loop_io(), lineups=[first[e].lineup for e in range(num_envs)],
        weight_sync_interval=config.rollout.weight_sync_interval_sec, checkpoint_state_dicts_by_agent=checkpoints,
        seed=worker_seed(seed, node, worker, incarnation), vec_env_kind=config.rollout.vec_env,
        subproc_workers=config.rollout.subproc_workers, max_idle_steps=config.env.max_idle_steps,
        fixed_players=fixed, teachers=teachers, run_id=run_id,
        wait_for_first_weights=lambda: io.stopped() or io.ready(),
        cell_seed_fn=lambda env, epoch: cell_seed(seed, epoch, node, worker, env),
        uid_prefix=f"{run_id}/train/n{node}/", snapshot_fallback=False)
    io.stats_fn = lambda: loop.stats

    def should_stop() -> bool:
        io.idle()
        return io.stopped()

    try:
        loop.run(should_stop=should_stop)
    finally:
        try:
            io.close()                         # the last results and the final progress go to the node first
        finally:
            loop.close()
    logger.info(f"Worker {worker}: finished. {io.counters()}")
