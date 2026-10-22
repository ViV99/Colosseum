"""Evaluator process of a node (SP5 spec 5.3; T5.1).

``evaluator_target`` (spawn target, ``ChildSpec(role="evaluator")``, name ``evaluator-<i>``): ``run_child`` (log
``<log dir>/evaluator-<i>.log``, the child signal policy, the parent-death watch of T4.4 as every node child), then
``evaluator_main``: one control connection to the node (``ChildHello(role="evaluator", channel="control")``), a
read-only ``WeightCacheReader`` of the machine's cache, and ``evaluator_loop``:
- ``EvalTask``: play the lease's lineups (all its whole blocks) with SP4's ``play_lineups`` on a ``CachePlayerPool``
  (``source`` = the lease's: ``pulse`` / ``tournament``; ``match_id_prefix`` the same; the torch sampling seeded
  with the lease's first lineup seed (P-PF4-20: it follows the lease grouping; env seeds are the blocks'); ``num_envs``
  and ``deterministic`` from the lease; the error context ``<source>, node N, evaluator E, env ...``); report
  ``EvalDone`` with the results grouped by block (each block's sorted by uid), never a part of a lease;
- a stop (``Stop``, or the node's connection closed) and an ``EvalCancel`` of the playing lease are checked before
  every env step: a stop ends the process without a report (the node reports the lease ``abandoned``), a cancel
  answers ``EvalDone(cancelled=True)``;
- a ``ConfigError`` / ``EnvContractError`` / ``PlayerError`` of a lease is ``EvalFailed(kind, message, where)``
  (``where``: the match uid and block, ``MatchPlacement``), weights missing from the cache ``EvalFailed("missing_
  weights")``; the evaluator keeps running. A typed evaluator error is never a ``ChildError`` (it travels to the hub
  as a failed ``BlockResult``). Any other exception crashes the process (``run_child`` logs the traceback; the node
  restarts it within ``WORKER_RESTARTS`` and reports the lease ``crash``).
"""

from __future__ import annotations

import functools
import json
import logging
import os
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from colosseum.arena.tournament.engine import MODEL_CACHE_SIZE, MatchPlacement, error_kind
from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError, EnvContractError, PlayerError
from colosseum.core.types import MatchResult
from colosseum.envs.game import GameSpec
from colosseum.net.local import ChildHello, EvalCancel, EvalDone, EvalFailed, EvalTask, Stop, connect
from colosseum.net.messages import PlayerRef
from colosseum.players.registry import FrozenSpec, build_frozen_model, make_bot
from colosseum.utils.process import run_child
from colosseum.worker.match_runner import ScriptedPlayer

__all__ = ["CONNECT_TIMEOUT_SEC", "POLL_SEC", "CachePlayerPool", "MissingWeights", "evaluator_loop", "evaluator_main",
           "evaluator_target", "play_task"]

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SEC = 30.0
POLL_SEC = 0.2
_KINDS = {"EnvContractError": "contract", "PlayerError": "player", "ConfigError": "config"}


class MissingWeights(RuntimeError):
    """A player's weights are not in the machine cache (the node hands a lease over only when they are)."""


class CachePlayerPool:
    """``PlayerPool`` of one lease's players over the machine's weight cache: neural players are built from the
    cache's read-only views on first use (``build_frozen_model`` copies them) and kept in an LRU of ``capacity``
    models keyed by blob (two tournaments' players with equal seat names never share a model; a lease's players
    are reserved); scripted players are ``ScriptedPlayer`` factories."""

    def __init__(self, spec: GameSpec, reader: Any, capacity: int = MODEL_CACHE_SIZE) -> None:
        self._spec, self._reader, self._capacity = spec, reader, int(capacity)
        self._current: dict[tuple[str, str], PlayerRef] = {}
        self._models: OrderedDict[Any, Any] = OrderedDict()
        self._bots: dict[tuple[str, str, str], ScriptedPlayer] = {}
        self._reserved: set[Any] = set()
        self.loads = 0

    def reserve(self, players: Sequence[PlayerRef]) -> None:
        """The players of the next lease: their models are loaded now (MissingWeights) and kept until the next
        ``reserve``."""
        self._current = {(p.agent_id, p.network_id): p for p in players}
        self._reserved = {p.blob for p in players if p.kind == "neural"}
        for player in players:
            if player.kind == "neural":
                self._model(player)
        self._trim()

    def get(self, agent_id: str, network_id: str) -> Any:
        player = self._current.get((agent_id, network_id))
        if player is None:
            return None
        if player.kind == "scripted":
            key = (player.agent_id, player.bot.class_path, json.dumps(player.bot.kwargs, sort_keys=True))
            if key not in self._bots:
                self._bots[key] = ScriptedPlayer(functools.partial(make_bot, player.bot, self._spec))
            return self._bots[key]
        return self._model(player)

    def _model(self, player: PlayerRef) -> Any:
        model = self._models.get(player.blob)
        if model is not None:
            self._models.move_to_end(player.blob)
            return model
        cached = self._reader.get(player.blob)
        if cached is None:
            raise MissingWeights(f"the weights of {player.name} (blob {player.blob.ident}) are not in this machine's "
                                 f"weight cache")
        networks = player.networks or cached.meta.architecture or {}
        frozen = FrozenSpec(agent_id=player.agent_id, roles=tuple(player.roles), networks=dict(networks),
                            model_state=cached.state_dict, source=f"{player.name} (blob {player.blob.ident})",
                            networks_source=player.name)
        model = build_frozen_model(None, frozen, self._spec)
        self.loads += 1
        self._models[player.blob] = model
        self._trim()
        return model

    def _trim(self) -> None:
        limit = max(self._capacity, len(self._reserved))
        for key in list(self._models):
            if len(self._models) <= limit:
                break
            if key not in self._reserved:
                del self._models[key]


class _Control:
    """The control connection; messages read while a lease plays wait in ``pending``."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self.pending: deque[Any] = deque()
        self.stop = False

    def _read(self, timeout: float) -> None:
        try:
            while self._conn.poll(timeout):
                message = self._conn.recv()
                if isinstance(message, Stop):
                    self.stop = True
                self.pending.append(message)
                timeout = 0
        except (EOFError, OSError):
            self.stop = True                       # the node is gone (the parent-death watch fires as well)

    def next(self, timeout: float) -> Any:
        if not self.pending:
            self._read(timeout)
        return self.pending.popleft() if self.pending else None

    def cancelled(self, lease_id: str) -> bool:
        """Read what arrived (no wait): a stop, or an ``EvalCancel`` of ``lease_id``."""
        self._read(0)
        return self.stop or any(isinstance(m, EvalCancel) and m.lease_id == lease_id for m in self.pending)

    def drop_cancels(self, lease_id: str) -> None:
        self.pending = deque(m for m in self.pending if not (isinstance(m, EvalCancel) and m.lease_id == lease_id))

    def send(self, message: Any) -> None:
        try:
            self._conn.send(message)
        except (EOFError, OSError):
            self.stop = True


def play_task(task: EvalTask, *, pool: CachePlayerPool, control: _Control, env_fn: Callable[[], Any],
              max_idle_steps: int, node: int, index: int) -> None:
    """Play one lease and report it on ``control`` (module docstring)."""
    from colosseum.eval import play_lineups

    lineups = [lu for block in task.blocks for lu in block.lineups]
    context = f"{task.source}, node {node}, evaluator {index}, "
    placement = MatchPlacement(lineups, task.num_envs)
    players = {(p.agent_id, p.network_id, p.blob.ident if p.blob else ""): p
               for block in task.blocks for p in block.players}
    try:
        pool.reserve(list(players.values()))
        played = play_lineups(env_fn=env_fn, pool=pool, lineups=lineups, num_envs=task.num_envs,
                              seed=lineups[0].seed, deterministic=task.deterministic, max_idle_steps=max_idle_steps,
                              observer=placement, source=task.source, context=context, match_id_prefix=task.source,
                              should_stop=lambda: control.cancelled(task.lease_id))
    except MissingWeights as e:
        control.send(EvalFailed(task.lease_id, "missing_weights", str(e), f"lease {task.lease_id}"))
        return
    except (ConfigError, EnvContractError, PlayerError) as e:
        control.send(EvalFailed(task.lease_id, _KINDS[error_kind(e)], str(e),  # type: ignore[arg-type]
                                placement.where(str(e), context, [b.block_id for b in task.blocks])))
        return
    if len(played) < len(lineups):                   # stopped or cancelled before the end
        if not control.stop:
            control.send(EvalDone(task.lease_id, cancelled=True))
        return
    by_block: dict[str, list[MatchResult]] = {}
    for result in played:
        by_block.setdefault(result.block, []).append(result)
    control.send(EvalDone(task.lease_id, {b: sorted(r, key=lambda x: x.uid) for b, r in by_block.items()}))
    logger.info(f"evaluator {index}: lease {task.lease_id} done ({len(task.blocks)} blocks, {len(played)} matches)")


def evaluator_loop(conn: Any, *, spec: GameSpec, reader: Any, env_fn: Callable[[], Any], max_idle_steps: int,
                   node: int, index: int) -> None:
    """The evaluator's main loop over its control connection; returns on ``Stop`` or a closed connection."""
    control = _Control(conn)
    pool = CachePlayerPool(spec, reader)
    while not control.stop:
        message = control.next(POLL_SEC)
        if message is None or isinstance(message, (Stop, EvalCancel)):
            continue                                  # a cancel of a lease that no longer plays: nothing to do
        if isinstance(message, EvalTask):
            play_task(message, pool=pool, control=control, env_fn=env_fn, max_idle_steps=max_idle_steps,
                      node=node, index=index)
            control.drop_cancels(message.lease_id)
        else:
            logger.warning(f"evaluator {index}: unexpected message {type(message).__name__} ignored")


def evaluator_main(*, name: str, index: int, incarnation: int, node: int, node_address: str, authkey: bytes,
                   cache_root: str, cache_prefix: str, config: ColosseumConfig) -> None:
    from colosseum.core.registry import env_spec, make_env
    from colosseum.core.threads import configure_torch_threads
    from colosseum.node.weights import WeightCacheReader

    configure_torch_threads(1)
    spec = env_spec(config)
    conn = connect(node_address, authkey, timeout=CONNECT_TIMEOUT_SEC)
    conn.send(ChildHello(role="evaluator", name=name, incarnation=incarnation, pid=os.getpid(), channel="control"))
    logger.info(f"{name}: node {node}, incarnation {incarnation}, cache {cache_prefix}")
    try:
        evaluator_loop(conn, spec=spec, reader=WeightCacheReader(Path(cache_root), cache_prefix),
                       env_fn=functools.partial(make_env, config), max_idle_steps=config.env.max_idle_steps,
                       node=node, index=index)
    finally:
        try:
            conn.close()
        except OSError:
            pass


def evaluator_target(*, name: str, log_dir: str | None = None, **kwargs: Any) -> None:
    """Spawn target of a node's evaluator child (module docstring)."""
    run_child(name, log_dir, functools.partial(evaluator_main, name=name), **kwargs)
