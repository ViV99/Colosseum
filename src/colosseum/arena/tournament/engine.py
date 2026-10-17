"""Tournament engine (SP4 spec block 3): whole blocks played in local worker processes on ``MatchRunner``.

``play_blocks`` checks the blocks (``check_blocks``: every lineup its own uid, the block's seed and id, every seat
a known player), splits them into tasks of whole blocks (``make_tasks``: at least ``TASK_MATCHES * num_envs``
matches each, so env batches are rarely half empty), starts ``min(num_workers, tasks)`` spawn processes and feeds
them the tasks through a queue. A worker (``run_child``: log ``<log_dir>/tournament-worker-<i>.log``, SIGINT
ignored, dies with the parent) plays a task with ``eval.play_lineups`` on a ``LazyPlayerPool`` (``source`` =
``tournament``; torch sampling seeded with the task's first block seed) and sends the results back grouped by
block. The main process calls ``on_block(block_id, results)`` for every block of a task once the whole task is
back (tasks in the order they finish, blocks of a task in schedule order, results sorted by uid): a block is never
reported in part. A worker's ConfigError, EnvContractError or PlayerError (or a subclass: the nearest of the
three) is re-raised in the main process with its message (context ``tournament, worker W, env E, ...``) plus the
worker, the match uid and block and the worker's log; a worker that dies otherwise raises ``TournamentError``
naming its log. On every exit (also a KeyboardInterrupt in the main process) the workers are stopped:
``stop_event``, ``STOP_GRACE_SEC``, then terminate / kill.

Results carry ``source``, the lineup's ``seed``, ``block`` and ``uid`` (T1.1); a seat's ``version_start`` /
``version_end`` is ``N`` for ``ckpt_v<N>`` and None for fixed players; ``env_steps`` is None.

``LazyPlayerPool`` is a ``PlayerPool`` keyed by ``(agent_id, network_id)``: scripted players are
``ScriptedPlayer`` factories; neural players are built on first use from their weights file
(``build_frozen_model``: eval mode) and kept in an LRU cache of ``MODEL_CACHE_SIZE`` models; ``reserve`` loads the
models of one task and protects them until the next ``reserve`` (no model is reloaded inside a task).
"""

from __future__ import annotations

import functools
import logging
import multiprocessing as mp
import queue
import re
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from colosseum.arena.tournament.players import TournamentPlayer
from colosseum.arena.tournament.schedule import Block
from colosseum.coordinator.checkpoint_manager import read_weights_file
from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError, EnvContractError, PlayerError
from colosseum.core.types import Lineup, MatchResult
from colosseum.envs.game import GameSpec
from colosseum.networks.model import PolicyModel
from colosseum.players.registry import FrozenSpec, build_frozen_model, make_bot
from colosseum.utils.process import ProcessSupervisor, run_child, start_process
from colosseum.worker.match_runner import ScriptedPlayer

logger = logging.getLogger(__name__)

MODEL_CACHE_SIZE = 16      # neural players kept per worker (LRU); a task's own players are never evicted
TASK_MATCHES = 4           # a task holds whole blocks with >= TASK_MATCHES * num_envs matches (or the rest)
STOP_GRACE_SEC = 2.0
_ERRORS: dict[str, type[Exception]] = {"ConfigError": ConfigError, "EnvContractError": EnvContractError,
                                       "PlayerError": PlayerError}


class TournamentError(RuntimeError):
    """A tournament worker died or broke the protocol (the message names its log)."""


def build_player_model(player: TournamentPlayer, spec: GameSpec) -> PolicyModel:
    """The neural player's model with its weights, in eval mode; ConfigError naming its origin."""
    try:
        state = read_weights_file(player.source.weights)
    except ValueError as e:
        raise ConfigError(f"{player.source.origin}: {e}") from e
    frozen = FrozenSpec(agent_id=player.agent_id, roles=tuple(player.roles), networks=dict(player.source.networks),
                        model_state=state, source=player.source.origin, networks_source=player.source.origin)
    return build_frozen_model(None, frozen, spec)


class LazyPlayerPool:
    """``PlayerPool`` of tournament players by ``(agent_id, network_id)`` (module docstring)."""

    def __init__(self, players: Iterable[TournamentPlayer], spec: GameSpec, capacity: int = MODEL_CACHE_SIZE) -> None:
        self._spec = spec
        self._players = {(p.agent_id, p.network_id): p for p in players}
        self._capacity = int(capacity)
        self._models: OrderedDict[tuple[str, str], PolicyModel] = OrderedDict()
        self._bots: dict[tuple[str, str], ScriptedPlayer] = {}
        self._reserved: set[tuple[str, str]] = set()
        self.loads = 0

    def reserve(self, keys: Iterable[tuple[str, str]]) -> None:
        keys = set(keys)
        unknown = sorted(k for k in keys if k not in self._players)
        if unknown:
            raise ValueError(f"LazyPlayerPool: unknown players {unknown}")
        self._reserved = {k for k in keys if self._players[k].source.kind == "model"}
        for key in sorted(self._reserved):
            self._model(key)
        self._trim()

    def get(self, agent_id: str, network_id: str) -> PolicyModel | ScriptedPlayer | None:
        key = (agent_id, network_id)
        player = self._players.get(key)
        if player is None:
            return None
        if player.source.kind == "bot":
            if key not in self._bots:
                self._bots[key] = ScriptedPlayer(functools.partial(make_bot, player.source.bot, self._spec))
            return self._bots[key]
        return self._model(key)

    def cached(self) -> list[tuple[str, str]]:
        return list(self._models)

    def _model(self, key: tuple[str, str]) -> PolicyModel:
        model = self._models.get(key)
        if model is not None:
            self._models.move_to_end(key)
            return model
        model = build_player_model(self._players[key], self._spec)
        self.loads += 1
        self._models[key] = model
        self._trim()
        return model

    def _trim(self) -> None:
        limit = max(self._capacity, len(self._reserved))
        for key in list(self._models):
            if len(self._models) <= limit:
                break
            if key not in self._reserved:
                del self._models[key]


def check_blocks(blocks: Sequence[Block], *, players: Iterable[TournamentPlayer] | None = None) -> None:
    """Every lineup of a block carries the block's seed and id and its own uid, unique over ``blocks``: one lineup is
    one episode (``play_lineups`` plays each once), and the journal treats a repeated uid as a duplicate, so a
    schedule must stage a new lineup for every episode (ValueError otherwise). With ``players``, every seat's
    ``(agent_id, network_id)`` must be one of theirs (TournamentError: a schedule bug, caught before any worker
    starts)."""
    seen: set[str] = set()
    known = None if players is None else {(p.agent_id, p.network_id) for p in players}
    for block in blocks:
        if len(block.lineups) != len(block.rotations):
            raise ValueError(f"block {block.id}: {len(block.lineups)} lineups for {len(block.rotations)} rotations")
        for lineup in block.lineups:
            if lineup.block != block.id or lineup.seed != block.seed or not lineup.uid:
                raise ValueError(f"block {block.id}: lineup {lineup.uid!r} needs block={block.id!r}, "
                                 f"seed={block.seed} and a uid (got block={lineup.block!r}, seed={lineup.seed})")
            if lineup.uid in seen:
                raise ValueError(f"block {block.id}: match uid {lineup.uid!r} is scheduled twice; every episode "
                                 f"needs a lineup of its own")
            seen.add(lineup.uid)
            unknown = [] if known is None else \
                sorted({(s.agent_id, s.network_id) for s in lineup.seats} - known)
            if unknown:
                raise TournamentError(f"match {lineup.uid}: seats {unknown} are not tournament players "
                                      f"({sorted(known)})")


def make_tasks(blocks: Sequence[Block], num_envs: int) -> list[list[Block]]:
    """Consecutive whole blocks, each task with at least ``TASK_MATCHES * num_envs`` matches (the last: the rest)."""
    target = max(1, TASK_MATCHES * int(num_envs))
    tasks: list[list[Block]] = []
    current: list[Block] = []
    matches = 0
    for block in blocks:
        current.append(block)
        matches += len(block.lineups)
        if matches >= target:
            tasks.append(current)
            current, matches = [], 0
    if current:
        tasks.append(current)
    return tasks


def _put(q: Any, item: Any, stop_event: Any) -> bool:
    """Blocking put that gives up when ``stop_event`` is set."""
    while not stop_event.is_set():
        try:
            q.put(item, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def _worker_target(*, worker_id: int, log_dir: str | None = None, **kwargs: Any) -> None:
    run_child(f"tournament-worker-{worker_id}", log_dir, _worker_main, worker_id=worker_id, **kwargs)


class _Placement:
    """``MatchObserver`` that knows which lineup every env plays (the match an error happened in)."""

    def __init__(self, lineups: Sequence[Lineup], num_envs: int) -> None:
        self.current = dict(enumerate(lineups[:min(num_envs, len(lineups))]))

    def on_act(self, env, seat, record) -> None:
        pass

    def on_rewards(self, env, rewards) -> None:
        pass

    def on_terminated(self, env, seats) -> None:
        pass

    def on_lineup_applied(self, env, old, new) -> None:
        self.current[env] = new

    def on_episode_end(self, env, end) -> None:
        pass

    def where(self, message: str, context: str, blocks: Sequence[Block]) -> str:
        """``match <uid> of block <id>`` for an error message that starts with ``<context>env <E>``; else the
        task's blocks."""
        found = re.match(re.escape(context) + r"env (\d+)", message)
        lineup = self.current.get(int(found.group(1))) if found is not None else None
        if lineup is not None:
            return f"match {lineup.uid} of block {lineup.block}"
        ids = [b.id for b in blocks]
        return f"one of blocks {ids[0]} .. {ids[-1]}" if len(ids) > 1 else f"block {ids[0]}"


def _error_kind(error: Exception) -> str:
    """The name of the nearest class of ``_ERRORS`` in the error's MRO (a ``DataError`` is a ``ConfigError``)."""
    return next(cls.__name__ for cls in type(error).__mro__ if _ERRORS.get(cls.__name__) is cls)


def _worker_main(*, worker_id: int, config: ColosseumConfig, players: list[TournamentPlayer], tasks: Any, results: Any,
                 stop_event: Any, num_envs: int, deterministic: bool, source: str) -> None:
    from colosseum.core.registry import env_spec, make_env
    from colosseum.core.threads import configure_torch_threads
    from colosseum.eval import play_lineups

    configure_torch_threads(1)
    spec = env_spec(config)
    pool = LazyPlayerPool(players, spec)
    context = f"{source}, worker {worker_id}, "
    while not stop_event.is_set():
        try:
            item = tasks.get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:
            return
        index, blocks = item
        lineups = [lu for b in blocks for lu in b.lineups]
        placement = _Placement(lineups, num_envs)
        try:
            pool.reserve({(s.agent_id, s.network_id) for lu in lineups for s in lu.seats})
            played = play_lineups(env_fn=lambda: make_env(config), pool=pool, lineups=lineups, num_envs=num_envs,
                                  seed=blocks[0].seed, deterministic=deterministic,
                                  max_idle_steps=config.env.max_idle_steps, observer=placement, source=source,
                                  context=context, match_id_prefix=source)
        except (ConfigError, EnvContractError, PlayerError) as e:
            _put(results, ("error", index, worker_id, _error_kind(e), str(e),
                           placement.where(str(e), context, blocks)), stop_event)
            raise
        by_block: dict[str, list[MatchResult]] = {}
        for result in played:
            by_block.setdefault(result.block, []).append(result)
        logger.info(f"tournament worker {worker_id}: task {index} done ({len(played)} matches)")
        if not _put(results, ("done", index, by_block), stop_event):
            return


def _typed_error(msg: tuple, supervisor: ProcessSupervisor) -> Exception:
    """The main-process error of a worker's ``("error", task, worker, kind, message, where)`` message."""
    _kind, _index, worker, kind, message, where = msg
    log = supervisor.log_path(f"tournament-worker-{worker}")
    return _ERRORS[kind](f"{message}\n(tournament-worker-{worker}, {where}"
                         f"{'' if log is None else f'; log {log}'})")


def _error_after_failure(results: Any, failure: Any, supervisor: ProcessSupervisor) -> Exception:
    """The error a dead worker sent before dying (a typed one), else a TournamentError naming its log."""
    try:
        while True:
            msg = results.get(timeout=1.0)
            if msg[0] == "error":
                return _typed_error(msg, supervisor)
    except queue.Empty:
        return TournamentError(f"tournament {failure.message()}")


def _drain(results: Any) -> list[tuple]:
    """Every message still in the results queue (workers that exited flushed theirs before exiting)."""
    out = []
    try:
        while True:
            out.append(results.get(timeout=0.2))
    except queue.Empty:
        return out


def play_blocks(config: ColosseumConfig, spec: GameSpec, players: Mapping[str, TournamentPlayer] | Sequence,
                blocks: Sequence[Block], *, num_workers: int, num_envs: int, deterministic: bool,
                on_block: Callable[[str, list[MatchResult]], None], log_dir: str | Path | None = None,
                source: str = "tournament") -> None:
    """Play ``blocks`` (module docstring)."""
    blocks = list(blocks)
    if not blocks:
        return
    if num_workers < 1 or num_envs < 1:
        raise ValueError(f"play_blocks: need num_workers >= 1 and num_envs >= 1, got {num_workers}, {num_envs}")
    player_list = list(players.values()) if isinstance(players, Mapping) else list(players)
    check_blocks(blocks, players=player_list)
    tasks = make_tasks(blocks, num_envs)
    ctx = mp.get_context("spawn")
    task_q, result_q, stop = ctx.Queue(), ctx.Queue(), ctx.Event()
    supervisor = ProcessSupervisor(stop, log_dir=log_dir)
    pending = set(range(len(tasks)))

    def handle(msg: tuple) -> None:
        if msg[0] == "error":
            raise _typed_error(msg, supervisor)
        _kind, index, by_block = msg
        for block in tasks[index]:
            got = sorted(by_block.get(block.id, []), key=lambda r: r.uid)
            if len(got) != len(block.lineups):
                raise TournamentError(f"block {block.id}: {len(got)} results for {len(block.lineups)} matches")
            on_block(block.id, got)
        pending.discard(index)

    try:
        for index, task in enumerate(tasks):
            task_q.put((index, task))
        workers = min(num_workers, len(tasks))
        for _ in range(workers):
            task_q.put(None)
        for w in range(workers):
            proc = ctx.Process(target=_worker_target, name=f"tournament-worker-{w}", daemon=True, kwargs=dict(
                worker_id=w, log_dir=None if log_dir is None else str(log_dir), config=config, players=player_list,
                tasks=task_q, results=result_q, stop_event=stop, num_envs=num_envs, deterministic=deterministic,
                source=source))
            start_process(proc)
            supervisor.add(f"tournament-worker-{w}", proc)
        while pending:
            try:
                msg = result_q.get(timeout=0.2)
            except queue.Empty:
                failure = supervisor.first_failure(nonzero_only=True)
                if failure is not None:
                    raise _error_after_failure(result_q, failure, supervisor) from None
                if not supervisor.alive():
                    for late in _drain(result_q):      # results sent just before the workers exited
                        handle(late)
                    if pending:
                        where = "" if log_dir is None else f"; see the tournament-worker-*.log files in {log_dir}"
                        raise TournamentError(f"tournament workers exited with {len(pending)} tasks unplayed"
                                              f"{where}") from None
                continue
            handle(msg)
    finally:
        stop.set()
        supervisor.wait_all(STOP_GRACE_SEC)
        supervisor.kill_remaining()
        for q in (task_q, result_q):
            q.cancel_join_thread()
            q.close()
