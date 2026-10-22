"""Evaluation test kit (SP5 T5.1): toy-game evaluation blocks as ``BlockTask``s, the machine weight cache (real
``WeightCacheWriter`` / ``WeightCacheReader`` under the test's ``COLOSSEUM_SHM_DIR``) filled with their players'
weights, a ``FakeBlobs`` registry standing in for ``BlobStore``'s evaluation blobs, and ``EvalNode``: an in-process
node that leases blocks from a ``HubEvaluation`` and plays them on an evaluator thread (``evaluator_loop`` over a
``multiprocessing.Pipe``; no process). ``drive`` is the test's main loop.

Eval blobs are cache keys ``blob_key("eval", agent_id, blob_id)`` (A18, P-PF4-2): ``FakeBlobs.live`` is indexed by
the raw ``blob_id`` (``split_blob_key(key)[1]``), as a ``BlobRequest(kind="eval")`` names it."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from colosseum.arena.tournament.players import PlayerSource, TournamentPlayer, config_player
from colosseum.arena.tournament.schedule import gauntlet_blocks
from colosseum.coordinator.checkpoint_manager import read_weights_file
from colosseum.core.registry import env_spec, make_env
from colosseum.hub.evaluation import HubEvaluation, player_ref
from colosseum.net.chunks import WeightsMeta, encode_weights
from colosseum.net.local import EvalCancel, EvalDone, EvalTask, Stop
from colosseum.net.messages import BlockTask, block_task_from_proto, result_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node.evaluator_main import evaluator_loop
from colosseum.node.leases import NodeLease, NodeLeases
from colosseum.node.weights import (
    CacheKey,
    WeightCacheReader,
    WeightCacheWriter,
    cache_prefix,
    shm_dir,
    split_blob_key,
)
from colosseum.players.registry import load_fixed_players
from game_helpers import agent_role_of, write_ckpt_dir

CA_FINGERPRINT = "ab" * 32
RUN_ID = "0123456789abcdef"          # cache_prefix needs a hex run id (P-PF4-11)


def blob_id_of(key: CacheKey) -> str:
    """The raw blob id of an eval cache key (what ``BlobRequest(kind="eval").ident`` carries)."""
    return split_blob_key(key)[1]


def snapshot_player(config, root: Path, agent: str, version: int, *, seed: int = 0) -> TournamentPlayer:
    """A stored snapshot ``root/ckpt_v<version>`` (``write_ckpt_dir``) as a tournament player."""
    path = write_ckpt_dir(root, config, agent, policy_version=version, seed=seed)
    roles, _role = agent_role_of(config, agent)
    networks = config.get_agent_config(agent).networks.model_dump(mode="json", by_alias=True)
    name = f"{agent}@v{version}"
    return TournamentPlayer(name=name, display=name, kind="snapshot", agent_id=agent, network_id=f"ckpt_v{version}",
                            roles=tuple(roles), lineage=agent, version=version,
                            source=PlayerSource("model", origin=str(path), weights=str(path / "model.pt"),
                                                networks=networks))


def fixed_players(config, names: Sequence[str]) -> dict[str, TournamentPlayer]:
    spec = env_spec(config)
    fixed = load_fixed_players(config, spec, list(names))
    return {n: config_player(config, spec, n, fixed) for n in names}


def gauntlet_tasks(config, candidate: TournamentPlayer, field: Mapping[str, TournamentPlayer], *, blocks: int,
                   owner: str = "tournament:t", priority: int = 0, num_envs: int = 2, seed: int = 0,
                   layout: str = "2p", blobs: Any = None) -> list[BlockTask]:
    """Stage-A gauntlet blocks of ``candidate`` against ``field`` as tasks of ``owner``; neural players get the blob
    id ``<owner name>-<agent>-<network>`` (registered with ``blobs`` when given)."""
    prefix = owner.split(":", 1)[1]
    spec = env_spec(config)
    players = {candidate.name: candidate, **field}
    sched = gauntlet_blocks(spec, layout, [candidate.name], list(field), players, blocks=blocks, seed=seed,
                            stage="A", prefix=prefix)
    refs = {}
    for p in players.values():
        blob_id = f"{prefix}-{p.agent_id}-{p.network_id}" if p.source.kind == "model" else None
        if blob_id is not None and blobs is not None:
            blobs.register_weights(blob_id, Path(p.source.weights), agent_id=p.agent_id,
                                   networks=dict(p.source.networks or {}), roles=p.roles)
        refs[(p.agent_id, p.network_id)] = player_ref(p, blob_id)
    source = "pulse" if owner.startswith("pulse:") else "tournament"
    out = []
    for b in sched:
        seats = sorted({(s.agent_id, s.network_id) for lu in b.lineups for s in lu.seats})
        out.append(BlockTask(b.id, source, owner, tuple(b.lineups), tuple(refs[k] for k in seats), priority, False,
                             num_envs))
    return out


class FakeBlobs:
    """``EvalBlobs`` stand-in: ``live[blob_id] = (weights file, agent id, networks, roles)``; ``log`` of calls."""

    def __init__(self) -> None:
        self.live: dict[str, tuple[Path, str, dict, tuple[str, ...]]] = {}
        self.log: list[tuple[str, str]] = []

    def register_weights(self, blob_id, weights_file, *, agent_id, networks, roles) -> None:
        assert Path(weights_file).is_file(), weights_file
        self.live[blob_id] = (Path(weights_file), agent_id, dict(networks), tuple(roles))
        self.log.append(("register", blob_id))

    def unregister(self, blob_id) -> None:
        self.live.pop(blob_id, None)
        self.log.append(("unregister", blob_id))


class EvalCache:
    """The machine's weight cache of one fake node (prefix of ``node`` and ``run_id``)."""

    def __init__(self, node: int = 1, run_id: str = RUN_ID) -> None:
        self.root, self.prefix = shm_dir(), cache_prefix(CA_FINGERPRINT, run_id, node)
        self.writer = WeightCacheWriter(self.root, self.prefix)
        self.reader = WeightCacheReader(self.root, self.prefix)

    def put(self, key: CacheKey, weights_file: Path, networks: Mapping, roles: Sequence[str]) -> None:
        meta = WeightsMeta(agent_id=key.agent_id, kind="fixed", generation=0, policy_version=0,
                           architecture=dict(networks), roles=tuple(roles))
        self.writer.put(key, encode_weights(read_weights_file(weights_file), meta))

    def put_from(self, blobs: FakeBlobs, key: CacheKey) -> None:
        path, _agent, networks, roles = blobs.live[blob_id_of(key)]
        self.put(key, path, networks, roles)

    def has(self, key: CacheKey) -> bool:
        return self.reader.has(key)


class EvalNode:
    """An in-process node: ``NodeLeases`` + one evaluator thread; ``inbox`` gets the hub's messages (``route``)."""

    def __init__(self, config, blobs: FakeBlobs, evaluation: HubEvaluation, *, number: int = 1,
                 slots: int = 1) -> None:
        self.number, self.blobs, self.ev = number, blobs, evaluation
        self.cache = EvalCache(number)
        self.leases = NodeLeases()
        self.inbox: list[pb.HubMessage] = []
        self.fetches = 0
        self.results_sent = 0
        self._conn, child = mp.Pipe()
        self._thread = threading.Thread(target=evaluator_loop, args=(child,), daemon=True, kwargs=dict(
            spec=env_spec(config), reader=self.cache.reader, env_fn=lambda: make_env(config),
            max_idle_steps=config.env.max_idle_steps, node=number, index=0))
        self._thread.start()
        self.leases.evaluator_up(0)
        self.slots = slots

    def join(self, *, fresh: bool = True) -> int:
        held, resend = self.leases.on_reconnect()
        reply = self.ev.node_joined(self.number, slots=self.slots, held_leases=held, fresh=fresh)
        self.inbox.extend(reply.after_welcome)
        for outcome in resend:
            self._report(outcome)
        return reply.evaluators

    def pump(self) -> None:
        free = self.leases.request(connected=True)
        if free:
            self.ev.on_lease_request(self.number, free)
        while self.inbox:
            msg = self.inbox.pop(0)
            kind = msg.WhichOneof("m")
            if kind == "lease":
                tasks = [block_task_from_proto(b, num_envs=msg.lease.num_envs) for b in msg.lease.blocks]
                lease = NodeLease(msg.lease.lease_id, tasks[0].source, tasks, msg.lease.num_envs,
                                  tasks[0].deterministic)
                for key in self.leases.on_lease(lease, have=self.cache.has):
                    found = blob_id_of(key) in self.blobs.live
                    if found:
                        self.cache.put_from(self.blobs, key)
                        self.fetches += 1
                    for outcome in self.leases.on_fetched(key, found):
                        self._report(outcome)
            elif kind == "block_ack":
                self.leases.on_ack(msg.block_ack.lease_id, list(msg.block_ack.block_ids))
            elif kind == "command" and msg.command.kind == "cancel_leases":
                for _index, lease_id in self.leases.on_cancel(list(msg.command.lease_ids)):
                    self._conn.send(EvalCancel(lease_id))
        for _index, lease in self.leases.assignments():
            self._conn.send(EvalTask(lease.lease_id, lease.source, lease.blocks, lease.num_envs, lease.deterministic))
        while self._conn.poll(0):
            m = self._conn.recv()
            outcomes = (self.leases.on_done(0, m.lease_id, m.results, m.cancelled) if isinstance(m, EvalDone)
                        else self.leases.on_failed(0, m.lease_id, m.kind, f"{m.message} ({m.where})"))
            for outcome in outcomes:
                self._report(outcome)

    def _report(self, outcome) -> None:
        msg = pb.BlockResult(lease_id=outcome.lease_id, block_id=outcome.block_id, failed=outcome.failed,
                             error_kind=outcome.error_kind, error=outcome.error)
        if outcome.results is not None:
            msg.results.extend(result_to_proto(r) for r in outcome.results)
        self.results_sent += 1
        self.ev.on_block_result(self.number, msg)

    def close(self) -> None:
        self._conn.send(Stop("test end"))
        self._thread.join(10)


def route(nodes: Mapping[int, EvalNode]) -> Callable[[int, pb.HubMessage], None]:
    """A ``HubEvaluation.send`` that puts messages in the nodes' inboxes (fill ``nodes`` after building the
    evaluation)."""
    return lambda number, msg: nodes[number].inbox.append(msg) if number in nodes else None


def drive(evaluation: HubEvaluation, nodes: Sequence[EvalNode], until: Callable[[], bool], *,
          inbox: Any = None, on_event: Callable[[Any], None] | None = None, timeout: float = 120.0) -> None:
    """The test's main loop: inbox events, ``evaluation.tick()``, every node's pump, until ``until()``."""
    deadline = time.monotonic() + timeout
    while not until():
        if time.monotonic() > deadline:
            raise AssertionError(f"drive: not done after {timeout} s ({evaluation.stats()})")
        while inbox is not None and not inbox.empty():
            on_event(inbox.get_nowait())
        evaluation.tick()
        for node in nodes:
            node.pump()
        time.sleep(0.005)
