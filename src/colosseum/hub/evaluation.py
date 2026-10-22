"""Evaluation on nodes (SP5 spec 5.3): block leases, the evaluation service of the hub, the pulse on blocks (T5.1),
hub-run tournaments (T5.2).

- ``BlockQueue`` (pure, the hub's monotonic clock passed in): every evaluation block from submission to its
  result (class docstring).
- ``check_block_results``: a block counts only whole, and only with results that are exactly its matches (node data
  is untrusted: the layout's roles and teams, the lineup's players and their version labels, finite numbers).
- ``HubEvaluation`` (main loop only; no gRPC, no threads): the queue, the owners of blocks (the pulse, hub-run
  tournaments), how many evaluators each node runs, and the lease protocol on the sessions:
  ``BlockLeaseRequest(free_slots)`` (an increment: slots the node has not asked for yet) -> ``BlockLease`` (one
  lease per grant, round robin over the asking nodes); ``BlockResult`` per block -> ``BlockAck`` always (the node
  forgets it; a result is never resent once acked); ``Hello.leases`` on a resumed session: leases the node no
  longer holds go back to the queue at once, leases the hub does not know are cancelled (``cancel_leases``); a new
  node process (fresh session) or node loss: its leases go back without using an attempt; ``set_evaluators``: a
  node runs its evaluator slots while any evaluation work can exist (pulse on, or an owner registered), else 0.
- Results of a pulse block go to the pulse (``HubPulse``, T5.1 (b)); when its snapshot is complete, to the journal
  writer only (SP4 amendment A20). Results of other owners go to their sink (T5.2). Every result is stamped with the
  node number of the evaluator that played it. A block that fails with a typed error (``FATAL_ERRORS``) fails its
  owner (the pulse: the run stops with exit 1, as an SP4 pulse crash did; a tournament: it fails); a block that
  fails ``MAX_BLOCK_ATTEMPTS`` times fails its owner too (the pulse: the snapshot is skipped with a WARNING).
"""

from __future__ import annotations

import heapq
import itertools
import logging
import math
import time
import uuid
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from colosseum.arena.entities import snapshot_version
from colosseum.arena.tournament.engine import TASK_MATCHES
from colosseum.arena.tournament.players import TournamentPlayer
from colosseum.core.types import MatchResult
from colosseum.envs.game import GameSpec
from colosseum.net.messages import (
    PULSE_PRIORITY,
    TOURNAMENT_PRIORITY,
    BlockTask,
    PlayerRef,
    lease_to_proto,
    result_from_proto,
)
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node.weights import blob_key

__all__ = ["FATAL_ERRORS", "MAX_BLOCK_ATTEMPTS", "PULSE_PREFIX", "PULSE_PRIORITY", "RETRY_ERRORS", "TOURNAMENT_PREFIX",
           "TOURNAMENT_PRIORITY", "BlockQueue", "BlockTask", "EvalBlobs", "HubEvaluation", "JoinReply", "Lease",
           "PlayerRef", "check_block_results", "player_ref", "pulse_owner"]

logger = logging.getLogger(__name__)

MAX_BLOCK_ATTEMPTS = 3          # leases of one block that may end without a result (expiry, crash, bad data)
FINISHED_MEMORY = 100_000       # finished block ids remembered to tell "late" / "cancelled" from "unknown"
COUNTER_KEYS = ("submitted", "leases", "leased", "completed", "late", "cancelled", "unknown", "expired", "requeued",
                "failed", "cancelled_blocks")   # BlockQueue.stats(): every key always present (stable system record)
PULSE_PREFIX = "pulse:"
TOURNAMENT_PREFIX = "tournament:"
RETRY_ERRORS = ("abandoned", "crash", "missing_weights")     # BlockResult.error_kind values that re-lease the block
FATAL_ERRORS = ("contract", "player", "config")               # the code is wrong: replaying cannot help


def pulse_owner(agent_id: str, checkpoint_id: str) -> str:
    return f"{PULSE_PREFIX}{agent_id}:{checkpoint_id}"


@dataclass(frozen=True)
class Lease:
    lease_id: str
    node: int
    block_ids: tuple[str, ...]
    deadline: float                 # hub monotonic clock


@dataclass
class _Entry:
    task: BlockTask
    seq: int
    state: Literal["queued", "leased", "done"] = "queued"
    lease: str | None = None
    attempts: int = 0
    issued: set[str] = field(default_factory=set)


@dataclass
class _LeaseState:
    lease: Lease
    owner: str
    remaining: set[str]


class BlockQueue:
    """Evaluation blocks waiting for, or leased to, node evaluators (spec 5.3). Pure.

    - ``submit``: blocks queue by ``(priority, submission order)``: tournaments (0) before pulse (1), FIFO within.
    - ``lease(node, slots)``: up to ``slots`` leases; a lease is consecutive queued blocks of ONE owner with at
      least ``task_matches * num_envs`` matches (SP4 ``make_tasks``), so an evaluator's envs are rarely half empty.
    - ``want`` / ``dispatch``: nodes ask for slots (an increment); ``dispatch`` grants one lease per waiting node in
      turn (round robin by node number, continuing after the last node served), so blocks spread over machines.
    - ``complete``: the first complete result of any lease that held the block wins ("accepted"; also after that
      lease expired), later ones are "late"; a cancelled block's are "cancelled"; a block (or lease) never issued
      "unknown". Every outcome is counted.
    - ``release`` / ``expire`` / ``drop_node`` / ``keep_only``: a block whose lease ended without a result goes back
      to its place in the queue; ``expire`` and counted releases (an evaluator crash, missing weights, invalid
      results) use one of ``max_attempts`` attempts, after which the block fails (``take_failed``); node loss and a
      node's lost leases do not count (the network, not the block).
    - ``cancel(owner)``: the owner's queued and leased blocks are gone; returns the block ids and the leases per
      node to tell the nodes about.
    """

    def __init__(self, *, task_matches: int = TASK_MATCHES, max_attempts: int = MAX_BLOCK_ATTEMPTS,
                 new_id: Callable[[], str] | None = None) -> None:
        self._task_matches = int(task_matches)
        self._max_attempts = int(max_attempts)
        self._new_id = new_id or (lambda: uuid.uuid4().hex)
        self._entries: dict[str, _Entry] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._leases: dict[str, _LeaseState] = {}
        self._finished: OrderedDict[str, str] = OrderedDict()
        self._failed: list[tuple[str, str, str]] = []
        self._waiting: dict[int, int] = {}
        self._cursor = -1
        self._seq = itertools.count()
        self.counters: Counter[str] = Counter()

    # --- submission ---
    def submit(self, tasks: Sequence[BlockTask]) -> None:
        tasks = list(tasks)
        seen: set[str] = set()
        for task in tasks:
            if task.block_id in self._entries or task.block_id in seen:
                raise ValueError(f"block {task.block_id} is already queued or leased")
            if not task.lineups:
                raise ValueError(f"block {task.block_id} has no lineups")
            seen.add(task.block_id)
        for task in tasks:
            self._finished.pop(task.block_id, None)          # a cancelled block queued again (a resumed tournament)
            entry = _Entry(task, next(self._seq))
            self._entries[task.block_id] = entry
            heapq.heappush(self._heap, (task.priority, entry.seq, task.block_id))
        self.counters["submitted"] += len(tasks)

    def _peek(self) -> _Entry | None:
        while self._heap:
            _priority, seq, block_id = self._heap[0]
            entry = self._entries.get(block_id)
            if entry is not None and entry.state == "queued" and entry.seq == seq:
                return entry
            heapq.heappop(self._heap)                          # stale: leased, done, cancelled or queued again
        return None

    # --- leases ---
    def lease(self, node: int, slots: int, now: float, lease_sec: float) -> list[tuple[Lease, list[BlockTask]]]:
        out: list[tuple[Lease, list[BlockTask]]] = []
        for _ in range(max(0, int(slots))):
            head = self._peek()
            if head is None:
                break
            owner = head.task.owner
            target = max(1, self._task_matches * int(head.task.num_envs))
            taken: list[_Entry] = []
            matches = 0
            while matches < target:
                entry = self._peek()
                if entry is None or entry.task.owner != owner:
                    break
                heapq.heappop(self._heap)
                taken.append(entry)
                matches += entry.task.matches
            lease = Lease(self._new_id(), int(node), tuple(e.task.block_id for e in taken),
                          float(now) + float(lease_sec))
            for entry in taken:
                entry.state, entry.lease = "leased", lease.lease_id
                entry.issued.add(lease.lease_id)
            self._leases[lease.lease_id] = _LeaseState(lease, owner, set(lease.block_ids))
            self.counters["leases"] += 1
            self.counters["leased"] += len(taken)
            out.append((lease, [e.task for e in taken]))
        return out

    def want(self, node: int, slots: int) -> None:
        if int(slots) > 0:
            self._waiting[int(node)] = self._waiting.get(int(node), 0) + int(slots)

    def unwant(self, node: int) -> None:
        self._waiting.pop(int(node), None)

    def waiting(self) -> dict[int, int]:
        return dict(self._waiting)

    def dispatch(self, now: float, lease_sec: float) -> list[tuple[int, Lease, list[BlockTask]]]:
        out: list[tuple[int, Lease, list[BlockTask]]] = []
        while self._waiting and self._peek() is not None:
            nodes = sorted(self._waiting)
            node = next((n for n in nodes if n > self._cursor), nodes[0])
            self._cursor = node
            (lease, tasks), = self.lease(node, 1, now, lease_sec)
            out.append((node, lease, tasks))
            self._waiting[node] -= 1
            if self._waiting[node] <= 0:
                del self._waiting[node]
        return out

    # --- outcomes ---
    def task(self, block_id: str) -> BlockTask | None:
        """The block's task while it is queued or leased (None: done, cancelled, failed or unknown)."""
        entry = self._entries.get(block_id)
        return entry.task if entry is not None and entry.state != "done" else None

    def complete(self, lease_id: str, block_id: str, results: Sequence[MatchResult],
                 now: float) -> Literal["accepted", "late", "cancelled", "unknown"]:
        entry = self._entries.get(block_id)
        outcome: Literal["accepted", "late", "cancelled", "unknown"]
        if entry is None:
            outcome = self._finished.get(block_id, "unknown")  # type: ignore[assignment]
        elif lease_id not in entry.issued:
            outcome = "unknown"
        elif entry.state == "done":
            outcome = "late"
        else:
            if entry.state == "leased":
                self._drop_from_lease(entry.lease, block_id)
            self._drop_from_lease(lease_id, block_id)
            entry.state, entry.lease = "done", None
            outcome = "accepted"
        self.counters["completed" if outcome == "accepted" else outcome] += 1
        return outcome

    def _drop_from_lease(self, lease_id: str | None, block_id: str) -> None:
        state = self._leases.get(lease_id) if lease_id is not None else None
        if state is not None:
            state.remaining.discard(block_id)
            if not state.remaining:
                del self._leases[lease_id]

    def release(self, lease_id: str, block_id: str, *, count: bool = True,
                reason: str = "released") -> Literal["requeued", "failed", "ignored"]:
        """The block's current lease ``lease_id`` ended without a result: back to the queue (``count``: one attempt
        used); "ignored" when ``lease_id`` does not hold the block (an old lease's late report)."""
        entry = self._entries.get(block_id)
        if entry is None or entry.state != "leased" or entry.lease != lease_id:
            return "ignored"
        self._drop_from_lease(lease_id, block_id)
        return self._requeue(entry, count=count, reason=reason)

    def _requeue(self, entry: _Entry, *, count: bool, reason: str) -> Literal["requeued", "failed"]:
        if count:
            entry.attempts += 1
        if entry.attempts >= self._max_attempts:
            del self._entries[entry.task.block_id]
            self._remember(entry.task.block_id, "late")
            self._failed.append((entry.task.block_id, entry.task.owner, reason))
            self.counters["failed"] += 1
            return "failed"
        entry.state, entry.lease = "queued", None
        heapq.heappush(self._heap, (entry.task.priority, entry.seq, entry.task.block_id))
        self.counters["requeued"] += 1
        return "requeued"

    def _release_leases(self, lease_ids: Sequence[str], *, count: bool, reason: str) -> list[str]:
        out: list[str] = []
        for lease_id in lease_ids:
            state = self._leases.pop(lease_id)
            for block_id in sorted(state.remaining):
                entry = self._entries.get(block_id)
                if entry is not None and entry.state == "leased" and entry.lease == lease_id and \
                        self._requeue(entry, count=count, reason=reason) == "requeued":
                    out.append(block_id)
        return out

    def expire(self, now: float) -> list[str]:
        """Leases past their deadline: their blocks go back to the queue (one attempt each); returns their ids."""
        due = [lid for lid, s in self._leases.items() if s.lease.deadline <= now]
        self.counters["expired"] += len(due)
        return self._release_leases(due, count=True, reason="its lease expired")

    def drop_node(self, node: int) -> list[str]:
        """The node is lost (or its process restarted): its leases go back without using an attempt."""
        self.unwant(node)
        return self._release_leases([lid for lid, s in self._leases.items() if s.lease.node == node], count=False,
                                    reason="node lost")

    def keep_only(self, node: int, lease_ids: Sequence[str]) -> list[str]:
        """A node's session resumed: the leases it no longer holds (``Hello.leases``) go back to the queue."""
        keep = set(lease_ids)
        return self._release_leases([lid for lid, s in self._leases.items()
                                     if s.lease.node == node and lid not in keep],
                                    count=False, reason="the node no longer holds the lease")

    def cancel(self, owner: str) -> tuple[list[str], dict[int, list[str]]]:
        blocks = sorted(bid for bid, e in self._entries.items() if e.task.owner == owner and e.state != "done")
        leases: dict[int, list[str]] = {}
        for lease_id, state in list(self._leases.items()):
            if state.owner == owner:
                del self._leases[lease_id]
                leases.setdefault(state.lease.node, []).append(lease_id)
        for block_id in blocks:
            del self._entries[block_id]
            self._remember(block_id, "cancelled")
        self.forget(owner)
        self.counters["cancelled_blocks"] += len(blocks)
        return blocks, leases

    def forget(self, owner: str) -> None:
        """Done blocks of a finished owner: remembered by id only (their later results are "late")."""
        for block_id in [b for b, e in self._entries.items() if e.task.owner == owner and e.state == "done"]:
            del self._entries[block_id]
            self._remember(block_id, "late")

    def _remember(self, block_id: str, what: str) -> None:
        self._finished[block_id] = what
        self._finished.move_to_end(block_id)
        while len(self._finished) > FINISHED_MEMORY:
            self._finished.popitem(last=False)

    def take_failed(self) -> list[tuple[str, str, str]]:
        """``(block id, owner, reason)`` of blocks that used up their attempts since the last call."""
        out, self._failed = self._failed, []
        return out

    # --- views ---
    def leases_of(self, node: int) -> list[str]:
        return sorted(lid for lid, s in self._leases.items() if s.lease.node == node)

    def queued(self, owner_prefix: str = "") -> int:
        return sum(1 for e in self._entries.values() if e.state == "queued" and e.task.owner.startswith(owner_prefix))

    def live(self, owner: str) -> int:
        return sum(1 for e in self._entries.values() if e.task.owner == owner and e.state != "done")

    def stats(self) -> dict[str, int]:
        states = Counter(e.state for e in self._entries.values())
        return {"queued": states["queued"], "leased_now": states["leased"], "leases_now": len(self._leases),
                **{k: int(self.counters[k]) for k in COUNTER_KEYS}}


def check_block_results(task: BlockTask, results: Sequence[MatchResult], *, spec: GameSpec) -> str | None:
    """None when ``results`` are exactly the block's matches: one per lineup (by uid, each once), with the block's
    id, the lineup's layout and seed, the block's source, the layout's outcome kind, one seat result per seat in
    seat order naming the lineup's player with the layout's role and team, version labels ``N`` / ``N`` for a
    ``ckpt_v<N>`` seat and none for a fixed one, the layout's teams, and finite numbers (P-PF4-14). Else a short
    reason (the block is played again). Never raises (node data is untrusted)."""
    try:
        return _check_block_results(task, results, spec)
    except Exception as e:  # noqa: BLE001 - malformed node data of any shape is a reason, never an error
        return f"malformed results ({type(e).__name__}: {e})"


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check_block_results(task: BlockTask, results: Sequence[MatchResult], spec: GameSpec) -> str | None:
    lineups = {lu.uid: lu for lu in task.lineups}
    if len(results) != len(lineups):
        return f"{len(results)} results for {len(lineups)} matches"
    seen: set[str] = set()
    for r in results:
        lineup = lineups.get(r.uid)
        if lineup is None or r.uid in seen:
            return f"match uid {r.uid!r} is not one of the block's or repeats"
        seen.add(r.uid)
        if r.block != task.block_id or r.layout != lineup.layout or r.seed != lineup.seed or r.source != task.source:
            return f"match {r.uid}: block, layout, seed or source differ from the lineup"
        seat_specs = spec.layouts.get(lineup.layout)
        if seat_specs is None:
            return f"match {r.uid}: layout {lineup.layout!r} is not a layout of the game"
        expected_kind = spec.outcome_kind(lineup.layout)
        if r.outcome_kind != expected_kind:
            return f"match {r.uid}: outcome kind {r.outcome_kind!r}, the layout's is {expected_kind!r}"
        if not isinstance(r.episode_length, int) or r.episode_length < 0:
            return f"match {r.uid}: malformed episode length {r.episode_length!r}"
        if [s.seat for s in r.seats] != list(range(len(seat_specs))) or len(lineup.seats) != len(seat_specs):
            return f"match {r.uid}: seats {[s.seat for s in r.seats]} for {len(seat_specs)} seats"
        for s, seat_spec, seat in zip(r.seats, seat_specs, lineup.seats, strict=True):
            if (s.agent_id, s.network_id) != (seat.agent_id, seat.network_id):
                return f"match {r.uid}: seat {s.seat} is {s.agent_id}/{s.network_id}, not the lineup's player"
            if (s.role, s.team) != (seat_spec.role, seat_spec.team):
                return (f"match {r.uid}: seat {s.seat} has role {s.role!r} and team {s.team}, the layout's seat has "
                        f"role {seat_spec.role!r} and team {seat_spec.team}")
            version = snapshot_version(seat.network_id)
            if s.version_start != version or s.version_end != version:
                return (f"match {r.uid}: seat {s.seat} has version labels {s.version_start}/{s.version_end}, "
                        f"expected {version}/{version}")
            if s.eliminated_step is not None and (not isinstance(s.eliminated_step, int) or s.eliminated_step < 0):
                return f"match {r.uid}: seat {s.seat} has a malformed elimination step"
        teams = sorted({s.team for s in seat_specs})
        if sorted(t.team for t in r.teams) != teams:
            return f"match {r.uid}: teams {sorted(t.team for t in r.teams)}, the layout's teams are {teams}"
        numbers = [s.reward for s in r.seats] + [x for t in r.teams for x in (t.rank, t.score)]
        if not all(_finite(x) for x in numbers):
            return f"match {r.uid}: non-finite rewards, ranks or scores"
    return None


class EvalBlobs(Protocol):
    """``BlobStore``'s evaluation blobs (T5.1 (b)): served by ``Download(kind="eval", ident=blob_id)``."""

    def register_weights(self, blob_id: str, weights_file: Path, *, agent_id: str, networks: Mapping[str, Any],
                         roles: Sequence[str]) -> None: ...

    def unregister(self, blob_id: str) -> None: ...


def player_ref(player: TournamentPlayer, blob_id: str | None) -> PlayerRef:
    """The evaluator's view of a tournament / pulse player (a neural one needs its registered ``blob_id``; its cache
    key is ``blob_key("eval", agent_id, blob_id)``, A18)."""
    if player.source.kind == "bot":
        return PlayerRef(player.name, player.agent_id, player.network_id, tuple(player.roles), "scripted",
                         bot=player.source.bot)
    if not blob_id:
        raise ValueError(f"player {player.name!r}: a neural player needs a blob id")
    return PlayerRef(player.name, player.agent_id, player.network_id, tuple(player.roles), "neural",
                     blob=blob_key("eval", player.agent_id, blob_id), networks=dict(player.source.networks or {}))


@dataclass(frozen=True)
class JoinReply:
    evaluators: int                               # Welcome.evaluators
    after_welcome: list                           # HubMessages to send right after the Welcome (cancel_leases)


def cancel_leases_message(lease_ids: Sequence[str]) -> pb.HubMessage:
    return pb.HubMessage(command=pb.Command(kind="cancel_leases", lease_ids=list(lease_ids)))


class HubEvaluation:
    """The hub's evaluation service (module docstring). ``send(node, HubMessage)`` puts a message on that node's
    session (``SessionHandle.send``; a message to a node without a session is dropped: its leases are reconciled
    by ``Hello.leases`` when it returns). ``spec`` is the game's (result checks, P-PF4-14). ``journal(results)`` is
    the journal writer's ``submit`` (pulse results); ``fail_run(text)`` stops the run with exit 1 (a typed error in
    a pulse block; ``Hub._fail_run``, A6). ``pulse`` is the ``HubPulse`` (T5.1 (b)) or None: it provides
    ``accepting``, ``start_next()``, ``on_block(block_id, results)`` (the snapshot's results once complete, else
    None), ``on_failed(owner, text)``, ``stop()`` and ``finish()`` (owners still in flight)."""

    def __init__(self, *, lease_sec: float, send: Callable[[int, pb.HubMessage], None], spec: GameSpec,
                 pulse: Any = None, journal: Callable[[list[MatchResult]], Any] | None = None,
                 fail_run: Callable[[str], None] | None = None, clock: Callable[[], float] = time.monotonic,
                 queue: BlockQueue | None = None) -> None:
        self.queue = queue if queue is not None else BlockQueue()
        self.pulse = pulse
        self._spec = spec
        self._lease_sec = float(lease_sec)
        self._send, self._journal, self._fail_run, self._clock = send, journal, fail_run, clock
        self._slots: dict[int, int] = {}
        self._running: dict[int, int] = {}
        self._sinks: dict[str, Callable[[str, str, Any], None]] = {}
        self._warned_invalid: set[int] = set()
        self._waiting_logged = False
        self._dispatching = True
        self.invalid_results = 0
        self.pulse_results = 0                    # pulse matches given to the journal writer

    # --- demand: how many evaluators each node runs ---
    @property
    def active(self) -> bool:
        return (self.pulse is not None and self.pulse.accepting) or bool(self._sinks)

    def evaluators_for(self, node: int) -> int:
        return self._slots.get(node, 0) if self.active else 0

    def _update_demand(self) -> None:
        for node in sorted(self._slots):
            wanted = self.evaluators_for(node)
            if self._running.get(node) != wanted:
                self._running[node] = wanted
                self._send(node, pb.HubMessage(command=pb.Command(kind="set_evaluators", evaluators=wanted)))

    # --- node lifecycle ---
    def node_joined(self, node: int, *, slots: int, held_leases: Sequence[str], fresh: bool) -> JoinReply:
        """A session opened (``Hello.slots.evaluators``, ``Hello.leases``; ``fresh``: a new node process)."""
        self._slots[node] = max(0, int(slots))
        self.queue.unwant(node)
        if fresh:
            self.queue.drop_node(node)                  # a new node process: its old evaluators are gone
        else:
            self.queue.keep_only(node, held_leases)
        unknown = sorted(set(held_leases) - set(self.queue.leases_of(node)))   # an earlier hub's, or cancelled
        self._running[node] = wanted = self.evaluators_for(node)
        self._dispatch()
        return JoinReply(wanted, [cancel_leases_message(unknown)] if unknown else [])

    def session_closed(self, node: int) -> None:
        self.queue.unwant(node)                          # its leases stay until it returns or is lost

    def node_lost(self, node: int) -> None:
        self.queue.drop_node(node)
        self._slots.pop(node, None)
        self._running.pop(node, None)
        self._dispatch()

    # --- messages from nodes ---
    def on_lease_request(self, node: int, free_slots: int) -> None:
        """``BlockLeaseRequest.free_slots`` (an increment); a node never waits for more leases than its evaluator
        slots (node data is untrusted)."""
        if node in self._slots and self._dispatching:
            room = self._slots[node] - self.queue.waiting().get(node, 0)
            self.queue.want(node, min(int(free_slots), room))
            self._dispatch()

    def on_block_result(self, node: int, msg: pb.BlockResult) -> None:
        self._send(node, pb.HubMessage(block_ack=pb.BlockAck(lease_id=msg.lease_id, block_ids=[msg.block_id])))
        task = self.queue.task(msg.block_id)
        if msg.failed:
            if task is not None:
                self._on_failed(node, msg, task)
            return
        try:
            results = [result_from_proto(p) for p in msg.results]
        except Exception as e:  # noqa: BLE001 - node data is untrusted (spec 7.1): counted, never fatal
            results, reason = [], f"malformed results ({type(e).__name__})"
        else:
            reason = check_block_results(task, results, spec=self._spec) if task is not None else None
        if task is not None and reason is not None:
            self.invalid_results += 1
            if node not in self._warned_invalid:
                self._warned_invalid.add(node)
                logger.warning(f"evaluation: node {node} sent invalid results for block {msg.block_id} ({reason}); "
                               f"the block is played again (further invalid results of this node are only counted)")
            self.queue.release(msg.lease_id, msg.block_id, count=True, reason=f"invalid results ({reason})")
            self._after_change()
            return
        if self.queue.complete(msg.lease_id, msg.block_id, results, self._clock()) != "accepted":
            return
        for result in results:
            result.node = node
        self._deliver(task, results)
        self._after_change()

    def _on_failed(self, node: int, msg: pb.BlockResult, task: BlockTask) -> None:
        if msg.error_kind in FATAL_ERRORS:
            self._owner_failed(task.owner, msg.error_kind,
                               f"{task.source} block {task.block_id} failed on node {node}: {msg.error}")
        else:
            self.queue.release(msg.lease_id, msg.block_id, count=msg.error_kind != "abandoned",
                               reason=f"{msg.error_kind or 'failed'} on node {node}")
        self._after_change()

    def _owner_failed(self, owner: str, kind: str, text: str) -> None:
        self.cancel(owner)
        if owner.startswith(PULSE_PREFIX):
            if self.pulse is not None:
                self.pulse.on_failed(owner, text)
            if kind in FATAL_ERRORS and self._fail_run is not None:
                self._fail_run(text)                     # SP4: a pulse error stops the run (exit 1)
        else:
            sink = self._sinks.get(owner)
            if sink is not None:
                sink("failed", kind, text)

    def _deliver(self, task: BlockTask, results: list[MatchResult]) -> None:
        if task.owner.startswith(PULSE_PREFIX):
            played = self.pulse.on_block(task.block_id, results) if self.pulse is not None else None
            if played is not None:
                self.queue.forget(task.owner)
                self.pulse_results += len(played)
                if self._journal is not None:
                    self._journal(played)                # A20: the journal only; the writer counts its drops
        else:
            sink = self._sinks.get(task.owner)
            if sink is not None:
                sink("block", task.block_id, results)

    # --- owners other than the pulse (T5.2) ---
    def add_owner(self, owner: str, sink: Callable[[str, str, Any], None]) -> None:
        """``sink(kind, key, value)``: ``("block", block id, results)`` per accepted block, ``("failed", error kind
        or "attempts", text)`` once when the owner fails (its blocks are cancelled)."""
        self._sinks[owner] = sink
        self._update_demand()

    def submit(self, tasks: Sequence[BlockTask]) -> None:
        self.queue.submit(tasks)
        self._dispatch()

    def remove_owner(self, owner: str) -> None:
        self.cancel(owner)
        self._sinks.pop(owner, None)
        self._update_demand()

    def cancel(self, owner: str) -> list[str]:
        blocks, leases = self.queue.cancel(owner)
        for node, lease_ids in sorted(leases.items()):
            self._send(node, cancel_leases_message(lease_ids))
        return blocks

    # --- main loop ---
    def tick(self) -> None:
        """Every main-loop pass (P-PF4-8): lease expiry, failed blocks, the next pulse snapshot, dispatch, the
        waiting INFO."""
        self.queue.expire(self._clock())
        self._handle_failed()
        if self.pulse is not None and self._dispatching and self.queue.queued(PULSE_PREFIX) == 0:
            tasks = self.pulse.start_next()      # the next snapshot once the last one's blocks are all leased
            if tasks:
                self.queue.submit(tasks)
        self._dispatch()
        self._log_waiting()

    def _after_change(self) -> None:
        self._handle_failed()
        self._dispatch()

    def _handle_failed(self) -> None:
        for block_id, owner, reason in self.queue.take_failed():
            self._owner_failed(owner, "attempts",
                               f"block {block_id} failed {MAX_BLOCK_ATTEMPTS} times (last: {reason})")

    def _dispatch(self) -> None:
        if not self._dispatching:
            return
        for node, lease, tasks in self.queue.dispatch(self._clock(), self._lease_sec):
            self._send(node, pb.HubMessage(lease=lease_to_proto(lease.lease_id, self._lease_sec, tasks)))

    def _log_waiting(self) -> None:
        waiting = self.queue.queued() > 0 and sum(self._running.values()) == 0
        if waiting and not self._waiting_logged:
            logger.info(f"evaluation: {self.queue.queued()} blocks wait: no connected node runs an evaluator (nodes "
                        f"with evaluator slots take them: colosseum join --evaluators N)")
        self._waiting_logged = waiting

    def stop(self) -> None:
        """The run stops: no new lease, no new pulse snapshot (leased blocks may still report until ``finish``)."""
        self._dispatching = False
        if self.pulse is not None:
            self.pulse.stop()

    def finish(self) -> None:
        """Before the journal writer stops: snapshots still in flight stay unevaluated; their blocks are cancelled."""
        if self.pulse is not None:
            for owner in self.pulse.finish():
                self.cancel(owner)

    def stats(self) -> dict[str, Any]:
        """The ``system`` record's ``evaluation`` value."""
        return {**self.queue.stats(), "invalid_results": self.invalid_results,
                "evaluators": sum(self._running.values()), "pulse_results": self.pulse_results}
