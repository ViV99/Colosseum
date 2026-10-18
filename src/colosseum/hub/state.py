"""Pure hub state (SP5 spec blocks 4.1-4.5, 4.7, 7.2, 7.6; plan rulings R7, R8, R16).

Everything here is owned by the hub's main loop (``colosseum.hub.hub.Hub``) and touched by no other thread; nothing
here does I/O except ``save_hub_state`` / ``load_hub_state`` and the ``on_reserve`` callback of ``EpochAllocator``.
Times are the hub's ``time.monotonic()`` values passed in by the caller (``now``); clocks of other machines never
appear here, only durations (``ttl_sec``).

- ``NodeRegistry``: the nodes the hub knows in this hub process: online / lost / removed, the node's session id, the
  serial of its current stream and whether that stream is open, the last heartbeat; ``connect`` tells a new node
  from a resumed session and a returned node (spec 4.1; pre-flight ruling P-PF2-1 (b): a closed stream keeps the
  session id, the same session id comes back as ``resumed``, and only the current stream's close counts).
- ``EpochAllocator`` (R7): epochs are reserved in blocks of ``EPOCH_BLOCK``; ``on_reserve(until)`` (the hub writes
  ``hub_state.json`` with fsync) runs before the first epoch of a new block is handed out, so a crash never reissues
  an epoch; a resumed hub starts after the saved reservation.
- ``CellTable``: cells (node, worker, env) with their epoch, the worker incarnation they belong to, their owner and
  previous owner, and when they were granted (spec 4.2). A request keeps a cell's epoch only for the same incarnation
  and an epoch this hub process granted; every other request gets a new epoch.
- ``OwnerRotation``: the deficit rule of spec 4.2. Every collecting agent's target share is equal, so the agent with
  the largest deficit is the one that owns the fewest live collecting cells; ties go to the first tied agent after the
  cell's previous owner in config order (no previous owner: the first tied agent). With one node whose cells all
  expire together this reproduces SP4's ``trainable[(g + round) % n]``.
- ``restrict_collecting``: seats of agents that cannot collect (no running learner) do not collect (spec 4.7).
- ``BudgetCounter``: the one budget (spec 4.4): a saved base plus the increments of each (node, session)'s cumulative
  env-step counter since its last saved value.
- ``VersionInfo`` (defined in ``colosseum.net.messages``, amendment A7) is re-exported here.
- ``VersionRegistry``: per agent the current learner generation, the highest accepted version, the version cap of
  every superseded generation and the latest announcement (spec 4.5, 7.2; R16: a superseded generation ``g`` is
  stale only above ``cap(g)``).
- ``ResultIntake``: checks of every match result a node sends (spec 4.3, 3.4: never raises) and the batch dedup / ack
  bookkeeping per (node, session, seq) (R8), with drop counters by reason and one WARNING per (node, reason).
- ``HubStateData`` / ``save_hub_state`` / ``load_hub_state``: ``runs/<name>/hub_state.json`` (amendment A3).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from colosseum.arena.entities import snapshot_version
from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID, Lineup, MatchResult
from colosseum.net.messages import VersionInfo  # A7: defined there, re-exported here

__all__ = [
    "DROP_REASONS", "EPOCH_BLOCK", "EVICTED_HOLD_FACTOR", "HUB_STATE_FILE", "HUB_STATE_VERSION", "BudgetCounter",
    "CellEntry", "CellKey", "CellTable", "EpochAllocator", "HubStateData", "NodeEntry", "NodeHello", "NodeRegistry",
    "OwnerRotation", "ResultIntake", "VersionInfo", "VersionRegistry", "load_hub_state", "restrict_collecting",
    "save_hub_state",
]

EPOCH_BLOCK = 1024                   # R7: epochs reserved per hub_state.json write
EVICTED_HOLD_FACTOR = 2              # spec 4.2: an evicted snapshot stays servable <= 2 x match_refresh_interval_sec
HUB_STATE_FILE = "hub_state.json"
HUB_STATE_VERSION = 1

DROP_REASONS = ("invalid", "stale_generation")


# ---------------------------------------------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeHello:
    """What the hub keeps of a node's ``Hello`` (``hub.server.hello_from_proto``): no protobuf here."""

    hardware: dict = field(default_factory=dict)
    slots: dict = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    data_addresses: tuple[str, ...] = ()
    live_cells: tuple[tuple[int, int, int, int], ...] = ()   # (worker, env, epoch, incarnation)
    env_steps: int = 0
    last_acked_seq: int = 0


@dataclass
class NodeEntry:
    """One node as the hub sees it (spec 4.1). ``last_heartbeat`` is hub monotonic time. ``session_id`` is the node
    process's session (kept while its stream is closed, P-PF2-1 (b)); ``serial`` the hub's per-stream serial of the
    current stream (``SessionHandle``, T2.3; None when the caller gives none); ``connected`` whether that stream is
    open."""

    node_id: str
    number: int
    name: str
    session_id: str | None
    state: Literal["online", "lost", "removed"]
    hardware: dict
    slots: dict
    tags: tuple[str, ...]
    data_addresses: tuple[str, ...]
    data_path: Literal["direct", "relay", "local"]
    last_heartbeat: float
    serial: int | None = None
    connected: bool = True


class NodeRegistry:
    """Nodes of this hub process by node id (spec 4.1). A closed stream does not make a node lost: only silence for
    ``node_lost_after_sec`` does (``lost``), so a node that reconnects within it with its session id is ``resumed``
    and keeps its cells. Streams are told apart by the caller's per-stream ``serial`` (P-PF2-1): the hub may learn of
    a cut only after the node's new stream is open, so a close of a superseded stream changes nothing."""

    def __init__(self) -> None:
        self._nodes: dict[str, NodeEntry] = {}

    def connect(self, node_id: str, number: int, name: str, session_id: str, hello: NodeHello, now: float, *,
                data_path: Literal["direct", "relay", "local"] = "relay",
                serial: int | None = None) -> Literal["new", "resumed", "returned"]:
        """A ``Hello`` on the stream ``serial``: ``new`` (first time in this hub process), ``resumed`` (the same
        session id of a node that is not lost, whether or not its old stream was seen to close: the stream was cut
        and came back; its cells stay), ``returned`` (a lost or removed node, or a new session id: the node process
        restarted or the node was given up; the caller drops the node's cells). The entry takes this stream's
        serial."""
        entry = self._nodes.get(node_id)
        if entry is None:
            kind: Literal["new", "resumed", "returned"] = "new"
        elif entry.state == "online" and entry.session_id == session_id:
            kind = "resumed"
        else:
            kind = "returned"
        self._nodes[node_id] = NodeEntry(
            node_id=node_id, number=int(number), name=name, session_id=session_id, state="online",
            hardware=dict(hello.hardware), slots=dict(hello.slots), tags=tuple(hello.tags),
            data_addresses=tuple(hello.data_addresses), data_path=data_path, last_heartbeat=now,
            serial=serial, connected=True)
        return kind

    def heartbeat(self, node_id: str, session_id: str, now: float) -> bool:
        """Record a heartbeat; False (ignored) for an unknown node, another session, or a node not online."""
        entry = self._nodes.get(node_id)
        if entry is None or entry.state != "online" or entry.session_id != session_id:
            return False
        entry.last_heartbeat = now
        return True

    def disconnect(self, node_id: str, serial: int | None) -> None:
        """The stream ``serial`` closed. Only the node's current stream counts (a superseded one's late close changes
        nothing); the node keeps its session id and stays online until ``lost`` (it may come back with it)."""
        entry = self._nodes.get(node_id)
        if entry is not None and entry.state == "online" and entry.serial == serial:
            entry.connected = False

    def lost(self, now: float, after_sec: float) -> list[NodeEntry]:
        """Mark and return the online nodes silent for more than ``after_sec`` (each once)."""
        out = []
        for entry in self._nodes.values():
            if entry.state == "online" and now - entry.last_heartbeat > after_sec:
                entry.state = "lost"
                entry.connected = False
                out.append(entry)
        return out

    def remove(self, node_id: str) -> NodeEntry | None:
        entry = self._nodes.get(node_id)
        if entry is not None:
            entry.state = "removed"
            entry.connected = False
        return entry

    def online(self) -> list[NodeEntry]:
        return sorted((e for e in self._nodes.values() if e.state == "online"), key=lambda e: e.number)

    def get(self, number: int) -> NodeEntry | None:
        return next((e for e in self._nodes.values() if e.number == number), None)

    def by_id(self, node_id: str) -> NodeEntry | None:
        return self._nodes.get(node_id)

    def all(self) -> list[NodeEntry]:
        return sorted(self._nodes.values(), key=lambda e: e.number)


# ---------------------------------------------------------------------------------------------------------------
# Cells, epochs, owners
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class CellKey:
    node: int
    worker: int
    env: int


class EpochAllocator:
    """Epoch numbers, unique and increasing within a run (R7). ``reserved_until``: the highest epoch an earlier hub
    process may have handed out (``hub_state.json``; 0 for a new run); this allocator starts at the next block, so the
    first epoch is ``reserved_until + 1``. ``on_reserve(until)`` must persist ``until`` durably (atomic write + fsync)
    before it returns; it runs before the first epoch above the current reservation is returned, and an exception
    from it propagates (no epoch is handed out unreserved)."""

    def __init__(self, reserved_until: int, on_reserve: Callable[[int], None], *, block: int = EPOCH_BLOCK) -> None:
        if reserved_until < 0 or block < 1:
            raise ValueError(f"EpochAllocator: reserved_until {reserved_until} and block {block}")
        self._reserved = int(reserved_until)
        self._next = int(reserved_until) + 1
        self._on_reserve = on_reserve
        self._block = int(block)

    @property
    def reserved_until(self) -> int:
        return self._reserved

    def next(self) -> int:
        epoch = self._next
        if epoch > self._reserved:
            until = self._reserved + self._block
            self._on_reserve(until)              # raises: nothing changes, the caller grants nothing
            self._reserved = until
        self._next = epoch + 1
        return epoch


class OwnerRotation:
    """Owner of every requested cell by the deficit rule (module docstring). ``agents``: the trainable agents in
    config order."""

    def __init__(self, agents: Sequence[str]) -> None:
        self._agents = list(agents)

    def assign(self, cells: Sequence[CellKey], *, collecting: Collection[str], owners: Mapping[CellKey, str],
               previous: Mapping[CellKey, str | None]) -> dict[CellKey, str]:
        """``cells``: the requested cells, in (node, worker, env) order; ``collecting``: agents that can collect now;
        ``owners``: the current owners of every other live collecting cell (other agents are ignored); ``previous``:
        the previous owner of a requested cell (absent or None: none). Returns an owner per requested cell; empty
        when no agent can collect."""
        order = [a for a in self._agents if a in set(collecting)]
        if not order:
            return {}
        requested = set(cells)
        owned = Counter(a for key, a in owners.items() if a in order and key not in requested)
        out: dict[CellKey, str] = {}
        for cell in cells:
            fewest = min(owned[a] for a in order)          # equal targets: the largest deficit owns the fewest
            tied = [a for a in order if owned[a] == fewest]
            prev = previous.get(cell)
            if prev in order:
                start = order.index(prev) + 1
                cyclic = order[start:] + order[:start]
                owner = next(a for a in cyclic if a in tied)
            else:
                owner = tied[0]
            out[cell] = owner
            owned[owner] += 1
        return out


@dataclass
class CellEntry:
    key: CellKey
    epoch: int
    incarnation: int
    owner: str | None = None
    previous: str | None = None
    granted_at: float | None = None      # hub monotonic time of the last grant
    ttl_sec: float = 0.0                 # of the last grant (0 = whole run)


class CellTable:
    """Live cells by key (spec 4.2); ``epochs`` gives new epochs."""

    def __init__(self, epochs: EpochAllocator) -> None:
        self._epochs = epochs
        self._cells: dict[CellKey, CellEntry] = {}
        self._granted: set[int] = set()        # epochs this hub process handed out

    def request(self, node: int, cells: Sequence[tuple[CellKey, int, int]], now: float) -> list[CellEntry]:
        """``cells``: ``(key, epoch, incarnation)`` the node asks a lineup for (epoch 0: none yet). Returns the
        entries in request order: the epoch stays for the same incarnation and an epoch this hub process granted to
        that cell; otherwise a new epoch. Every key must be of ``node`` (ValueError)."""
        out = []
        for key, epoch, incarnation in cells:
            if key.node != node:
                raise ValueError(f"cell {key} requested by node {node}")
            entry = self._cells.get(key)
            keep = (entry is not None and entry.incarnation == incarnation and entry.epoch == epoch
                    and epoch in self._granted)
            if not keep:
                new_epoch = self._epochs.next()
                self._granted.add(new_epoch)
                previous = None if entry is None else (entry.owner or entry.previous)
                entry = CellEntry(key=key, epoch=new_epoch, incarnation=int(incarnation), owner=None,
                                  previous=previous)
                self._cells[key] = entry
            out.append(entry)
        return out

    def grant(self, key: CellKey, owner: str, ttl_sec: float, now: float) -> CellEntry:
        """Record the lineup just granted: ``owner`` becomes the owner (the old one the previous owner)."""
        entry = self._cells[key]
        if entry.owner is not None:
            entry.previous = entry.owner
        entry.owner = owner
        entry.granted_at = now
        entry.ttl_sec = float(ttl_sec)
        return entry

    def owners(self, *, exclude: Collection[CellKey] = ()) -> dict[CellKey, str]:
        """Current owners of the live cells not in ``exclude``."""
        skip = set(exclude)
        return {k: e.owner for k, e in self._cells.items() if e.owner is not None and k not in skip}

    def drop_node(self, node: int) -> list[CellEntry]:
        dropped = [e for k, e in self._cells.items() if k.node == node]
        for entry in dropped:
            del self._cells[entry.key]
        return dropped

    def drop_cells(self, keys: Iterable[CellKey]) -> list[CellEntry]:
        out = [self._cells.pop(k) for k in list(keys) if k in self._cells]
        return out

    def get(self, key: CellKey) -> CellEntry | None:
        return self._cells.get(key)

    def live(self) -> list[CellEntry]:
        return [self._cells[k] for k in sorted(self._cells)]


def restrict_collecting(lineup: Lineup, collecting: Collection[str]) -> Lineup:
    """``lineup`` with ``collect=False`` for every seat whose agent cannot collect (spec 4.7); the lineup itself is
    not changed (a copy is returned when anything changes)."""
    can = set(collecting)
    if all(not s.collect or s.agent_id in can for s in lineup.seats):
        return lineup
    seats = [s if not s.collect or s.agent_id in can else replace(s, collect=False) for s in lineup.seats]
    return replace(lineup, seats=seats)


# ---------------------------------------------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------------------------------------------


class BudgetCounter:
    """The global env-step counter (spec 4.4): ``base`` plus every increment of a (node, session)'s cumulative
    counter since its last value. ``sessions`` (from ``hub_state.json``): the last value per
    ``"<node_id>/<session_id>"``, so a node of a saved session that reports again is counted from where the save
    left off."""

    def __init__(self, base: int = 0, sessions: Mapping[str, int] | None = None) -> None:
        if base < 0:
            raise ValueError(f"BudgetCounter: base {base} < 0")
        self._value = int(base)
        self._last: dict[str, int] = {str(k): int(v) for k, v in (sessions or {}).items()}

    def report(self, node_id: str, session_id: str, cumulative: int) -> int:
        """A node's cumulative env steps of its session; returns the increment applied (0 for a value that is not
        larger than the last one)."""
        key = f"{node_id}/{session_id}"
        cumulative = int(cumulative)
        last = self._last.get(key, 0)
        if cumulative <= last:
            return 0
        self._last[key] = cumulative
        self._value += cumulative - last
        return cumulative - last

    def add(self, n: int) -> None:
        """Steps that are not a node's (``training.resume_from``'s start point)."""
        if n < 0:
            raise ValueError(f"BudgetCounter.add({n})")
        self._value += int(n)

    @property
    def value(self) -> int:
        return self._value

    def state(self) -> dict:
        return {"env_steps": self._value, "sessions": dict(self._last)}


# ---------------------------------------------------------------------------------------------------------------
# Versions and generations
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class _AgentVersions:
    generation: int = 0
    max_version: int = -1                       # highest accepted version over all generations
    gen_max: int = -1                           # highest accepted version of the current generation
    caps: dict[int, int] = field(default_factory=dict)   # superseded generation -> its highest accepted version
    latest: VersionInfo | None = None


class VersionRegistry:
    """Agent -> generation -> versions (spec 4.5, 7.2; R16)."""

    def __init__(self) -> None:
        self._agents: dict[str, _AgentVersions] = {}

    def _get(self, agent_id: str) -> _AgentVersions:
        return self._agents.setdefault(agent_id, _AgentVersions())

    def new_generation(self, agent_id: str) -> int:
        """The next learner generation of ``agent_id`` (first: 1); the current one's cap is recorded."""
        a = self._get(agent_id)
        if a.generation > 0:
            a.caps[a.generation] = a.gen_max
        a.generation += 1
        a.gen_max = -1
        return a.generation

    def generation(self, agent_id: str) -> int:
        a = self._agents.get(agent_id)
        return 0 if a is None else a.generation

    def announce(self, info: VersionInfo) -> Literal["accepted", "stale_generation", "not_newer", "invalid"]:
        """A learner's announcement: accepted only from the current generation and above every accepted version."""
        if (info.version < 0 or info.generation < 1 or info.size < 0 or not info.digest
                or not info.agent_id):
            return "invalid"
        a = self._agents.get(info.agent_id)
        if a is None or info.generation > a.generation:
            return "invalid"                         # the hub never assigned this generation
        if info.generation < a.generation:
            return "stale_generation"
        if info.version <= a.max_version:
            return "not_newer"
        a.max_version = a.gen_max = info.version
        a.latest = info
        return "accepted"

    def latest(self, agent_id: str) -> VersionInfo | None:
        a = self._agents.get(agent_id)
        return None if a is None else a.latest

    def version_floor(self, agent_id: str) -> int:
        """The first version a newly assigned learner may publish: max accepted + 1 (0 without any; R9)."""
        a = self._agents.get(agent_id)
        return 0 if a is None else a.max_version + 1

    def is_stale(self, agent_id: str, generation: int | None, version: int | None) -> bool:
        """Whether data played with (``generation``, ``version``) of ``agent_id``'s latest weights must be rejected
        (R16): unknown generation (None or 0: SP4 path) -> no; the current one -> no; a superseded ``g`` -> only above
        ``cap(g)`` (or without a version); a generation never assigned -> yes."""
        if generation is None or generation == 0:
            return False
        a = self._agents.get(agent_id)
        if a is None or generation > a.generation or generation < 0:
            return True
        if generation == a.generation:
            return False
        cap = a.caps.get(generation)
        return version is None or cap is None or version > cap

    def state(self) -> dict:
        return {aid: {"generation": a.generation, "max_version": a.max_version, "gen_max": a.gen_max,
                      "caps": {str(g): c for g, c in sorted(a.caps.items())},
                      "latest": None if a.latest is None else a.latest.to_dict()}
                for aid, a in sorted(self._agents.items())}

    @classmethod
    def from_state(cls, d: Mapping[str, Any]) -> VersionRegistry:
        reg = cls()
        for aid, s in d.items():
            reg._agents[str(aid)] = _AgentVersions(
                generation=int(s["generation"]), max_version=int(s["max_version"]),
                gen_max=int(s.get("gen_max", -1)), caps={int(g): int(c) for g, c in s.get("caps", {}).items()},
                latest=None if s.get("latest") is None else VersionInfo.from_dict(s["latest"]))
        return reg


# ---------------------------------------------------------------------------------------------------------------
# Result intake
# ---------------------------------------------------------------------------------------------------------------

_NETWORK_RE = re.compile(r"ckpt_v\d+")


@dataclass
class _SessionSeqs:
    low: int = 0                                # every seq <= low is acked
    acked: set[int] = field(default_factory=set)        # acked seqs above low
    in_flight: set[int] = field(default_factory=set)

    def mark_acked(self, seq: int) -> None:
        self.in_flight.discard(seq)
        if seq > self.low:
            self.acked.add(seq)
        while self.low + 1 in self.acked:
            self.low += 1
            self.acked.discard(self.low)


class ResultIntake:
    """Result checks and batch bookkeeping (spec 4.3, R8). ``journal``: whether accounting follows the journal's
    commits (``accounts_at_intake`` is its negation: without the journal the hub accounts a batch when it arrives and
    acks it at once)."""

    def __init__(self, *, journal: bool) -> None:
        self.journal = bool(journal)
        self._sessions: dict[tuple[str, str], _SessionSeqs] = {}
        self.dropped: Counter[str] = Counter()
        self._warned: set[tuple[int, str]] = set()

    @property
    def accounts_at_intake(self) -> bool:
        return not self.journal

    # ---- checks -------------------------------------------------------------------------------------------------

    def check(self, result: MatchResult, *, spec, agents: Collection[str], versions: VersionRegistry,
              node: int | None = None, uid_prefix: str | None = None) -> str | None:
        """None for a valid result, else the drop reason (``invalid`` / ``stale_generation``); never raises.
        ``agents``: every agent a result may name (current and historical); ``node`` / ``uid_prefix``: the sending
        node's number and ``<run_id>/train/n<node>/`` (a node cannot write another node's uids)."""
        try:
            return self._check(result, spec, agents, versions, node, uid_prefix)
        except Exception:  # noqa: BLE001 - untrusted input: anything unexpected is invalid
            return "invalid"

    @staticmethod
    def _check(result: MatchResult, spec, agents: Collection[str], versions: VersionRegistry, node: int | None,
               uid_prefix: str | None) -> str | None:
        if not isinstance(result, MatchResult) or result.source != "train":
            return "invalid"
        if result.layout not in spec.layouts or result.outcome_kind != spec.outcome_kind(result.layout):
            return "invalid"
        seat_specs = spec.layouts[result.layout]
        if len(result.seats) != len(seat_specs):
            return "invalid"
        if node is not None and result.node != node:
            return "invalid"
        if uid_prefix is not None and not (result.uid.startswith(uid_prefix) and len(result.uid) > len(uid_prefix)):
            return "invalid"
        if not isinstance(result.episode_length, int) or result.episode_length < 0:
            return "invalid"
        for i, (seat, seat_spec) in enumerate(zip(result.seats, seat_specs, strict=True)):
            if seat.seat != i or seat.role != seat_spec.role or seat.team != seat_spec.team:
                return "invalid"
            if seat.agent_id not in agents:
                return "invalid"
            net = seat.network_id
            if net not in (LATEST_NETWORK_ID, FIXED_NETWORK_ID) and snapshot_version(net) is None:
                return "invalid"
            if not math.isfinite(float(seat.reward)):
                return "invalid"
            for v in (seat.version_start, seat.version_end, seat.eliminated_step):
                if v is not None and (not isinstance(v, int) or v < 0):
                    return "invalid"
        teams = {s.team for s in seat_specs}
        if sorted(t.team for t in result.teams) != sorted(teams):
            return "invalid"
        if not all(math.isfinite(float(t.rank)) and math.isfinite(float(t.score)) for t in result.teams):
            return "invalid"
        for seat in result.seats:
            if seat.network_id == LATEST_NETWORK_ID and versions.is_stale(seat.agent_id, seat.generation,
                                                                          seat.version_end):
                return "stale_generation"
        return None

    def count_drop(self, node: int, reason: str, n: int = 1) -> bool:
        """Count ``n`` dropped results; True the first time for (node, reason): the caller logs one WARNING."""
        self.dropped[reason] += int(n)
        if (node, reason) in self._warned:
            return False
        self._warned.add((node, reason))
        return True

    # ---- batches ------------------------------------------------------------------------------------------------

    def open(self, node_id: str, session_id: str, last_acked: int) -> None:
        """A session's Hello: seqs up to ``last_acked`` are acked (the node says so; a hub restart forgot them)."""
        seqs = self._sessions.setdefault((node_id, session_id), _SessionSeqs())
        if last_acked > seqs.low:
            seqs.low = int(last_acked)
            seqs.acked = {s for s in seqs.acked if s > seqs.low}
            seqs.in_flight = {s for s in seqs.in_flight if s > seqs.low}
            while seqs.low + 1 in seqs.acked:
                seqs.low += 1
                seqs.acked.discard(seqs.low)

    def begin(self, node_id: str, session_id: str, seq: int) -> bool:
        """A batch arrives: True when it is new (now in flight); False when it is acked or in flight (skip it; an
        acked one is acked again: ``is_acked``)."""
        seqs = self._sessions.setdefault((node_id, session_id), _SessionSeqs())
        if seq <= seqs.low or seq in seqs.acked or seq in seqs.in_flight:
            return False
        seqs.in_flight.add(int(seq))
        return True

    def is_acked(self, node_id: str, session_id: str, seq: int) -> bool:
        seqs = self._sessions.get((node_id, session_id))
        return seqs is not None and (seq <= seqs.low or seq in seqs.acked)

    def committed(self, node_id: str, session_id: str, seq: int) -> None:
        """Every result of the batch is accounted: ack it."""
        self._sessions.setdefault((node_id, session_id), _SessionSeqs()).mark_acked(int(seq))

    def failed(self, node_id: str, session_id: str, seq: int) -> None:
        """Nothing of the batch was accounted: not acked, the node resends it."""
        seqs = self._sessions.get((node_id, session_id))
        if seqs is not None:
            seqs.in_flight.discard(int(seq))

    def forget_node(self, node_id: str) -> None:
        """The node is gone for good (``nodes remove``); a lost node keeps its bookkeeping (it may resend)."""
        for key in [k for k in self._sessions if k[0] == node_id]:
            del self._sessions[key]


# ---------------------------------------------------------------------------------------------------------------
# hub_state.json
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class HubStateData:
    """``runs/<name>/hub_state.json`` (amendment A3, the one schema): ``env_steps`` / ``sessions`` are
    ``BudgetCounter.state()``, ``versions`` is ``VersionRegistry.state()``, ``counters`` is ``{"book":
    Hub._book.state(), "intake_book": MetricsHub.intake_book.state()}`` (two ``CounterBook`` states; T4.3 restores
    them with ``CounterBook.from_state``). ``counters`` is saved and loaded whole, so keys this version does not know
    survive a resume (D16). T4.1 adds ``learners``."""

    version: int
    run_id: str
    env_steps: int
    sessions: dict[str, int]
    epoch_reserved_until: int
    versions: dict
    counters: dict[str, Any]
    nodes_lost: int


def save_hub_state(path: str | Path, data: HubStateData, *, fsync: bool) -> None:
    """Write ``data`` atomically (temp file + ``os.replace``); with ``fsync`` the file and its directory are synced
    before this returns (epoch reservations, R7)."""
    path = Path(path)
    text = json.dumps(asdict(data), indent=2, sort_keys=True, allow_nan=False)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    if fsync:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def load_hub_state(path: str | Path) -> HubStateData | None:
    """The saved state, or None when the file does not exist; a malformed file or another version is a ValueError
    naming the file (the caller turns it into a ConfigError with a hint)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("version") != HUB_STATE_VERSION:
            raise ValueError(f"version {d.get('version')!r}, expected {HUB_STATE_VERSION}")
        return HubStateData(version=int(d["version"]), run_id=str(d["run_id"]), env_steps=int(d["env_steps"]),
                            sessions={str(k): int(v) for k, v in d["sessions"].items()},
                            epoch_reserved_until=int(d["epoch_reserved_until"]), versions=dict(d["versions"]),
                            counters=dict(d["counters"]), nodes_lost=int(d["nodes_lost"]))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"malformed {path}: {type(e).__name__}: {e}") from e
