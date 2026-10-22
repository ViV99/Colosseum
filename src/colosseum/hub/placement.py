"""Learner placement and the hub's learner table (SP5 spec blocks 1, 2.6, 7.1, 7.2; task T4.1).

Pure: no processes, no network, no clock (callers pass ``now`` from ``time.monotonic()`` of the hub). The hub's
main loop owns one ``LearnerTable`` and calls ``place`` whenever the set of nodes or the set of agents that need a
learner changes.

**Slots** (spec 2.6). A node offers one GPU learner slot per GPU index in ``NodeCapacity.gpus`` (CUDA, or MPS when
``mps``) and ``cpu_learners`` CPU learner slots; a node started with ``--no-learners`` offers none. ``unlimited``
(train's local node, ruling R18) takes any number of learners on each of its GPUs and on the CPU; on such a node a
GPU learner goes to its least used GPU (ties: the lower index), so one GPU takes every learner, as before SP5.

**Devices** (``learner.device``):
- ``cpu`` -> a CPU slot;
- ``cuda`` -> any CUDA GPU slot; ``cuda:N`` -> the GPU slot with index N (on any node that has that index);
- ``mps`` -> an MPS slot;
- ``auto`` -> a CUDA GPU slot when any online node offers CUDA GPU slots (free or taken), else a CPU slot, like
  ``learner.resolve_device("auto")`` on one machine; MPS only when asked for by name.

**Preference** (``learner.placement``): ``node`` (a node name) and ``tags`` (every one of them) mark the preferred
nodes. Without ``strict`` another node with a matching free slot is used when no preferred node has one; with
``strict`` only preferred nodes are used. Within a tier: ascending node number, then ascending GPU index.

**Stability:** a learner never moves while its node lives: ``current`` slots are kept as they are (also while their
node is lost but not yet past ``learner_relocate_after_sec``); only the agents in ``relocate`` are placed again.
Agents are placed in the order of ``agents`` (config order), so the earlier agent wins a contested slot.

**The table** (``LearnerTable``) holds per trainable agent the status of its learner:
``pending`` (no slot; its seats do not collect) -> ``starting`` (``start_learner`` sent with a new generation; its
seats already collect and it owns cells: a lineup reaches a worker only once the learner's first weights are in
the machine's cache, R11) -> ``running`` (the hub accepted a version announcement of that generation) and
``relocating`` (its node is lost; it waits ``learner_relocate_after_sec`` from the node's last heartbeat for the
node to come back, then becomes ``pending``) or ``failed`` (the node's restart limit is spent; never placed again in
this hub incarnation). Every assignment has a new generation (the caller takes it from
``VersionRegistry.new_generation``); ``reconcile`` compares a reconnecting node's running learners with the table
(adopt the current generation, stop anything else).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "CPU", "CUDA", "LEARNER_STATUSES", "MPS", "TABLE_STATE_VERSION", "LearnerEntry", "LearnerSlot", "LearnerTable",
    "NodeCapacity", "PlacementPlan", "Reconciled", "capacity_from_hello", "parse_device", "place", "requested_slot",
]

CPU, CUDA, MPS = "cpu", "cuda", "mps"
_CUDA_INDEX_RE = re.compile(r"cuda:(\d+)")
LEARNER_STATUSES = ("pending", "starting", "running", "relocating", "failed")
LearnerStatus = Literal["pending", "starting", "running", "relocating", "failed"]
TABLE_STATE_VERSION = 1


@dataclass(frozen=True)
class LearnerSlot:
    node: int
    device: str                   # "cpu" | "cuda:N" | "mps"


@dataclass(frozen=True)
class NodeCapacity:
    node: int
    name: str
    tags: tuple[str, ...]
    gpus: tuple[int, ...]         # GPU indices offered for learners (Slots.gpus; [] with --no-learners)
    cpu_learners: int             # CPU learner slots (Slots.cpu_learners; 0 with --no-learners)
    unlimited: bool = False       # train's local node (R18): any number of learners per GPU and on the CPU
    mps: bool = False             # the GPU slots are MPS (macOS), not CUDA


@dataclass(frozen=True)
class PlacementPlan:
    assigned: dict[str, LearnerSlot]      # every agent with a slot: the kept ones and the newly placed ones
    pending: list[str]                    # agents without a slot, in the order of ``agents``
    reasons: dict[str, str]               # pending agent -> why (one line, for logs and ``nodes list``)

    def new(self, current: Mapping[str, LearnerSlot]) -> dict[str, LearnerSlot]:
        """The slots that differ from ``current`` (agents to start)."""
        return {a: s for a, s in self.assigned.items() if current.get(a) != s}


def parse_device(device: str) -> tuple[str, int | None]:
    """``learner.device`` -> (kind, GPU index or None) for every value but ``auto``; ValueError otherwise."""
    if device in (CPU, CUDA, MPS):
        return device, None
    match = _CUDA_INDEX_RE.fullmatch(device)
    if match:
        return CUDA, int(match.group(1))
    raise ValueError(f"learner.device {device!r}: expected auto, cpu, cuda, cuda:N or mps")


def requested_slot(device: str, nodes: Sequence[NodeCapacity]) -> tuple[str, int | None]:
    """The slot kind ``device`` needs on this cluster (``auto`` resolved over ``nodes``, module docstring)."""
    if device == "auto":
        return (CUDA, None) if any(n.gpus and not n.mps for n in nodes) else (CPU, None)
    return parse_device(device)


def _preferred(node: NodeCapacity, placement: Any) -> bool:
    if placement is None:
        return True
    if placement.node is not None and node.name != placement.node:
        return False
    return all(tag in node.tags for tag in placement.tags)


def _has_preference(placement: Any) -> bool:
    return placement is not None and (placement.node is not None or bool(placement.tags))


class _Usage:
    """Slots taken per node (kept and newly placed learners)."""

    def __init__(self, nodes: Sequence[NodeCapacity]) -> None:
        self.nodes = {n.node: n for n in nodes}
        self.gpu: dict[int, dict[int, int]] = {n.node: {i: 0 for i in n.gpus} for n in nodes}
        self.cpu: dict[int, int] = {n.node: 0 for n in nodes}

    def take(self, slot: LearnerSlot) -> None:
        if slot.node not in self.nodes:          # a kept slot on a lost node holds nothing
            return
        if slot.device == CPU:
            self.cpu[slot.node] += 1
            return
        index = 0 if slot.device == MPS else parse_device(slot.device)[1]
        counts = self.gpu[slot.node]
        if index in counts:
            counts[index] += 1

    def free_slot(self, node: NodeCapacity, kind: str, index: int | None) -> LearnerSlot | None:
        if kind == CPU:
            if node.unlimited or self.cpu[node.node] < node.cpu_learners:
                return LearnerSlot(node.node, CPU)
            return None
        if (kind == MPS) != node.mps:
            return None
        counts = self.gpu[node.node]
        candidates = [i for i in sorted(counts) if index is None or i == index]
        if not node.unlimited:
            candidates = [i for i in candidates if counts[i] == 0]
        if not candidates:
            return None
        best = min(candidates, key=lambda i: (counts[i], i))
        return LearnerSlot(node.node, MPS if node.mps else f"{CUDA}:{best}")


def _what(kind: str, index: int | None) -> str:
    if kind == CPU:
        return "CPU learner slot"
    if kind == MPS:
        return "MPS slot"
    return "CUDA GPU slot" if index is None else f"slot on GPU {index}"


def place(agents: Mapping[str, Any], nodes: Sequence[NodeCapacity], *, current: Mapping[str, LearnerSlot],
          relocate: Collection[str] = ()) -> PlacementPlan:
    """Slots for every agent of ``agents`` (agent -> its ``LearnerConfig``, in config order) on the online
    ``nodes`` (module docstring). ``current`` slots are kept unless the agent is in ``relocate``; a kept slot
    counts against its node's capacity. Agents of ``agents`` without a kept slot are placed in order; the rest are
    pending with a reason. Deterministic for equal inputs."""
    ordered = sorted(nodes, key=lambda n: n.node)
    usage = _Usage(ordered)
    assigned: dict[str, LearnerSlot] = {}
    for agent in agents:
        slot = current.get(agent)
        if slot is not None and agent not in relocate:
            assigned[agent] = slot
            usage.take(slot)
    pending: list[str] = []
    reasons: dict[str, str] = {}
    offering = [n for n in ordered if n.unlimited or n.gpus or n.cpu_learners > 0]
    for agent, cfg in agents.items():
        if agent in assigned:
            continue
        placement = getattr(cfg, "placement", None)
        if not offering:
            pending.append(agent)
            reasons[agent] = "no online node offers learner slots"
            continue
        kind, index = requested_slot(cfg.device, ordered)
        preferred = [n for n in offering if _preferred(n, placement)]
        strict = bool(getattr(placement, "strict", False)) and _has_preference(placement)
        tiers = [preferred] if strict else [preferred, [n for n in offering if n not in preferred]]
        slot = None
        for tier in tiers:
            for node in tier:
                slot = usage.free_slot(node, kind, index)
                if slot is not None:
                    break
            if slot is not None:
                break
        if slot is None:
            pending.append(agent)
            if strict and not preferred:
                reasons[agent] = f"no online node matches learner.placement ({_describe(placement)}, strict)"
            else:
                where = f" on the nodes matching learner.placement ({_describe(placement)}, strict)" if strict else ""
                reasons[agent] = f"no free {_what(kind, index)}{where} (learner.device {cfg.device})"
            continue
        assigned[agent] = slot
        usage.take(slot)
    return PlacementPlan(assigned=assigned, pending=pending, reasons=reasons)


def _describe(placement: Any) -> str:
    parts = []
    if placement.node is not None:
        parts.append(f"node {placement.node!r}")
    if placement.tags:
        parts.append(f"tags {list(placement.tags)}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------------------------------------------
# The learner table
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class LearnerEntry:
    agent_id: str
    status: LearnerStatus = "pending"
    slot: LearnerSlot | None = None
    generation: int = 0                 # the current (or last) assignment's generation; 0 = never assigned
    lost_since: float | None = None     # hub monotonic: the last heartbeat of the slot's node (relocating only)
    reason: str = ""                    # why pending / relocating / failed
    assignments: int = 0                # assignments in this hub incarnation (learner_restarts = sum - 1 each)


@dataclass(frozen=True)
class Reconciled:
    """``LearnerTable.reconcile``: the node's learners the hub keeps, the ones it must stop, and the agents whose
    learner this node should run but does not (they became pending)."""

    adopted: list[str] = field(default_factory=list)
    stop: list[tuple[str, int]] = field(default_factory=list)          # (agent, generation)
    gone: list[str] = field(default_factory=list)


class LearnerTable:
    """Per trainable agent: the status of its learner (module docstring). Main loop only."""

    def __init__(self, agents: Sequence[str]) -> None:
        self._entries: dict[str, LearnerEntry] = {a: LearnerEntry(a) for a in agents}

    def __contains__(self, agent: str) -> bool:
        return agent in self._entries

    def __getitem__(self, agent: str) -> LearnerEntry:
        return self._entries[agent]

    def entries(self) -> list[LearnerEntry]:
        return list(self._entries.values())

    def to_place(self) -> list[str]:
        """Agents waiting for a slot, in config order."""
        return [a for a, e in self._entries.items() if e.status == "pending"]

    def current(self) -> dict[str, LearnerSlot]:
        """Slots of every learner that exists or is awaited (starting, running, relocating)."""
        return {a: e.slot for a, e in self._entries.items()
                if e.status in ("starting", "running", "relocating") and e.slot is not None}

    def on_node(self, node: int) -> list[str]:
        return [a for a, e in self._entries.items()
                if e.slot is not None and e.slot.node == node and e.status in ("starting", "running", "relocating")]

    def can_collect(self, agent: str) -> bool:
        """The agent's seats collect and it owns cells while its learner starts or runs (spec 4.7)."""
        entry = self._entries.get(agent)
        return entry is not None and entry.status in ("starting", "running")

    def collecting(self) -> list[str]:
        """Agents that can collect, in config order (the owner rotation's agents)."""
        return [a for a, e in self._entries.items() if e.status in ("starting", "running")]

    def assign(self, agent: str, slot: LearnerSlot, generation: int) -> None:
        """A ``start_learner`` with ``generation`` goes to ``slot``'s node: ``starting``."""
        entry = self._entries[agent]
        if entry.status == "failed":
            raise ValueError(f"learner of {agent!r} failed; it is not placed again")
        if generation <= entry.generation:
            raise ValueError(f"generation {generation} of {agent!r} is not above {entry.generation}")
        entry.status, entry.slot, entry.generation = "starting", slot, int(generation)
        entry.lost_since, entry.reason = None, ""
        entry.assignments += 1

    def announced(self, agent: str, generation: int) -> bool:
        """The hub accepted a version of ``generation``: a ``starting`` learner of that generation is ``running``.
        True when the status changed."""
        entry = self._entries.get(agent)
        if entry is None or entry.generation != generation or entry.status != "starting":
            return False
        entry.status = "running"
        return True

    def node_lost(self, node: int, last_heartbeat: float) -> list[str]:
        """``node`` is lost: its learners wait for it (``relocating``); returns them."""
        moved = []
        for agent in self.on_node(node):
            entry = self._entries[agent]
            if entry.status != "relocating":
                entry.status, entry.lost_since = "relocating", float(last_heartbeat)
                entry.reason = f"node {node} lost"
                moved.append(agent)
        return moved

    def relocation_due(self, now: float, after_sec: float) -> list[str]:
        """``relocating`` learners whose node stayed away ``after_sec`` since its last heartbeat become ``pending``
        (the caller places them again with a new generation); returns them."""
        due = []
        for agent, entry in self._entries.items():
            if entry.status == "relocating" and entry.lost_since is not None and now - entry.lost_since >= after_sec:
                entry.status, entry.slot, entry.lost_since = "pending", None, None
                entry.reason = f"relocating: its node stayed away {after_sec:g} s"
                due.append(agent)
        return due

    def reconcile(self, node: int, running: Mapping[str, int], *, announced: Mapping[str, int]) -> Reconciled:
        """A node (re)connected with ``running`` = {agent: generation} of its live learners (``Hello.learners``).
        ``announced`` = {agent: generation of the newest accepted version} (``VersionRegistry``).

        - a learner of the agent's current generation on its slot's node is adopted: ``running`` when a version of
          that generation was accepted, else ``starting`` (a ``relocating`` one is back);
        - any other learner (a stale generation, an agent placed elsewhere, failed or unknown) is stopped;
        - an agent whose learner this node should run (starting / running / relocating here) but does not becomes
          ``pending`` (``gone``): the caller places it again with a new generation."""
        out = Reconciled()
        for agent, generation in running.items():
            entry = self._entries.get(agent)
            ours = (entry is not None and entry.slot is not None and entry.slot.node == node
                    and entry.generation == generation and entry.status in ("starting", "running", "relocating"))
            if not ours:
                out.stop.append((agent, int(generation)))
                continue
            entry.status = "running" if announced.get(agent) == generation else "starting"
            entry.lost_since, entry.reason = None, ""
            out.adopted.append(agent)
        for agent in self.on_node(node):
            if agent not in out.adopted:
                entry = self._entries[agent]
                entry.status, entry.slot, entry.lost_since = "pending", None, None
                entry.reason = f"node {node} came back without it"
                out.gone.append(agent)
        return out

    def died(self, agent: str, generation: int) -> LearnerSlot | None:
        """The learner of ``generation`` died and its node may restart it: the slot to restart on (the caller
        assigns a new generation there), or None when the report is about an old generation (ignored)."""
        entry = self._entries.get(agent)
        if entry is None or entry.generation != generation or entry.status not in ("starting", "running"):
            return None
        return entry.slot

    def fail(self, agent: str, reason: str) -> None:
        """The restart limit is spent (spec 7.1): ``failed``, slot released, never placed again."""
        entry = self._entries[agent]
        entry.status, entry.slot, entry.lost_since, entry.reason = "failed", None, None, reason

    def failed(self) -> list[str]:
        return [a for a, e in self._entries.items() if e.status == "failed"]

    def set_pending(self, agent: str, reason: str) -> None:
        entry = self._entries[agent]
        if entry.status != "failed":
            entry.status, entry.slot, entry.lost_since, entry.reason = "pending", None, None, reason

    def state(self, local_nodes: Collection[int] = ()) -> dict:
        """``hub_state.json`` ``learners`` (JSON-ready); ``local_nodes``: numbers of nodes that die with the hub
        (train's local node), so a resume does not wait for them."""
        return {"version": TABLE_STATE_VERSION, "agents": {
            a: {"status": e.status, "node": None if e.slot is None else e.slot.node,
                "device": None if e.slot is None else e.slot.device, "generation": e.generation, "reason": e.reason,
                "local": e.slot is not None and e.slot.node in local_nodes}
            for a, e in self._entries.items()}}

    def load_resumed(self, state: Mapping[str, Any] | None, *, now: float) -> list[str]:
        """In-place resume (T4.3): learners the previous hub incarnation had (starting, running or relocating) on a
        remote node wait for it as ``relocating`` from ``now`` (the node reconnects and is reconciled, or they
        relocate after ``learner_relocate_after_sec``); learners of a local node (it died with the hub) and
        ``failed`` agents are ``pending`` at once (failed ones are retried); generations continue; agents not in
        this table are dropped. Returns the awaited agents."""
        awaited = []
        for agent, saved in (state or {}).get("agents", {}).items():
            entry = self._entries.get(agent)
            if entry is None:
                continue
            entry.generation = max(entry.generation, int(saved.get("generation") or 0))
            if (saved.get("status") in ("starting", "running", "relocating") and saved.get("node") is not None
                    and not saved.get("local")):
                entry.status = "relocating"
                entry.slot = LearnerSlot(int(saved["node"]), str(saved["device"]))
                entry.lost_since = float(now)
                entry.reason = "waiting for its node after the hub restart"
                awaited.append(agent)
        return awaited


def capacity_from_hello(node: int, name: str, tags: Sequence[str], slots: Mapping[str, Any],
                        gpu_kinds: Mapping[int, str], *, unlimited: bool) -> NodeCapacity:
    """A ``NodeCapacity`` from a node's ``Hello.slots`` (as a dict: ``gpus``, ``cpu_learners``, ``no_learners``) and
    its GPU kinds by index (``Hello.hardware.gpus``). A negative ``cpu_learners`` (train's local node sends -1:
    unlimited, R18) counts as 0 slots: only ``unlimited`` makes a node take any number."""
    if slots.get("no_learners"):
        return NodeCapacity(node, name, tuple(tags), (), 0, unlimited=False)
    gpus = tuple(sorted({int(i) for i in slots.get("gpus", ()) if int(i) >= 0}))
    mps = bool(gpus) and all(gpu_kinds.get(i) == MPS for i in gpus)
    return NodeCapacity(node, name, tuple(tags), gpus, max(0, int(slots.get("cpu_learners", 0) or 0)),
                        unlimited=unlimited, mps=mps)
