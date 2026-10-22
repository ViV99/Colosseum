"""Evaluator leases on a node (SP5 spec 5.3; T5.1): the node agent's evaluation bookkeeping, pure (the agent turns
its answers into messages and child commands).

- Evaluators are indexes ``0 .. n-1``; ``evaluator_up`` / ``evaluator_down`` follow the children.
- ``request``: how many more slots to ask the hub for (``BlockLeaseRequest.free_slots``, an increment): idle
  evaluators minus leases waiting for one minus slots already asked; ``on_reconnect`` forgets the asked slots (a
  new session: the hub forgot them too).
- A lease (``on_lease``) waits for the weights of its players (``fetching``; the agent fetches the keys it returns,
  one fetch per key and machine), then for an idle evaluator (``ready`` -> ``assignments``: ``playing``).
- Every block outcome (``BlockOutcome``: results, or a failure kind) stays unacked until the hub's ``BlockAck``
  and is resent after a reconnect; ``on_reconnect`` names the leases held (``Hello.leases``).
- A dead evaluator's lease reports ``crash`` per block, a stopped one's ``abandoned``, a failed fetch
  ``missing_weights``; ``on_cancel`` drops a lease without reporting it (the evaluator stays busy until it answers).
- ``drop_candidates``: evaluation blobs no lease uses, beyond the newest ``EVAL_BLOBS_KEPT``, to remove from the
  machine cache.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from colosseum.core.types import MatchResult
from colosseum.net.messages import BlockTask, PlayerRef
from colosseum.node.weights import CacheKey

__all__ = ["EVAL_BLOBS_KEPT", "BlockOutcome", "NodeLease", "NodeLeases"]

EVAL_BLOBS_KEPT = 8        # unused evaluation blobs kept in the machine cache (newest first)


@dataclass
class NodeLease:
    """One lease as the node holds it: ``fetching`` its weights, ``ready`` for an evaluator, ``playing``."""

    lease_id: str
    source: str
    blocks: list[BlockTask]
    num_envs: int
    deterministic: bool
    state: Literal["fetching", "ready", "playing"] = "fetching"
    evaluator: int | None = None
    missing: set[CacheKey] = field(default_factory=set)

    @property
    def players(self) -> list[PlayerRef]:
        out: dict[tuple[str, str, str], PlayerRef] = {}
        for task in self.blocks:
            for p in task.players:
                out.setdefault((p.agent_id, p.network_id, p.blob.ident if p.blob else ""), p)
        return list(out.values())

    @property
    def keys(self) -> set[CacheKey]:
        return {p.blob for p in self.players if p.blob is not None}


@dataclass(frozen=True)
class BlockOutcome:
    """What the node reports for one block (``BlockResult``): results, or a failure with its kind."""

    lease_id: str
    block_id: str
    results: list[MatchResult] | None = None
    error_kind: str = ""          # abandoned | crash | missing_weights | contract | player | config
    error: str = ""

    @property
    def failed(self) -> bool:
        return self.results is None


def _key_order(key: CacheKey) -> tuple[str, str]:
    return key.agent_id, key.ident


class NodeLeases:
    """The node's evaluation bookkeeping (module docstring)."""

    def __init__(self) -> None:
        self._evaluators: dict[int, str | None] = {}     # up evaluators: index -> lease id playing (None: idle)
        self._leases: dict[str, NodeLease] = {}
        self._unacked: dict[tuple[str, str], BlockOutcome] = {}
        self._asked = 0
        self._eval_keys: OrderedDict[CacheKey, None] = OrderedDict()

    # --- evaluators ---
    def evaluator_up(self, index: int) -> None:
        self._evaluators[index] = None

    def evaluator_down(self, index: int, *, error_kind: str = "crash", error: str = "") -> list[BlockOutcome]:
        """An evaluator exited (``crash``) or is being stopped (``abandoned``): its lease's blocks are released."""
        lease_id = self._evaluators.pop(index, None)
        return [] if lease_id is None else self._abandon(lease_id, error_kind, error)

    def idle(self) -> list[int]:
        return sorted(i for i, lease in self._evaluators.items() if lease is None)

    def up(self) -> list[int]:
        return sorted(self._evaluators)

    # --- leases ---
    def on_lease(self, lease: NodeLease, have: Callable[[CacheKey], bool]) -> list[CacheKey]:
        """A lease from the hub; returns the cache keys to fetch (the agent dedups fetches already running)."""
        self._asked = max(0, self._asked - 1)
        lease.missing = {k for k in lease.keys if not have(k)}
        lease.state = "fetching" if lease.missing else "ready"
        self._leases[lease.lease_id] = lease
        for key in sorted(lease.keys, key=_key_order):
            self._eval_keys[key] = None
            self._eval_keys.move_to_end(key)
        return sorted(lease.missing, key=_key_order)

    def on_fetched(self, key: CacheKey, ok: bool) -> list[BlockOutcome]:
        """A fetch ended: leases waiting for ``key`` become ready (``ok``) or fail with ``missing_weights``."""
        out: list[BlockOutcome] = []
        for lease in list(self._leases.values()):
            if lease.state != "fetching" or key not in lease.missing:
                continue
            if ok:
                lease.missing.discard(key)
                if not lease.missing:
                    lease.state = "ready"
            else:
                out += self._abandon(lease.lease_id, "missing_weights",
                                     f"the weights {key.ident} could not be fetched from the hub")
        return out

    def assignments(self) -> list[tuple[int, NodeLease]]:
        """Ready leases handed to idle evaluators (lowest index first, leases in arrival order)."""
        out: list[tuple[int, NodeLease]] = []
        idle = self.idle()
        for lease in self._leases.values():
            if not idle:
                break
            if lease.state == "ready":
                index = idle.pop(0)
                lease.state, lease.evaluator = "playing", index
                self._evaluators[index] = lease.lease_id
                out.append((index, lease))
        return out

    def on_done(self, index: int, lease_id: str, by_block: Mapping[str, Sequence[MatchResult]],
                cancelled: bool) -> list[BlockOutcome]:
        """An evaluator's ``EvalDone``: one outcome per block of the lease (a block without results: ``crash``)."""
        if self._evaluators.get(index) == lease_id:
            self._evaluators[index] = None
        lease = self._leases.pop(lease_id, None)
        if lease is None or cancelled:
            return []
        out = []
        for task in lease.blocks:
            got = by_block.get(task.block_id)
            outcome = (BlockOutcome(lease_id, task.block_id, list(got)) if got is not None else
                       BlockOutcome(lease_id, task.block_id, None, "crash", "the evaluator returned no results"))
            self._unacked[(lease_id, task.block_id)] = outcome
            out.append(outcome)
        return out

    def on_failed(self, index: int, lease_id: str, kind: str, message: str) -> list[BlockOutcome]:
        """An evaluator's ``EvalFailed``: every block of the lease fails with ``kind``."""
        if self._evaluators.get(index) == lease_id:
            self._evaluators[index] = None
        return self._abandon(lease_id, kind, message)

    def abandon(self, lease_id: str, kind: str, error: str) -> list[BlockOutcome]:
        """Give a lease back with ``kind`` per block (e.g. ``abandoned``: no evaluator will play it)."""
        return self._abandon(lease_id, kind, error)

    def _abandon(self, lease_id: str, kind: str, error: str) -> list[BlockOutcome]:
        lease = self._leases.pop(lease_id, None)
        if lease is None:
            return []
        if lease.evaluator is not None and self._evaluators.get(lease.evaluator) == lease_id:
            self._evaluators[lease.evaluator] = None
        out = [BlockOutcome(lease_id, task.block_id, None, kind, error) for task in lease.blocks]
        for outcome in out:
            self._unacked[(lease_id, outcome.block_id)] = outcome
        return out

    def on_ack(self, lease_id: str, block_ids: Sequence[str]) -> None:
        for block_id in block_ids:
            self._unacked.pop((lease_id, block_id), None)

    def on_cancel(self, lease_ids: Sequence[str]) -> list[tuple[int, str]]:
        """The hub withdrew these leases: ``(evaluator, lease id)`` to send ``EvalCancel`` to; nothing of them is
        reported."""
        out: list[tuple[int, str]] = []
        for lease_id in lease_ids:
            lease = self._leases.pop(lease_id, None)
            if lease is not None and lease.state == "playing" and lease.evaluator is not None:
                out.append((lease.evaluator, lease_id))
            for key in [k for k in self._unacked if k[0] == lease_id]:
                del self._unacked[key]
        return out

    # --- the session ---
    def request(self, *, connected: bool) -> int:
        """Slots to ask the hub for now (an increment); 0 while no session is open."""
        if not connected:
            return 0
        waiting = sum(1 for lease in self._leases.values() if lease.state in ("fetching", "ready"))
        free = len(self.idle()) - waiting - self._asked
        if free <= 0:
            return 0
        self._asked += free
        return free

    def on_reconnect(self) -> tuple[list[str], list[BlockOutcome]]:
        """A new session: the leases to name in ``Hello.leases`` and the outcomes to send again."""
        self._asked = 0
        return self.held(), list(self._unacked.values())

    def held(self) -> list[str]:
        """The leases this node holds (playing or waiting, or with unacked outcomes): ``Hello.leases``."""
        return sorted({*self._leases, *(lease_id for lease_id, _ in self._unacked)})

    @property
    def asked(self) -> int:
        """Slots asked for that no lease answered yet."""
        return self._asked

    def reset_asked(self) -> None:
        """Forget the asked slots (the next ``request`` asks again): the hub caps a node's held plus waiting leases
        at its evaluator slots, so asking again never grants more than the node can play."""
        self._asked = 0

    def missing(self) -> set[CacheKey]:
        """The cache keys leases still wait for (the agent fetches those not in flight)."""
        return {k for lease in self._leases.values() if lease.state == "fetching" for k in lease.missing}

    # --- the machine cache ---
    def keys_in_use(self) -> set[CacheKey]:
        return {k for lease in self._leases.values() for k in lease.keys}

    def drop_candidates(self, busy: Callable[[CacheKey], bool] | None = None) -> list[CacheKey]:
        """Evaluation blobs to remove from the cache: unused by every lease, beyond the newest ``EVAL_BLOBS_KEPT``
        (``busy``: keys kept anyway, e.g. a fetch of them in flight)."""
        used = self.keys_in_use() | {k for k in self._eval_keys if busy is not None and busy(k)}
        unused = [k for k in self._eval_keys if k not in used]
        drop = unused[:max(0, len(unused) - EVAL_BLOBS_KEPT)]
        for key in drop:
            del self._eval_keys[key]
        return drop
