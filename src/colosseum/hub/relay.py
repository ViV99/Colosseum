"""Chunk and weight relay through the hub (SP5 spec 4.8; T3.2). Thread-safe state shared by the hub's gRPC
servicer threads; the hub main loop only publishes which nodes are online (``set_online``) and the version
registry's latest entries (``LatestRelay.publish_versions``), and reads ``stats()`` when it saves its counters (the
book source ``hub/<Hub._incarnation>``, amendment A2).

Chunks. A sender node streams the frames for one target node with ``Hub.RelaySend``; the target node keeps one
``Hub.RelayInbox`` stream open and grants credits (``RELAY_INITIAL_CREDITS`` at the start, then one per frame it
handled). Between the two the hub keeps one bounded buffer per (source, target) pair (``RELAY_PAIR_FRAMES`` frames and
``RELAY_PAIR_BYTES`` bytes; a single frame larger than the byte bound is still admitted into an empty buffer). Frames
are never decoded: the hub forwards the ``ChunkFrame`` message as it came. Flow control is end to end:
- the receiver grants no credit while its learner does not read -> the hub sends nothing -> the pair buffer fills
  -> ``put`` waits -> the RelaySend servicer stops reading -> gRPC flow control stops the sender's stream -> the
  sender's queue fills -> the worker's ``send_bytes`` blocks (the worker loop stalls, as on one machine);
- a frame is dropped only when its target is not online (``disconnect``), when it found no room for
  ``RELAY_PUT_TIMEOUT_SEC`` (the receiver stopped draining: ``overflow``), or when it was sent down an inbox that
  broke before the frame was credited back (``disconnect`` at detach, an upper bound: pre-flight ruling P-PF3-14);
  all counted.
Round-robin over sources keeps one busy sender from starving another.

Weights. ``LatestRelay`` serves ``Hub.Download(kind="latest")`` for learners whose data port other machines cannot
reach: the first request for a version the hub does not hold sends one ``LatestRequest`` (a "pull") down the
learner node's RelayInbox; the learner node answers with ``Hub.Upload(kind="relay_latest")`` (handled in the servicer:
no hub state is touched, no ``UploadIn``); every waiting and later download of that version is served from the one
uploaded file (one upload per version, whatever the number of fetching nodes). The newest file per agent is kept
(``runs/<name>/blobs/relay/latest-<agent digest>-v<N>.frame``: no agent id in a file name, as every blob of the hub),
the previous one deleted after the replace (open readers keep their file: POSIX).
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

__all__ = [
    "RELAY_INITIAL_CREDITS", "RELAY_PAIR_BYTES", "RELAY_PAIR_FRAMES", "RELAY_PULL_WAIT_SEC", "RELAY_PUT_TIMEOUT_SEC",
    "InboxHandle", "LatestRelay", "RelayHub", "RelayedVersion",
]

RELAY_PAIR_FRAMES = 64
RELAY_PAIR_BYTES = 64 << 20
RELAY_PUT_TIMEOUT_SEC = 10.0
RELAY_PULL_WAIT_SEC = 10.0
RELAY_INITIAL_CREDITS = 16              # what a receiver grants when its inbox opens (node.router)
_WAIT_SLICE_SEC = 0.25


@dataclass
class _Pair:
    items: collections.deque = field(default_factory=collections.deque)   # (item, nbytes)
    nbytes: int = 0


@dataclass(eq=False)
class InboxHandle:
    """One RelayInbox stream of a target node (``RelayHub.attach``). ``sent``: frames handed to it; ``returned``:
    credits granted after the initial ones (one per frame the receiver handled); their difference at detach is what
    the stream lost (P-PF3-14)."""

    dest: int
    credits: int = 0
    detached: bool = False
    sent: int = 0
    returned: int = 0
    accounted: bool = False


@dataclass
class _Dest:
    """One target node: its pair buffers, its attached inbox, pending control items."""

    cond: threading.Condition
    pairs: dict[int, _Pair] = field(default_factory=dict)
    order: collections.deque = field(default_factory=collections.deque)   # round-robin over source numbers
    inbox: InboxHandle | None = None
    controls: collections.deque = field(default_factory=collections.deque)


class RelayHub:
    """Pair buffers between RelaySend and RelayInbox streams (module docstring). ``enabled`` False:
    ``distributed.relay: false`` (the servicers refuse the relay streams)."""

    def __init__(self, *, enabled: bool = True, pair_frames: int = RELAY_PAIR_FRAMES,
                 pair_bytes: int = RELAY_PAIR_BYTES, put_timeout_sec: float = RELAY_PUT_TIMEOUT_SEC,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.enabled = bool(enabled)
        self._pair_frames = int(pair_frames)
        self._pair_bytes = int(pair_bytes)
        self._put_timeout = float(put_timeout_sec)
        self._clock = clock
        self._lock = threading.Lock()                  # guards _dests, _online, _numbers and the counters
        self._dests: dict[int, _Dest] = {}
        self._online: set[int] = set()
        self._numbers: dict[str, int] = {}
        self._counters: collections.Counter = collections.Counter()

    # --- main loop -----------------------------------------------------------------------------------------------

    def set_online(self, numbers: Mapping[str, int]) -> None:
        """``node_id -> number`` of the online nodes. Buffers towards nodes that left are dropped (``disconnect``)
        and their inboxes detached (their frames in flight counted too)."""
        with self._lock:
            self._numbers = dict(numbers)
            self._online = set(numbers.values())
            gone = [d for d in self._dests if d not in self._online]
        for number in gone:
            dest = self._dest(number)
            with dest.cond:
                dropped = sum(len(p.items) for p in dest.pairs.values())
                dest.pairs.clear()
                dest.order.clear()
                dest.controls.clear()
                if dest.inbox is not None:
                    dropped += self._retire(dest.inbox)
                    dest.inbox = None
                dest.cond.notify_all()
            self._count("dropped_chunks.disconnect", dropped)

    def stats(self) -> dict[str, int]:
        """Cumulative counters of this hub process: ``relay_bytes``, ``relay_frames``,
        ``dropped_chunks.overflow``, ``dropped_chunks.disconnect`` (fed into the hub's CounterBook)."""
        with self._lock:
            return dict(self._counters)

    def number_of(self, node_id: str) -> int | None:
        with self._lock:
            return self._numbers.get(node_id)

    # --- sender side (RelaySend servicer threads) -------------------------------------------------------------------

    def put(self, src: int, dst: int, item: Any, nbytes: int, *,
            should_stop: Callable[[], bool] = lambda: False) -> str | None:
        """Queue one frame from node ``src`` for node ``dst``. Blocks while the pair buffer is full (backpressure).
        Returns None when queued, else the drop reason (counted): ``disconnect`` (``dst`` not online, or the
        stream ended while waiting) or ``overflow`` (no room for ``RELAY_PUT_TIMEOUT_SEC``)."""
        with self._lock:
            online = dst in self._online
        if not online:
            self._count("dropped_chunks.disconnect")
            return "disconnect"
        dest = self._dest(dst)
        deadline = self._clock() + self._put_timeout
        with dest.cond:
            while True:
                with self._lock:
                    online = dst in self._online
                if not online or should_stop():
                    reason = "disconnect"
                    break
                pair = dest.pairs.get(src)
                if pair is None:
                    pair = dest.pairs[src] = _Pair()
                    dest.order.append(src)
                if not pair.items or (len(pair.items) < self._pair_frames
                                      and pair.nbytes + nbytes <= self._pair_bytes):
                    pair.items.append((item, int(nbytes)))
                    pair.nbytes += int(nbytes)
                    dest.cond.notify_all()
                    return None
                remaining = deadline - self._clock()
                if remaining <= 0:
                    reason = "overflow"
                    break
                dest.cond.wait(min(remaining, _WAIT_SLICE_SEC))
        self._count(f"dropped_chunks.{reason}")
        return reason

    # --- receiver side (RelayInbox servicer threads) ----------------------------------------------------------------

    def attach(self, dst: int, initial_credits: int = 0) -> InboxHandle:
        """A RelayInbox stream of node ``dst`` opened with ``initial_credits``; an older stream of the same node is
        detached (it ends; its frames in flight are counted)."""
        dest = self._dest(dst)
        handle = InboxHandle(dst, credits=max(0, int(initial_credits)))
        lost = 0
        with dest.cond:
            if dest.inbox is not None:
                lost = self._retire(dest.inbox)
            dest.inbox = handle
            dest.cond.notify_all()
        self._count("dropped_chunks.disconnect", lost)
        return handle

    def detach(self, handle: InboxHandle) -> None:
        dest = self._dest(handle.dest)
        with dest.cond:
            lost = self._retire(handle)
            if dest.inbox is handle:
                dest.inbox = None
            dest.cond.notify_all()
        self._count("dropped_chunks.disconnect", lost)

    def add_credits(self, handle: InboxHandle, frames: int) -> None:
        """Credits a receiver returned (one per frame it handled)."""
        if frames <= 0:
            return
        dest = self._dest(handle.dest)
        with dest.cond:
            handle.credits += int(frames)
            handle.returned += int(frames)
            dest.cond.notify_all()

    def push_control(self, dst: int, item: Any) -> bool:
        """A control item (a weights pull) for node ``dst``; sent before chunks and without credit. False when the
        node has no inbox attached."""
        dest = self._dest(dst)
        with dest.cond:
            if dest.inbox is None:
                return False
            dest.controls.append(item)
            dest.cond.notify_all()
            return True

    def next_item(self, handle: InboxHandle, timeout: float) -> tuple[str, Any] | None:
        """The next item for ``handle``'s node: ``("control", item)`` or ``("chunk", item)`` (one credit spent);
        None after ``timeout`` or when the handle was detached (check ``handle.detached``)."""
        dest = self._dest(handle.dest)
        deadline = self._clock() + timeout
        with dest.cond:
            while not handle.detached:
                if dest.controls:
                    return "control", dest.controls.popleft()
                if handle.credits > 0:
                    picked = self._pop_round_robin(dest)
                    if picked is not None:
                        item, nbytes = picked
                        handle.credits -= 1
                        handle.sent += 1
                        dest.cond.notify_all()        # room for a waiting put
                        self._count("relay_frames")
                        self._count("relay_bytes", nbytes)
                        return "chunk", item
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return None
                dest.cond.wait(min(remaining, _WAIT_SLICE_SEC))
        return None

    # --- internals -----------------------------------------------------------------------------------------------

    @staticmethod
    def _retire(handle: InboxHandle) -> int:
        """Mark ``handle`` detached (under its destination's condition); the frames it lost, counted once."""
        handle.detached = True
        if handle.accounted:
            return 0
        handle.accounted = True
        return max(0, handle.sent - handle.returned)

    def _dest(self, number: int) -> _Dest:
        with self._lock:
            dest = self._dests.get(number)
            if dest is None:
                dest = self._dests[number] = _Dest(threading.Condition(threading.Lock()))
            return dest

    @staticmethod
    def _pop_round_robin(dest: _Dest) -> tuple[Any, int] | None:
        for _ in range(len(dest.order)):
            src = dest.order[0]
            dest.order.rotate(-1)
            pair = dest.pairs[src]
            if pair.items:
                item, nbytes = pair.items.popleft()
                pair.nbytes -= nbytes
                return item, nbytes
        return None

    def _count(self, name: str, n: int = 1) -> None:
        if n:
            with self._lock:
                self._counters[name] += n


@dataclass(frozen=True)
class RelayedVersion:
    agent_id: str
    version: int
    generation: int
    path: Path
    teacher_active: bool = False
    size: int = 0
    digest: str = ""


def relay_file_name(agent_id: str, version: int) -> str:
    """``latest-<32 hex of the agent id's sha256>-v<N>.frame`` (no agent id in a file name)."""
    return f"latest-{hashlib.sha256(agent_id.encode()).hexdigest()[:32]}-v{int(version)}.frame"


class LatestRelay:
    """Latest weights of relay-only learners, uploaded once per version on demand (module docstring). ``stats()``:
    ``relay_uploads`` / ``relay_uploads_refused`` (completed uploads) and ``relay_bytes`` of relayed downloads."""

    def __init__(self, root: Path, relay: RelayHub, *, wait_sec: float = RELAY_PULL_WAIT_SEC,
                 make_pull: Callable[[str, int, str], Any] = lambda agent, since, rid: (agent, since, rid),
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._relay = relay
        self._wait = float(wait_sec)
        self._make_pull = make_pull
        self._clock = clock
        self._cond = threading.Condition()
        self._versions: dict[str, Any] = {}               # agent -> VersionInfo (registry latest)
        self._cached: dict[str, RelayedVersion] = {}
        self._pending: dict[str, tuple[int, float]] = {}  # agent -> (version asked, when)
        self._ids = itertools.count(1)
        self._counters: collections.Counter = collections.Counter()

    def publish_versions(self, versions: Mapping[str, Any]) -> None:
        """Main loop: the registry's latest ``VersionInfo`` per agent (after every accepted announcement)."""
        with self._cond:
            self._versions = dict(versions)
            self._cond.notify_all()

    def latest(self, agent_id: str) -> Any | None:
        with self._cond:
            return self._versions.get(agent_id)

    def stats(self) -> dict[str, int]:
        with self._cond:
            return dict(self._counters)

    def count_bytes(self, n: int) -> None:
        """Bytes of a relayed latest download (``relay_bytes``)."""
        with self._cond:
            self._counters["relay_bytes"] += int(n)

    def open_newer(self, agent_id: str, since_version: int, *,
                   timeout: float | None = None) -> tuple[RelayedVersion, BinaryIO] | None:
        """An open file of the newest relayed version above ``since_version``, pulling it from the learner node
        when the hub holds none (one pull in flight per agent; a pull unanswered for ``wait_sec`` is resent).
        None after ``timeout`` (default ``wait_sec``), or at once when the learner node has no relay inbox."""
        timeout = self._wait if timeout is None else float(timeout)
        deadline = self._clock() + timeout
        with self._cond:
            while True:
                cached = self._cached.get(agent_id)
                if cached is not None and cached.version > since_version:
                    try:
                        return cached, open(cached.path, "rb")
                    except FileNotFoundError:     # replaced between the lookup and the open: look again
                        continue
                info = self._versions.get(agent_id)
                if info is None or info.version <= since_version:
                    return None
                now = self._clock()
                pending = self._pending.get(agent_id)
                if pending is None or pending[0] < info.version or now - pending[1] >= self._wait:
                    pull = self._make_pull(agent_id, cached.version if cached else -1, f"pull-{next(self._ids)}")
                    if self._relay.push_control(info.node, pull):
                        self._pending[agent_id] = (info.version, now)
                    else:
                        return None                   # the learner node has no relay inbox: nothing to wait for
                remaining = deadline - now
                if remaining <= 0:
                    return None
                self._cond.wait(min(remaining, _WAIT_SLICE_SEC))

    def complete(self, agent_id: str, version: int, generation: int, spooled: Path, *,
                 teacher_active: bool = False, digest: str = "") -> str:
        """An upload of ``agent_id`` v``version`` arrived (digest and frame header already checked;
        ``teacher_active`` read from the header). ``accepted`` (it becomes the served file), ``not_newer`` or
        ``stale_generation`` (the file is deleted)."""
        with self._cond:
            info = self._versions.get(agent_id)
            cached = self._cached.get(agent_id)
            if info is not None and generation != info.generation:
                reason = "stale_generation"
            elif cached is not None and version <= cached.version:
                reason = "not_newer"
            else:
                final = self._root / relay_file_name(agent_id, version)
                size = os.path.getsize(spooled)
                os.replace(spooled, final)
                self._cached[agent_id] = RelayedVersion(agent_id, int(version), int(generation), final,
                                                        bool(teacher_active), int(size), str(digest))
                if cached is not None and cached.path != final:
                    cached.path.unlink(missing_ok=True)
                pending = self._pending.get(agent_id)
                if pending is not None and pending[0] <= version:
                    self._pending.pop(agent_id, None)
                self._counters["relay_uploads"] += 1
                self._cond.notify_all()
                return "accepted"
            self._counters["relay_uploads_refused"] += 1
        spooled.unlink(missing_ok=True)
        return reason

    def count_refused(self) -> None:
        """An upload refused before ``complete`` (not the learner node, a wrong digest or header)."""
        with self._cond:
            self._counters["relay_uploads_refused"] += 1
