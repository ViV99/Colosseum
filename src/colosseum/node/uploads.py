"""Checkpoint uploads of a node (SP5 spec 5.2, 7.4; task T4.2): which learner checkpoint frame goes to the hub next.

A learner hands its node three kinds of checkpoint frames (``CheckpointHeader`` + frame on its control connection);
the node uploads each with its own ``Hub/Upload`` call (never on the session, never through a bounded queue that
could drop a final snapshot: spec 7.4). Per agent:

- ``final``: one per learner generation; never replaced, never dropped by the queue; a failed upload is retried
  until it is delivered or the caller gives up at the stop deadline (``drop_waiting``);
- ``recovery``: at most one waiting; a newer one replaces it (only the newest recovery point matters);
- ``periodic``: at most one waiting; a newer one replaces it and the replaced one is counted in ``replaced`` (that
  snapshot never reaches the pool: the hub is slower than the learner's ``checkpoint.interval``).

At most one upload per agent is in flight (frames of one agent arrive in order) and at most ``concurrency`` per
node. Order: finals, then recovery points, then periodic snapshots; within a kind the oldest offer first. A failed
upload (the hub is away, the call broke) waits again, unless a newer frame of its kind arrived meanwhile (then it
is dropped; a periodic one is counted). A refused upload (the hub answered ``accepted: false``) is never retried.
Pure and single-threaded: the node's main loop owns it, the transfer pool runs the calls and posts the outcome.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

__all__ = ["FINAL", "PERIODIC", "RECOVERY", "UPLOAD_CONCURRENCY", "Upload", "UploadQueue"]

FINAL, RECOVERY, PERIODIC = "final", "recovery", "periodic"
_PRIORITY = {FINAL: 0, RECOVERY: 1, PERIODIC: 2}
UPLOAD_CONCURRENCY = 2


@dataclass(frozen=True)
class Upload:
    ticket: int
    agent_id: str
    kind: str                 # final | recovery | periodic
    generation: int
    version: int
    frame: bytes


class UploadQueue:
    """The node's checkpoint uploads (module docstring)."""

    def __init__(self, concurrency: int = UPLOAD_CONCURRENCY) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._concurrency = int(concurrency)
        self._next = 1
        self._finals: dict[str, deque[Upload]] = {}
        self._latest: dict[tuple[str, str], Upload] = {}
        self._flight: dict[int, Upload] = {}
        self.replaced = 0          # periodic snapshots that never left the node (replaced by a newer one)

    def offer(self, agent_id: str, kind: str, generation: int, version: int, frame: bytes) -> int:
        """Queue a frame; returns its ticket."""
        if kind not in _PRIORITY:
            raise ValueError(f"unknown checkpoint kind {kind!r}")
        upload = Upload(self._next, agent_id, kind, int(generation), int(version), frame)
        self._next += 1
        if kind == FINAL:
            self._finals.setdefault(agent_id, deque()).append(upload)
        else:
            if (agent_id, kind) in self._latest and kind == PERIODIC:
                self.replaced += 1
            self._latest[(agent_id, kind)] = upload
        return upload.ticket

    def take(self) -> list[Upload]:
        """The uploads to start now (marked in flight)."""
        busy = {u.agent_id for u in self._flight.values()}
        waiting = [q[0] for q in self._finals.values() if q] + list(self._latest.values())
        waiting.sort(key=lambda u: (_PRIORITY[u.kind], u.ticket))
        started = []
        for upload in waiting:
            if len(self._flight) >= self._concurrency:
                break
            if upload.agent_id in busy:
                continue
            if upload.kind == FINAL:
                self._finals[upload.agent_id].popleft()
            else:
                del self._latest[(upload.agent_id, upload.kind)]
            self._flight[upload.ticket] = upload
            busy.add(upload.agent_id)
            started.append(upload)
        return started

    def done(self, ticket: int, ok: bool, *, retry: bool = True) -> Upload | None:
        """The outcome of an upload: ``ok`` (delivered and accepted), or failed (``retry``: the call broke or the
        hub is away; ``retry=False``: the hub refused it). Returns the upload (None for an unknown ticket)."""
        upload = self._flight.pop(ticket, None)
        if upload is None or ok or not retry:
            return upload
        if upload.kind == FINAL:
            self._finals.setdefault(upload.agent_id, deque()).appendleft(upload)
        elif (upload.agent_id, upload.kind) not in self._latest:
            self._latest[(upload.agent_id, upload.kind)] = upload
        elif upload.kind == PERIODIC:
            self.replaced += 1
        return upload

    def waiting(self, kind: str | None = None) -> int:
        """Frames not yet delivered (waiting or in flight), of ``kind`` or all."""
        items = [*self._flight.values(), *self._latest.values(), *(u for q in self._finals.values() for u in q)]
        return sum(1 for u in items if kind is None or u.kind == kind)

    def in_flight(self) -> list[Upload]:
        """The uploads started and not yet done."""
        return list(self._flight.values())

    def drop_waiting(self) -> list[Upload]:
        """Give up on everything not in flight (the stop deadline passed); returns what was dropped."""
        dropped = [u for q in self._finals.values() for u in q] + list(self._latest.values())
        self._finals.clear()
        self._latest.clear()
        return dropped

    def clear(self) -> None:
        """Forget everything (the node released its run)."""
        self._finals.clear()
        self._latest.clear()
        self._flight.clear()
