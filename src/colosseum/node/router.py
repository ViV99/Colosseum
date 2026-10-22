"""Chunk routing between machines (SP5 spec 4.7, 4.8; T3.2): node side.

- ``RouteTable`` (thread-safe snapshot, written by the node main loop): agent -> ``RemoteRoute`` (target node,
  generation, data addresses, relay-only) for learners on other nodes.
- ``RouteReader`` threads (one per worker route connection, ``net.local``): ``recv_bytes`` a frame (counted
  ``chunks_routed``), ``peek_meta`` for its agent (a malformed header: ``invalid``), put it on the target's
  ``TargetSender`` queue (blocking: the worker's ``send_bytes`` then blocks, spec 4.7 backpressure); no route ->
  dropped ``no_learner``. The frame is never decoded.
- ``TargetSender`` threads (one per target node, started by ``ChunkRouter.set_routes`` as soon as the target is known,
  before any frame: the node reports a path in ``Heartbeat.paths`` before the hub grants it collecting seats,
  pre-flight ruling P-PF3-1): path choice — direct ``NodeData.Chunks`` when one announced data address answers within
  ``connect_timeout_sec`` (the channel's TLS server name names the target node, so only that node's certificate is
  accepted), else ``Hub.RelaySend`` when the relay is available, else ``down`` (frames are dropped and counted at
  once, never queued: a worker never waits for a path that does not exist). Frames are forwarded undecoded in
  ``ChunkFrame`` messages; a stream is rotated every ``STREAM_ROTATE_FRAMES`` frames or ``STREAM_ROTATE_SEC`` seconds so
  its ``ChunkSummary`` confirms what arrived. When a stream fails, the frames handed to it since its start are counted
  ``disconnect`` (an upper bound: some may have arrived; ruling W03-1), the sender waits a backoff (``RETRY_MIN_SEC``
  doubling to ``RETRY_MAX_SEC``; a stream that carried a frame resets it; frames that wait meanwhile stay queued:
  backpressure, not loss; P-PF3-7) and chooses the path again (a failed direct path is probed again after
  ``DIRECT_RETRY_SEC`` while the relay carries the frames, at every attempt without a relay).
- ``LocalDelivery`` (receiving side, used by the data-port ``Chunks`` servicer; T3.2 (b) adds the relay receiver):
  the frame goes to the local learner's chunk connection with ``send_bytes`` under that connection's lock (blocking
  while the learner does not read: gRPC flow control then holds the sending node back); no learner -> ``no_learner``,
  a broken connection -> ``disconnect`` (its entry is removed; the node opens a new one, at most once a second). A
  replaced connection is retired, never closed under a writer (P-PF3-6).

Counters are cumulative for the router (the node's run; ``counters()``; the node sends them in ``Heartbeat.counters``,
amendment A2 names): ``chunks_routed``, ``chunks_sent.direct``, ``chunks_sent.relay``, ``chunks_delivered``,
``dropped_chunks.{disconnect,no_learner,invalid}``.

Seam for T3.2 (b): ``relay_call`` (None until the hub relay exists, or with ``distributed.relay: false``) and the
relay receiver, which hands relayed frames to the same ``LocalDelivery``.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import grpc

from colosseum.net.frame import FrameError, peek_meta

logger = logging.getLogger(__name__)

__all__ = [
    "DIRECT_RETRY_SEC", "PATHS", "RETRY_MAX_SEC", "RETRY_MIN_SEC", "SENDER_QUEUE_FRAMES", "STREAM_ROTATE_FRAMES",
    "STREAM_ROTATE_SEC", "ChunkRouter", "Counters", "LocalDelivery", "RemoteRoute", "RouteReader", "RouteTable",
    "TargetSender",
]

SENDER_QUEUE_FRAMES = 32
STREAM_ROTATE_FRAMES = 256
STREAM_ROTATE_SEC = 2.0
DIRECT_RETRY_SEC = 30.0
RETRY_MIN_SEC, RETRY_MAX_SEC = 0.5, 10.0
_POLL_SEC = 0.1
PATHS = ("direct", "relay", "down", "none")    # "none": no route to the target (any more)


class Counters:
    """Thread-safe cumulative counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c: collections.Counter = collections.Counter()

    def add(self, name: str, n: int = 1) -> None:
        if n:
            with self._lock:
                self._c[name] += n

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._c)


@dataclass(frozen=True)
class RemoteRoute:
    node: int
    generation: int
    addresses: tuple[str, ...]
    relay_only: bool


class RouteTable:
    """agent -> RemoteRoute for learners on other nodes (immutable snapshots swapped under a lock)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._remote: dict[str, RemoteRoute] = {}

    def set_remote(self, routes: dict[str, RemoteRoute]) -> None:
        with self._lock:
            self._remote = dict(routes)

    def remote(self, agent_id: str) -> RemoteRoute | None:
        with self._lock:
            return self._remote.get(agent_id)

    def targets(self) -> dict[int, RemoteRoute]:
        with self._lock:
            return {r.node: r for r in self._remote.values()}


@dataclass(eq=False)
class _Stream:
    lock: threading.Lock = field(default_factory=threading.Lock)
    n: int = 0
    closed: bool = False


class TargetSender(threading.Thread):
    """Frames for one target node (module docstring). ``make_channel(address)`` opens a data channel to the target
    (None: this node cannot reach data ports, e.g. no node certificate); ``direct_stub(channel)`` and ``relay_call()``
    return a callable ``call(request_iterator) -> ChunkSummary`` (``NodeData.Chunks`` / ``Hub.RelaySend``, a
    client-streaming RPC); ``relay_call`` None: no relay. ``on_path(target, path)`` is called from this thread when the
    path changes (``PATHS``); the node posts it into its inbox."""

    def __init__(self, target: int, *, route: Callable[[], RemoteRoute | None], make_frame: Callable[..., Any],
                 make_channel: Callable[[str], grpc.Channel] | None,
                 direct_stub: Callable[[grpc.Channel], Callable] | None, relay_call: Callable[[], Callable] | None,
                 connect_timeout_sec: float, counters: Counters,
                 on_path: Callable[[int, str], None] = lambda t, p: None) -> None:
        super().__init__(name=f"chunks-to-n{target}", daemon=True)
        self.target = target
        self.queue: queue.Queue = queue.Queue(maxsize=SENDER_QUEUE_FRAMES)
        self._route = route
        self._make_frame = make_frame
        self._make_channel = make_channel
        self._direct_stub = direct_stub
        self._relay_call = relay_call
        self._timeout = float(connect_timeout_sec)
        self._counters = counters
        self._on_path = on_path
        self._halt = threading.Event()          # not "_stop": that shadows Thread._stop and breaks join
        self._channel: grpc.Channel | None = None
        self._direct_failed_at = -float("inf")
        self.path = "none"

    def stop(self) -> None:
        self._halt.set()

    def put(self, item: tuple, should_stop: Callable[[], bool]) -> bool:
        """Blocking put (backpressure); a frame for a ``down`` path is dropped and counted at once. False when
        stopping (the caller counts the frame)."""
        while not should_stop() and not self._halt.is_set():
            if self.path == "down":
                self._counters.add("dropped_chunks.disconnect")
                return True
            try:
                self.queue.put(item, timeout=_POLL_SEC)
                return True
            except queue.Full:
                continue
        return False

    def _choose(self) -> tuple[str, Callable | None, RemoteRoute | None]:
        """direct when an announced address answers within ``connect_timeout_sec``; else relay (when available); else
        down. While on the relay, the direct path is probed again only every ``DIRECT_RETRY_SEC``; without a relay it
        is probed on every attempt (``run`` backs off between attempts)."""
        from colosseum.node.dataport import probe_direct

        route = self._route()
        if route is None:
            return "none", None, None
        now = time.monotonic()
        direct_possible = (bool(route.addresses) and not route.relay_only and self._make_channel is not None
                           and self._direct_stub is not None)
        recently_failed = now - self._direct_failed_at < DIRECT_RETRY_SEC
        if direct_possible and not (recently_failed and self._relay_call is not None):
            found = probe_direct(route.addresses, self._make_channel, self._timeout)
            if found is not None:
                self._channel = found[1]
                return "direct", self._direct_stub(found[1]), route
            self._direct_failed_at = now
        if self._relay_call is not None:
            return "relay", self._relay_call(), route
        return "down", None, route

    def _set_path(self, path: str) -> None:
        if path != self.path:
            self.path = path
            try:
                self._on_path(self.target, path)
            except Exception:  # noqa: BLE001 - a callback bug never stops the sender
                logger.exception(f"chunks to node {self.target}: the path callback failed")

    def _frames(self, stream: _Stream) -> Iterator[Any]:
        """Request iterator of one stream. gRPC pulls it from its own thread, which may still be inside ``next``
        when the call has already failed: a frame taken after ``stream.closed`` is counted here, every frame
        counted in ``stream.n`` is counted by ``run`` (under ``stream.lock``), so none is lost uncounted."""
        started = time.monotonic()
        while not self._halt.is_set():
            if stream.n >= STREAM_ROTATE_FRAMES or time.monotonic() - started >= STREAM_ROTATE_SEC:
                return
            try:
                item = self.queue.get(timeout=_POLL_SEC)
            except queue.Empty:
                continue
            with stream.lock:
                if stream.closed:
                    self._counters.add("dropped_chunks.disconnect")
                    return
                stream.n += 1
            yield self._make_frame(*item)

    def _same_route(self, chosen: RemoteRoute | None) -> bool:
        now = self._route()
        return now is not None and chosen is not None and (now.addresses, now.relay_only) == (chosen.addresses,
                                                                                              chosen.relay_only)

    def run(self) -> None:
        backoff = RETRY_MIN_SEC
        while not self._halt.is_set():
            path, call, chosen = self._choose()
            self._set_path(path)
            if call is None:                       # no route or path down: drop what waits, try again later
                self._drop_waiting("disconnect" if path == "down" else "no_learner")
                self._halt.wait(backoff)
                backoff = min(RETRY_MAX_SEC, backoff * 2)
                continue
            while not self._halt.is_set():
                stream = _Stream()
                try:
                    call(self._frames(stream))
                except Exception as e:  # noqa: BLE001 - grpc.RpcError, or a failure of the call itself
                    with stream.lock:
                        stream.closed = True
                        lost = stream.n
                    self._counters.add("dropped_chunks.disconnect", lost)
                    code = e.code() if isinstance(e, grpc.RpcError) and hasattr(e, "code") else type(e).__name__
                    logger.info(f"chunk stream to node {self.target} ({path}) failed: {code}; {lost} frames "
                                f"counted as lost; retrying in {backoff:.1f} s")
                    if path == "direct":
                        self._direct_failed_at = time.monotonic()
                        self._close_channel()
                    self._halt.wait(backoff)       # P-PF3-7: never a tight retry loop; frames stay queued
                    backoff = min(RETRY_MAX_SEC, backoff * 2)
                    break
                with stream.lock:
                    stream.closed = True
                    sent = stream.n
                self._counters.add(f"chunks_sent.{path}", sent)
                if sent:
                    backoff = RETRY_MIN_SEC
                if path == "relay" and time.monotonic() - self._direct_failed_at >= DIRECT_RETRY_SEC:
                    break                          # time to try the direct path again
                if not self._same_route(chosen):
                    break                          # the target moved (new addresses) or left
            if path == "direct" and not self._halt.is_set():
                self._close_channel()
        self._close_channel()
        self._drop_waiting("disconnect")

    def _close_channel(self) -> None:
        channel, self._channel = self._channel, None
        if channel is not None:
            channel.close()

    def _drop_waiting(self, reason: str) -> None:
        n = 0
        while True:
            try:
                self.queue.get_nowait()
                n += 1
            except queue.Empty:
                break
        self._counters.add(f"dropped_chunks.{reason}", n)


class LocalDelivery:
    """Frames from other machines to this node's learners (module docstring). ``set_learner`` is called by the node
    main loop; ``deliver`` by servicer threads (several at once)."""

    def __init__(self, counters: Counters) -> None:
        self._lock = threading.Lock()
        self._learners: dict[str, tuple[Any, threading.Lock]] = {}
        self._counters = counters

    def set_learner(self, agent_id: str, conn: Any | None) -> None:
        """The chunk connection to ``agent_id``'s local learner (None: no learner). A replaced connection is retired:
        a short thread closes it once its writer released it (a blocked write ends when the old learner is gone)."""
        with self._lock:
            old = self._learners.pop(agent_id, None)
            if conn is not None:
                self._learners[agent_id] = (conn, threading.Lock())
        if old is not None and old[0] is not conn:
            _retire(old)

    def connected(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._learners

    def clear(self) -> None:
        with self._lock:
            old, self._learners = list(self._learners.values()), {}
        for entry in old:
            _retire(entry)

    def count(self, name: str, n: int = 1) -> None:
        self._counters.add(name, n)

    def deliver(self, agent_id: str, frame: bytes) -> str | None:
        """Hand ``frame`` to the learner (blocking while it does not read). None when written, else the drop reason
        (counted)."""
        for _ in range(2):                         # a second look when the entry was retired before the write
            with self._lock:
                entry = self._learners.get(agent_id)
            if entry is None:
                self._counters.add("dropped_chunks.no_learner")
                return "no_learner"
            conn, lock = entry
            retired = False
            try:
                with lock:
                    retired = conn.closed          # closed by its retirement while this thread waited for the lock
                    if not retired:
                        conn.send_bytes(frame)
            except (OSError, EOFError, ValueError, TypeError):   # TypeError: a handle closed under the write
                with self._lock:
                    if self._learners.get(agent_id) is entry:
                        del self._learners[agent_id]
                        _retire(entry)
                self._counters.add("dropped_chunks.disconnect")
                return "disconnect"
            if retired:
                with self._lock:
                    current = self._learners.get(agent_id)
                    if current is entry:            # closed, not replaced: broken
                        del self._learners[agent_id]
                if current is not entry:
                    continue
                self._counters.add("dropped_chunks.disconnect")
                return "disconnect"
            self._counters.add("chunks_delivered")
            return None
        self._counters.add("dropped_chunks.disconnect")
        return "disconnect"


def _retire(entry: tuple[Any, threading.Lock]) -> None:
    """Close a learner connection once no writer holds it (P-PF3-6), off the caller's thread."""
    conn, lock = entry

    def close() -> None:
        with lock:
            try:
                conn.close()
            except OSError:
                pass

    threading.Thread(target=close, name="retire-learner-conn", daemon=True).start()


class RouteReader(threading.Thread):
    """Frames from one worker's route connection (``net.local``) to the target senders (module docstring)."""

    def __init__(self, conn: Any, *, worker: int, routes: RouteTable, sender_for: Callable[[int], TargetSender | None],
                 counters: Counters, stop: threading.Event, max_frame_bytes: int | None = None) -> None:
        super().__init__(name=f"route-w{worker}", daemon=True)
        self._conn = conn
        self._routes = routes
        self._sender_for = sender_for
        self._counters = counters
        self._halt = stop
        self._max = max_frame_bytes

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                if not self._conn.poll(_POLL_SEC):
                    continue
                frame = self._conn.recv_bytes(self._max) if self._max else self._conn.recv_bytes()
            except (EOFError, OSError):
                break                                   # the worker is gone (or sent an oversized frame)
            self._counters.add("chunks_routed")
            try:
                _kind, meta = peek_meta(frame)
                agent_id = str(meta["agent_id"])
                generation = int(meta.get("generation", 0))
            except (FrameError, KeyError, TypeError, ValueError, AttributeError):
                self._counters.add("dropped_chunks.invalid")
                continue
            route = self._routes.remote(agent_id)
            sender = None if route is None else self._sender_for(route.node)
            if sender is None:
                self._counters.add("dropped_chunks.no_learner")
                continue
            if not sender.put((agent_id, generation, frame), self._halt.is_set):
                self._counters.add("dropped_chunks.disconnect")
        try:
            self._conn.close()
        except OSError:
            pass


class ChunkRouter:
    """The node's outbound chunk path and its delivery to local learners (module docstring). The main loop calls
    ``set_routes`` whenever the version registry or this node's learners change, ``add_worker`` for every accepted
    worker route connection, and reads ``paths()`` / ``counters()`` for the heartbeat; ``stop`` at the end.
    ``make_frame(agent, generation, frame, target)`` builds the ``ChunkFrame``; ``make_channel(address, target)`` opens
    a data channel to ``target``'s data port (None: this node cannot); ``direct_stub(channel)`` / ``relay_call()``
    return the client-streaming callables (``relay_call`` None: no relay)."""

    def __init__(self, *, make_frame: Callable[..., Any], make_channel: Callable[[str, int], grpc.Channel] | None,
                 direct_stub: Callable[[grpc.Channel], Callable] | None, relay_call: Callable[[], Callable] | None,
                 connect_timeout_sec: float, on_path: Callable[[int, str], None] = lambda t, p: None,
                 max_frame_bytes: int | None = None) -> None:
        self.routes = RouteTable()
        self.counters_obj = Counters()
        self.delivery = LocalDelivery(self.counters_obj)
        self._make_frame = make_frame
        self._make_channel = make_channel
        self._direct_stub = direct_stub
        self._relay_call = relay_call
        self._timeout = float(connect_timeout_sec)
        self._on_path = on_path
        self._max_frame = max_frame_bytes
        self._lock = threading.Lock()
        self._senders: dict[int, TargetSender] = {}
        self._readers: list[RouteReader] = []
        self._halt = threading.Event()

    def set_routes(self, remote: dict[str, RemoteRoute]) -> None:
        """The learners on other nodes; a sender starts (and probes its path) for every new target at once (P-PF3-1),
        the sender of a target no route names any more stops."""
        self.routes.set_remote(remote)
        targets = {r.node for r in remote.values()}
        with self._lock:
            gone = [t for t in self._senders if t not in targets]
            stopped = [self._senders.pop(t) for t in gone]
        for sender in stopped:
            sender.stop()
        for target in sorted(targets):
            self.sender_for(target)

    def sender_for(self, target: int) -> TargetSender | None:
        if self._halt.is_set():
            return None
        with self._lock:
            sender = self._senders.get(target)
            if sender is None:
                if target not in self.routes.targets():
                    return None
                make_channel = (None if self._make_channel is None
                                else lambda address, t=target: self._make_channel(address, t))
                sender = TargetSender(
                    target, route=lambda t=target: self.routes.targets().get(t),
                    make_frame=lambda agent, gen, frame, t=target: self._make_frame(agent, gen, frame, t),
                    make_channel=make_channel, direct_stub=self._direct_stub, relay_call=self._relay_call,
                    connect_timeout_sec=self._timeout, counters=self.counters_obj, on_path=self._on_path)
                self._senders[target] = sender
                sender.start()
            return sender

    def add_worker(self, conn: Any, worker: int) -> None:
        reader = RouteReader(conn, worker=worker, routes=self.routes, sender_for=self.sender_for,
                             counters=self.counters_obj, stop=self._halt, max_frame_bytes=self._max_frame)
        with self._lock:
            self._readers = [r for r in self._readers if r.is_alive()] + [reader]
        reader.start()

    def paths(self) -> dict[int, str]:
        with self._lock:
            return {t: s.path for t, s in self._senders.items()}

    def counters(self) -> dict[str, int]:
        return self.counters_obj.snapshot()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop every thread (frames still queued are counted ``disconnect``); idempotent."""
        self._halt.set()
        with self._lock:
            senders, self._senders = list(self._senders.values()), {}
            readers = list(self._readers)
        for sender in senders:
            sender.stop()
        deadline = time.monotonic() + timeout
        for thread in [*senders, *readers]:
            thread.join(max(0.0, deadline - time.monotonic()))
        self.delivery.clear()
