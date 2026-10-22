"""Same-machine process connections (SP5 spec 4.7, 7.8; plan rulings R3, R4, R10, R11).

A node talks to each of its children (workers, learners, evaluators) over ``multiprocessing.connection`` Unix-socket
connections authenticated with the node's ``authkey``; no ``mp.Queue``, ``mp.Value`` or ``mp.Event`` is shared
between processes that restart, so a ``kill -9`` of any process can only break its own connections.

Addresses. ``address(ca8, run8, node, proc)`` names a listener: ``colosseum-{ca8}-{run8}-n{node}-{proc}`` with
``proc`` = ``node`` (the node's listener for every child) or ``learner_proc_name(agent, incarnation)`` =
``learner-{agent}-i{incarnation}`` (one per learner incarnation; an agent id longer than 24 characters is replaced by
16 hex digits of its sha256, so no config can make a name too long, P-PF2-18). Linux uses abstract sockets
(``"\\0" + name``: no file, released with the process); elsewhere (macOS) a socket file in ``$COLOSSEUM_RUNTIME_DIR``
or ``<tempdir>/cst-<uid>/`` whose path must be at most 104 characters (ValueError otherwise).

Connections. Each child opens one ``control`` connection to the node; a worker also opens one ``route`` connection
to the node (frames for learners on other machines, T3.2) and one ``chunks`` connection per local learner it collects
for; the node opens one ``chunks`` connection to each local learner (frames from other machines, T3.2). The first
message on every connection is a ``ChildHello``. An ``Acceptor`` accepts a listener's connections off the
accepting thread: the authentication handshake and the hello of each connection run in a short-lived thread bounded
by ``HANDSHAKE_TIMEOUT_SEC``, and any failure closes only that connection (P-PF2-6). Control messages are pickled
dataclasses (R3, the trust boundary of today's queues); chunks and checkpoints travel as frames (``send_bytes``);
weights only through the shared-memory cache (R10). A ``ControlChannel`` serializes the sends of several threads of
one process on one connection.

Worker side: ``WorkerIO`` is the ``LoopIO`` of ``RolloutLoop`` (R4). Each learner connection has a ``FrameSender``
thread with a one-frame slot, so ``send_chunk`` waits in short slices (it keeps flushing results and reading control
messages, and sees a stop) while a busy learner applies backpressure; a broken or missing learner drops the chunk and
counts it (``disconnect`` / ``no_learner``, spec 4.7: at most once); a connection that has not opened within
``OPEN_TIMEOUT_SEC`` counts as broken (P-PF2-6). Every drop counter of a worker is updated under one lock (P-PF2-15).
A broken sender to a local learner whose route is unchanged is reopened at a later chunk after a backoff (R34).
Frames to a local learner are never compressed (lz4 is only for frames that leave the machine, T3.2).

Learner side: ``ConnectionIntake`` accepts the chunk connections, reads whole frames into a bounded deque and stops
starting new frames while it is full (backpressure through the sockets); ``get`` reads the frame's header meta,
applies the misroute, generation (R16) and policy-lag (``learner.max_policy_lag``, T3.3: ``stale``) rules on it, then
decodes and checks the frame (``decode_chunk(..., validator=)``, A17), counting every drop by reason. Its one reader
thread never blocks on a connection: frames are assembled from non-blocking sockets, so a sender stopped inside a
frame holds back only itself, and after ``FRAME_STALL_SEC`` without a byte its connection is closed (``disconnect``).
An EOF closes only that connection; a frame cut by a dying sender closes it and counts ``disconnect``.
``NodeOutputs`` publishes weights into the shared-memory cache and reports them, sends checkpoints and metrics to the
node. ``QueueIntake`` / ``QueueOutputs`` keep SP4's queue path for the distributed roles (until T7.2).
"""

from __future__ import annotations

import collections
import hashlib
import logging
import multiprocessing.connection as mpc
import os
import queue
import struct
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from colosseum.core.types import Lineup, MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand

if TYPE_CHECKING:
    from colosseum.net.chunks import ChunkValidator
    from colosseum.node.weights import CacheKey, WeightCacheReader, WeightCacheWriter

logger = logging.getLogger(__name__)

RUNTIME_DIR_ENV = "COLOSSEUM_RUNTIME_DIR"
MAX_SOCKET_PATH = 104            # macOS sun_path limit
MAX_ABSTRACT_NAME = 107          # Linux: 108 bytes of sun_path minus the leading NUL
CONNECT_RETRY_SEC = 0.05
HANDSHAKE_TIMEOUT_SEC = 5.0      # an accepted connection's authentication + ChildHello (P-PF2-6)
OPEN_TIMEOUT_SEC = 10.0          # a FrameSender whose connection has not opened by then is broken (P-PF2-6)
SENDER_REOPEN_SEC = 1.0          # a broken sender to a local learner is reopened after this (R34), doubling while
SENDER_REOPEN_MAX_SEC = 30.0     # reopened senders never open, up to this
LEARNER_AGENT_CHARS = 24         # longer agent ids are hashed in learner socket names (P-PF2-18)
SEND_SLICE_SEC = 0.5             # a blocked send_chunk wakes this often (results, control messages, stop)
PROGRESS_INTERVAL_SEC = 0.5      # WorkerProgress cadence (spec 4.4: cumulative counters)
INTAKE_REPORT_SEC = 1.0          # IntakeReport cadence
FRAME_STALL_SEC = 10.0           # a chunk connection silent this long inside a frame is closed (frame: disconnect)
_GENERATIONS_KEPT = 256          # version -> generation entries a worker keeps per agent


# ---------------------------------------------------------------------------------------------------------------
# Addresses, listeners, connections
# ---------------------------------------------------------------------------------------------------------------


def address(ca8: str, run8: str, node: int, proc: str) -> str:
    """The name of a node-local listener (module docstring)."""
    return f"colosseum-{ca8}-{run8}-n{int(node)}-{proc}"


def learner_proc_name(agent_id: str, incarnation: int) -> str:
    """The ``proc`` part of a learner incarnation's listener name (module docstring, P-PF2-18)."""
    part = agent_id if len(agent_id) <= LEARNER_AGENT_CHARS else hashlib.sha256(agent_id.encode()).hexdigest()[:16]
    return f"learner-{part}-i{int(incarnation)}"


def _socket_address(name: str) -> str:
    if sys.platform.startswith("linux"):
        if len(name.encode()) > MAX_ABSTRACT_NAME:
            raise ValueError(f"socket name {name!r} is longer than {MAX_ABSTRACT_NAME} bytes")
        return "\0" + name
    root = Path(os.environ.get(RUNTIME_DIR_ENV) or Path(tempfile.gettempdir()) / f"cst-{os.getuid()}")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = str(root / name)
    if len(path) > MAX_SOCKET_PATH:
        raise ValueError(f"socket path {path!r} is longer than {MAX_SOCKET_PATH} characters; set "
                         f"{RUNTIME_DIR_ENV} to a shorter directory")
    return path


def listener(name: str, authkey: bytes) -> mpc.Listener:
    """A listener for ``name`` (a stale socket file of a dead process is replaced; abstract names cannot be stale)."""
    addr = _socket_address(name)
    if not addr.startswith("\0") and os.path.exists(addr):
        os.unlink(addr)
    return mpc.Listener(addr, family="AF_UNIX", backlog=64, authkey=authkey)


def connect(name: str, authkey: bytes, *, timeout: float) -> mpc.Connection:
    """Connect to ``name``, retrying while the listener does not exist yet; ``ConnectionError`` after ``timeout``."""
    addr = _socket_address(name)
    deadline = time.monotonic() + timeout
    while True:
        try:
            return mpc.Client(addr, family="AF_UNIX", authkey=authkey)
        except (FileNotFoundError, ConnectionRefusedError) as e:
            if time.monotonic() >= deadline:
                raise ConnectionError(f"no listener {name!r} after {timeout:.1f} s ({e})") from e
            time.sleep(CONNECT_RETRY_SEC)


def listener_socket(lst: mpc.Listener) -> Any:
    """The listening socket of ``lst`` (CPython's ``Listener._listener._socket``), for ``wait`` with a timeout, so an
    accept thread can be stopped (closing a socket does not wake a thread blocked in ``accept`` on Linux)."""
    return lst._listener._socket


class _HandshakeDeadline:
    """``send_bytes`` / ``recv_bytes`` of a connection for ``multiprocessing.connection``'s challenge functions, with
    every read bounded by one deadline (a client that never answers cannot hold the handshake forever)."""

    def __init__(self, conn: mpc.Connection, deadline: float) -> None:
        self._conn = conn
        self._deadline = deadline

    def wait(self) -> None:
        if not self._conn.poll(max(0.0, self._deadline - time.monotonic())):
            raise TimeoutError(f"no answer within {HANDSHAKE_TIMEOUT_SEC:.0f} s")

    def send_bytes(self, buf: bytes) -> None:
        self._conn.send_bytes(buf)

    def recv_bytes(self, maxlength: int | None = None) -> bytes:
        self.wait()
        return self._conn.recv_bytes(maxlength)


class Acceptor:
    """Accepts ``lst``'s connections in a thread (P-PF2-6). Each accepted connection is authenticated with
    ``authkey`` (the node's key; mandatory: an abstract Unix socket has no file permissions, so the challenge is the
    only gate in front of the pickled control messages, R3) and reads its ``ChildHello`` in a short-lived daemon
    thread bounded by ``HANDSHAKE_TIMEOUT_SEC``; then ``on_connection(conn, hello)`` runs in that thread. The
    challenge always runs here, whatever key ``lst`` was created with (``Listener.accept``, which would run it on the
    accepting thread, is never called). A failure of one connection (a client that dies or stalls in the handshake,
    a wrong key, no hello, an error in the callback) closes only that connection; the accept thread never stops
    before ``close``. ``close`` does not close the listener."""

    def __init__(self, lst: mpc.Listener, on_connection: Callable[[mpc.Connection, ChildHello], None], *,
                 authkey: bytes, name: str) -> None:
        if not isinstance(authkey, bytes) or not authkey:
            raise ValueError(f"{name}: an Acceptor needs a non-empty authkey (bytes)")
        self._listener = lst
        self._authkey = authkey
        self._on_connection = on_connection
        self._name = name
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._accept_loop, name=f"accept-{name}", daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        sock = listener_socket(self._listener)
        while not self._closed.is_set():
            try:
                if not mpc.wait([sock], timeout=0.5):
                    continue
                conn = self._listener._listener.accept()   # the raw connection, not yet authenticated
            except Exception as e:  # noqa: BLE001 - one failed accept never stops the loop
                if self._closed.is_set():
                    return
                logger.debug(f"{self._name}: accept failed ({type(e).__name__}: {e})")
                self._closed.wait(0.05)
                continue
            threading.Thread(target=self._handshake, args=(conn,), name=f"handshake-{self._name}",
                             daemon=True).start()

    def _handshake(self, conn: mpc.Connection) -> None:
        try:
            bounded = _HandshakeDeadline(conn, time.monotonic() + HANDSHAKE_TIMEOUT_SEC)
            mpc.deliver_challenge(bounded, self._authkey)
            mpc.answer_challenge(bounded, self._authkey)
            bounded.wait()
            hello = conn.recv()                            # authenticated: a pickled control message (R3)
            if not isinstance(hello, ChildHello):
                raise TypeError(f"the first message is a {type(hello).__name__}, not a ChildHello")
            if self._closed.is_set():
                raise ConnectionError("the acceptor is closed")
            self._on_connection(conn, hello)
        except Exception as e:  # noqa: BLE001 - only this connection fails
            logger.debug(f"{self._name}: a connection failed its handshake ({type(e).__name__}: {e})")
            try:
                conn.close()
            except OSError:
                pass

    def close(self, timeout: float = 2.0) -> None:
        self._closed.set()
        self._thread.join(timeout)


# ---------------------------------------------------------------------------------------------------------------
# Control messages (pickled dataclasses, R3)
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class ChildHello:
    role: Literal["worker", "learner", "evaluator", "node"]
    name: str
    incarnation: int
    pid: int
    channel: Literal["control", "route", "chunks"]


@dataclass
class Stop:
    reason: str


@dataclass
class ChildError:
    """Sent by a child before it exits with an error (``contract``: EnvContractError, ``player``: PlayerError,
    ``crash``: any other exception); ``context`` is the error's location text."""

    kind: Literal["contract", "player", "crash"]
    message: str
    context: str


@dataclass
class CellLineup:
    """A lineup for one env of a worker; ``lineup.epoch`` is set; ``keys`` the cache key of every seat's weights
    (R11: all in the cache when this is sent)."""

    env: int
    lineup: Lineup
    keys: dict[int, CacheKey]


@dataclass
class WorkerLineups:
    cells: list[CellLineup]


@dataclass
class WorkerEvict:
    keys: list[CacheKey]


@dataclass
class LearnerRoute:
    """Where a worker sends ``agent_id``'s chunks: a local learner's listener ``address``, or the node's route
    connection (``via_node``); neither: the agent has no learner (chunks are dropped as ``no_learner``)."""

    agent_id: str
    address: str | None
    via_node: bool
    generation: int


@dataclass
class WeightsReady:
    """The first-weights wait of a worker ends (``agents``: those with weights in the cache)."""

    agents: list[str]


@dataclass
class WorkerResults:
    results: list[MatchResult]


@dataclass
class WorkerProgress:
    """A worker incarnation's cumulative counters (env steps and stats), every ``PROGRESS_INTERVAL_SEC``."""

    incarnation: int
    env_steps: int
    counters: dict[str, int]


@dataclass
class LearnerEnvSteps:
    env_steps: int


@dataclass
class VersionPublished:
    agent_id: str
    generation: int
    version: int
    segment: str
    size: int
    digest: str
    teacher_active: bool


@dataclass
class CheckpointHeader:
    """Followed on the same connection by one ``send_bytes(encode_checkpoint(...))`` of ``size`` bytes."""

    kind: Literal["periodic", "final", "recovery"]
    agent_id: str
    generation: int
    version: int
    size: int


@dataclass
class LearnerMetrics:
    metrics: dict


@dataclass
class IntakeReport:
    depth: int
    dropped: dict[str, int]


class ControlChannel:
    """The sending side of one connection, safe for several threads (one lock); ``closed`` after a failed send.
    Receiving stays with one thread of the process."""

    def __init__(self, conn: mpc.Connection) -> None:
        self.conn = conn
        self._lock = threading.Lock()
        self.closed = False

    def send(self, message: Any) -> bool:
        with self._lock:
            if self.closed:
                return False
            try:
                self.conn.send(message)
                return True
            except (OSError, EOFError, ValueError):
                self.closed = True
                return False

    def send_with_frame(self, message: Any, frame: bytes | memoryview) -> bool:
        """``message`` then ``frame`` as one unit (no other send in between)."""
        with self._lock:
            if self.closed:
                return False
            try:
                self.conn.send(message)
                self.conn.send_bytes(frame)
                return True
            except (OSError, EOFError, ValueError):
                self.closed = True
                return False

    def close(self) -> None:
        with self._lock:
            self.closed = True
            try:
                self.conn.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------------------------------------------


class FrameSender:
    """One connection's sender thread with a one-frame slot (module docstring). ``opener`` makes the connection in
    the thread (the hello is sent there too). ``put`` waits up to ``timeout`` for the slot; after a failure the sender
    is ``broken`` and every later or queued frame is counted in ``dropped`` (reason ``disconnect``). A connection
    that has not opened ``OPEN_TIMEOUT_SEC`` after the start breaks the sender at the next ``put`` (P-PF2-6); a
    connection the opener returns after that is closed unused. ``lock`` guards every update of ``dropped`` (the
    owner's lock when several senders and the owner share one counter, P-PF2-15)."""

    def __init__(self, name: str, opener: Callable[[], mpc.Connection], dropped: Counter, *,
                 lock: threading.Lock | None = None) -> None:
        self.name = name
        self._opener = opener
        self._dropped = dropped
        self._lock = lock if lock is not None else threading.Lock()
        self._state = threading.Lock()          # broken / opened / the connection (never held while counting)
        self._slot: queue.Queue = queue.Queue(maxsize=1)
        self.broken = False
        self.broken_at: float | None = None     # time.monotonic() of the failure (WorkerIO's reopen backoff, R34)
        self._closed = False
        self._opened = False
        self._started = time.monotonic()
        self._conn: mpc.Connection | None = None
        self._thread = threading.Thread(target=self._run, name=f"frames-{name}", daemon=True)
        self._thread.start()

    def put(self, frame: bytes, timeout: float) -> bool:
        """True: the frame was consumed (queued, or dropped and counted); False: the slot stayed busy."""
        if not self._opened and not self.broken and time.monotonic() - self._started >= OPEN_TIMEOUT_SEC:
            self._fail(f"its connection did not open within {OPEN_TIMEOUT_SEC:.0f} s")
        if self.broken or self._closed:
            self._count(1)
            return True
        try:
            self._slot.put(frame, timeout=timeout)
            return True
        except queue.Full:
            return False

    def _count(self, n: int) -> None:
        with self._lock:
            self._dropped["disconnect"] += n

    def _run(self) -> None:
        try:
            conn = self._opener()
        except Exception as e:  # noqa: BLE001 - the learner is gone or never came up
            if not self._closed:                # after stop() nothing is sent any more: nothing to report
                self._fail(f"cannot connect ({type(e).__name__}: {e})")
            return
        with self._state:
            if self.broken or self._closed:
                late = True
            else:
                late = False
                self._conn, self._opened = conn, True
        if late:                                # broken by the open bound, or closed, while connecting
            try:
                conn.close()
            except OSError:
                pass
            return
        while not self._closed:
            try:
                frame = self._slot.get(timeout=SEND_SLICE_SEC)
            except queue.Empty:
                continue
            if frame is None:
                break
            try:
                conn.send_bytes(frame)
            except (OSError, EOFError, ValueError) as e:
                self._count(1)
                self._fail(f"the receiver is gone ({type(e).__name__}: {e})")
                return
        self._close_conn()

    @property
    def opened(self) -> bool:
        """Its connection opened (it may have broken since)."""
        return self._opened

    def _fail(self, why: str) -> None:
        with self._state:
            first = not self.broken
            self.broken = True
            if first:
                self.broken_at = time.monotonic()
        if first:
            logger.warning(f"frame sender {self.name}: {why}; its chunks are dropped (dropped_chunks.disconnect)")
        n = 0
        while True:
            try:
                if self._slot.get_nowait() is not None:
                    n += 1
            except queue.Empty:
                break
        if n:
            self._count(n)
        self._close_conn()

    def _close_conn(self) -> None:
        with self._state:
            conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    def close(self, timeout: float = 1.0) -> None:
        """Stop the thread; a frame still in the slot is counted as dropped."""
        self.stop()
        self.join(timeout)

    def join(self, timeout: float) -> None:
        self._thread.join(max(0.0, timeout))

    def stop(self) -> None:
        """Tell the thread to stop (without waiting); a frame still in the slot is counted as dropped."""
        with self._state:
            self._closed = True
        try:
            pending = self._slot.get_nowait()
            if pending is not None:
                self._count(1)
        except queue.Empty:
            pass
        try:
            self._slot.put_nowait(None)
        except queue.Full:
            pass


class _ResultSink:
    """``ResultBatcher``'s queue over the control channel: a failed send is a drop (``queue.Full``)."""

    def __init__(self, control: ControlChannel) -> None:
        self._control = control

    def put_nowait(self, batch: list[MatchResult]) -> None:
        if not self._control.send(WorkerResults(batch)):
            raise queue.Full


class WorkerIO:
    """``LoopIO`` of a node's worker over its connections and the shared-memory cache (module docstring).

    ``pump`` reads every pending control message (lineups, evictions, routes, ``WeightsReady``, ``Stop``); the loop
    callbacks call it. ``stats_fn`` (set by the worker body once the loop exists) gives the loop's counters for
    ``WorkerProgress``; this object adds its chunk drops (``dropped_chunks_no_learner``, ``dropped_chunks_disconnect``)
    and the result batcher's drops (into ``dropped_results``). ``dropped`` is updated by this object and its sender
    threads under one lock (P-PF2-15). ``results_batch`` / ``results_flush_sec`` default to SP4's FX1 batching
    (``RESULTS_BATCH_MAX`` / ``RESULTS_BATCH_SEC``). A broken sender to a local learner whose route is unchanged is
    reopened at the next chunk once ``SENDER_REOPEN_SEC`` passed since it broke (R34: e.g. after the learner's
    intake cut a stalled connection); the wait doubles, up to ``SENDER_REOPEN_MAX_SEC``, while reopened senders never
    open (a learner that is gone), and resets after one opens; chunks while broken stay dropped and counted. The
    route connection to the node is never reopened (the node closing it ends the worker anyway)."""

    def __init__(self, *, node: int, worker: int, incarnation: int, control: mpc.Connection,
                 route: mpc.Connection | None, cache: WeightCacheReader, routes: Mapping[str, LearnerRoute],
                 authkey: bytes, results_batch: int = 256, results_flush_sec: float = 0.1,
                 clock: Callable[[], float] = time.monotonic) -> None:
        from colosseum.worker.rollout_worker import ResultBatcher

        self.node, self.worker, self.incarnation = int(node), int(worker), int(incarnation)
        self._control = ControlChannel(control)
        self._route_conn = route
        self._cache = cache
        self._authkey = authkey
        self._clock = clock
        self._routes: dict[str, LearnerRoute] = dict(routes)
        self._senders: dict[str, FrameSender] = {}       # by agent (local learner) or "<route>"
        self._reopen_after: dict[str, float] = {}        # by agent: the reopen wait of a broken sender (R34)
        self._lz4: dict[str, Any] = {}                   # agent -> frame.Lz4Probe of its frames to other machines
        self.dropped: Counter = Counter()                # chunk drops by reason (under _drop_lock)
        self._drop_lock = threading.Lock()
        self._batcher = ResultBatcher(_ResultSink(self._control), worker, max_results=results_batch,
                                      max_age_sec=results_flush_sec, clock=clock)
        self._lineups: dict[int, CellLineup] = {}        # pending, by env (a later one replaces an earlier one)
        self._evict: list[CacheKey] = []
        self._given: set[tuple[str, str]] = set()        # (agent, checkpoint id) handed to the loop
        self._loaded: dict[str, int] = {}                # agent -> version loaded in the loop
        self._loaded_generation: dict[str, int] = {}     # agent -> generation of those weights (R40)
        self._generations: dict[str, collections.OrderedDict[int, int]] = {}
        self._ready = False
        self._stopped = False
        self._env_steps = 0
        self._last_progress = clock()
        self.stats_fn: Callable[[], dict[str, int]] | None = None

    # ---- control messages ----------------------------------------------------------------------------------

    def pump(self) -> None:
        conn = self._control.conn
        while not self._stopped:
            try:
                if not conn.poll(0):
                    return
                msg = conn.recv()
            except (OSError, EOFError):
                self._stopped = True            # the node is gone
                return
            if isinstance(msg, WorkerLineups):
                for cell in msg.cells:
                    self._lineups[cell.env] = cell
            elif isinstance(msg, WorkerEvict):
                self._evict.extend(msg.keys)
            elif isinstance(msg, LearnerRoute):
                self._set_route(msg)
            elif isinstance(msg, WeightsReady):
                self._ready = True
            elif isinstance(msg, Stop):
                self._stopped = True
            else:
                logger.warning(f"worker {self.worker}: unexpected control message {type(msg).__name__}")

    def _set_route(self, route: LearnerRoute) -> None:
        old = self._routes.get(route.agent_id)
        self._routes[route.agent_id] = route
        if old is not None and (old.address, old.via_node, old.generation) != (route.address, route.via_node,
                                                                              route.generation):
            sender = self._senders.pop(route.agent_id, None)
            if sender is not None:
                sender.close()
            self._reopen_after.pop(route.agent_id, None)   # a new route starts with the base reopen wait (R34)

    def ready(self) -> bool:
        self.pump()
        return self._ready

    def stopped(self) -> bool:
        self.pump()
        return self._stopped

    def note_snapshot(self, agent_id: str, checkpoint_id: str) -> None:
        """The worker body loaded this snapshot itself (from its first lineups): a later lineup naming it does not
        read and load it again."""
        self._given.add((agent_id, checkpoint_id))

    def take_lineups(self) -> dict[int, CellLineup]:
        """The pending lineups by env (the worker body reads its first ones before the loop exists)."""
        self.pump()
        out, self._lineups = self._lineups, {}
        return out

    # ---- LoopIO --------------------------------------------------------------------------------------------

    def loop_io(self) -> Any:
        from colosseum.worker.rollout_loop import LoopIO

        return LoopIO(send_chunk=self.send_chunk, poll_weights=self.poll_weights, report_result=self.report_result,
                      poll_command=self.poll_command, add_env_steps=self.add_env_steps)

    def send_chunk(self, chunk: TrajectoryChunk) -> None:
        from colosseum.net.chunks import encode_chunk

        route = self._routes.get(chunk.agent_id)
        if route is None or (route.address is None and not route.via_node):
            self._count_drop("no_learner")
            return
        sender = self._sender(route)
        probe = self._lz4_probe(chunk.agent_id) if route.via_node else None     # frames that leave the machine
        compress = probe is not None and probe.compress()
        frame = encode_chunk(chunk, generation=self._generation_of(chunk.agent_id, chunk.policy_version),
                             node=self.node, worker=self.worker, compress=compress)
        if compress:                                 # every compressed frame is measured, sent or not (P-PF1-8)
            from colosseum.net.frame import frame_sizes

            probe.observe(*frame_sizes(frame))
        while not sender.put(frame, timeout=SEND_SLICE_SEC):
            self.idle()
            if self._stopped:
                self._count_drop("disconnect")
                return

    def _count_drop(self, reason: str) -> None:
        with self._drop_lock:
            self.dropped[reason] += 1

    def _lz4_probe(self, agent_id: str) -> Any:
        """Adaptive lz4 per agent for frames to learners on other machines (spec 4.9; T3.2)."""
        probe = self._lz4.get(agent_id)
        if probe is None:
            from colosseum.net.frame import Lz4Probe

            probe = self._lz4[agent_id] = Lz4Probe(clock=self._clock)
        return probe

    def _sender(self, route: LearnerRoute) -> FrameSender:
        # Every agent routed via the node shares one sender over the route connection ("<route>"): a route change of
        # one agent never closes it (``_set_route`` closes senders by agent key), and its failure means the node is gone
        # (every later via-node frame is counted ``disconnect``) (T3.2).
        key = route.agent_id if not route.via_node else "<route>"
        sender = self._senders.get(key)
        if sender is not None and not sender.broken and sender.opened:
            self._reopen_after.pop(key, None)            # it opened: the next break waits the base value (R34)
        if sender is not None and sender.broken and not route.via_node and sender.broken_at is not None:
            wait = self._reopen_after.get(key, SENDER_REOPEN_SEC)
            elapsed = time.monotonic() - sender.broken_at
            if elapsed >= wait:
                self._reopen_after[key] = (SENDER_REOPEN_SEC if sender.opened
                                           else min(2.0 * wait, SENDER_REOPEN_MAX_SEC))
                logger.info(f"worker {self.worker}: reopening the chunk connection to the learner of "
                            f"{route.agent_id!r} (broken for {elapsed:.1f} s; reopen wait {wait:.1f} s)")
                sender.close(0.0)
                del self._senders[key]
                sender = None
        if sender is None:
            if route.via_node:
                conn = self._route_conn
                if conn is None:
                    raise RuntimeError(f"worker {self.worker}: no route connection for {route.agent_id!r}")
                opener: Callable[[], mpc.Connection] = lambda: conn  # noqa: E731
            else:
                name = route.address

                def opener() -> mpc.Connection:
                    c = connect(name, self._authkey, timeout=5.0)
                    c.send(ChildHello("worker", f"worker-{self.worker}", self.incarnation, os.getpid(), "chunks"))
                    return c
            sender = self._senders[key] = FrameSender(f"{key}@worker-{self.worker}", opener, self.dropped,
                                                      lock=self._drop_lock)
        return sender

    def poll_weights(self, agent_id: str) -> WeightPayload | None:
        """The agent's latest weights if they are newer than the loaded ones, or of another generation (controller
        ruling R40: after a relocation the new learner generation may publish below a dead one's versions; within
        one generation versions only go up, R9)."""
        cached = self._cache.read_latest(agent_id, newer_than=self._loaded.get(agent_id, -1),
                                         loaded_generation=self._loaded_generation.get(agent_id))
        if cached is None:
            return None
        meta = cached.meta
        self._loaded[agent_id] = int(meta.policy_version)
        self._loaded_generation[agent_id] = int(meta.generation)
        gens = self._generations.setdefault(agent_id, collections.OrderedDict())
        gens[int(meta.policy_version)] = int(meta.generation)
        while len(gens) > _GENERATIONS_KEPT:
            gens.popitem(last=False)
        return WeightPayload(agent_id=agent_id, policy_version=int(meta.policy_version),
                             state_dict=cached.state_dict, teacher_active=bool(meta.teacher_active),
                             generation=int(meta.generation))

    def _generation_of(self, agent_id: str, version: int | None) -> int:
        if version is None:
            return 0
        return self._generations.get(agent_id, {}).get(int(version), 0)

    def poll_command(self) -> WorkerCommand | None:
        self.pump()
        if not self._lineups and not self._evict:
            return None
        evict: dict[str, list[str]] = {}
        for key in self._evict:
            if key.kind == "snapshot":
                evict.setdefault(key.agent_id, []).append(key.ident)
                self._given.discard((key.agent_id, key.ident))
        self._evict = []
        new: dict[str, dict[str, dict]] = {}
        lineups: list[Lineup | None] = []
        for env, cell in sorted(self._lineups.items()):
            for key in cell.keys.values():
                if key.kind != "snapshot" or (key.agent_id, key.ident) in self._given:
                    continue
                cached = self._cache.get(key)
                if cached is None:   # the node sends a lineup only when its weights are cached (R11)
                    raise RuntimeError(f"worker {self.worker}: snapshot {key} of a granted lineup is not in the "
                                       f"weight cache")
                new.setdefault(key.agent_id, {})[key.ident] = cached.state_dict
                self._given.add((key.agent_id, key.ident))
            lineups.extend([None] * (env + 1 - len(lineups)))
            lineups[env] = cell.lineup
        self._lineups = {}
        return WorkerCommand(lineups=lineups, new_checkpoints=new, evict=evict)

    def report_result(self, result: MatchResult) -> None:
        result.node = self.node
        for seat in result.seats:
            if seat.network_id == "latest":
                seat.generation = self._generation_of(seat.agent_id, seat.version_end) or None
        self._batcher.add(result)

    def add_env_steps(self, n: int) -> None:
        self._env_steps += int(n)
        self.idle()

    # ---- housekeeping ----------------------------------------------------------------------------------------

    def idle(self) -> None:
        """Flush due results and progress, read control messages (every loop step and every blocked-send slice)."""
        self._batcher.flush_if_due()
        if self._clock() - self._last_progress >= PROGRESS_INTERVAL_SEC:
            self.send_progress()
        self.pump()

    def counters(self) -> dict[str, int]:
        stats = dict(self.stats_fn()) if self.stats_fn is not None else {}
        stats["dropped_results"] = int(stats.get("dropped_results", 0)) + self._batcher.dropped
        with self._drop_lock:
            stats["dropped_chunks_no_learner"] = int(self.dropped["no_learner"])
            stats["dropped_chunks_disconnect"] = int(self.dropped["disconnect"])
        return {k: int(v) for k, v in stats.items()}

    def send_progress(self) -> None:
        self._last_progress = self._clock()
        self._control.send(WorkerProgress(self.incarnation, self._env_steps, self.counters()))

    def close(self, timeout: float = 1.0) -> None:
        """Flush the pending results, send the final progress, stop the sender threads (all at once, then joined
        within one shared ``timeout``)."""
        self._batcher.flush()
        self.send_progress()
        senders, self._senders = list(self._senders.values()), {}
        for sender in senders:
            sender.stop()
        deadline = time.monotonic() + timeout
        for sender in senders:
            sender.join(deadline - time.monotonic())

    def send_error(self, error: ChildError) -> None:
        self._control.send(error)


# ---------------------------------------------------------------------------------------------------------------
# Learner side
# ---------------------------------------------------------------------------------------------------------------


class _FrameCut(Exception):
    """The connection ended inside a frame (after its first length byte): the frame is lost."""


class _FrameTooLarge(Exception):
    """A frame whose announced length exceeds the cap (or is malformed)."""


_LEN = struct.Struct("!i")       # multiprocessing.connection's message header: signed 32-bit length, or -1 and then
_LEN64 = struct.Struct("!Q")     # an unsigned 64-bit length (CPython's Connection._send_bytes on Unix)


class _FrameReader:
    """Reads one chunk connection's ``send_bytes`` messages (CPython's Unix framing: a signed 32-bit length, or -1
    and an unsigned 64-bit length, then the body) incrementally from its non-blocking socket, with explicit outcomes
    instead of ``recv_bytes``'s, which reports a sender cut between the length and the body as a clean EOF.

    ``read(now)`` takes what the socket holds without blocking and returns a whole frame, or None while the frame is
    incomplete; EOFError for an end between messages, _FrameCut for an end inside one (after its first length byte),
    _FrameTooLarge for a length above ``cap`` (the body stays unread: the caller closes the connection), OSError for
    socket errors. ``started``: a frame is in progress; ``last_progress``: when its last bytes arrived (``now`` of
    that ``read``; the intake's stall deadline)."""

    def __init__(self, conn: mpc.Connection, cap: int) -> None:
        self.conn = conn
        self._fd = conn.fileno()
        os.set_blocking(self._fd, False)
        self._cap = int(cap)
        self.last_progress = 0.0
        self._clear()

    def _clear(self) -> None:
        self._head = bytearray()
        self._need = _LEN.size
        self._body: bytearray | None = None
        self._pos = 0

    @property
    def started(self) -> bool:
        return bool(self._head)

    def read(self, now: float) -> bytearray | None:
        while self._body is None:                 # the length: 4 bytes, or 4 + 8 for a 64-bit length
            try:
                data = os.read(self._fd, self._need - len(self._head))
            except BlockingIOError:
                return None
            if not data:
                if not self._head:
                    raise EOFError
                raise _FrameCut(f"end of stream after {len(self._head)} bytes of a frame's length")
            self._head += data
            self.last_progress = now
            if len(self._head) < self._need:
                continue
            (size,) = _LEN.unpack_from(self._head)
            if size == -1:
                if self._need == _LEN.size:
                    self._need += _LEN64.size
                    continue
                (size,) = _LEN64.unpack_from(self._head, _LEN.size)
            if size < 0 or size > self._cap:
                raise _FrameTooLarge(f"a frame of {size} bytes, the cap is {self._cap}")
            self._body = bytearray(size)
        body = self._body
        with memoryview(body) as view:
            while self._pos < len(body):
                try:
                    got = os.readv(self._fd, [view[self._pos:]])
                except BlockingIOError:
                    return None
                if got == 0:
                    raise _FrameCut(f"end of stream after {self._pos} of {len(body)} bytes")
                self._pos += got
                self.last_progress = now
        self._clear()
        return body


class ChunkIntake(Protocol):
    def get(self, timeout: float) -> TrajectoryChunk: ...       # queue.Empty on timeout

    def note_policy_version(self, version: int) -> None:
        """The learner's current policy version, before each batch: the reference of ``max_policy_lag``."""

    @property
    def depth(self) -> int: ...

    @property
    def dropped(self) -> dict[str, int]: ...


class LearnerOutputs(Protocol):
    checkpoints: bool                      # whether checkpoints are wanted (SP4: a checkpoint queue was given)

    def publish_weights(self, payload: WeightPayload) -> None: ...

    def send_checkpoint(self, payload: dict, *, kind: Literal["periodic", "final", "recovery"], block: bool) -> bool:
        ...

    def send_metrics(self, metrics: dict) -> None: ...

    def finish(self, stop_event: Any) -> None: ...


class ConnectionIntake:
    """``ChunkIntake`` of a node's learner (module docstring). ``control``: the learner's control channel to its
    node (this object's reader thread is its only reader: ``LearnerEnvSteps`` -> ``value`` / ``env_steps``,
    ``Stop`` or EOF -> ``stop_flag``); ``capacity``: frames buffered (``learner.queue_size``); ``cap_bytes``: the
    largest frame accepted (``payload_byte_cap``). ``version_floor``: the first version of this generation; a chunk
    of an older generation is kept only below it (R16). ``max_policy_lag`` (``learner.max_policy_lag``, T3.3): with
    a bound, a chunk whose version is below the learner's current version (``note_policy_version``) minus the bound
    is dropped as ``stale`` by its header, never decoded (so a stale frame's body is not checked). Only
    ``ChildHello(channel="chunks")`` connections are kept (an ``Acceptor`` authenticates them with ``authkey``, the
    node's key, P-PF2-6). A frame goes through one path (A17): header meta -> misroute, generation and lag rules ->
    ``decode_chunk(..., validator=)``; every drop is counted by reason. An unreadable control message or a failure
    of the reader thread logs an ERROR and sets ``stop_flag``; an unexpected error on one chunk connection closes
    only that connection (``disconnect``).

    One reader thread serves every connection without ever blocking on one of them: chunk sockets are non-blocking
    and each frame is assembled from what its socket holds (``_FrameReader``). A frame in progress holds one slot of
    ``capacity`` (a new frame is started only while the deque plus the frames in progress are below it, so they
    never exceed it); a connection with nothing to read for ``stall_sec`` inside a frame (a stopped sender) is closed
    and its frame counted ``disconnect`` (data already in the socket is always read first, so a pause of the reader
    itself cuts nobody). Costs: the stalled frame holds its slot until then (with ``capacity`` 1 every other sender
    waits up to ``stall_sec``); a sender that continues later counts the same frame as lost too (an over-count of one)
    and is broken until its worker reopens it (``WorkerIO``, R34: at the next chunk after ``SENDER_REOPEN_SEC``), so
    that worker's chunks for this learner meanwhile are dropped and counted ``disconnect``."""

    def __init__(self, lst: mpc.Listener, control: ControlChannel, validator: ChunkValidator, *, authkey: bytes,
                 agent_id: str, generation: int, version_floor: int, capacity: int, cap_bytes: int,
                 max_policy_lag: int | None = None, stall_sec: float = FRAME_STALL_SEC,
                 clock: Callable[[], float] = time.monotonic) -> None:
        from colosseum.net.chunks import decode_chunk

        self._listener = lst
        self._control = control
        self._validator = validator
        self.agent_id = agent_id
        self.generation = int(generation)
        self.version_floor = int(version_floor)
        self.stop_reason: str | None = None       # the reason of the first Stop the node sent (T4.1)
        self._capacity = max(1, int(capacity))
        self._cap = int(cap_bytes)
        self._max_lag = None if max_policy_lag is None else int(max_policy_lag)
        self._min_version: int | None = None
        self._stall_sec = float(stall_sec)
        self._clock = clock                  # the reader's clock (stall deadlines, report cadence)
        self._decode = decode_chunk
        self._frames: collections.deque[bytes] = collections.deque()
        self._cond = threading.Condition()
        self._dropped: Counter = Counter()
        self._new: list[mpc.Connection] = []
        self._readers: dict[mpc.Connection, _FrameReader] = {}      # used by the reader thread only
        self._in_progress = 0         # frames in progress when the reader last decided whether it is full
        self._wake_r, self._wake_w = mpc.Pipe(duplex=False)
        self._closed = threading.Event()
        self.stop_flag = threading.Event()
        self.env_steps = 0
        self._read_thread = threading.Thread(target=self._read_loop, name=f"intake-read-{agent_id}", daemon=True)
        self._read_thread.start()
        self._acceptor = Acceptor(lst, self._on_connection, authkey=authkey, name=f"intake-{agent_id}")

    # ---- ChunkIntake ---------------------------------------------------------------------------------------

    @property
    def value(self) -> int:
        """The hub's global env-step counter (``learner_process``'s ``progress_counter``)."""
        return self.env_steps

    @property
    def depth(self) -> int:
        with self._cond:
            return len(self._frames)

    @property
    def dropped(self) -> dict[str, int]:
        with self._cond:
            return dict(self._dropped)

    def note_policy_version(self, version: int) -> None:
        """The learner's current version (before each batch): chunks below ``version - max_policy_lag`` are stale."""
        self._min_version = None if self._max_lag is None else int(version) - self._max_lag

    def get(self, timeout: float) -> TrajectoryChunk:
        """The next valid chunk; queue.Empty after ``timeout`` (the whole call's budget: dropped frames never
        extend it)."""
        deadline = time.monotonic() + timeout
        while True:
            with self._cond:
                while not self._frames:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise queue.Empty
                    self._cond.wait(left)
                frame = self._frames.popleft()
                was_full = len(self._frames) + 1 + self._in_progress >= self._capacity   # as _full
            if was_full:
                self._wake()
            chunk = self._accept(frame)
            if chunk is not None:
                return chunk

    def _accept(self, frame: bytes) -> TrajectoryChunk | None:
        from colosseum.net.chunks import ChunkError, chunk_meta
        from colosseum.net.frame import FrameError, peek_meta

        try:
            kind, raw_meta = peek_meta(frame)
            if kind != "chunk":
                raise FrameError("kind", f"a {kind!r} frame on a chunk connection")
            meta = chunk_meta(raw_meta)
        except (FrameError, ChunkError):
            self._drop("invalid")
            return None
        if meta.agent_id != self.agent_id:
            self._drop("invalid")                     # a misrouted chunk
            return None
        gen = meta.generation
        if gen != 0 and gen != self.generation and (gen > self.generation or meta.policy_version >= self.version_floor):
            self._drop("stale_generation")
            return None
        min_version = self._min_version
        if min_version is not None and meta.policy_version < min_version:
            self._drop("stale")                       # by its header: a stale frame is never decoded
            return None
        try:
            chunk, _ = self._decode(frame, cap=self._cap, validator=self._validator)
        except ChunkError as e:
            self._drop(e.reason)
            return None
        return chunk

    def _drop(self, reason: str, n: int = 1) -> None:
        with self._cond:
            first = self._dropped[reason] == 0
            self._dropped[reason] += n
        if first:
            logger.warning(f"learner {self.agent_id}: dropped a chunk ({reason}); further drops are only counted "
                           f"(dropped_chunks)")

    # ---- threads --------------------------------------------------------------------------------------------

    def _wake(self) -> None:
        try:
            self._wake_w.send_bytes(b"w")
        except OSError:
            pass

    def _on_connection(self, conn: mpc.Connection, hello: ChildHello) -> None:
        if hello.channel != "chunks" or self._closed.is_set():
            conn.close()
            return
        with self._cond:
            self._new.append(conn)
        self._wake()

    def _read_loop(self) -> None:
        try:
            self._read_forever()
        except Exception:  # noqa: BLE001 - a reader bug: the learner must not starve without a stop
            logger.exception(f"learner {self.agent_id}: the chunk intake's reader failed; stopping the learner")
            self.stop_flag.set()
            self._closed.set()

    def _read_forever(self) -> None:
        last_report = self._clock()
        control = self._control.conn
        while not self._closed.is_set():
            with self._cond:
                new, self._new = self._new, []
            for conn in new:
                self._add_reader(conn)
            now = self._clock()
            self._cut_stalled(now)
            started = [conn for conn, reader in self._readers.items() if reader.started]
            self._in_progress = len(started)
            full = self._full(len(started))
            waitables: list[Any] = [control, self._wake_r] + (started if full else list(self._readers))
            try:
                ready = mpc.wait(waitables, timeout=self._wait_timeout(now))
            except OSError:
                ready = []
            for obj in ready:
                if obj is self._wake_r:
                    self._drain_wake()
                elif obj is control:
                    self._read_control()
                elif obj in self._readers:
                    self._read_frame(obj)
            if self._clock() - last_report >= INTAKE_REPORT_SEC:
                last_report = self._clock()
                self._control.send(IntakeReport(self.depth, self.dropped))

    def _full(self, started: int) -> bool:
        """No room for one more frame: the deque plus the ``started`` frames in progress fill ``capacity``."""
        with self._cond:
            return len(self._frames) + started >= self._capacity

    def _add_reader(self, conn: mpc.Connection) -> None:
        try:
            self._readers[conn] = _FrameReader(conn, self._cap)
        except (OSError, ValueError):             # closed already: nothing of it was read
            self._close_conn(conn)

    @staticmethod
    def _readable(conn: mpc.Connection) -> bool:
        try:
            return bool(conn.poll(0))
        except (OSError, ValueError, EOFError):
            return False

    def _wait_timeout(self, now: float) -> float:
        """The reader's wait: at most 0.5 s, and no later than the earliest stall deadline of a frame in progress."""
        timeout = 0.5
        for reader in self._readers.values():
            if reader.started:
                timeout = min(timeout, reader.last_progress + self._stall_sec - now)
        return max(0.01, timeout)

    def _cut_stalled(self, now: float) -> None:
        """Close every connection silent for ``stall_sec`` inside a frame; the frame is lost (``disconnect``). A
        connection whose socket holds data now is never cut (its bytes are read first): after a pause of this reader
        itself (a stopped or frozen learner) healthy senders are past the deadline too."""
        for conn, reader in list(self._readers.items()):
            if reader.started and now - reader.last_progress >= self._stall_sec and not self._readable(conn):
                logger.warning(f"learner {self.agent_id}: a chunk sender stopped inside a frame for "
                               f"{now - reader.last_progress:.1f} s; its connection is closed and the frame lost")
                self._close_conn(conn)
                self._drop("disconnect")

    def _drain_wake(self) -> None:
        try:
            while self._wake_r.poll(0):
                self._wake_r.recv_bytes()
        except (OSError, EOFError):
            pass

    def _read_control(self) -> None:
        try:
            msg = self._control.conn.recv()
        except (OSError, EOFError):
            self.stop_flag.set()                  # the node is gone: stop (final snapshot is best effort)
            self._closed.set()
            return
        except Exception:  # noqa: BLE001 - a malformed message: the learner's only link to its node is broken
            logger.exception(f"learner {self.agent_id}: unreadable control message from the node; stopping")
            self.stop_flag.set()
            self._closed.set()
            return
        if isinstance(msg, LearnerEnvSteps):
            self.env_steps = int(msg.env_steps)
        elif isinstance(msg, Stop):
            if self.stop_reason is None:
                self.stop_reason = str(msg.reason)    # SP5 T4.1: superseded / hub_lost send no final snapshot
            self.stop_flag.set()
        else:
            logger.warning(f"learner {self.agent_id}: unexpected control message {type(msg).__name__}")

    def _read_frame(self, conn: mpc.Connection) -> None:
        reader = self._readers[conn]
        if not reader.started and self._full(sum(r.started for r in self._readers.values())):
            return                                # full again since the wait: leave the frame to its sender
        try:
            frame = reader.read(self._clock())
        except EOFError:                          # the sender closed between frames: nothing lost
            self._close_conn(conn)
            return
        except _FrameTooLarge:                    # its body is never read: the connection closes
            self._close_conn(conn)
            self._drop("invalid")
            return
        except (_FrameCut, OSError):              # cut mid-frame (a killed sender), or a socket error
            self._close_conn(conn)
            self._drop("disconnect")
            return
        except Exception:  # noqa: BLE001 - only this connection fails
            logger.exception(f"learner {self.agent_id}: reading a chunk connection failed; it is closed")
            self._close_conn(conn)
            self._drop("disconnect")
            return
        if frame is None:                         # the rest of the frame is still on its way
            return
        with self._cond:
            self._frames.append(frame)
            self._cond.notify()

    def _close_conn(self, conn: mpc.Connection) -> None:
        self._readers.pop(conn, None)
        try:
            conn.close()
        except OSError:
            pass

    def close(self, timeout: float = 2.0) -> None:
        self._closed.set()
        self._wake()
        self._acceptor.close(timeout)
        self._read_thread.join(timeout)
        for conn in list(self._readers) + self._new + [self._wake_r, self._wake_w]:
            try:
                conn.close()
            except OSError:
                pass
        try:
            self._listener.close()
        except OSError:
            pass


class CheckpointTooLargeError(ValueError):
    """A checkpoint frame above the message cap: neither the node nor the hub would take it."""


def checkpoint_too_large(agent_id: str, kind: str, version: int, size: int, cap: int) -> str:
    """The one text for a checkpoint above the cap (learner and node): agent, kind, version, size, cap and the
    setting to raise."""
    from colosseum.net.frame import DECOMPRESSED_SIZE_FACTOR

    mb = cap // (DECOMPRESSED_SIZE_FACTOR * 2**20)
    return (f"the {kind} checkpoint v{version} of learner {agent_id!r} is {size / 2**20:.1f} MiB, above the cap of "
            f"{cap / 2**20:.0f} MiB ({DECOMPRESSED_SIZE_FACTOR} x distributed.max_message_mb = {mb}); raise "
            f"distributed.max_message_mb")


class NodeOutputs:
    """``LearnerOutputs`` of a node's learner (module docstring): weights into the cache + ``VersionPublished``,
    checkpoints as ``CheckpointHeader`` + frame, metrics as ``LearnerMetrics``; all on ``control``. ``cap_bytes``:
    a checkpoint frame above it raises ``CheckpointTooLargeError`` (the learner then fails with that reason instead of
    a frame the node cannot read)."""

    checkpoints = True

    def __init__(self, control: ControlChannel, cache: WeightCacheWriter, *, agent_id: str, generation: int,
                 cap_bytes: int | None = None) -> None:
        self._control = control
        self._cache = cache
        self.agent_id = agent_id
        self.generation = int(generation)
        self._cap = cap_bytes
        self._cache_error_logged = False

    def publish_weights(self, payload: WeightPayload) -> None:
        from colosseum.net.chunks import WeightsMeta, encode_weights
        from colosseum.node.weights import CacheFullError

        meta = WeightsMeta(agent_id=self.agent_id, kind="latest", generation=self.generation,
                           policy_version=int(payload.policy_version), teacher_active=bool(payload.teacher_active))
        frame = encode_weights(payload.state_dict, meta)
        try:
            segment, digest = self._cache.publish_latest(self.agent_id, frame, version=meta.policy_version,
                                                         generation=self.generation,
                                                         teacher_active=meta.teacher_active)
        except CacheFullError as e:
            logger.error(f"learner {self.agent_id}: v{meta.policy_version} not published: {e}",
                         exc_info=not self._cache_error_logged)
            self._cache_error_logged = True
            return
        self._control.send(VersionPublished(self.agent_id, self.generation, meta.policy_version, segment,
                                            len(frame), digest, meta.teacher_active))

    def send_checkpoint(self, payload: dict, *, kind: Literal["periodic", "final", "recovery"], block: bool) -> bool:
        """Header + frame on the control connection (blocking while the node reads; ``block`` is accepted for the
        protocol: a socket send has no non-blocking form that keeps the frame whole). False: the node is gone."""
        from colosseum.net.chunks import encode_checkpoint

        frame = encode_checkpoint(payload, kind=kind, generation=self.generation)
        if self._cap is not None and len(frame) > self._cap:
            raise CheckpointTooLargeError(checkpoint_too_large(self.agent_id, kind, int(payload["policy_version"]),
                                                               len(frame), self._cap))
        header = CheckpointHeader(kind, self.agent_id, self.generation, int(payload["policy_version"]), len(frame))
        return self._control.send_with_frame(header, frame)

    def send_metrics(self, metrics: dict) -> None:
        self._control.send(LearnerMetrics(dict(metrics)))

    def finish(self, stop_event: Any) -> None:
        """Nothing to release (the cache segments stay for the readers)."""


class QueueIntake:
    """``ChunkIntake`` over SP4's trajectory ``mp.Queue`` of payload dicts (decode + slot check, misrouted agent ->
    ValueError, as SP4's ``collect_batch``). ``max_policy_lag`` as ``ConnectionIntake``'s, checked on the decoded
    chunk's version (queue items are in memory already); stale chunks are counted in ``dropped``. Picklable (SP4's
    learner process receives it): no lock; the counter has one writer, the thread calling ``get``."""

    def __init__(self, q: Any, *, agent_id: str | None = None, max_policy_lag: int | None = None) -> None:
        self._q = q
        self.agent_id = agent_id
        self._max_lag = None if max_policy_lag is None else int(max_policy_lag)
        self._min_version: int | None = None
        self._stale = 0

    def note_policy_version(self, version: int) -> None:
        """The learner's current version (before each batch): chunks below ``version - max_policy_lag`` are stale."""
        self._min_version = None if self._max_lag is None else int(version) - self._max_lag

    def get(self, timeout: float) -> TrajectoryChunk:
        """The next admissible chunk; queue.Empty after ``timeout`` (the whole call's budget)."""
        deadline = time.monotonic() + timeout
        while True:
            chunk = self._take(max(0.0, deadline - time.monotonic()))
            min_version = self._min_version
            if min_version is None or chunk.policy_version >= min_version:
                return chunk
            self._stale += 1
            if self._stale == 1:
                logger.warning(f"learner {self.agent_id}: dropped a chunk (stale); further drops are only counted "
                               f"(dropped_chunks)")

    def _take(self, timeout: float) -> TrajectoryChunk:
        from colosseum.core.types import validate_slot_structure

        item = self._q.get(timeout=timeout)
        if isinstance(item, TrajectoryChunk):
            return item
        if not isinstance(item, dict):
            raise TypeError(f"trajectory queue item must be a chunk payload dict, got {type(item).__name__}")
        chunk = TrajectoryChunk.from_payload(item)
        if self.agent_id is not None and chunk.agent_id != self.agent_id:
            raise ValueError(f"learner of agent {self.agent_id!r} received a chunk of agent {chunk.agent_id!r}; "
                             f"check the workers' learner addresses (run-workers -l AGENT=HOST:PORT)")
        validate_slot_structure(chunk)
        return chunk

    @property
    def depth(self) -> int:
        try:
            return int(self._q.qsize())
        except (NotImplementedError, AttributeError):
            return -1

    @property
    def dropped(self) -> dict[str, int]:
        stale = self._stale
        return {"stale": stale} if stale else {}


@dataclass
class QueueOutputs:
    """``LearnerOutputs`` over SP4's queues: newest-wins weight mailboxes, the checkpoint queue (final: blocking put
    and flush), the metrics queue (never blocks); ``finish`` is SP4's ``_release_weight_queues``."""

    weight_queues: list = field(default_factory=list)
    checkpoint_queue: Any = None
    metrics_queue: Any = None
    trajectory_queue: Any = None
    weight_sync_interval: float = 5.0

    @property
    def checkpoints(self) -> bool:
        return self.checkpoint_queue is not None

    def publish_weights(self, payload: WeightPayload) -> None:
        from colosseum.core.ipc import put_latest

        for wq in self.weight_queues:
            if not put_latest(wq, payload):
                logger.debug(f"Learner [{payload.agent_id}]: weight mailbox busy, v{payload.policy_version} not "
                             f"delivered")

    def send_checkpoint(self, payload: dict, *, kind: Literal["periodic", "final", "recovery"], block: bool) -> bool:
        from colosseum.learner.learner import FINAL_CHECKPOINT_TIMEOUT_SEC, send_checkpoint
        from colosseum.utils.process import SHUTDOWN_GRACE_SEC, flush_queue

        if self.checkpoint_queue is None:
            return False
        sent = send_checkpoint(self.checkpoint_queue, payload, block=block, timeout=FINAL_CHECKPOINT_TIMEOUT_SEC)
        if kind == "final":
            flush_queue(self.checkpoint_queue, warn_after=SHUTDOWN_GRACE_SEC)
        return sent

    def send_metrics(self, metrics: dict) -> None:
        if self.metrics_queue is None:
            return
        try:
            self.metrics_queue.put_nowait(metrics)
        except queue.Full:
            pass

    def finish(self, stop_event: Any) -> None:
        from colosseum.learner.learner import _release_weight_queues, _weight_flush_timeout

        trajectory = self.trajectory_queue if self.trajectory_queue is not None else queue.Queue()
        _release_weight_queues(self.weight_queues, trajectory, stop_event,
                               timeout=_weight_flush_timeout(self.weight_sync_interval))
