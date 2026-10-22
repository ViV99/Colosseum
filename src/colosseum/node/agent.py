"""The node agent: one per machine (SP5 spec 1, 2.6, 4.1-4.4, 4.6, 4.7, 7.1, 7.7; rulings R3, R10-R13).

``NodeAgent.run`` is one loop on the calling thread (the "main-loop thread"; it alone starts children, because
``PR_SET_PDEATHSIG`` fires when the creating thread exits). What it does:

1. **Identity.** ``options.local`` (``train``'s node, R12): ``credentials`` = ``(node_id, number, node_secret)`` from
   the hub, no ``Register``. Otherwise ``NodeIdentity.load_or_create(token.ca_fingerprint, identity_root)``; without
   a registration: the hub CA (``fetch_hub_ca``, unless ``insecure`` / ``loopback_plain``), ``Register`` with the
   token secret, ``save_registration``. A refusal (``UNAUTHENTICATED`` / ``PERMISSION_DENIED``) is exit 1.
2. **Session.** A ``HubSession`` whose ``hello`` carries the hardware, the slots, the live cells, the session's
   cumulative env steps and the last acked result seq (built from a snapshot the loop refreshes every pass: the
   session thread never reads the loop's books).
3. **Welcome** (once per run; a second one after a reconnect starts nothing new): the config, the env spec, the
   agents, the shared-memory cache (``cleanup_dead``, the node lock once per run, writer and reader), the children's
   listener (``Acceptor`` with an ``os.urandom(32)`` key), then every ``Welcome.fixed`` blob fetched into the cache;
   then ``workers`` workers (``worker-<i>``) and the learners of ``Welcome.learners`` (``learner-<agent>``; their
   ``recovery`` / ``trainer_state`` start blobs, A28, are downloaded first; teacher and ``init`` weights come from the
   ``Welcome.fixed`` blobs in the cache). A second Welcome of the same run starts the learners it lists that do not
   run here and stops the ones it no longer lists.
4. **Children** run under ``ChildSupervisor``. Each pass waits on the wake pipe, the children's connections and
   their sentinels, reads the ready connections first, then the inbox, then the exits (P-PF2-11: an exited child's
   control connection is read to its end, bounded by ``EXIT_DRAIN_SEC``, before it counts as "reported" or "died").
   A worker that reported a ``ChildError`` is not restarted (the hub stops the run, A12); one that died without a
   report is ``proc_died`` and restarts with a new incarnation (new cells, new epochs) within ``WORKER_RESTARTS``,
   else its slot is ``slot_disabled``. A learner that ends without being stopped is ``learner_failed`` (until T4.2:
   the hub stops the run); workers are routed away from it (``LearnerRoute(agent, None)``).
5. **Messages from children.** ``WorkerResults`` -> ``ResultBuffer``; ``WorkerProgress`` -> per (worker,
   incarnation) cumulative env steps (their sum is the session's counter: it never decreases) and counters
   (``MetricsBatch.workers``); ``ChildError`` -> ``Event(child_error)`` with ``node <n> (<name>), `` prepended to the
   child's context. Learners (``node.learners``): their first ``VersionPublished`` -> ``LearnerRoute`` to every
   worker (and to every worker that connects later); ``VersionPublished`` -> ``VersionAnnounce`` (the latest one
   kept while the hub is away);
   ``CheckpointHeader`` + frame -> an ``Upload`` in the transfer pool (``checkpoint`` / ``final``), its reply logged (a
   refused periodic one is a WARNING, a refused final an ERROR); ``LearnerMetrics`` -> ``MetricsBatch.train``;
   ``IntakeReport`` -> ``MetricsBatch.intake``.
6. **Lineups (R11).** ``CellTracker.due()`` -> one ``LineupRequest`` per pass; a grant waits until every key of its
   seats is in the cache (a ``latest`` pointer, a snapshot, a frozen agent's blob), fetching the missing snapshots and
   blobs in the transfer pool; a failed fetch keeps the old lineup, counts ``lineup_fetch_failed`` per grant and
   re-requests after ``LINEUP_RETRY_SEC`` (WARNING once per key); a grant waiting longer than
   ``max(ttl, LINEUP_WAIT_SEC)`` is given up the same way (P-PF2-12). A worker gets its routes and ``WeightsReady``
   when its control connection opens, before its first ``WorkerLineups``. Snapshots no cell references any more
   leave the cache after ``CACHE_DROP_DELAY_SEC`` (safe by an ordering invariant, see ``_prune_cache``).
7. **Announces.** ``evictions`` -> ``WorkerEvict`` to every worker; ``env_steps`` -> ``LearnerEnvSteps`` to every
   learner (LR progress).
8. **Heartbeat** every ``distributed.heartbeat_sec``: env steps, ``dropped_results``, processes, load, cached keys
   and the node counters (``lineup_fetch_failed``, ``dropped_results_node_buffer``, ``worker_restarts``: A2).
   ``MetricsBatch`` every ``METRICS_FLUSH_SEC``; due result batches every pass, unacked ones again after
   ``RESULT_RESEND_SEC`` on a live session (P-PF2-4) and all of them after every reconnect.
9. **Commands.** ``stop_run`` stops (item 10); ``start_learner`` / ``stop_learner`` start or stop one learner
   (stopped, it sends its final snapshot, which is uploaded); ``release`` stops like item 10, then forgets the run
   (its children, cache files, lock and listener) and keeps the session (``once``: the node then exits 0; serving
   another run is T6.3's); ``set_evaluators`` / ``update_code`` (T5.1 / T8.3) are logged and ignored.
10. **Stop** (``stop_run`` or ``request_stop``): ``Stop`` to every child; the loop keeps reading results until every
    child exited or the grace passed (``LOCAL_CHILD_GRACE_SEC`` for the local node, ``distributed.shutdown_grace_sec -
    5`` otherwise); the rest is killed; the result buffer is flushed and its acks awaited until the grace's deadline +
    ``ACK_WAIT_EXTRA_SEC``, as are the uploads in flight (the learners' finals); what is still unacked then is counted
    in ``dropped_results_node_buffer`` (P-PF2-4 b); a last ``Heartbeat``, ``Event(run_stopped)``; the transfers
    still running are cancelled (``TRANSFER_JOIN_SEC``); the node's cache files and lock are removed; the session
    closes. Worst case of a local node (``LOCAL_CHILD_GRACE_SEC`` 5 s): 5 s grace + 2 s kill + 1 s acks / uploads +
    1 s transfer join = 9 s, plus the session's close (immediate while the hub answers; ``HubSession`` bounds it).

Threads besides the loop: the session reader (``HubSession``), the listener's ``Acceptor`` and the transfer pool;
they only post into the inbox (a ``SimpleQueue`` plus a non-blocking wake pipe in the loop's ``wait`` list). The
loop never blocks on the network and reads only connections ``wait`` reported ready.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing.connection as mpc
import os
import queue
import re
import shutil
import socket
import sys
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import grpc

from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.security import Token
from colosseum.node.books import CellTracker, ResultBuffer
from colosseum.node.learners import LearnerBlobsDone, NodeLearners
from colosseum.node.procs import ChildSpec, ChildSupervisor
from colosseum.utils.process import SHUTDOWN_GRACE_SEC

logger = logging.getLogger(__name__)

__all__ = [
    "ACK_WAIT_EXTRA_SEC", "CACHE_DROP_DELAY_SEC", "EXIT_DRAIN_SEC", "LINEUP_RETRY_SEC", "LINEUP_WAIT_SEC",
    "LOCAL_CHILD_GRACE_SEC", "METRICS_FLUSH_SEC", "RESULT_RESEND_SEC", "CellTracker", "NodeAgent", "NodeOptions",
    "ResultBuffer", "local_node_target", "main",
]

LINEUP_RETRY_SEC = 1.0          # a requested cell without an answer, or after a failed fetch, is asked again
LINEUP_WAIT_SEC = 30.0          # a grant waiting longer than max(ttl, this) for its weights is given up (P-PF2-12)
RESULT_RESEND_SEC = 10.0        # an unacked result batch is sent again after this long on a live session (P-PF2-4)
METRICS_FLUSH_SEC = 0.5         # MetricsBatch cadence
LOCAL_CHILD_GRACE_SEC = SHUTDOWN_GRACE_SEC - 2.0   # children of train's local node: before the hub's 7 s
REMOTE_GRACE_MARGIN_SEC = 5.0   # remote nodes: distributed.shutdown_grace_sec minus this
ACK_WAIT_EXTRA_SEC = 1.0        # result acks are awaited until the children's deadline plus this
CACHE_DROP_DELAY_SEC = 10.0     # an unreferenced snapshot stays this long in the cache before it is removed
EXIT_DRAIN_SEC = 1.0            # an exited child's control connection is read to its end within this (P-PF2-11)
FIXED_RETRY_SEC = 2.0           # a failed fetch of a Welcome.fixed blob is retried after this
TRANSFER_THREADS = 4
TRANSFER_JOIN_SEC = 1.0         # at the end, cancelled transfers are waited for this long before the cache cleanup
DOWNLOAD_TIMEOUT_SEC = 300.0
UPLOAD_TIMEOUT_SEC = 300.0
_UPLOAD_KINDS = {"periodic": "checkpoint", "final": "final", "recovery": "recovery"}
REGISTER_TIMEOUT_SEC = 30.0
FETCH_CA_TIMEOUT_SEC = 10.0
CONNECT_RETRY_SEC = (0.5, 10.0)  # registration retries while the hub is unreachable (doubling)
PENDING_EVENTS_KEPT = 100       # events kept while the hub is away (oldest dropped)
LOOP_WAIT_SEC = 0.05
CHILD_MESSAGES_PER_PASS = 256   # messages read from one child connection per pass (others get their turn)
CACHE_PRUNE_SEC = 1.0
_REFUSALS = (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED,
             grpc.StatusCode.FAILED_PRECONDITION)


# ---------------------------------------------------------------------------------------------------------------
# Options, inbox events, per-run state
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeOptions:
    hub_address: str
    token: Token
    insecure: bool = False
    loopback_plain: bool = False
    workers: int | None = None
    evaluators: int | None = None
    cpu_learners: int | None = None
    no_learners: bool = False
    gpus: tuple[int, ...] | None = None
    name: str | None = None
    tags: tuple[str, ...] = ()
    once: bool = False
    identity_root: Path | None = None
    log_dir: Path | None = None
    local: bool = False                # train's local node (R12): unlimited learner slots, no Bootstrap, hub's sys.path
    data_port: int | None = None       # None: no data port; 0 ephemeral; -1 distributed.data_port (A23; T3.1)
    credentials: tuple[str, int, str] | None = None   # the local node's (node_id, number, node_secret) from the hub
    shutdown_grace_sec: float | None = None           # None: LOCAL_CHILD_GRACE_SEC / shutdown_grace_sec - 5


@dataclass
class _Accepted:
    conn: mpc.Connection
    hello: Any


@dataclass
class _FetchDone:
    key: Any
    error: str | None
    generation: int                    # the run this fetch belongs to (a fetch of an older run is ignored)


@dataclass
class _StopRequested:
    reason: str


@dataclass
class _UploadDone:
    token: str                         # the upload's key in NodeAgent._uploads
    kind: str                          # the checkpoint kind (periodic / final / recovery)
    agent_id: str
    version: int
    reply: Any                         # pb.UploadReply, or None after a transfer error
    error: str | None


class _BadSeat(ValueError):
    """A grant names a player this node cannot seat; ``player`` = (agent, network) (one WARNING per player)."""

    def __init__(self, message: str, agent: str, network: str) -> None:
        super().__init__(message)
        self.player = (agent, network)


class _Refused(Exception):
    """The hub refused this node (registration or identity); ``run`` returns 1."""


@dataclass
class _Run:
    """What the node knows of the run it serves (from the first ``Welcome``)."""

    serial: int
    run_id: str
    number: int
    config: Any
    spec: Any
    agent_ids: list[str]
    agent_roles: dict[str, list[str]]
    agent_configs: dict[str, Any]
    role_specs: dict[str, Any]
    bots: dict[str, Any]
    fixed_roles: dict[str, tuple[str, ...]]
    teachers: dict[str, Any]
    fixed_keys: dict[Any, str]          # every Welcome.fixed blob key -> its kind (kept in the cache for the run)
    frozen_keys: dict[str, Any]         # frozen agent -> its fixed blob key
    root: Path
    prefix: str
    writer: Any
    reader: Any
    lock: Path | None
    authkey: bytes
    node_address: str
    listener: Any
    acceptor: Any
    log_dir: Path
    results: ResultBuffer
    heartbeat_sec: float
    envs: int
    workers: int
    grace: float
    started: bool = False
    fixed_retry_at: dict = field(default_factory=dict)
    closed: bool = False                # torn down: transfers of this run put nothing into the cache


def _hostname_name() -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", socket.gethostname())[:63].lstrip("-_.")
    return name or "node"


def _split_address(address: str) -> tuple[str, int]:
    host, _, port = address.rpartition(":")
    return host.strip("[]") or "127.0.0.1", int(port)


def _rpc_text(e: BaseException) -> str:
    if isinstance(e, grpc.RpcError) and hasattr(e, "code"):
        return f"{e.code().name}: {e.details() if hasattr(e, 'details') else ''}".rstrip(": ")
    return f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------------------------------------------


class NodeAgent:
    """The node agent (module docstring). ``run`` blocks; ``request_stop`` is thread-safe. ``supervisor``,
    ``cells`` and ``results`` are the loop's books (tests read them; only the loop changes them)."""

    def __init__(self, options: NodeOptions) -> None:
        self.options = options
        self.session_id = uuid.uuid4().hex
        self.supervisor = ChildSupervisor()
        self.cells = CellTracker(retry_sec=LINEUP_RETRY_SEC, clock=time.monotonic)
        self._counters: Counter = Counter(lineup_fetch_failed=0, worker_restarts=0)
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._wake_r, self._wake_w = os.pipe()
        os.set_blocking(self._wake_r, False)
        os.set_blocking(self._wake_w, False)
        self._closed = False
        self._wake_lock = threading.Lock()
        self._run: _Run | None = None
        self._run_serial = 0
        self._session: Any = None
        self._channel: grpc.Channel | None = None
        self._node_id = ""
        self._node_secret = ""
        self._number = -1
        self._name = options.name or ("local" if options.local else _hostname_name())
        self._max_mb = 64
        self._hw: Any = None
        self._connected = False
        self._pool = ThreadPoolExecutor(max_workers=TRANSFER_THREADS, thread_name_prefix="node-transfer")
        self._fetching: dict[Any, Future] = {}
        self._fetch_warned: set = set()
        self._refused_players: set = set()        # (agent, network) of refused grants (warned once)
        self._cached: set = set()                   # snapshot keys this node put into the cache
        self._unref_since: dict[Any, float] = {}
        self._control: dict[str, mpc.Connection] = {}
        self._child_conns: dict[mpc.Connection, str] = {}
        self._route_conns: set[mpc.Connection] = set()
        self._reported: set[str] = set()           # children that sent a ChildError
        self._disabled: set[str] = set()
        self._progress: dict[tuple[int, int], int] = {}            # (worker, incarnation) -> cumulative env steps
        self._worker_stats: dict[tuple[int, int], dict[str, int]] = {}
        self._stats_changed: set[tuple[int, int]] = set()
        self._learner_routes: dict[str, Any] = {}   # agent -> LearnerRoute sent to every worker
        self.learners = NodeLearners(_LearnerHost(self))   # the learners this node runs (node.learners)
        self._uploads: dict[str, tuple[Future, str, str, int]] = {}   # token -> (future, kind, agent, version)
        self._released: set[str] = set()            # run ids this node released (never served again)
        self._releasing = False
        self._dropped_carry = 0                     # dropped_results_node_buffer of released runs (per session)
        self._heartbeat_sec = 1.0
        self._events: deque = deque(maxlen=PENDING_EVENTS_KEPT)
        self._hello_lock = threading.Lock()
        self._hello_state: tuple = ("", (), 0, 0, 0, ())
        self._next_heartbeat = 0.0
        self._next_metrics = 0.0
        self._next_prune = 0.0
        self._stopping = False
        self._stop_phase = ""
        self._children_deadline = 0.0
        self._ack_deadline = 0.0
        self._done = False
        self._exit_code = 0

    # ---- public ---------------------------------------------------------------------------------------------

    def request_stop(self) -> None:
        self._post(_StopRequested("stop requested"))

    @property
    def results(self) -> ResultBuffer | None:
        return None if self._run is None else self._run.results

    @property
    def run_id(self) -> str | None:
        return None if self._run is None else self._run.run_id

    @property
    def number(self) -> int:
        return self._number

    @property
    def cache_prefix(self) -> str | None:
        return None if self._run is None else self._run.prefix

    def run(self) -> int:
        """The node's loop until its stop; returns the exit code (0 after a stop, 1 after a refusal or a failure)."""
        try:
            try:
                identity = self._identity()
            except _Refused as e:
                logger.error(f"node: {e}")
                return 1
            if identity is None:
                return 0
            self._open_session(*identity)
            while not self._done:
                self._pass()
            return self._exit_code
        except Exception:  # noqa: BLE001 - a bug in the loop: stop the children, report, exit 1
            logger.exception("node: the main loop failed")
            return 1
        finally:
            self._close()

    # ---- identity and session -------------------------------------------------------------------------------

    def _post(self, item: object) -> None:
        """Any thread: put ``item`` into the inbox and wake the loop. The wake pipe is written under ``_wake_lock``,
        which ``_close`` holds while it marks the node closed, so no byte reaches a reused descriptor number."""
        self._inbox.put(item)
        with self._wake_lock:
            if self._closed:
                return
            try:
                os.write(self._wake_w, b"w")
            except (BlockingIOError, OSError):    # a wake is pending already
                pass

    def _plain(self) -> bool:
        return self.options.insecure or self.options.loopback_plain

    def _identity(self) -> tuple[str, str, grpc.ChannelCredentials | None] | None:
        """``(node_id, node_secret, channel credentials)``; None when stopped meanwhile; _Refused on a refusal."""
        from colosseum.net.security import (
            NodeIdentity,
            SecurityError,
            fetch_hub_ca,
            hub_channel_credentials,
        )

        opts = self.options
        self._hw = self._probe_hardware()
        if opts.local:
            if opts.credentials is None:
                raise _Refused("a local node needs the hub's credentials (NodeOptions.credentials)")
            node_id, number, secret = opts.credentials
            self._number = int(number)
            if self._plain():
                return node_id, secret, None
            host, port = _split_address(opts.hub_address)
            try:
                ca_pem = fetch_hub_ca(host, port, fingerprint=opts.token.ca_fingerprint, timeout=FETCH_CA_TIMEOUT_SEC)
            except SecurityError as e:
                raise _Refused(str(e)) from None
            return node_id, secret, hub_channel_credentials(ca_pem)
        try:
            identity = NodeIdentity.load_or_create(opts.token.ca_fingerprint, opts.identity_root)
        except SecurityError as e:
            raise _Refused(str(e)) from None
        if not identity.registered:
            reply = self._register(identity)
            if reply is None:
                return None
            try:
                identity.save_registration(reply, hub_addresses=[opts.hub_address])
            except SecurityError as e:
                raise _Refused(str(e)) from None
            logger.info(f"node: registered with the hub as node {identity.number} ({identity.name}, "
                        f"{identity.node_id[:8]})")
        self._number = int(identity.number)
        self._name = opts.name or identity.name or self._name
        if self._plain():
            return identity.node_id, identity.secret, None
        return identity.node_id, identity.secret, hub_channel_credentials((identity.path / "ca.crt").read_bytes())

    def _register(self, identity: Any) -> Any:
        """``Register`` with the token (retried while the hub is unreachable); None when stopped meanwhile."""
        from colosseum.net.grpc_util import channel_options, token_metadata
        from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc
        from colosseum.net.security import (
            HUB_SERVER_NAME,
            HubUnreachableError,
            SecurityError,
            fetch_hub_ca,
            hub_channel_credentials,
        )
        from colosseum.node.hardware import machine_id

        opts = self.options
        request = pb.RegisterRequest(public_key_pem=identity.public_key_pem().decode("ascii"), machine_id=machine_id(),
                                     name=self._name, tags=list(opts.tags), node_id=identity.node_id or "")
        delay = CONNECT_RETRY_SEC[0]
        while True:
            channel = None
            try:
                if self._plain():
                    channel = grpc.insecure_channel(opts.hub_address, options=channel_options(self._max_mb))
                else:
                    host, port = _split_address(opts.hub_address)
                    ca_pem = fetch_hub_ca(host, port, fingerprint=opts.token.ca_fingerprint,
                                          timeout=FETCH_CA_TIMEOUT_SEC)
                    channel = grpc.secure_channel(opts.hub_address, hub_channel_credentials(ca_pem),
                                                  options=channel_options(self._max_mb, server_name=HUB_SERVER_NAME))
                return pb_grpc.HubStub(channel).Register(request, metadata=token_metadata(opts.token),
                                                         timeout=REGISTER_TIMEOUT_SEC)
            except grpc.RpcError as e:
                code = e.code() if hasattr(e, "code") else None
                if code in _REFUSALS:
                    raise _Refused(f"the hub refused the registration ({_rpc_text(e)})") from None
                logger.warning(f"node: cannot register with the hub at {opts.hub_address} yet ({_rpc_text(e)}); "
                               f"retrying in {delay:.1f} s")
            except HubUnreachableError as e:                             # nothing refused: the hub is not up (yet)
                logger.warning(f"node: {e}; retrying in {delay:.1f} s")
            except SecurityError as e:
                raise _Refused(str(e)) from None
            finally:
                if channel is not None:
                    channel.close()
            if self._wait_or_stop(delay):
                return None
            delay = min(CONNECT_RETRY_SEC[1], delay * 2)

    def _wait_or_stop(self, delay: float) -> bool:
        """Sleep ``delay`` unless a stop is requested (True: stop)."""
        end = time.monotonic() + delay
        while True:
            try:
                item = self._inbox.get(timeout=max(0.0, min(0.1, end - time.monotonic())))
                if isinstance(item, _StopRequested):
                    return True
            except queue.Empty:
                pass
            if time.monotonic() >= end:
                return False

    def _open_session(self, node_id: str, secret: str, credentials: grpc.ChannelCredentials | None) -> None:
        from colosseum.net.grpc_util import channel_options
        from colosseum.net.security import HUB_SERVER_NAME
        from colosseum.node.session import HubSession

        self._node_id, self._node_secret = node_id, secret
        if credentials is None:
            self._channel = grpc.insecure_channel(self.options.hub_address, options=channel_options(self._max_mb))
        else:
            self._channel = grpc.secure_channel(self.options.hub_address, credentials,
                                                options=channel_options(self._max_mb, server_name=HUB_SERVER_NAME))
        self._session = HubSession(self.options.hub_address, credentials=credentials, node_id=node_id,
                                   node_secret=secret, hello=self._hello, inbox=self._post,
                                   max_message_mb=self._max_mb)
        self._session.start()

    def _probe_hardware(self) -> Any:
        from colosseum.node.hardware import probe

        return probe()

    def _slots(self) -> pb.Slots:
        from colosseum.node.hardware import default_slots

        opts, envs = self.options, self._hello_state[4]
        if opts.local:
            return pb.Slots(workers=int(opts.workers or 0), evaluators=int(opts.evaluators or 0), cpu_learners=-1,
                            gpus=list(opts.gpus or ()), no_learners=opts.no_learners, envs_per_worker=envs)
        d = default_slots(self._hw)
        return pb.Slots(workers=int(opts.workers if opts.workers is not None else d["workers"]),
                        evaluators=int(opts.evaluators if opts.evaluators is not None else d["evaluators"]),
                        cpu_learners=int(opts.cpu_learners if opts.cpu_learners is not None else d["cpu_learners"]),
                        gpus=list(opts.gpus if opts.gpus is not None else d["gpus"]), no_learners=opts.no_learners,
                        envs_per_worker=envs)

    def _hardware(self) -> pb.Hardware:
        from colosseum.net.security import cache_home
        from colosseum.node.weights import shm_dir

        hw = self._hw

        def free(path: Path) -> int:
            for p in (path, *path.parents):              # the nearest existing ancestor
                try:
                    return int(shutil.disk_usage(p).free)
                except OSError:
                    continue
            return 0

        return pb.Hardware(logical_cores=hw.logical_cores, physical_cores=hw.physical_cores,
                           memory_bytes=hw.memory_bytes,
                           gpus=[pb.Gpu(index=int(g.get("index", 0)), name=str(g.get("name", "")),
                                        memory_bytes=int(g.get("memory_bytes", 0)), kind=str(g.get("kind", "")))
                                 for g in hw.gpus],
                           os=hw.os, platform=hw.platform, python=hw.python, torch_version=hw.torch_version,
                           torch_build=hw.torch_build, shm_free_bytes=free(shm_dir()),
                           cache_free_bytes=free(cache_home()))

    def _hello(self) -> pb.Hello:
        """Called by the session thread at every (re)connect: built from the loop's snapshot only."""
        from colosseum.net.messages import PROTOCOL_VERSION

        with self._hello_lock:
            run_id, live, env_steps, last_acked, _envs, learners = self._hello_state
        return pb.Hello(node_id=self._node_id, session_id=self.session_id, protocol_major=PROTOCOL_VERSION[0],
                        protocol_minor=PROTOCOL_VERSION[1], code_hash="", hardware=self._hardware(),
                        slots=self._slots(), data_addresses=[], run_id=run_id,
                        live_cells=[pb.LiveCell(worker=w, env=e, epoch=ep, incarnation=i) for w, e, ep, i in live],
                        env_steps=env_steps, last_acked_seq=last_acked,
                        learners=[pb.RunningLearner(agent_id=a, generation=g, version=v) for a, g, v in learners],
                        name=self._name,
                        tags=list(self.options.tags))

    def _update_hello_state(self) -> None:
        run = self._run
        state = ("" if run is None else run.run_id, tuple(self.cells.live()), self._env_steps(),
                 0 if run is None else run.results.last_acked, 0 if run is None else run.envs,
                 self.learners.hello_entries())
        with self._hello_lock:
            self._hello_state = state

    def _send(self, message: pb.NodeMessage) -> bool:
        return self._session is not None and self._session.send(message)

    def _event(self, kind: str, **fields: Any) -> None:
        """An ``Event`` to the hub; kept (up to ``PENDING_EVENTS_KEPT``) while the hub is away."""
        event = pb.Event(kind=kind, **fields)
        if not (self._connected and self._send(pb.NodeMessage(event=event))):
            self._events.append(event)

    # ---- the loop ---------------------------------------------------------------------------------------------

    def _pass(self) -> None:
        conns = list(self._child_conns)
        try:
            ready = mpc.wait([self._wake_r, *conns, *self.supervisor.sentinels()], timeout=LOOP_WAIT_SEC)
        except OSError:
            ready = []
        for obj in ready:                           # P-PF2-11: ready connections before the exits
            if isinstance(obj, mpc.Connection) and obj in self._child_conns:
                self._read_child(obj)
        if self._wake_r in ready:
            self._drain_wake()
        self._drain_inbox()
        for name, code in self.supervisor.exited():
            self._on_exit(name, code)
        self._tick(time.monotonic())
        self._update_hello_state()

    def _drain_wake(self) -> None:
        try:
            while os.read(self._wake_r, 4096):
                pass
        except (BlockingIOError, OSError):
            pass

    def _drain_inbox(self) -> None:
        while True:
            try:
                item = self._inbox.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, pb.HubMessage):
                self._on_hub_message(item)
            elif isinstance(item, _Accepted):
                self._on_child_hello(item.conn, item.hello)
            elif isinstance(item, _FetchDone):
                self._on_fetch_done(item)
            elif isinstance(item, _StopRequested):
                self._begin_stop(item.reason)
            elif isinstance(item, LearnerBlobsDone):
                self.learners.on_blobs(item)
            elif isinstance(item, _UploadDone):
                self._on_upload_done(item)
            else:
                self._on_session_event(item)

    def _on_session_event(self, item: object) -> None:
        from colosseum.node.session import SessionLost, SessionUp

        if isinstance(item, SessionUp):
            self._connected = True
            run = self._run
            if run is not None:
                resend = run.results.resend_all()
                for message in resend:
                    self._send(message)
                if resend:
                    logger.info(f"node: resent {len(resend)} unacknowledged result batches after the reconnect")
            while self._events:
                if not self._send(pb.NodeMessage(event=self._events[0])):
                    break
                self._events.popleft()
            self.learners.resend_pending()
        elif isinstance(item, SessionLost):
            if self._connected:
                logger.warning(f"node: the session with the hub ended ({item.reason}); reconnecting")
            self._connected = False

    def _tick(self, now: float) -> None:
        run = self._run
        if self._stopping:
            self._stop_step(now)
            if self._run is not run:                 # released in this pass: nothing of the old run is sent
                return
        elif run is not None:
            if not run.started:
                self._maybe_start_workers(run, now)
            self.learners.start_pending(run)
            self._request_lineups()
            self._expire_waiting()
            self._deliver(run)
            if now >= self._next_prune:
                self._next_prune = now + CACHE_PRUNE_SEC
                self._prune_cache(run, now)
        if self._done:
            return
        if run is None:                              # released (or no Welcome yet): heartbeats keep the session
            if self._released and self._connected and now >= self._next_heartbeat:
                self._next_heartbeat = now + self._heartbeat_sec
                self._send(pb.NodeMessage(heartbeat=pb.Heartbeat(
                    env_steps=self._env_steps(), dropped_results=self._dropped_carry,
                    counters=self._node_counters(None))))
            return
        for message in run.results.due():
            self._send(message)
        if self._connected:
            for message in run.results.overdue(RESULT_RESEND_SEC):
                self._send(message)
            if now >= self._next_heartbeat:
                self._next_heartbeat = now + run.heartbeat_sec
                self._send_heartbeat(run)
            if now >= self._next_metrics:
                self._next_metrics = now + METRICS_FLUSH_SEC
                self._send_metrics()

    # ---- hub messages -----------------------------------------------------------------------------------------

    def _on_hub_message(self, msg: pb.HubMessage) -> None:
        kind = msg.WhichOneof("m")
        if kind == "welcome":
            self._on_welcome(msg.welcome)
        elif kind == "grants":
            self._on_grants(msg.grants)
        elif kind == "ack":
            if self._run is not None:
                self._run.results.ack(msg.ack.seqs)
        elif kind == "command":
            self._on_command(msg.command)
        elif kind == "announce":
            self._on_announce(msg.announce)
        else:
            logger.debug(f"node: ignored a {kind} message (not handled by this node yet)")

    def _on_welcome(self, welcome: pb.Welcome) -> None:
        if self._stopping:
            return
        run = self._run
        if run is None:
            if self._released:
                logger.warning(f"node: run {welcome.run_id[:8]} not served: this node released its run (serving "
                               f"another run is the join command's, T6.3)")
                return
            try:
                self._start_run(welcome)
            except Exception as e:  # noqa: BLE001 - a config, env or cache problem: this node cannot serve the run
                logger.exception(f"node: cannot serve run {welcome.run_id[:8]} ({type(e).__name__}: {e})")
                self._exit_code = 1
                self._begin_stop("the run could not start on this node")
            return
        if welcome.run_id != run.run_id:
            logger.warning(f"node: the hub now runs {welcome.run_id[:8]}, this node serves {run.run_id[:8]}; "
                           f"ignored (switching runs is the join command's, T6.3)")
            return
        logger.info(f"node: session resumed for run {run.run_id[:8]} (node {run.number})")
        listed = {la.agent_id for la in welcome.learners}
        for agent in [a for a in self.learners.agents() if a not in listed]:
            self.learners.stop(agent, "the hub no longer assigns it here")
        for la in welcome.learners:
            self.learners.assign(run, la)

    def _start_run(self, welcome: pb.Welcome) -> None:
        from colosseum.core.config import ColosseumConfig
        from colosseum.core.registry import env_spec
        from colosseum.core.roles import agent_role_spec
        from colosseum.net.local import Acceptor, address, listener
        from colosseum.node.logs import node_log_dir
        from colosseum.node.weights import (
            WeightCacheReader,
            WeightCacheWriter,
            acquire_node_lock,
            blob_key,
            cache_prefix,
            cleanup_dead,
            shm_dir,
        )
        from colosseum.players.registry import BotSpec, resolve_player_roles

        config = ColosseumConfig.model_validate_json(welcome.config_json)
        run_id, number = welcome.run_id, int(welcome.node_number)
        if self.options.local or self._number < 0:
            self._number = number
        elif number != self._number:
            logger.warning(f"node: the hub calls this node {number}, its registration says {self._number}")
            self._number = number
        spec = env_spec(config)
        trainable = config.get_trainable_agent_ids()
        scripted = [a for a in config.fixed_agent_ids() if config.agent_kind(a) == "scripted"]
        roles = resolve_player_roles(config, spec, scripted)     # frozen agents' roles come from their blobs
        bots = {a: BotSpec(config.agent_entry(a).class_path, dict(config.agent_entry(a).kwargs)) for a in scripted}
        fixed_keys: dict = {}
        frozen_keys: dict = {}
        for fb in welcome.fixed:
            key = blob_key(fb.kind, fb.agent_id, fb.blob_id)
            fixed_keys[key] = fb.kind
            if fb.kind == "fixed":
                frozen_keys[fb.agent_id] = key
        dist = config.distributed
        teachers = _scripted_teachers(config)
        agent_roles = {a: list(roles[a]) for a in trainable}
        agent_configs = {a: config.get_agent_config(a) for a in trainable}
        role_specs = {a: agent_role_spec(spec, roles[a]) for a in trainable}
        ca8, run8 = self.options.token.ca_fingerprint[:8], run_id[:8]
        log_dir = Path(self.options.log_dir) if self.options.log_dir is not None else node_log_dir(ca8, run_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        if self.options.shutdown_grace_sec is not None:
            grace = float(self.options.shutdown_grace_sec)
        elif self.options.local:
            grace = LOCAL_CHILD_GRACE_SEC
        else:
            grace = max(1.0, float(dist.shutdown_grace_sec) - REMOTE_GRACE_MARGIN_SEC)
        workers = self.options.workers
        if workers is None:
            from colosseum.node.hardware import default_slots

            workers = default_slots(self._hw)["workers"]
        results = ResultBuffer(batch=dist.results_batch, flush_sec=dist.results_flush_sec,
                               limit_bytes=int(dist.node_result_buffer_mb * 2**20), make=_result_batch,
                               size_of=lambda m: m.ByteSize(), clock=time.monotonic)
        root = shm_dir()
        prefix = cache_prefix(self.options.token.ca_fingerprint, run_id, number)
        cleanup_dead(root)
        # Resources from here on: everything opened is released again if the start fails (a retrying node must be
        # able to take its lock and bind its listener's name again).
        lock = lst = acceptor = None
        try:
            lock = acquire_node_lock(root, prefix)                # once per run (P-PF2 brief note)
            authkey = os.urandom(32)
            node_address = address(ca8, run8, number, "node")
            lst = listener(node_address, authkey)
            acceptor = Acceptor(lst, lambda conn, hello: self._post(_Accepted(conn, hello)), authkey=authkey,
                                name=f"node-{number}")
            run = _Run(
                serial=self._run_serial + 1, run_id=run_id, number=number, config=config, spec=spec,
                agent_ids=trainable, agent_roles=agent_roles, agent_configs=agent_configs, role_specs=role_specs,
                bots=bots, fixed_roles={a: tuple(roles[a]) for a in scripted}, teachers=teachers,
                fixed_keys=fixed_keys, frozen_keys=frozen_keys, root=root, prefix=prefix,
                writer=WeightCacheWriter(root, prefix), reader=WeightCacheReader(root, prefix), lock=lock,
                authkey=authkey, node_address=node_address, listener=lst, acceptor=acceptor, log_dir=log_dir,
                results=results, heartbeat_sec=float(dist.heartbeat_sec), envs=int(config.rollout.envs_per_worker),
                workers=int(workers), grace=grace)
        except BaseException:
            _release_partial_start(acceptor, lst, root if lock is not None else None, prefix)
            raise
        self._run_serial += 1
        self._run = run
        for la in welcome.learners:
            self.learners.assign(run, la)
        logger.info(f"node {number} ({self._name}): serving run {run_id[:8]} with {workers} workers; "
                    f"{len(fixed_keys)} fixed blobs to fetch; logs in {log_dir}")
        self._maybe_start_workers(run, time.monotonic())

    def _maybe_start_workers(self, run: _Run, now: float) -> None:
        """Fetch the missing ``Welcome.fixed`` blobs; once all are cached, start the workers."""
        missing = [k for k in run.fixed_keys if not self._has(run, k)]
        for key in missing:
            if key not in self._fetching and now >= run.fixed_retry_at.get(key, 0.0):
                self._fetch(run, key)
        if not missing:
            self._start_workers(run)

    def _start_workers(self, run: _Run) -> None:
        from colosseum.node.worker_main import worker_target

        for agent, key in run.frozen_keys.items():
            cached = run.reader.get(key)
            if cached is None:
                raise RuntimeError(f"the fixed blob of {agent!r} vanished from the weight cache")
            run.fixed_roles[agent] = tuple(cached.meta.roles) if cached.meta.roles else tuple(run.spec.roles)
        daemon = run.config.rollout.vec_env != "subprocess"            # P-PF2-7
        for i in range(run.workers):
            kwargs = dict(name=f"worker-{i}", log_dir=str(run.log_dir), node=run.number, worker=i,
                          node_address=run.node_address, authkey=run.authkey, config=run.config,
                          agent_ids=list(run.agent_ids), agent_roles=run.agent_roles,
                          agent_configs=run.agent_configs, role_specs=run.role_specs, bots=run.bots,
                          frozen_keys=dict(run.frozen_keys), fixed_roles=dict(run.fixed_roles),
                          teachers=run.teachers, cache_root=str(run.root), cache_prefix=run.prefix,
                          run_id=run.run_id)
            incarnation = self.supervisor.start(ChildSpec(f"worker-{i}", "worker", worker_target, kwargs,
                                                          daemon=daemon))
            self.cells.add_worker(i, run.envs, incarnation)
        run.started = True
        logger.info(f"node: started {run.workers} workers")

    def _on_grants(self, grants: pb.LineupGrants) -> None:
        from colosseum.net.messages import lineup_from_proto

        run = self._run
        if run is None or self._stopping:
            return
        for grant in grants.grants:
            try:
                lineup = lineup_from_proto(grant.lineup)
                lineup.epoch = int(grant.epoch)
                keys = self._seat_keys(run, lineup)
            except ValueError as e:                  # the cell asks again after LINEUP_RETRY_SEC: warn once
                player = getattr(e, "player", None) or ("", str(e))
                if player not in self._refused_players:
                    self._refused_players.add(player)
                    logger.warning(f"node: refused a lineup grant for worker {grant.worker}, env {grant.env}: {e} "
                                   f"(further grants naming it are refused silently)")
                continue
            self.cells.granted(int(grant.worker), int(grant.env), int(grant.epoch), float(grant.ttl_sec), lineup,
                               keys)
        self._deliver(run)

    def _seat_keys(self, run: _Run, lineup: Any) -> dict:
        """The cache key of every seat's weights (none for a scripted seat); ValueError for an unknown player."""
        from colosseum.arena.entities import snapshot_version
        from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID
        from colosseum.node.weights import CacheKey, segment_name

        keys = {}
        for i, seat in enumerate(lineup.seats):
            agent, net = seat.agent_id, seat.network_id
            if net == FIXED_NETWORK_ID:
                if agent in run.frozen_keys:
                    keys[i] = run.frozen_keys[agent]
                elif agent not in run.bots:
                    raise _BadSeat(f"seat {i}: {agent!r} is no fixed player of this run", agent, net)
            elif agent not in run.agent_ids:
                raise _BadSeat(f"seat {i}: {agent!r} is no trainable agent of this run", agent, net)
            elif net == LATEST_NETWORK_ID:
                keys[i] = CacheKey("latest", agent)
            elif snapshot_version(net) is not None:
                key = CacheKey("snapshot", agent, net)
                try:
                    segment_name(run.prefix, key)
                except ValueError as e:
                    raise _BadSeat(f"seat {i}: {e}", agent, net) from None
                keys[i] = key
            else:
                raise _BadSeat(f"seat {i}: unknown network {net!r}", agent, net)
        return keys

    def _on_command(self, cmd: pb.Command) -> None:
        if cmd.kind == "stop_run":
            self._begin_stop(f"the hub stopped the run ({cmd.reason or 'stop_run'})")
        elif cmd.kind == "release":
            self._begin_stop(f"the hub released this node ({cmd.reason or 'release'})", release=True)
        elif cmd.kind == "start_learner":
            if self._run is not None and not self._stopping:
                self.learners.assign(self._run, cmd.learner)
        elif cmd.kind == "stop_learner":
            self.learners.stop(cmd.learner.agent_id, cmd.reason or "stop_learner")
        else:
            logger.info(f"node: command {cmd.kind!r} is not handled by this node yet; ignored")

    def _on_announce(self, announce: pb.Announce) -> None:
        from colosseum.net.local import WorkerEvict
        from colosseum.node.weights import CacheKey

        self.learners.forward_env_steps(int(announce.env_steps))
        keys = [CacheKey("snapshot", ev.agent_id, ev.checkpoint_id) for ev in announce.evictions]
        if keys and self._run is not None:
            for name, conn in list(self._control.items()):
                if self.supervisor.role(name) == "worker":
                    self._child_send(name, conn, WorkerEvict(keys))

    # ---- lineups and the cache --------------------------------------------------------------------------------

    def _has(self, run: _Run, key: Any) -> bool:
        if key.kind == "latest":
            return run.reader.latest_version(key.agent_id) is not None
        return key in self._cached or run.reader.has(key)

    def _request_lineups(self) -> None:
        if not self._connected:
            return
        due = self.cells.due()
        if due:
            self._send(pb.NodeMessage(lineup_request=pb.LineupRequest(cells=[
                pb.CellRequest(worker=w, env=e, epoch=ep, incarnation=i) for w, e, ep, i in due])))

    def _expire_waiting(self) -> None:
        for worker, env, keys in self.cells.stale_waiting(LINEUP_WAIT_SEC):
            self.cells.failed(worker, env)
            self._counters["lineup_fetch_failed"] += 1
            run = self._run
            missing = [k for k in keys.values() if run is None or not self._has(run, k)]
            for key in missing:
                self._warn_fetch(key, f"its weights did not arrive within {LINEUP_WAIT_SEC:.0f} s")

    def _deliver(self, run: _Run) -> None:
        """Send every waiting grant whose weights are all cached to its worker (R11); fetch what is missing."""
        from colosseum.net.local import CellLineup, WorkerLineups

        ready: dict[int, list] = {}
        for worker, env, _lineup, keys in self.cells.waiting():
            missing = [k for k in set(keys.values()) if not self._has(run, k)]
            for key in missing:
                if key.kind != "latest" and key not in self._fetching:
                    self._fetch(run, key)
            if missing or f"worker-{worker}" not in self._control:
                continue
            lineup, keys = self.cells.deliver(worker, env)
            ready.setdefault(worker, []).append(CellLineup(env, lineup, dict(keys)))
        for worker, cells in ready.items():
            name = f"worker-{worker}"
            self._child_send(name, self._control[name], WorkerLineups(cells))

    def _fetch(self, run: _Run, key: Any) -> None:
        future = self._pool.submit(self._download, run, key)
        self._fetching[key] = future
        serial = run.serial

        def done(f: Future) -> None:
            error = None
            if f.cancelled():
                error = "cancelled"
            elif f.exception() is not None:
                error = _rpc_text(f.exception())
            self._post(_FetchDone(key, error, serial))

        future.add_done_callback(done)

    def _download_bytes(self, run: _Run, kind: str, agent_id: str, ident: str) -> bytes:
        """A transfer-pool thread: one ``Download`` (checked: size, offsets, digest; capped)."""
        from colosseum.net.frame import payload_byte_cap
        from colosseum.net.grpc_util import node_metadata, receive_parts
        from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc

        parts = pb_grpc.HubStub(self._channel).Download(
            pb.BlobRequest(kind=kind, agent_id=agent_id, ident=ident),
            metadata=node_metadata(self._node_id, self._node_secret), timeout=DOWNLOAD_TIMEOUT_SEC)
        data, _meta = receive_parts(parts, cap=payload_byte_cap(run.config.distributed.max_message_mb))
        if data is None:
            raise ValueError("the hub answered not_modified")
        return data

    def _download(self, run: _Run, key: Any) -> None:
        """A transfer-pool thread: ``Download`` the key's frame, check it, put it into the cache."""
        from colosseum.net.chunks import decode_weights
        from colosseum.node.weights import split_blob_key

        if key.kind == "snapshot":
            kind, ident = "snapshot", key.ident
        else:
            kind, ident = split_blob_key(key)
        data = self._download_bytes(run, kind, key.agent_id, ident)
        _state, meta = decode_weights(data, cap=len(data))
        if meta.agent_id != key.agent_id:
            raise ValueError(f"the frame holds weights of {meta.agent_id!r}")
        if run.closed:
            raise RuntimeError("the run was torn down (stopped or released) meanwhile")
        run.writer.put(key, data)

    def _on_fetch_done(self, done: _FetchDone) -> None:
        self._fetching.pop(done.key, None)
        run = self._run
        if run is None or run.serial != done.generation:
            return
        if done.error is None:
            if done.key not in run.fixed_keys:
                self._cached.add(done.key)
            self._fetch_warned.discard(done.key)
            if not self._stopping:
                self._deliver(run)
            return
        if done.key in run.fixed_keys:
            run.fixed_retry_at[done.key] = time.monotonic() + FIXED_RETRY_SEC
            self._warn_fetch(done.key, f"{done.error}; retrying (the workers start once every fixed blob is here)")
            return
        failed = [(w, e) for w, e, _lineup, keys in self.cells.waiting() if done.key in keys.values()]
        for worker, env in failed:
            self.cells.failed(worker, env)
            self._counters["lineup_fetch_failed"] += 1
        self._warn_fetch(done.key, f"{done.error}; {len(failed)} lineups keep their old one")

    def _warn_fetch(self, key: Any, why: str) -> None:
        if key in self._fetch_warned:
            return
        self._fetch_warned.add(key)
        logger.warning(f"node: fetching {key.kind} {key.agent_id}/{key.ident} failed: {why} (lineup_fetch_failed; "
                       f"further failures of this key are only counted)")

    def _prune_cache(self, run: _Run, now: float) -> None:
        """Snapshots no cell references for ``CACHE_DROP_DELAY_SEC`` leave the cache (``Welcome.fixed`` blobs and
        latest pointers stay).

        Invariant (a worker never misses a delivered snapshot): a key stops being referenced only when ``deliver``
        replaces its cell's lineup, and ``_deliver`` sends the replacing ``WorkerLineups`` on the same ordered control
        connection at once; a worker pumps every pending control message before each cache read
        (``WorkerIO.poll_command``), so it never reads a lineup that was replaced, and the snapshot of the lineup it
        does read is still referenced. Any new delivery path must keep this (replace a cell's keys only together
        with sending the replacing lineup). The delay is a margin, not what makes it safe."""
        referenced = self.cells.referenced()
        for key in list(self._cached):
            if key in referenced or key in self._fetching:
                self._unref_since.pop(key, None)
                continue
            since = self._unref_since.setdefault(key, now)
            if now - since >= CACHE_DROP_DELAY_SEC:
                try:
                    run.writer.drop(key)
                except OSError as e:
                    logger.warning(f"node: cannot remove {key} from the weight cache: {e}")
                self._cached.discard(key)
                self._unref_since.pop(key, None)

    # ---- children ---------------------------------------------------------------------------------------------

    def _on_child_hello(self, conn: mpc.Connection, hello: Any) -> None:
        from colosseum.net.local import Stop

        name = hello.name
        proc = self.supervisor.get(name)
        if (self._run is None or proc is None or proc.exitcode is not None
                or int(hello.incarnation) != self.supervisor.incarnation(name)):
            logger.debug(f"node: closed a connection of {name!r} (incarnation {hello.incarnation}): not a live child")
            conn.close()
            return
        if hello.channel == "control":
            old = self._control.pop(name, None)
            if old is not None:
                self._close_conn(old)
            self._control[name] = conn
            self._child_conns[conn] = name
            if self._stopping:
                self._child_send(name, conn, Stop("the node stops"))
            elif self.supervisor.role(name) == "worker":
                self._on_worker_connected(name, conn)
            elif self.supervisor.role(name) == "learner":
                self.learners.on_connected(name)
        elif hello.channel == "route":
            self._child_conns[conn] = name
            self._route_conns.add(conn)
        else:
            conn.close()

    def _on_worker_connected(self, name: str, conn: mpc.Connection) -> None:
        from colosseum.net.local import LearnerRoute, WeightsReady

        run = self._run
        for agent in run.agent_ids:
            self._child_send(name, conn, self._learner_routes.get(agent, LearnerRoute(agent, None, False, 0)))
        ready = [a for a in run.agent_ids if run.reader.latest_version(a) is not None]
        self._child_send(name, conn, WeightsReady(ready))
        self._deliver(run)

    def _child_send(self, name: str, conn: mpc.Connection, message: Any) -> bool:
        try:
            conn.send(message)
            return True
        except (OSError, EOFError, ValueError) as e:
            logger.debug(f"node: cannot send {type(message).__name__} to {name}: {e}")
            return False

    def _frame_cap(self) -> int:
        from colosseum.net.frame import payload_byte_cap

        return payload_byte_cap(self._run.config.distributed.max_message_mb if self._run is not None else self._max_mb)

    def _read_child(self, conn: mpc.Connection) -> None:
        from colosseum.net.local import CheckpointHeader

        name = self._child_conns.get(conn, "?")
        for _ in range(CHILD_MESSAGES_PER_PASS):
            try:
                if not conn.poll(0):
                    return
                if conn in self._route_conns:
                    conn.recv_bytes()               # frames for other machines: T3.2
                    continue
                message = conn.recv()
                frame = None
                if isinstance(message, CheckpointHeader):     # its frame follows at once (one send under a lock)
                    cap = self._frame_cap()
                    if int(message.size) > cap:
                        self._oversized_checkpoint(name, message, cap)
                        self._close_conn(conn)     # the frame stays unread: the learner's connection ends
                        return
                    # A blocking read of one whole frame: a learner SIGSTOPped in the middle of it stalls the loop
                    # (SIGKILL gives EOF at once); bounding it is T4.4's.
                    frame = conn.recv_bytes(cap)
            except (EOFError, ConnectionResetError, BrokenPipeError):
                # The peer is gone: a clean end, or (Linux AF_UNIX) a peer that exited with data of ours unread,
                # which reads as ECONNRESET; its exit is reported by the exit path, not as a read failure.
                self._close_conn(conn)
                return
            except OSError as e:                         # not an end of the peer: say what happened
                logger.error(f"node: reading from {name} failed ({type(e).__name__}: {e}); its connection is closed")
                self.learners.note_failure(name, f"the node could not read its message ({e})")
                self._close_conn(conn)
                return
            except Exception:  # noqa: BLE001 - an unreadable message: only this connection closes
                logger.exception(f"node: unreadable message from {name}; its connection is closed")
                self._close_conn(conn)
                return
            if frame is not None:
                self._on_checkpoint(name, message, frame)
            else:
                self._on_child_message(name, message)

    def _oversized_checkpoint(self, name: str, header: Any, cap: int) -> None:
        """A checkpoint header announcing a frame above the cap (the learner checks first, so this is a learner of
        another cap or a broken one): ERROR, a ``warning`` event, and the text for its ``learner_failed``."""
        from colosseum.net.local import checkpoint_too_large

        text = checkpoint_too_large(header.agent_id, header.kind, int(header.version), int(header.size), cap)
        logger.error(f"node: {name}: {text}")
        self._event("warning", proc=name, message=text)
        self.learners.note_failure(name, text)

    def _on_child_message(self, name: str, message: Any) -> None:
        from colosseum.net.local import ChildError, WorkerProgress, WorkerResults

        run = self._run
        if isinstance(message, WorkerResults):
            if run is not None:
                for batch in run.results.add(message.results):
                    self._send(batch)
        elif isinstance(message, WorkerProgress):
            worker = _slot_index(name)
            key = (worker, int(message.incarnation))
            self._progress[key] = max(self._progress.get(key, 0), int(message.env_steps))
            self._worker_stats[key] = dict(message.counters)
            self._stats_changed.add(key)
        elif isinstance(message, ChildError):
            self._reported.add(name)
            log_path = "" if run is None else str(run.log_dir / f"{name}.log")
            logger.error(f"node: {name} failed ({message.kind}): {message.message}")
            self._event("child_error", proc=name, error_kind=message.kind, message=message.message,
                        context=f"node {self._number} ({self._name}), {message.context}", log_path=log_path)
        else:
            self.learners.on_message(name, message)

    def _close_conn(self, conn: mpc.Connection) -> None:
        name = self._child_conns.pop(conn, None)
        self._route_conns.discard(conn)
        if name is not None and self._control.get(name) is conn:
            del self._control[name]
        try:
            conn.close()
        except OSError:
            pass

    def _drain_to_eof(self, name: str) -> None:
        """Read an exited child's control connection to its end, bounded by ``EXIT_DRAIN_SEC`` (P-PF2-11)."""
        conn = self._control.get(name)
        deadline = time.monotonic() + EXIT_DRAIN_SEC
        while conn is not None and self._control.get(name) is conn:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0 or not conn.poll(remaining):
                    break
            except (EOFError, OSError):
                break
            self._read_child(conn)
        for c in [c for c, n in self._child_conns.items() if n == name]:
            self._close_conn(c)

    def _on_exit(self, name: str, code: int | None) -> None:
        self._drain_to_eof(name)
        if self._stopping:
            return
        run = self._run
        role = self.supervisor.role(name)
        log_path = "" if run is None else str(run.log_dir / f"{name}.log")
        if role == "learner":
            self.learners.on_exit(name, code, log_path, self._learner_routes)
            return
        if name in self._reported:
            logger.info(f"node: {name} exited (code {code}) after reporting its error; not restarted")
            return
        logger.warning(f"node: {name} died (exit {code}) without reporting an error; see {log_path}")
        self._event("proc_died", proc=name, exit_code=int(code or 0), log_path=log_path)
        if role == "worker" and run is not None:
            worker = _slot_index(name)
            self.cells.drop_worker(worker)
            if self.supervisor.may_restart(name):
                incarnation = self.supervisor.restart(name)
                self.cells.add_worker(worker, run.envs, incarnation)
                self._counters["worker_restarts"] += 1
                logger.info(f"node: restarted {name} (incarnation {incarnation})")
            else:
                self._disabled.add(name)
                logger.error(f"node: {name} died too often; its slot is disabled")
                self._event("slot_disabled", proc=name, exit_code=int(code or 0), log_path=log_path)

    # ---- learners (node.learners) and uploads ------------------------------------------------------------------

    def _set_route(self, route: Any) -> None:
        """Where every worker (and every worker that connects later) sends ``route.agent_id``'s chunks."""
        self._learner_routes[route.agent_id] = route
        for name, conn in list(self._control.items()):
            if self.supervisor.role(name) == "worker":
                self._child_send(name, conn, route)

    def _on_checkpoint(self, name: str, header: Any, frame: bytes) -> None:
        """A learner's snapshot: an ``Upload`` in the transfer pool (``checkpoint`` / ``final``)."""
        entry = self.learners.of(name)
        kind = _UPLOAD_KINDS.get(header.kind)
        if entry is None or kind is None or header.agent_id != entry.agent_id:
            logger.warning(f"node: dropped a {header.kind!r} checkpoint of {header.agent_id!r} from {name}")
            return
        token = uuid.uuid4().hex
        future = self._pool.submit(self._upload, kind, header, frame, token)
        self._uploads[token] = (future, header.kind, header.agent_id, int(header.version))

        def done(f: Future) -> None:
            error = "cancelled" if f.cancelled() else (None if f.exception() is None else _rpc_text(f.exception()))
            self._post(_UploadDone(token, header.kind, header.agent_id, int(header.version),
                                   None if error else f.result(), error))

        future.add_done_callback(done)

    def _upload(self, kind: str, header: Any, frame: bytes, token: str) -> pb.UploadReply:
        """A transfer-pool thread: one ``Upload`` (header, then the frame in parts)."""
        # The transfer pool runs uploads until T4.2's node.uploads.UploadQueue (A29) replaces this path.
        from colosseum.net.grpc_util import blob_digest, node_metadata, send_parts
        from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc

        digest = blob_digest(frame)
        up = pb.UploadHeader(kind=kind, agent_id=header.agent_id, generation=int(header.generation),
                             version=int(header.version), total_size=len(frame), digest=digest, request_id=token)

        def parts():
            yield pb.UploadPart(header=up)
            for part in send_parts(frame, digest=digest):
                if part.data:
                    yield pb.UploadPart(data=part.data)

        return pb_grpc.HubStub(self._channel).Upload(parts(), metadata=node_metadata(self._node_id, self._node_secret),
                                                     timeout=UPLOAD_TIMEOUT_SEC)

    def _on_upload_done(self, done: _UploadDone) -> None:
        self._uploads.pop(done.token, None)
        what = f"the {done.kind} snapshot v{done.version} of {done.agent_id}"
        if done.error is None and done.reply.accepted:
            logger.info(f"node: uploaded {what} ({done.reply.checkpoint_id})")
            return
        why = done.error if done.error is not None else f"refused: {done.reply.reason}"
        level = logging.ERROR if done.kind == "final" else logging.WARNING
        logger.log(level, f"node: the hub did not take {what} ({why})")

    # ---- heartbeats and metrics -------------------------------------------------------------------------------

    def _env_steps(self) -> int:
        return sum(self._progress.values())

    def _proc_state(self, name: str) -> str:
        if name in self._disabled:
            return "disabled"
        proc = self.supervisor.get(name)
        return "alive" if proc is not None and proc.exitcode is None else "dead"

    def _node_counters(self, run: _Run | None) -> dict[str, int]:
        """The A2 node counters, cumulative per session (a released run's drops are carried)."""
        dropped = self._dropped_carry + (0 if run is None else int(run.results.dropped))
        return {"lineup_fetch_failed": int(self._counters["lineup_fetch_failed"]),
                "dropped_results_node_buffer": dropped, "worker_restarts": int(self._counters["worker_restarts"])}

    def _send_heartbeat(self, run: _Run) -> None:
        try:
            load = float(os.getloadavg()[0])
        except (AttributeError, OSError):
            load = 0.0
        cached = sorted(f"{k.kind}:{k.agent_id}:{k.ident}"
                        for k in (*self._cached, *(k for k in run.fixed_keys if self._has(run, k))))
        self._send(pb.NodeMessage(heartbeat=pb.Heartbeat(
            env_steps=self._env_steps(), dropped_results=self._dropped_carry + int(run.results.dropped),
            procs=[pb.ProcState(name=n, state=self._proc_state(n), restarts=max(0, self.supervisor.incarnation(n) - 1))
                   for n in self.supervisor.names()],
            load=load, cached_keys=cached, counters=self._node_counters(run))))

    def _send_metrics(self) -> None:
        learners = self.learners
        if not (self._stats_changed or learners.train_metrics or learners.intake_stats):
            return
        workers = [pb.WorkerStats(worker=w, incarnation=i, counters=self._worker_stats.get((w, i), {}))
                   for w, i in sorted(self._stats_changed)]
        batch = pb.MetricsBatch(workers=workers, train=list(learners.train_metrics),
                                intake=list(learners.intake_stats.values()))
        if self._send(pb.NodeMessage(metrics=batch)):
            self._stats_changed.clear()
            learners.train_metrics.clear()
            learners.intake_stats.clear()

    # ---- the stop ---------------------------------------------------------------------------------------------

    def _begin_stop(self, reason: str, *, release: bool = False) -> None:
        """Item 10 (``release``: then forget the run and keep the session)."""
        from colosseum.net.local import Stop

        if self._stopping:
            return
        self._stopping = True
        self._releasing = release and self._run is not None
        now = time.monotonic()
        run = self._run
        logger.info(f"node: stopping ({reason})")
        if run is None:
            self._stop_phase = "acks"
            self._ack_deadline = now
            return
        for name, conn in list(self._control.items()):
            self._child_send(name, conn, Stop(reason))
        self._children_deadline = now + run.grace
        self._stop_phase = "children"

    def _stop_step(self, now: float) -> None:
        run = self._run
        if self._stop_phase == "children":
            if self.supervisor.alive() and now < self._children_deadline:
                return
            if self.supervisor.alive():
                self.supervisor.stop_all(0.0)
            for name, code in self.supervisor.exited():
                self._on_exit(name, code)
            if run is not None:
                for message in run.results.flush():
                    self._send(message)
            self._ack_deadline = max(now, self._children_deadline) + ACK_WAIT_EXTRA_SEC
            self._stop_phase = "acks"
            return
        if self._stop_phase == "acks":
            waiting = run is not None and (run.results.unacked() or self._uploads)
            if waiting and time.monotonic() < self._ack_deadline:
                return
            if self._uploads:
                logger.error(f"node: {len(self._uploads)} snapshot uploads did not finish before the stop's deadline; "
                             f"they are not waited for any longer")
                for _future, kind, agent, version in self._uploads.values():
                    if kind == "final":                  # the hub's log says so too (A1 "warning")
                        self._event("warning", proc=f"learner-{agent}",
                                    message=f"final v{version} not uploaded before the stop deadline")
            if run is not None:
                run.results.give_up()
                if self._connected:
                    self._send_metrics()
                    self._send_heartbeat(run)
            if self._connected:
                self._send(pb.NodeMessage(event=pb.Event(kind="run_stopped")))
            if self._releasing and run is not None:
                self._forget_run(run)
                if self.options.once:
                    self._done = True
                return
            self._done = True

    def _forget_run(self, run: _Run) -> None:
        """After ``release``: the run's resources go (``_teardown_run``) and so do its books; the session stays.
        This node serves nothing further of this run (``_released``); serving another run is T6.3's."""
        self._teardown_run(run)
        self._dropped_carry += int(run.results.dropped)
        self._heartbeat_sec = run.heartbeat_sec
        self._released.add(run.run_id)
        self._run = None
        self.supervisor = ChildSupervisor()
        self.cells = CellTracker(retry_sec=LINEUP_RETRY_SEC, clock=time.monotonic)
        self.learners.clear()
        for book in (self._learner_routes, self._uploads, self._fetching, self._unref_since, self._control,
                     self._child_conns):
            book.clear()
        for book in (self._cached, self._route_conns, self._reported, self._disabled):
            book.clear()
        self._stopping, self._releasing, self._stop_phase = False, False, ""
        logger.info(f"node: released run {run.run_id[:8]}; the session stays open")

    def _close(self) -> None:
        """Children, the session, transfers, then the run's resources and cache files (every path out of ``run``).
        The order matters: the node is marked closed first (thread callbacks no longer wake the loop), the transfers
        are cancelled (the channel closes) and waited for (``TRANSFER_JOIN_SEC``) before the prefix's files are
        removed, so no download puts a segment after the cleanup; the wake pipe closes last."""
        with self._wake_lock:
            self._closed = True
        if self.supervisor.alive():
            self.supervisor.stop_all(0.0)
        if self._session is not None:
            self._session.close()
        if self._channel is not None:
            self._channel.close()                    # cancels the downloads and uploads in flight
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self._run is not None:
            self._teardown_run(self._run)
        for fd in (self._wake_r, self._wake_w):
            try:
                os.close(fd)
            except OSError:
                pass

    def _teardown_run(self, run: _Run) -> None:
        """Close the run's acceptor, listener and child connections; wait ``TRANSFER_JOIN_SEC`` for the transfers
        in flight; remove the node's cache files and lock (a segment a transfer writes later is a stray file of a
        prefix without a lock, which the next node's ``cleanup_dead`` removes)."""
        from colosseum.node.weights import release_node_lock

        run.acceptor.close()
        try:
            run.listener.close()
        except OSError:
            pass
        for conn in list(self._child_conns):
            self._close_conn(conn)
        run.closed = True                            # a download still running puts nothing any more
        in_flight = [*self._fetching.values(), *(u[0] for u in self._uploads.values())]
        for future in in_flight:
            future.cancel()                          # not started yet: never runs
        if in_flight:
            _, stuck = wait_futures(in_flight, timeout=TRANSFER_JOIN_SEC)
            if stuck:
                logger.warning(f"node: {len(stuck)} transfers did not end within {TRANSFER_JOIN_SEC:.0f} s")
        if run.lock is not None:
            try:
                release_node_lock(run.root, run.prefix)
            except Exception as e:  # noqa: BLE001 - leftovers are removed by the next node's cleanup_dead
                logger.warning(f"node: cannot remove the weight cache files of {run.prefix}: {e}")


class _LearnerHost:
    """``node.learners.LearnerHost`` over the agent: the narrow interface ``NodeLearners`` uses."""

    def __init__(self, agent: NodeAgent) -> None:
        self._agent = agent

    @property
    def supervisor(self) -> ChildSupervisor:
        return self._agent.supervisor

    @property
    def number(self) -> int:
        return self._agent._number

    @property
    def ca8(self) -> str:
        return self._agent.options.token.ca_fingerprint[:8]

    def current_run(self) -> Any:
        return self._agent._run

    def post(self, item: object) -> None:
        self._agent._post(item)

    def send_child(self, name: str, message: Any) -> bool:
        conn = self._agent._control.get(name)
        return conn is not None and self._agent._child_send(name, conn, message)

    def set_route(self, route: Any) -> None:
        self._agent._set_route(route)

    def event(self, kind: str, **fields: Any) -> None:
        self._agent._event(kind, **fields)

    def send_if_connected(self, message: pb.NodeMessage) -> bool:
        return self._agent._connected and self._agent._send(message)

    def send(self, message: pb.NodeMessage) -> bool:
        return self._agent._send(message)

    def submit(self, fn: Any, *args: Any) -> Future:
        return self._agent._pool.submit(fn, *args)

    def download_bytes(self, run: Any, kind: str, agent_id: str, ident: str) -> bytes:
        return self._agent._download_bytes(run, kind, agent_id, ident)

    def fixed_cached(self, run: Any) -> bool:
        return all(self._agent._has(run, k) for k in run.fixed_keys)

    def rpc_text(self, error: BaseException) -> str:
        return _rpc_text(error)


# ---------------------------------------------------------------------------------------------------------------
# Helpers and entry points
# ---------------------------------------------------------------------------------------------------------------


def _release_partial_start(acceptor: Any, lst: Any, root: Path | None, prefix: str) -> None:
    """Undo what a failed run start opened: the acceptor thread, the listener (its socket name) and the node lock
    with the prefix's files (``root`` None: the lock was not taken)."""
    from colosseum.node.weights import release_node_lock

    if acceptor is not None:
        acceptor.close()
    if lst is not None:
        try:
            lst.close()
        except OSError:
            pass
    if root is not None:
        try:
            release_node_lock(root, prefix)
        except Exception as e:  # noqa: BLE001 - leftovers are removed by the next node's cleanup_dead
            logger.warning(f"node: cannot remove the weight cache files of {prefix}: {e}")


def _slot_index(name: str) -> int:
    return int(name.rsplit("-", 1)[1])


def _result_batch(seq: int, results: list) -> pb.NodeMessage:
    """One ``ResultBatch`` message; a result that cannot be converted is left out with a WARNING (never raises)."""
    from colosseum.net.messages import result_to_proto

    protos = []
    for result in results:
        try:
            protos.append(result_to_proto(result))
        except Exception as e:  # noqa: BLE001 - one malformed result never stops the node
            logger.warning(f"node: a match result could not be encoded and is left out ({type(e).__name__}: {e})")
    return pb.NodeMessage(results=pb.ResultBatch(seq=int(seq), results=protos))


def _scripted_teachers(config: Any) -> dict[str, Any]:
    """The scripted kickstart teachers (DAgger) of the trainable agents, as ``BotSpec`` (no file is read)."""
    from colosseum.players.registry import BotSpec

    out = {}
    agents = set(config.agent_ids())
    for agent in config.get_trainable_agent_ids():
        teacher = config.get_agent_config(agent).kickstart.teacher
        if teacher in agents and config.agent_kind(teacher) == "scripted":
            entry = config.agent_entry(teacher)
            out[agent] = BotSpec(entry.class_path, dict(entry.kwargs))
    return out


def local_node_target(*, options: NodeOptions, log_dir: str) -> None:
    """The hub's spawn target for ``train``'s local node (R12): process logging, the child signal policy (it dies
    with the hub), then the node until its stop."""
    from colosseum.utils.process import run_child

    def body() -> None:
        code = NodeAgent(options).run()
        if code:
            sys.exit(code)

    run_child("node", log_dir, body)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m colosseum.node.agent --hub ADDR --token-file F [--insecure] [--workers N] [--name NAME]`` (the
    loader's target, T6.2; ``join`` defines the full flags in T6.3)."""
    import signal

    from colosseum.net.security import parse_token
    from colosseum.utils.logging import setup_process_logging

    parser = argparse.ArgumentParser(prog="python -m colosseum.node.agent")
    parser.add_argument("--hub", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    setup_process_logging(None, "node", console_level=logging.INFO)
    options = NodeOptions(hub_address=args.hub, token=parse_token(Path(args.token_file).read_text()),
                          insecure=args.insecure, workers=args.workers, name=args.name)
    agent = NodeAgent(options)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: agent.request_stop())
    return agent.run()


if __name__ == "__main__":
    sys.exit(main())
