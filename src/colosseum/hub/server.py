"""The gRPC servicers of the hub port (SP5 spec 4.1, 3.2-3.4; rulings R1, R13).

The servicers only parse, authenticate (``AuthInterceptor`` does it before a method runs), convert, read and write
files and put events on the main loop's inbox (``colosseum.hub.hub.Hub`` from T2.5; ``tests/hub_kit.MiniHubLoop``
before). The replies the main loop gives come back through futures (``RegisterIn``, ``UploadIn``, ``AdminIn``) or a
session's bounded outbound queue (``SessionHandle.send``). A servicer never touches hub state.

Sessions (pre-flight ruling P-PF2-1 (a)): every ``Session`` stream gets a ``SessionHandle`` with a per-stream
``serial``; ``SessionOpened`` / ``SessionClosed`` carry the handle and ``NodeMessageIn`` the serial, so the main loop
tells a node's current stream from a superseded one (the hub may learn of a cut only after the node's new stream is
open). ``SessionClosed`` is posted exactly once per opened stream, by the stream's reader thread.

Files (pre-flight ruling P-PF2-9 (a, c); every string a node sends is untrusted): uploads are spooled under
``runs/<name>/.incoming/<uuid4 hex>.part`` (the header's ``request_id`` is only echoed); ``BlobStore`` validates agent
ids (``core.config.check_agent_id``, the config's own rule) and idents before it forms any path, serves snapshots
from ``checkpoints/<agent>/ckpt_v<N>/model.pt`` (converted once into a frame under ``blobs/``) and every other kind
only from the ``(kind, agent, ident)`` triples the main loop registered (``register_fixed``). Blob file names never
contain the agent id (``BlobStore._file_name``).

Lifecycle: a servicer that waits for the main loop's reply (``Register``, ``Upload``, ``Admin``) waits in
``POLL_SEC`` slices while its RPC is active, so ``server.stop`` (which cancels every RPC) ends it at once even when
the main loop no longer drains the inbox.

``RelaySend`` / ``RelayInbox`` answer ``UNIMPLEMENTED`` until T3.2; ``Bootstrap`` is registered by T6.2.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc

from colosseum.coordinator.checkpoint_manager import read_weights_file
from colosseum.core.config import check_agent_id
from colosseum.core.errors import ConfigError
from colosseum.core.run_dir import RunDir
from colosseum.core.types import MatchResult
from colosseum.hub.state import NodeHello
from colosseum.net.chunks import WeightsMeta, encode_weights
from colosseum.net.frame import payload_byte_cap
from colosseum.net.grpc_util import BLOB_DIGEST_SIZE, AuthInterceptor, node_id_of, send_parts, server_options
from colosseum.net.messages import PROTOCOL_VERSION, results_from_batch
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc
from colosseum.net.security import HubIdentity, SecurityError, private_dir

__all__ = [
    "ADMIN_REPLY_TIMEOUT_SEC", "CLOSE_GRACE_SEC", "LOOPBACK_KEY", "OUTBOUND_QUEUE_SIZE", "REGISTER_REPLY_TIMEOUT_SEC",
    "SERVER_THREADS", "UPLOAD_KINDS", "UPLOAD_REPLY_TIMEOUT_SEC", "AdminIn", "BlobStore", "HubServicer",
    "NodeMessageIn", "RegisterIn", "SessionClosed", "SessionHandle", "SessionOpened", "StopRequested", "UploadIn",
    "blob_ident", "build_server", "hello_from_proto",
]

log = logging.getLogger(__name__)

SERVER_THREADS = 128                 # every live session and relay stream holds one thread: 20 nodes x 2 + transfers
OUTBOUND_QUEUE_SIZE = 10_000         # per session; a full queue closes the session (the node reconnects)
UPLOAD_REPLY_TIMEOUT_SEC = 60.0
REGISTER_REPLY_TIMEOUT_SEC = 30.0
ADMIN_REPLY_TIMEOUT_SEC = 30.0
CLOSE_GRACE_SEC = 5.0                # a closed session whose writer is stuck (a dead peer) is cancelled after this
LOOPBACK_KEY = "loopback"            # the plain loopback port in build_server's port map
POLL_SEC = 0.5

UPLOAD_KINDS = ("checkpoint", "final", "recovery")              # relay_latest: T3.2; agent_file: T8.2
REGISTERED_KINDS = ("fixed", "teacher", "init", "recovery", "trainer_state")
BLOB_IDENT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,159}")  # A18's blob id
SNAPSHOT_IDENT_RE = re.compile(r"ckpt_v([0-9]{1,18})")
SESSION_ID_RE = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_serials = itertools.count(1)
_AGENT_DIGEST_SIZE = 16              # bytes of the agent-id digest in blob file names (32 hex chars)


def _valid_agent_id(value: object) -> bool:
    """An agent id the config accepts (``check_agent_id``: one source of truth, no length limit of its own)."""
    if not isinstance(value, str):
        return False
    try:
        check_agent_id(value)
    except ConfigError:
        return False
    return True


class _Cancelled(Exception):
    """The RPC whose reply a servicer waited for was cancelled (``server.stop``, or the client went away)."""


def _await_reply(reply: Future, context: grpc.ServicerContext, timeout: float) -> Any:
    """``reply.result(timeout)`` in ``POLL_SEC`` slices while the RPC is active: FutureTimeout after ``timeout``,
    ``_Cancelled`` as soon as the RPC is not active, else the result (or the main loop's exception)."""
    deadline = time.monotonic() + timeout
    while True:
        if not context.is_active():
            raise _Cancelled
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FutureTimeout
        try:
            return reply.result(timeout=min(POLL_SEC, remaining))
        except FutureTimeout:
            continue


# ---------------------------------------------------------------------------------------------------------------
# Inbox events (servicer threads -> main loop)
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class SessionOpened:
    handle: SessionHandle
    hello: pb.Hello


@dataclass
class SessionClosed:
    node_id: str
    session_id: str
    reason: str
    handle: SessionHandle

    @property
    def serial(self) -> int:
        return self.handle.serial


@dataclass
class NodeMessageIn:
    """One session message after the ``Hello`` (a second ``Hello`` too: the main loop ignores it). ``results``: the
    converted results of a ``ResultBatch``; ``error``: the conversion's failure (then ``results`` is None)."""

    node_id: str
    session_id: str
    message: pb.NodeMessage
    results: list[MatchResult] | None = None
    error: str | None = None
    serial: int | None = None


@dataclass
class RegisterIn:
    request: pb.RegisterRequest
    peer: str
    reply: Future                    # pb.RegisterReply, or an exception (SecurityError -> PERMISSION_DENIED)


@dataclass
class UploadIn:
    """A complete upload, checked for size and digest, spooled at ``path``. The main loop handles it, deletes the file
    and resolves ``reply`` with a ``pb.UploadReply``. After ``UPLOAD_REPLY_TIMEOUT_SEC`` the servicer gives up,
    answers ``timeout`` and deletes the file; when the RPC is cancelled first (``server.stop``, a node that went away)
    it deletes the file and returns at once (a late handler must tolerate a missing file)."""

    header: pb.UploadHeader
    path: Path
    node_id: str
    reply: Future


@dataclass
class AdminIn:
    command: str
    args: dict
    reply: Future | queue.SimpleQueue   # Admin: a Future of pb.AdminReply; AdminStream: replies until one is done


@dataclass
class StopRequested:
    """Put by ``Hub.request_stop`` (T2.5)."""


def _plain(msg: Any) -> dict:
    """Every field of a protobuf message as plain Python values (int64 stays int, nested messages dicts)."""
    out: dict = {}
    for f in msg.DESCRIPTOR.fields:
        value = getattr(msg, f.name)
        if f.message_type is not None:
            out[f.name] = [_plain(v) for v in value] if f.is_repeated else _plain(value)
        else:
            out[f.name] = list(value) if f.is_repeated else value
    return out


def hello_from_proto(hello: pb.Hello) -> NodeHello:
    """What the hub keeps of a ``Hello`` (``hub.state`` holds no protobuf)."""
    return NodeHello(hardware=_plain(hello.hardware), slots=_plain(hello.slots), tags=tuple(hello.tags),
                     data_addresses=tuple(hello.data_addresses),
                     live_cells=tuple((c.worker, c.env, c.epoch, c.incarnation) for c in hello.live_cells),
                     env_steps=int(hello.env_steps), last_acked_seq=int(hello.last_acked_seq))


# ---------------------------------------------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------------------------------------------

_CLOSE = object()


class SessionHandle:
    """One live session stream (contract). ``serial`` is unique per stream in this process (P-PF2-1). ``send`` never
    blocks and is thread-safe: a full outbound queue closes the session at once (``closed_reason`` = ``"outbound
    overflow"``; the node reconnects and gets a full ``Welcome``). ``close`` lets the queued messages go out first; a
    stream whose writer is stuck is cancelled after ``CLOSE_GRACE_SEC``."""

    def __init__(self, node_id: str, session_id: str) -> None:
        self.node_id, self.session_id = node_id, session_id
        self.serial = next(_serials)
        self._out: queue.Queue = queue.Queue(maxsize=OUTBOUND_QUEUE_SIZE)
        self.closed_reason: str | None = None
        self._abandon = False
        self._lock = threading.Lock()
        self._context: grpc.ServicerContext | None = None

    def __repr__(self) -> str:
        return f"SessionHandle({self.node_id[:8]}, {self.session_id!r}, serial={self.serial})"

    def send(self, msg: pb.HubMessage) -> None:
        if self.closed_reason is not None:
            return
        try:
            self._out.put_nowait(msg)
        except queue.Full:
            self._close("outbound overflow", abandon=True)

    def close(self, reason: str) -> None:
        self._close(reason, abandon=False)

    def _close(self, reason: str, *, abandon: bool) -> None:
        with self._lock:
            if self.closed_reason is not None:
                return
            self.closed_reason = reason
            self._abandon = abandon
        if abandon:
            self._cancel()
            return
        try:
            self._out.put_nowait(_CLOSE)
        except queue.Full:                       # the generator sees closed_reason once the queue drains
            pass
        timer = threading.Timer(CLOSE_GRACE_SEC, self._cancel)
        timer.daemon = True
        timer.start()

    def _cancel(self) -> None:
        context = self._context
        if context is not None and context.is_active():
            context.cancel()

    def _next(self, timeout: float) -> Any:
        if self._abandon:
            return _CLOSE
        try:
            return self._out.get(timeout=timeout)
        except queue.Empty:
            return _CLOSE if self.closed_reason is not None else None


def _rpc_reason(e: grpc.RpcError) -> str:
    code = e.code() if hasattr(e, "code") else None    # a cut client: a bare RpcError without code()
    return "stream cut" if code is None else f"stream cut ({code.name})"


class HubServicer(pb_grpc.HubServicer):
    """The ``colosseum.v1.Hub`` methods (module docstring)."""

    def __init__(self, inbox: queue.SimpleQueue, blobs: BlobStore, max_message_mb: int) -> None:
        self._inbox = inbox
        self._blobs = blobs
        self._upload_cap = payload_byte_cap(max_message_mb)

    # ---- Register -----------------------------------------------------------------------------------------------

    def Register(self, request: pb.RegisterRequest, context: grpc.ServicerContext) -> pb.RegisterReply:
        reply: Future = Future()
        self._inbox.put(RegisterIn(request, context.peer(), reply))
        try:
            return _await_reply(reply, context, REGISTER_REPLY_TIMEOUT_SEC)
        except _Cancelled:                             # nobody reads the reply (the hub stops or the node left)
            return pb.RegisterReply()
        except FutureTimeout:
            context.abort(grpc.StatusCode.UNAVAILABLE, "the hub did not answer the registration in time; retry")
        except SecurityError as e:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, str(e))
        except Exception as e:  # noqa: BLE001 - one bad registration never stops the hub
            log.warning("registration from %s failed: %s: %s", context.peer(), type(e).__name__, e)
            context.abort(grpc.StatusCode.INTERNAL, f"registration failed ({type(e).__name__})")

    # ---- Session ------------------------------------------------------------------------------------------------

    def Session(self, request_iterator, context: grpc.ServicerContext):
        node_id = node_id_of(context)                  # verified by the interceptor
        try:
            first = next(request_iterator)
        except (StopIteration, grpc.RpcError):
            return
        if first.WhichOneof("m") != "hello":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "the first session message must be Hello")
        hello = first.hello
        if hello.node_id != node_id:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "Hello names another node id than the credentials")
        if hello.protocol_major != PROTOCOL_VERSION[0]:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          f"protocol {hello.protocol_major}.{hello.protocol_minor} != hub protocol "
                          f"{PROTOCOL_VERSION[0]}.{PROTOCOL_VERSION[1]}: install the hub's version of colosseum")
        if not SESSION_ID_RE.fullmatch(hello.session_id):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Hello needs a session id (1-128 chars of [A-Za-z0-9_.:-])")
        handle = SessionHandle(node_id, hello.session_id)
        handle._context = context
        self._inbox.put(SessionOpened(handle, hello))
        threading.Thread(target=self._read, args=(handle, request_iterator), name=f"session-{node_id[:8]}",
                         daemon=True).start()
        while context.is_active():
            msg = handle._next(POLL_SEC)
            if msg is _CLOSE:
                return
            if msg is not None:
                yield msg

    def _read(self, handle: SessionHandle, request_iterator) -> None:
        reason = "end of stream"
        try:
            for msg in request_iterator:
                self._inbox.put(self._event(handle, msg))
        except grpc.RpcError as e:
            reason = _rpc_reason(e)
        except Exception as e:  # noqa: BLE001 - one bad stream never stops the hub
            reason = f"{type(e).__name__}: {e}"
        handle.close(reason)
        self._inbox.put(SessionClosed(handle.node_id, handle.session_id, handle.closed_reason or reason, handle))

    @staticmethod
    def _event(handle: SessionHandle, msg: pb.NodeMessage) -> NodeMessageIn:
        if msg.WhichOneof("m") != "results":
            return NodeMessageIn(handle.node_id, handle.session_id, msg, serial=handle.serial)
        try:
            results = results_from_batch(msg.results)
        except Exception as e:  # noqa: BLE001 - untrusted input
            return NodeMessageIn(handle.node_id, handle.session_id, msg, error=f"{type(e).__name__}: {e}",
                                 serial=handle.serial)
        return NodeMessageIn(handle.node_id, handle.session_id, msg, results=results, serial=handle.serial)

    # ---- Upload -------------------------------------------------------------------------------------------------

    def Upload(self, request_iterator, context: grpc.ServicerContext) -> pb.UploadReply:
        try:
            return self._upload(request_iterator, node_id_of(context) or "", context)
        except grpc.RpcError:                          # the node went away mid-upload (the spool file is removed)
            return pb.UploadReply(accepted=False, reason="cut")

    def _upload(self, request_iterator, node_id: str, context: grpc.ServicerContext) -> pb.UploadReply:
        def refuse(reason: str) -> pb.UploadReply:
            return pb.UploadReply(accepted=False, reason=reason)

        first = next(request_iterator, None)
        if first is None or first.WhichOneof("p") != "header":
            return refuse("header")
        header = first.header
        if not _valid_agent_id(header.agent_id):
            return refuse("header")
        if header.kind not in UPLOAD_KINDS:
            return refuse("kind")
        total = int(header.total_size)
        if total < 0 or total > self._upload_cap:
            return refuse("too_large")
        spool = private_dir(self._blobs.incoming) / f"{uuid.uuid4().hex}.part"     # P-PF2-9 (a)
        handed = False
        try:
            h = hashlib.blake2b(digest_size=BLOB_DIGEST_SIZE)
            size = 0
            with open(spool, "xb") as f:
                for part in request_iterator:
                    if part.WhichOneof("p") != "data":
                        return refuse("header")
                    size += len(part.data)
                    if size > total:
                        return refuse("size")
                    h.update(part.data)
                    f.write(part.data)
            if size != total:
                return refuse("size")
            if h.hexdigest() != header.digest:
                return refuse("digest")
            reply: Future = Future()
            self._inbox.put(UploadIn(header, spool, node_id, reply))
            handed = True
            try:
                return _await_reply(reply, context, UPLOAD_REPLY_TIMEOUT_SEC)
            except _Cancelled:                         # the hub stops or the node left: nobody reads the reply
                spool.unlink(missing_ok=True)
                return refuse("cut")
            except FutureTimeout:
                spool.unlink(missing_ok=True)
                return refuse("timeout")
            except Exception as e:  # noqa: BLE001 - the main loop failed this upload
                spool.unlink(missing_ok=True)
                return refuse(f"error: {type(e).__name__}")
        finally:
            if not handed:
                spool.unlink(missing_ok=True)

    # ---- Download -----------------------------------------------------------------------------------------------

    def Download(self, request: pb.BlobRequest, context: grpc.ServicerContext):
        path = self._blobs.frame_path(request.kind, request.agent_id, request.ident)
        if path is None:
            context.abort(grpc.StatusCode.NOT_FOUND,
                          f"no {request.kind[:32]!r} blob {request.agent_id[:64]!r} / {request.ident[:64]!r}")
        try:
            parts = send_parts(path)
            first = next(parts)
        except FileNotFoundError:
            context.abort(grpc.StatusCode.NOT_FOUND, f"blob {path.name} vanished")
        yield first
        yield from parts

    # ---- Relay (T3.2) -------------------------------------------------------------------------------------------

    def RelaySend(self, request_iterator, context: grpc.ServicerContext) -> pb.ChunkSummary:
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "relay: T3.2")

    def RelayInbox(self, request_iterator, context: grpc.ServicerContext):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "relay: T3.2")
        yield  # pragma: no cover - abort raises

    # ---- Admin --------------------------------------------------------------------------------------------------

    @staticmethod
    def _admin_args(request: pb.AdminRequest, context: grpc.ServicerContext) -> dict:
        try:
            args = json.loads(request.args_json or "{}")
        except ValueError:
            args = None
        if not isinstance(args, dict):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "args_json must be a JSON object")
        return args

    def Admin(self, request: pb.AdminRequest, context: grpc.ServicerContext) -> pb.AdminReply:
        args = self._admin_args(request, context)
        reply: Future = Future()
        self._inbox.put(AdminIn(request.command, args, reply))
        try:
            return _await_reply(reply, context, ADMIN_REPLY_TIMEOUT_SEC)
        except _Cancelled:                             # nobody reads the reply (the hub stops or the caller left)
            return pb.AdminReply()
        except FutureTimeout:
            context.abort(grpc.StatusCode.UNAVAILABLE, "the hub did not answer in time")
        except Exception as e:  # noqa: BLE001 - the main loop failed this command
            context.abort(grpc.StatusCode.INTERNAL, f"{request.command[:64]}: {type(e).__name__}: {e}")

    def AdminStream(self, request: pb.AdminRequest, context: grpc.ServicerContext):
        args = self._admin_args(request, context)
        replies: queue.SimpleQueue = queue.SimpleQueue()
        self._inbox.put(AdminIn(request.command, args, replies))
        while context.is_active():
            try:
                reply = replies.get(timeout=POLL_SEC)
            except queue.Empty:
                continue
            yield reply
            if reply.done:
                return


# ---------------------------------------------------------------------------------------------------------------
# Blobs
# ---------------------------------------------------------------------------------------------------------------


def blob_ident(kind: str, agent_id: str, data: bytes | str) -> str:
    """The ``ident`` of a registered blob: 16 hex chars of a blake2b over ``kind``, ``agent_id`` and the content, so
    it is unique per (kind, agent, content) (node caches name blob files by kind and ident only, A18)."""
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    h = hashlib.blake2b(digest_size=BLOB_DIGEST_SIZE)
    for piece in (kind.encode("utf-8"), b"\0", agent_id.encode("utf-8"), b"\0", raw):
        h.update(piece)
    return h.hexdigest()[:16]


def _is_file(path: Path) -> bool:
    """``path.is_file()`` that is False for any OSError (Python 3.12's raises e.g. ENAMETOOLONG for a very long
    agent id)."""
    try:
        return path.is_file()
    except OSError:
        return False


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with open(tmp, "xb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class BlobStore:
    """Read-only file serving and lazy snapshot conversion for ``Download`` (servicer threads) plus the registered
    blobs of the main loop (module docstring). Files: ``runs/<name>/blobs/<kind>-<agent digest>-<ident>.frame``
    (``trainer_state``: ``.bin``; ``_file_name``). It never deletes snapshot frames itself (T3.1 removes those of
    evicted snapshots after their hold). Agent ids follow the config's rule (``check_agent_id``); the snapshot source
    ``checkpoints/<agent>/ckpt_v<N>/model.pt`` keeps the agent's directory."""

    def __init__(self, run_dir: RunDir) -> None:
        self.run_dir = run_dir
        self._registered: set[tuple[str, str, str]] = set()
        self._lock = threading.Lock()
        self._convert_locks: dict[tuple[str, str], threading.Lock] = {}

    @property
    def dir(self) -> Path:
        return self.run_dir.root / "blobs"

    @property
    def incoming(self) -> Path:
        """``runs/<name>/.incoming/``: upload spool files."""
        return self.run_dir.root / ".incoming"

    @staticmethod
    def _file_name(kind: str, agent_id: str, ident: str) -> str:
        """``<kind>-<agent digest>-<ident>.frame`` (``trainer_state``: ``.bin``), the agent digest being 32 hex chars
        of a blake2b of the agent id. Injective: kinds contain no ``-`` and the digest has a fixed length, so the name
        splits back into (kind, digest, ident) although agent ids and idents may contain ``-``. Bounded: at most
        13 + 1 + 32 + 1 + 160 + 6 chars whatever the agent id's length (the config sets no limit on it)."""
        agent = hashlib.blake2b(agent_id.encode("utf-8"), digest_size=_AGENT_DIGEST_SIZE).hexdigest()
        return f"{kind}-{agent}-{ident}" + (".bin" if kind == "trainer_state" else ".frame")

    def frame_path(self, kind: str, agent_id: str, ident: str) -> Path | None:
        """The file to serve for a ``BlobRequest``, or None (unknown, invalid or unreadable). Ids are checked before
        any path is formed (P-PF2-9 (c))."""
        if not (isinstance(kind, str) and isinstance(agent_id, str) and isinstance(ident, str)):
            return None
        if not _valid_agent_id(agent_id):
            return None
        if kind == "snapshot":
            m = SNAPSHOT_IDENT_RE.fullmatch(ident)
            return None if m is None else self._snapshot(agent_id, ident, int(m.group(1)))
        if kind not in REGISTERED_KINDS or not BLOB_IDENT_RE.fullmatch(ident):
            return None
        with self._lock:
            if (kind, agent_id, ident) not in self._registered:
                return None
        path = self.dir / self._file_name(kind, agent_id, ident)
        return path if _is_file(path) else None

    def _snapshot(self, agent_id: str, ident: str, version: int) -> Path | None:
        path = self.dir / self._file_name("snapshot", agent_id, ident)
        if _is_file(path):
            return path
        model = self.run_dir.checkpoints / agent_id / ident / "model.pt"
        if not _is_file(model):                       # no lock entry for a snapshot that does not exist
            return None
        with self._lock:
            lock = self._convert_locks.setdefault((agent_id, ident), threading.Lock())
        with lock:                                    # two concurrent downloads convert once
            if _is_file(path):
                return path
            if not _is_file(model):
                return None
            try:
                state = read_weights_file(model)
            except ValueError as e:
                log.warning("snapshot %s/%s cannot be served: %s", agent_id, ident, e)
                return None
            frame = encode_weights(state, WeightsMeta(agent_id=agent_id, kind="snapshot", generation=0,
                                                      policy_version=version, checkpoint_id=ident))
            self.dir.mkdir(parents=True, exist_ok=True)
            _write_atomic(path, frame)
            return path

    def register_fixed(self, blob_id: str, frame: bytes, *, kind: str, agent_id: str) -> None:
        """Register (and write atomically) a blob the main loop serves: a frozen agent's or a teacher's weights, an
        ``init`` state, a recovery frame or trainer state (``kind``). ValueError for an invalid kind or id."""
        if kind not in REGISTERED_KINDS:
            raise ValueError(f"blob kind must be one of {REGISTERED_KINDS}, got {kind!r}")
        if not _valid_agent_id(agent_id):
            raise ValueError(f"invalid agent id {agent_id!r} for a blob")
        if not isinstance(blob_id, str) or not BLOB_IDENT_RE.fullmatch(blob_id):
            raise ValueError(f"invalid blob id {blob_id!r}")
        self.dir.mkdir(parents=True, exist_ok=True)
        _write_atomic(self.dir / self._file_name(kind, agent_id, blob_id), bytes(frame))
        with self._lock:
            self._registered.add((kind, agent_id, blob_id))


# ---------------------------------------------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------------------------------------------


def _bind(server: grpc.Server, address: str, credentials: grpc.ServerCredentials | None) -> int:
    try:
        port = server.add_insecure_port(address) if credentials is None else server.add_secure_port(address,
                                                                                                     credentials)
    except RuntimeError as e:                          # grpcio raises for a port it cannot bind
        raise OSError(f"cannot listen on {address}: {e}") from None
    if port == 0:
        raise OSError(f"cannot listen on {address} (the port is taken or the address is not local)")
    return port


def build_server(*, inbox: queue.SimpleQueue, identity: HubIdentity, insecure: bool, listen: Sequence[str],
                 loopback_plain: bool, blobs: BlobStore, max_message_mb: int,
                 bootstrap: Callable[[grpc.Server], None] | None = None) -> tuple[grpc.Server, dict[str, int]]:
    """The hub's gRPC server (unstarted) and ``{address or LOOPBACK_KEY: bound port}``. Every address of ``listen``
    is TLS with the hub identity's certificate, or plain with ``insecure``; ``loopback_plain`` adds a plain
    ``127.0.0.1:0`` port (``train``'s local node, R12). The token / node secrets are required in every mode (R1).
    ``bootstrap`` registers the Bootstrap service on the server (T6.2). OSError names an address it cannot bind."""
    server = grpc.server(ThreadPoolExecutor(max_workers=SERVER_THREADS, thread_name_prefix="hub-rpc"),
                         interceptors=[AuthInterceptor(lambda: identity.token.secret, identity.nodes)],
                         options=server_options(max_message_mb))
    pb_grpc.add_HubServicer_to_server(HubServicer(inbox, blobs, max_message_mb), server)
    if bootstrap is not None:
        bootstrap(server)
    ports: dict[str, int] = {}
    try:
        for address in listen:
            ports[address] = _bind(server, address, None if insecure else identity.server_credentials())
        if loopback_plain:
            ports[LOOPBACK_KEY] = _bind(server, "127.0.0.1:0", None)
    except OSError:
        server.stop(None)
        raise
    return server, ports
