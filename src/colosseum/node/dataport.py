"""The node's data port (SP5 spec 3.1-3.3, 4.1, 4.5, 4.7; amendment A25): the ``NodeData`` gRPC service, mutual
TLS (plain + the token secret with ``--insecure``).

- ``GetLatest(agent, since_version)``: the newest latest weights of an agent whose learner runs on this node,
  straight from the machine's shared-memory cache: one part ``not_modified`` when the cache holds nothing newer
  than ``since_version``, else the weights frame in parts (``grpc_util.send_parts``) whose first part's
  ``meta_json`` is ``{"agent_id", "version", "generation", "size", "digest", "teacher_active"}`` (the cache
  pointer; ``digest`` = blake2b-16 of the frame). Answered without the hub (spec 4.5). Only this node's own
  learners' weights are served (no re-serving of fetched weights: one owner per version).
- ``Chunks(stream ChunkFrame)``: frames for this node's learners (T3.2; refused with ``UNIMPLEMENTED`` until then).

Peers: with TLS the client certificate must be issued by the hub's CA (``require_client_auth``) and its common name
(the node id) must not be banned (``banned()``, from ``Welcome`` / ``Announce``; the check is
``grpc_util.DataPortInterceptor``, pre-flight ruling P-PF3-12); with ``--insecure`` the metadata must carry the run
token's secret (``x-colosseum-token``). Refusals: ``UNAUTHENTICATED`` / ``PERMISSION_DENIED`` (counted in
``peer_refused``); an agent without a learner here: ``NOT_FOUND``. A certificate of another CA, or none, fails the
TLS handshake (the client sees ``UNAVAILABLE``).

Binding (spec 3.1): ``distributed.data_port``, the next free port when it is taken (``DATA_PORT_TRIES`` in all;
``grpc.so_reuseport`` is off in ``server_options``, otherwise a second node on the machine would silently share the
port); port 0 binds an ephemeral port (tests). ``data_addresses`` lists what the node announces.

``probe_direct`` (also used by the chunk router, T3.2) and ``fetch_latest`` are the client side.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import mmap
import socket
import threading
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from concurrent import futures
from dataclasses import dataclass
from typing import Any

import grpc

from colosseum.net.frame import FrameError, payload_byte_cap, peek_meta
from colosseum.net.grpc_util import (
    DataPortInterceptor,
    metadata_value,
    not_modified_part,
    receive_parts,
    send_parts,
    server_options,
)
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.proto import colosseum_v1_pb2_grpc as pbg
from colosseum.net.security import MD_TOKEN
from colosseum.node.weights import frame_digest

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_PORT_TRIES", "DATA_SERVER_THREADS", "DataServer", "FetchError", "FetchedLatest", "check_fetched",
    "data_addresses", "fetch_latest", "frame_digest", "is_loopback", "open_data_port", "probe_direct",
    "read_latest_reply",
]

DATA_PORT_TRIES = 16
DATA_SERVER_THREADS = 64


def data_addresses(port: int, hub_host: str) -> list[str]:
    """``host:port`` addresses a node announces for its data port: first the local address of the route to the
    hub (a UDP ``connect`` sends nothing), then the machine's other IPv4 addresses; loopback only when the hub
    itself is on loopback (one-machine tests and ``train``'s local node), and then first."""
    hosts: list[str] = []
    loopback_hub = is_loopback(hub_host)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((hub_host, 9))
            hosts.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            hosts.append(str(info[4][0]))
    except OSError:
        pass
    out: list[str] = []
    for host in hosts:
        if is_loopback(host) and not loopback_hub:
            continue
        address = f"{host}:{int(port)}"
        if address not in out:
            out.append(address)
    if loopback_hub:
        first = f"127.0.0.1:{int(port)}"
        if first in out:
            out.remove(first)
        out.insert(0, first)
    return out


def is_loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class DataServer(pbg.NodeDataServicer):
    """The ``NodeData`` servicer and its gRPC server (module docstring). ``cache``: the machine's
    ``WeightCacheReader`` (``set_cache`` replaces it for another run); ``own_agents()``: agents whose learner runs on
    this node (a snapshot the main loop publishes); ``banned()``: node ids refused; ``delivery``: the T3.2
    ``LocalDelivery`` (None: ``Chunks`` is refused with ``UNIMPLEMENTED``). ``counters`` (``latest_served``,
    ``latest_not_modified``, ``peer_refused``) are cumulative for the server's life (the node process)."""

    def __init__(self, *, cache: Any, own_agents: Callable[[], Collection[str]], max_message_mb: int,
                 insecure: bool, token_secret: str, banned: Callable[[], Collection[str]] = frozenset,
                 delivery: Any | None = None, node: int = -1) -> None:
        self._cache = cache
        self._own = own_agents
        self._max_mb = int(max_message_mb)
        self._insecure = bool(insecure)
        self._token_secret = token_secret
        self._banned = banned
        self._delivery = delivery
        self._node = int(node)
        self._server: grpc.Server | None = None
        self.port: int | None = None
        self._lock = threading.Lock()
        self.counters: dict[str, int] = {"latest_served": 0, "latest_not_modified": 0, "peer_refused": 0}

    def set_node(self, node: int) -> None:
        """The node number, known from ``Welcome`` (frames for another node are refused by ``Chunks``, T3.2)."""
        self._node = int(node)

    def set_cache(self, cache: Any) -> None:
        self._cache = cache

    def counters_snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counters)

    # --- lifecycle -----------------------------------------------------------------------------------------------

    def start(self, port: int, *, credentials: grpc.ServerCredentials | None, host: str = "0.0.0.0") -> int:
        """Bind ``port`` (or the next free one, ``DATA_PORT_TRIES`` in all; 0 = ephemeral) on ``host`` and start
        serving. ``credentials`` None means plain (``--insecure``: the token secret is checked per call). Raises
        ``OSError`` when no port could be bound."""
        interceptors = [] if credentials is None else [DataPortInterceptor(self._peer_allowed)]
        candidates = [0] if int(port) == 0 else range(int(port), int(port) + DATA_PORT_TRIES)
        for candidate in candidates:
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=DATA_SERVER_THREADS,
                                                            thread_name_prefix="data-port"),
                                 interceptors=interceptors, options=server_options(self._max_mb))
            pbg.add_NodeDataServicer_to_server(self, server)
            address = f"{host}:{candidate}"
            try:
                bound = (server.add_insecure_port(address) if credentials is None
                         else server.add_secure_port(address, credentials))
            except RuntimeError:            # grpcio raises on a failed bind ("Failed to bind to address ...")
                bound = 0
            if bound:
                server.start()
                self._server, self.port = server, int(bound)
                if port and bound != port:
                    logger.info(f"node: data port {port} is taken; listening on {bound}")
                return int(bound)
            server.stop(None)
        raise OSError(f"no free data port in {port}..{int(port) + DATA_PORT_TRIES - 1} on {host}")

    def stop(self, grace: float = 1.0) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.stop(grace).wait(grace + 1.0)

    # --- NodeData ------------------------------------------------------------------------------------------------

    def GetLatest(self, request: pb.LatestRequest, context: grpc.ServicerContext) -> Iterator[pb.BlobPart]:  # noqa: N802
        self._check_token(context)
        agent_id = request.agent_id
        if agent_id not in set(self._own()):
            context.abort(grpc.StatusCode.NOT_FOUND, f"node {self._node} runs no learner of agent {agent_id[:64]!r}")
        opened = self._cache.open_latest(agent_id)
        if opened is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"no weights of agent {agent_id!r} in the cache yet")
        f, pointer = opened
        with f:
            meta = json.dumps({"agent_id": agent_id, **{k: pointer.get(k) for k in (
                "version", "generation", "size", "digest", "teacher_active")}})
            if int(pointer["version"]) <= int(request.since_version):
                self._count("latest_not_modified")
                yield not_modified_part(meta)
                return
            mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                yield from send_parts(mapping, meta_json=meta, digest=str(pointer["digest"]))
            finally:
                try:
                    mapping.close()
                except BufferError:          # a part view still alive (a cancelled stream): freed with it
                    pass
        self._count("latest_served")

    def Chunks(self, request_iterator: Iterable[pb.ChunkFrame], context: grpc.ServicerContext) -> pb.ChunkSummary:  # noqa: N802
        self._check_token(context)
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "chunks over the data port arrive with T3.2")

    # --- internals -----------------------------------------------------------------------------------------------

    def _peer_allowed(self, node_id: str) -> bool:
        """``DataPortInterceptor``'s check of a TLS peer (its certificate's CN): not banned."""
        if node_id in set(self._banned()):
            self._count("peer_refused")
            return False
        return True

    def _check_token(self, context: grpc.ServicerContext) -> None:
        """``--insecure``: the run token's secret in the metadata (TLS peers were checked by the interceptor)."""
        if not self._insecure:
            return
        secret = metadata_value(context.invocation_metadata(), MD_TOKEN)
        if secret is None or not hmac.compare_digest(secret.encode(), self._token_secret.encode()):
            self._count("peer_refused")
            context.abort(grpc.StatusCode.UNAUTHENTICATED,
                          "data port: missing or wrong token secret (x-colosseum-token)")

    def _count(self, name: str) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + 1


def open_data_port(*, port: int, hub_host: str, insecure: bool, token_secret: str, identity: Any, cache: Any,
                   own_agents: Callable[[], Collection[str]], banned: Callable[[], Collection[str]],
                   max_message_mb: int, node: int,
                   announced: Sequence[str] | None = None) -> tuple[DataServer, tuple[str, ...]] | None:
    """A node's data port (``NodeAgent``, item 11): mutual TLS with the node's certificate (``identity``, a
    ``NodeIdentity`` with its CA in ``ca.crt``), or plain + the token secret (``insecure``); bound on loopback when the
    hub is on loopback, else on every interface (pre-flight ruling P-PF3-21), at ``port`` or the next free one.
    Returns the started server and the addresses to announce (``announced``, else ``data_addresses``); None after one
    WARNING when it cannot be opened (no node certificate: ``train``'s local node; no free port)."""
    from colosseum.net.security import data_server_credentials

    hint = "other machines reach this node's learners only through the hub"
    credentials = None
    if not insecure:
        if identity is None or identity.cert_pem is None:
            logger.warning(f"node: no data port: this node has no node certificate; {hint}")
            return None
        try:
            credentials = data_server_credentials(identity, (identity.path / "ca.crt").read_bytes())
        except Exception as e:  # noqa: BLE001 - the node runs without a data port
            logger.warning(f"node: no data port ({type(e).__name__}: {e}); {hint}")
            return None
    server = DataServer(cache=cache, own_agents=own_agents, max_message_mb=max_message_mb, insecure=insecure,
                        token_secret=token_secret, banned=banned, node=node)
    host = "127.0.0.1" if is_loopback(hub_host) else "0.0.0.0"
    try:
        bound = server.start(port, credentials=credentials, host=host)
    except OSError as e:
        logger.warning(f"node: no data port ({e}); {hint}")
        return None
    addresses = tuple(announced) if announced else tuple(data_addresses(bound, hub_host))
    logger.info(f"node: data port {host}:{bound} ({'plain + token' if insecure else 'mutual TLS'}); announced as "
                f"{', '.join(addresses)}")
    return server, addresses


def probe_direct(addresses: Sequence[str], make_channel: Callable[[str], grpc.Channel],
                 timeout: float) -> tuple[str, grpc.Channel] | None:
    """The first of ``addresses`` whose channel becomes READY within ``timeout`` seconds (all probed at once;
    READY includes the TLS handshake); the other channels are closed. None when none answers (spec 4.8: then the
    relay). ``grpc.channel_ready_future`` returns a ``grpc.Future``, not a ``concurrent.futures`` one: completion
    is signalled through ``add_done_callback``."""
    if not addresses:
        return None
    channels = {a: make_channel(a) for a in addresses}
    changed = threading.Event()
    ready = {a: grpc.channel_ready_future(ch) for a, ch in channels.items()}
    for fut in ready.values():
        fut.add_done_callback(lambda _f: changed.set())
    winner = None
    deadline = time.monotonic() + timeout
    while winner is None:
        for address, fut in ready.items():
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                winner = address
                break
        remaining = deadline - time.monotonic()
        if winner is not None or remaining <= 0 or all(f.done() for f in ready.values()):
            break
        changed.wait(remaining)
        changed.clear()
    for address, fut in ready.items():
        if address != winner:
            fut.cancel()
            channels[address].close()
    return None if winner is None else (winner, channels[winner])


class FetchError(Exception):
    """A latest-weights fetch that failed; ``reason``: ``rpc`` (unreachable, refused, not found: the text has the
    status), ``truncated``, ``blob`` (a malformed part stream), ``digest``, ``invalid`` (metadata or frame)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass(frozen=True)
class FetchedLatest:
    agent_id: str
    version: int
    generation: int
    teacher_active: bool
    frame: bytes
    source: str            # "direct" | "relay"


def check_fetched(agent_id: str, frame: bytes, meta: dict, source: str) -> FetchedLatest:
    """Hostile-input checks of a fetched latest frame (spec 3.4): the digest of the bytes, the frame header's agent
    and version against the reply's metadata."""
    try:
        version, generation = int(meta["version"]), int(meta["generation"])
        digest, teacher_active = str(meta["digest"]), bool(meta.get("teacher_active", False))
    except (KeyError, TypeError, ValueError) as e:
        raise FetchError("invalid", f"metadata {str(meta)[:200]}") from e
    if frame_digest(frame) != digest:
        raise FetchError("digest", f"agent {agent_id} v{version}")
    try:
        kind, header = peek_meta(frame)
    except FrameError as e:
        raise FetchError("invalid", f"frame header: {e}") from e
    if (kind != "weights" or header.get("agent_id") != agent_id or header.get("kind") != "latest"
            or int(header.get("policy_version", -1)) != version):
        raise FetchError("invalid", f"frame of {header.get('agent_id')!r} v{header.get('policy_version')} for "
                                    f"{agent_id!r} v{version}")
    return FetchedLatest(agent_id, version, generation, teacher_active, frame, source)


def read_latest_reply(parts: Iterator[pb.BlobPart], agent_id: str, *, max_message_mb: int,
                      source: str) -> FetchedLatest | None:
    """A ``GetLatest`` / ``Download(kind="latest")`` reply: None for "not modified", else the checked frame.
    Raises ``FetchError`` (``grpc.RpcError`` and malformed parts included)."""
    try:
        frame, meta_json = receive_parts(parts, cap=payload_byte_cap(max_message_mb))
    except grpc.RpcError as e:
        code = e.code() if hasattr(e, "code") else None
        details = e.details() if hasattr(e, "details") else str(e)
        raise FetchError("rpc", f"{code}: {details}") from e
    except ValueError as e:            # grpc_util.BlobError: size cap, offsets, digest, empty
        raise FetchError("blob", str(e)) from e
    if frame is None:
        return None
    try:
        meta = json.loads(meta_json)
    except ValueError as e:
        raise FetchError("invalid", "metadata is not JSON") from e
    if not isinstance(meta, dict):
        raise FetchError("invalid", "metadata is not an object")
    return check_fetched(agent_id, bytes(frame), meta, source)


def fetch_latest(channel: grpc.Channel, agent_id: str, since_version: int, *, max_message_mb: int,
                 metadata: Sequence[tuple[str, str]] = (), timeout: float = 60.0) -> FetchedLatest | None:
    """``GetLatest`` on a node's data port. None: not modified. Raises ``FetchError`` (unreachable, refused,
    truncated, wrong digest, wrong frame)."""
    stub = pbg.NodeDataStub(channel)
    try:
        parts = stub.GetLatest(pb.LatestRequest(agent_id=agent_id, since_version=int(since_version)),
                               metadata=tuple(metadata), timeout=timeout)
    except grpc.RpcError as e:
        raise FetchError("rpc", f"{e.code()}: {e.details()}") from e
    return read_latest_reply(parts, agent_id, max_message_mb=max_message_mb, source="direct")
