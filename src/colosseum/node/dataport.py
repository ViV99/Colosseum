"""The node's data port (SP5 spec 3.1-3.3, 4.1, 4.5, 4.7; amendment A25): the ``NodeData`` gRPC service, mutual
TLS (plain + the token secret with ``--insecure``).

- ``GetLatest(agent, since_version)``: the newest latest weights of an agent whose learner runs on this node,
  straight from the machine's shared-memory cache: one part ``not_modified`` when the cache holds nothing newer
  than ``since_version``, else the weights frame in parts (``grpc_util.send_parts``) whose first part's
  ``meta_json`` is ``{"agent_id", "version", "generation", "size", "digest", "teacher_active"}`` (the cache
  pointer; ``digest`` = blake2b-16 of the frame). Answered without the hub (spec 4.5). Only this node's own
  learners' weights are served (no re-serving of fetched weights: one owner per version).
- ``Chunks(stream ChunkFrame)``: frames for this node's learners (T3.2): each frame goes undecoded to its learner's
  chunk connection (``router.LocalDelivery``, blocking: end-to-end backpressure); one without a learner here, or for
  another node, is dropped and counted; ``UNIMPLEMENTED`` while the node serves no run.

Peers: with TLS the client certificate must be issued by the hub's CA (``require_client_auth``) and its common name
(the node id) must not be banned (``banned()``, from ``Welcome`` / ``Announce``; the check is
``grpc_util.DataPortInterceptor``, pre-flight ruling P-PF3-12); with ``--insecure`` the metadata must carry the run
token's secret (``x-colosseum-token``). Refusals: ``UNAUTHENTICATED`` / ``PERMISSION_DENIED`` (counted in
``peer_refused``); an agent without a learner here: ``NOT_FOUND``. A certificate of another CA, or none, fails the
TLS handshake (the client sees ``UNAVAILABLE``).

Binding (spec 3.1): ``distributed.data_port``, the next free port when it is taken (``DATA_PORT_TRIES`` in all;
``grpc.so_reuseport`` is off in ``server_options``, otherwise a second node on the machine would silently share the
port); port 0 binds an ephemeral port (tests). A node announces only addresses its port is bound on
(``open_data_port``): ``data_addresses`` for a port on every interface. A client checks the serving node's identity:
its channel's TLS server name is ``security.node_server_name(<the source's number>)``, a SAN only that node's
certificate carries.

``probe_direct`` (also used by the chunk router, T3.2) and ``fetch_latest`` are the client side.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import mmap
import socket
import sys
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
    "data_addresses", "fetch_latest", "frame_digest", "is_loopback", "is_wildcard", "machine_addresses",
    "open_data_port", "probe_direct",
    "read_latest_reply",
]

DATA_PORT_TRIES = 16
DATA_SERVER_THREADS = 64


# A destination for the "default route" probe: a UDP ``connect`` sends nothing, it only picks the source address.
_ROUTE_PROBE_DEST = ("192.0.2.1", 9)              # TEST-NET-1 (RFC 5737): never answered, never needed
_SIOCGIFCONF = 0x8912
_IFREQ_BYTES = 40                                 # struct ifreq on 64-bit Linux


def data_addresses(port: int, hub_host: str) -> list[str]:
    """``host:port`` addresses a node announces for a data port bound on every interface: the machine's non-loopback
    IPv4 addresses (the source of the route to the hub, the source of the default route, the host name's addresses,
    every interface's address: ``machine_addresses``); loopback only when the hub itself is on loopback (one-machine
    tests and ``train``'s local node), and then first (the hub shows loopback addresses only to nodes of the same
    machine, ``hub.state.visible_version``)."""
    out = [f"{host}:{int(port)}" for host in machine_addresses(hub_host)]
    if is_loopback(hub_host):
        out.insert(0, f"127.0.0.1:{int(port)}")
    return out


def machine_addresses(hub_host: str = "") -> list[str]:
    """This machine's non-loopback IPv4 addresses, the likeliest first: the source of the route to ``hub_host`` (when
    it is not loopback), the source of the default route, the host name's addresses (``/etc/hosts`` often maps it to
    127.0.1.1: skipped), then every interface's (Linux ``SIOCGIFCONF``). Stdlib only; nothing is sent."""
    hosts: list[str] = []
    if hub_host and not is_loopback(hub_host):
        hosts.append(_route_source((hub_host, 9)) or "")
    hosts.append(_route_source(_ROUTE_PROBE_DEST) or "")
    try:
        hosts.extend(str(info[4][0]) for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    hosts.extend(_interface_addresses())
    out: list[str] = []
    for host in hosts:
        if host and not is_loopback(host) and host != "0.0.0.0" and host not in out:
            out.append(host)
    return out


def _route_source(dest: tuple[str, int]) -> str | None:
    """The local address of the route to ``dest`` (a UDP ``connect`` sends no packet); None without such a route."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(dest)
            return str(s.getsockname()[0])
    except OSError:
        return None


def _interface_addresses() -> list[str]:
    """Every interface's IPv4 address (Linux ``SIOCGIFCONF``; elsewhere, or on any error, none)."""
    if not sys.platform.startswith("linux"):
        return []
    try:
        import array
        import fcntl
        import struct

        size = _IFREQ_BYTES * 128
        buf = array.array("B", bytes(size))
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            got = struct.unpack("iL", fcntl.ioctl(s.fileno(), _SIOCGIFCONF,
                                                  struct.pack("iL", size, buf.buffer_info()[0])))[0]
        data = buf.tobytes()[:got]
        return [socket.inet_ntoa(data[i + 20:i + 24]) for i in range(0, len(data) - _IFREQ_BYTES + 1, _IFREQ_BYTES)]
    except Exception:  # noqa: BLE001 - discovery is best effort (struct.error, OSError, ...)
        return []


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

    def set_delivery(self, delivery: Any | None) -> None:
        """The run's ``router.LocalDelivery`` (the node binds its data port once per process and its chunk router is
        per run, T3.2); None: ``Chunks`` is refused (``UNIMPLEMENTED``)."""
        self._delivery = delivery

    def counters_snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counters)

    # --- lifecycle -----------------------------------------------------------------------------------------------

    def start(self, port: int, *, credentials: grpc.ServerCredentials | None, host: str = "0.0.0.0") -> int:
        """Bind ``port`` (or the next free one, ``DATA_PORT_TRIES`` in all; 0 = ephemeral) on ``host`` and start
        serving. ``credentials`` None means plain, which only an ``insecure`` server may do (the token secret is then
        checked per call; a TLS server without credentials would serve plain without any check: ValueError). Raises
        ``OSError`` when no port could be bound."""
        if credentials is None and not self._insecure:
            raise ValueError("a data port without --insecure needs TLS credentials (data_server_credentials)")
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
        """Frames for this node's learners (spec 4.7): each goes to its learner's chunk connection undecoded
        (``LocalDelivery.deliver``, blocking while the learner does not read: gRPC flow control then holds the
        sender back). A frame for another node or an agent without a learner here is dropped (counted); the stream
        itself never fails because of one frame."""
        self._check_token(context)
        delivery = self._delivery
        if delivery is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "this node takes no chunks (no run yet)")
        received = dropped = 0
        for frame in request_iterator:
            received += 1
            if frame.target_node != self._node:
                delivery.count("dropped_chunks.no_learner")
                dropped += 1
                continue
            if delivery.deliver(frame.agent_id, frame.frame) is not None:
                dropped += 1
        return pb.ChunkSummary(received=received, dropped=dropped)

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


def is_wildcard(host: str) -> bool:
    """``host`` binds every interface (``0.0.0.0``, ``[::]``)."""
    return host.strip("[]") in ("0.0.0.0", "::", "")


def open_data_port(*, port: int, hub_host: str, insecure: bool, token_secret: str, identity: Any, cache: Any,
                   own_agents: Callable[[], Collection[str]], banned: Callable[[], Collection[str]],
                   max_message_mb: int, node: int, announced: Sequence[str] | None = None,
                   local: bool = False, host: str | None = None,
                   delivery: Any | None = None) -> tuple[DataServer, tuple[str, ...]] | None:
    """A node's data port (``NodeAgent``, item 11): mutual TLS with the node's certificate (``identity``: a
    ``NodeIdentity``, or the ``MemoryNodeIdentity`` the hub minted for ``train``'s local node, R30), or plain + the
    token secret (``insecure``); bound on ``host`` (default: loopback when the hub is on loopback, else every interface,
    pre-flight ruling P-PF3-21), at ``port`` or the next free one; ``delivery``: the run's ``router.LocalDelivery``
    (``Chunks``). Returns the started server and the addresses to announce: ``announced`` when given (port forwards),
    else only addresses the port is bound on (carried decision 3a of T3.1): every interface's (``data_addresses``)
    for a wildcard host, else ``host:port``. None after one WARNING when it cannot be opened (no node certificate, no
    free port). ``local``: ``train``'s node, which has no node certificate without ``--listen`` (an INFO, not a
    WARNING: its learners need no data port then)."""
    from colosseum.net.security import data_server_credentials

    hint = "other machines reach this node's learners only through the hub"
    credentials = None
    if not insecure:
        if identity is None or identity.cert_pem is None:
            logger.log(logging.INFO if local else logging.WARNING,
                       f"node: no data port: this node has no node certificate; {hint}")
            return None
        try:
            credentials = data_server_credentials(identity, identity.ca_pem)
        except Exception as e:  # noqa: BLE001 - the node runs without a data port
            logger.warning(f"node: no data port ({type(e).__name__}: {e}); {hint}")
            return None
        _check_own_name(identity.cert_pem, node)
    server = DataServer(cache=cache, own_agents=own_agents, max_message_mb=max_message_mb, insecure=insecure,
                        token_secret=token_secret, banned=banned, node=node, delivery=delivery)
    if host is None:
        host = "127.0.0.1" if is_loopback(hub_host) else "0.0.0.0"
    try:
        bound = server.start(port, credentials=credentials, host=host)
    except OSError as e:
        logger.warning(f"node: no data port ({e}); {hint}")
        return None
    if announced:
        addresses = tuple(announced)
    elif is_wildcard(host):
        addresses = tuple(data_addresses(bound, hub_host))
        if not any(not is_loopback(a.rpartition(":")[0]) for a in addresses):
            logger.warning(f"node: no non-loopback address of this machine was found for the data port on {host}:"
                           f"{bound}; other machines reach this node's learners only through the hub. Announce the "
                           f"addresses to use (NodeOptions.data_addresses, e.g. a port forward) or listen on a "
                           f"specific address (--listen ADDR)")
    else:
        addresses = (f"{host}:{bound}",)
    logger.info(f"node: data port {host}:{bound} ({'plain + token' if insecure else 'mutual TLS'}); announced as "
                f"{', '.join(addresses)}")
    return server, addresses


def _check_own_name(cert_pem: bytes, node: int) -> None:
    """A node certificate issued before SP5 T3.2 lacks its per-node name (``security.node_server_name``): every peer
    refuses its data port (a failed handshake, seen there as "unreachable"). Say so once, with the remedy."""
    from colosseum.net.security import certificate_dns_names, node_server_name

    if node < 0:
        return
    try:
        names = certificate_dns_names(cert_pem)
    except ValueError:
        names = []
    if node_server_name(node) not in names:
        logger.warning(f"node: this node's certificate does not name it as node {node} ({node_server_name(node)}): "
                       f"other nodes refuse its data port and reach its learners only through the hub. Re-register "
                       f"this node (forget its node identity and join again) to get a current certificate")


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
    digest: str = ""       # blake2b-16 of the frame (checked against the reply's metadata)


def check_fetched(agent_id: str, frame: bytes, meta: dict, source: str) -> FetchedLatest:
    """Hostile-input checks of a fetched latest frame (spec 3.4, 4.6 "checked on arrival"): the digest of the bytes,
    the frame header's agent, kind, version, generation and ``teacher_active`` against the reply's metadata."""
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
            or header.get("policy_version") != version or header.get("generation") != generation
            or bool(header.get("teacher_active")) != teacher_active):
        raise FetchError("invalid", f"frame of {header.get('agent_id')!r} v{header.get('policy_version')} "
                                    f"(generation {header.get('generation')}, teacher_active "
                                    f"{header.get('teacher_active')}) for {agent_id!r} v{version} (generation "
                                    f"{generation}, teacher_active {teacher_active})")
    return FetchedLatest(agent_id, version, generation, teacher_active, frame, source, digest)


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
