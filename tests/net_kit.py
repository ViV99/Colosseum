"""Test support for the SP5 wire layer (T1.3): hub / node identities under ``tmp_path``, registration without a hub
process, loopback gRPC servers on port 0 with echo handlers of the ``colosseum.v1`` method paths, and channels.

The servers bind ``127.0.0.1:0`` themselves (no bind-then-release race); a test stops them in a ``finally`` (or
through the ``servers`` fixture of its module)."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import grpc

from colosseum.net.grpc_util import channel_options, server_options
from colosseum.net.security import HUB_SERVER_NAME, HubIdentity, NodeIdentity, NodeRecord

MAX_MB = 4


def hub_identity(tmp_path: Path, name: str = "hub-identity") -> HubIdentity:
    return HubIdentity.load_or_create(Path(tmp_path) / name)


def node_identity(tmp_path: Path, ca_fingerprint: str, name: str = "node") -> NodeIdentity:
    return NodeIdentity.load_or_create(ca_fingerprint, root=Path(tmp_path) / f"{name}-identity")


def register(hub: HubIdentity, node: NodeIdentity, *, name: str = "laptop", machine_id: str = "machine-1",
             tags: Sequence[str] = ()) -> NodeRecord:
    """What the hub's Register handler does (T2.3), without a server: book the node, issue its certificate, and let
    the node keep the reply."""
    record, secret = hub.nodes.register(machine_id=machine_id, name=name, tags=tags, node_id=node.node_id)
    cert = hub.issue_node_certificate(record.node_id, node.public_key_pem())
    reply = SimpleNamespace(node_id=record.node_id, number=record.number, name=record.name,
                            certificate_pem=cert.decode("ascii"), node_secret=secret, ca_pem=hub.ca_pem.decode("ascii"))
    node.save_registration(reply)
    return hub.nodes.get(record.node_id)


def create_hub_identity_child(path: str, barrier, results) -> None:
    """A spawned child's body for the concurrent-creation test: wait for the sibling, then load or create."""
    barrier.wait(timeout=60)
    try:
        identity = HubIdentity.load_or_create(Path(path))
        results.put(("ok", identity.fingerprint, identity.token.secret))
    except Exception as e:  # noqa: BLE001 - reported to the test
        results.put(("error", f"{type(e).__name__}: {e}", ""))


def _echo(request: bytes, context) -> bytes:
    return b"ok:" + request


def _echo_stream(requests: Iterable[bytes], context) -> Iterable[bytes]:
    for request in requests:
        yield b"ok:" + request


def _count(requests: Iterable[bytes], context) -> bytes:
    return str(sum(1 for _ in requests)).encode()


def _two(request: bytes, context) -> Iterable[bytes]:
    yield b"part1"
    yield b"part2"


def echo_handlers() -> list[grpc.GenericRpcHandler]:
    """Raw-bytes handlers on the contract's method paths (one of each streaming kind)."""
    def h(kind, fn):
        return getattr(grpc, f"{kind}_rpc_method_handler")(fn)

    return [
        grpc.method_handlers_generic_handler("colosseum.v1.Bootstrap", {
            "Info": h("unary_unary", _echo), "Script": h("unary_unary", _echo), "Bundle": h("unary_stream", _two)}),
        grpc.method_handlers_generic_handler("colosseum.v1.Hub", {
            "Register": h("unary_unary", _echo), "Admin": h("unary_unary", _echo),
            "AdminStream": h("unary_stream", _two), "Session": h("stream_stream", _echo_stream),
            "Upload": h("stream_unary", _count), "Download": h("unary_stream", _two),
            "RelaySend": h("stream_unary", _count), "RelayInbox": h("stream_stream", _echo_stream)}),
        grpc.method_handlers_generic_handler("colosseum.v1.NodeData", {
            "Chunks": h("stream_unary", _count), "GetLatest": h("unary_stream", _two)}),
    ]


def loopback_server(credentials: grpc.ServerCredentials | None = None, *,
                    interceptors: Sequence[grpc.ServerInterceptor] = (),
                    handlers: Sequence[grpc.GenericRpcHandler] | None = None,
                    port: int = 0) -> tuple[grpc.Server, int]:
    """A started server on ``127.0.0.1:<port>`` (default 0: a free port; plain without ``credentials``); returns
    ``(server, port)``; the port is 0 when the bind failed (the server is then stopped)."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), interceptors=list(interceptors),
                         options=server_options(MAX_MB))
    server.add_generic_rpc_handlers(list(handlers if handlers is not None else echo_handlers()))
    address = f"127.0.0.1:{port}"
    try:
        if credentials is None:
            bound = server.add_insecure_port(address)
        else:
            bound = server.add_secure_port(address, credentials)
    except RuntimeError:                      # grpcio raises for a port it cannot bind
        bound = 0
    if bound == 0:
        server.stop(None)
        return server, 0
    server.start()
    return server, bound


def channel(port: int, credentials: grpc.ChannelCredentials | None = None,
            server_name: str | None = HUB_SERVER_NAME) -> grpc.Channel:
    target = f"127.0.0.1:{port}"
    if credentials is None:
        return grpc.insecure_channel(target, options=channel_options(MAX_MB))
    return grpc.secure_channel(target, credentials, options=channel_options(MAX_MB, server_name=server_name))


def call(ch: grpc.Channel, method: str, payload: bytes = b"x", *, metadata=(), timeout: float = 5.0) -> bytes:
    """One unary call of a raw-bytes method (``/colosseum.v1.Hub/Register`` style)."""
    return ch.unary_unary(method)(payload, metadata=tuple(metadata), timeout=timeout)


def call_stream(ch: grpc.Channel, method: str, payloads: Sequence[bytes] = (b"a", b"b"), *, metadata=(),
                timeout: float = 5.0) -> list[bytes]:
    return list(ch.stream_stream(method)(iter(payloads), metadata=tuple(metadata), timeout=timeout))
