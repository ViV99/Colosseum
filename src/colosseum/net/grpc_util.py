"""gRPC helpers of the SP5 wire layer: channel / server options, auth metadata and the server-side auth interceptor
(ruling R1), and blobs in parts (T2.3; spec 4.1 "Большие данные": ``PART_BYTES``, ``send_parts``, ``receive_parts``).

Which RPC needs what (:class:`AuthInterceptor`; refusals are UNAUTHENTICATED / PERMISSION_DENIED with a hint):
- open: ``Bootstrap/Info`` and ``Bootstrap/Script`` (the installer: version check and script, no secrets);
- the token secret (``x-colosseum-token``): ``Bootstrap/Bundle`` (the project code), ``Hub/Register``,
  ``Hub/Admin``, ``Hub/AdminStream``;
- a node id + its node secret (``x-colosseum-node``, ``x-colosseum-node-secret``), not banned: every other ``Hub``
  method (deny by default);
- ``NodeData`` (a node's data port, mutual TLS, :class:`DataPortInterceptor`): the client certificate's CN must be a
  node id the port accepts (the node agent's current set of the run's nodes, from the hub; T3.2).
Services outside ``colosseum.v1`` are passed through untouched; ``NodeData`` on the hub port is refused.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import uuid
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import grpc

from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.security import MD_NODE_ID, MD_NODE_SECRET, MD_TOKEN, NodeBook, Token

KEEPALIVE_MS = 10_000                    # spec 4.1: a dead peer is noticed within ~10 s
KEEPALIVE_TIMEOUT_MS = 5_000
SERVICE_PREFIX = "/colosseum.v1."
OPEN_METHODS = frozenset({"/colosseum.v1.Bootstrap/Info", "/colosseum.v1.Bootstrap/Script"})
TOKEN_METHODS = frozenset({"/colosseum.v1.Bootstrap/Bundle", "/colosseum.v1.Hub/Register", "/colosseum.v1.Hub/Admin",
                           "/colosseum.v1.Hub/AdminStream"})
DATA_SERVICE = "/colosseum.v1.NodeData/"


def _sizes(max_message_mb: int) -> list[tuple[str, Any]]:
    limit = int(max_message_mb) * 2**20
    return [("grpc.max_send_message_length", limit), ("grpc.max_receive_message_length", limit)]


def channel_options(max_message_mb: int, *, server_name: str | None = None) -> list[tuple[str, Any]]:
    """Client options: message sizes, keepalive pings every 10 s (also without calls), the fixed TLS server name."""
    options = _sizes(max_message_mb) + [
        ("grpc.keepalive_time_ms", KEEPALIVE_MS), ("grpc.keepalive_timeout_ms", KEEPALIVE_TIMEOUT_MS),
        ("grpc.keepalive_permit_without_calls", 1), ("grpc.http2.max_pings_without_data", 0),
    ]
    if server_name is not None:
        options.append(("grpc.ssl_target_name_override", server_name))
    return options


def server_options(max_message_mb: int) -> list[tuple[str, Any]]:
    """Server options: message sizes, clients may ping every 10 s without calls, and no ``SO_REUSEPORT`` (grpc's
    default would let a second server bind a taken port and split the connections between the two; A8)."""
    return _sizes(max_message_mb) + [
        ("grpc.keepalive_time_ms", KEEPALIVE_MS), ("grpc.keepalive_timeout_ms", KEEPALIVE_TIMEOUT_MS),
        ("grpc.keepalive_permit_without_calls", 1), ("grpc.http2.min_ping_interval_without_data_ms", 5_000),
        ("grpc.http2.max_ping_strikes", 0), ("grpc.so_reuseport", 0),
    ]


def token_metadata(token: Token) -> tuple[tuple[str, str], ...]:
    return ((MD_TOKEN, token.secret),)


def node_metadata(node_id: str, node_secret: str) -> tuple[tuple[str, str], ...]:
    return ((MD_NODE_ID, node_id), (MD_NODE_SECRET, node_secret))


def metadata_value(metadata: Any, key: str) -> str | None:
    """The value of ``key`` in invocation metadata (a sequence of (key, value) pairs), or None."""
    for item in metadata or ():
        k, v = (item.key, item.value) if hasattr(item, "key") else item
        if k == key:
            return v if isinstance(v, str) else None
    return None


def node_id_of(context: grpc.ServicerContext) -> str | None:
    """The node id an authenticated hub RPC was made with (the interceptor checked it)."""
    return metadata_value(context.invocation_metadata(), MD_NODE_ID)


def peer_common_name(context: grpc.ServicerContext) -> str | None:
    """The CN of the client certificate of a mutual-TLS call, or None."""
    names = context.auth_context().get("x509_common_name") or []
    return names[0].decode("utf-8", "replace") if names else None


def _refusing(handler: grpc.RpcMethodHandler, code: grpc.StatusCode, message: str) -> grpc.RpcMethodHandler:
    """A handler of the same kind as ``handler`` that aborts every call with ``code``."""

    def abort(request_or_iterator: Any, context: grpc.ServicerContext) -> Any:
        context.abort(code, message)

    def abort_stream(request_or_iterator: Any, context: grpc.ServicerContext) -> Any:
        context.abort(code, message)
        yield  # pragma: no cover - abort raises

    kwargs = {"request_deserializer": handler.request_deserializer, "response_serializer": handler.response_serializer}
    if handler.request_streaming and handler.response_streaming:
        return grpc.stream_stream_rpc_method_handler(abort_stream, **kwargs)
    if handler.request_streaming:
        return grpc.stream_unary_rpc_method_handler(abort, **kwargs)
    if handler.response_streaming:
        return grpc.unary_stream_rpc_method_handler(abort_stream, **kwargs)
    return grpc.unary_unary_rpc_method_handler(abort, **kwargs)


def _checking_peer(handler: grpc.RpcMethodHandler, allow: Callable[[grpc.ServicerContext], str | None]
                   ) -> grpc.RpcMethodHandler:
    """``handler`` with a check of the caller's TLS identity first (``allow`` returns a refusal message or None)."""
    kwargs = {"request_deserializer": handler.request_deserializer, "response_serializer": handler.response_serializer}

    def guard(context: grpc.ServicerContext) -> None:
        refusal = allow(context)
        if refusal is not None:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, refusal)

    if handler.request_streaming and handler.response_streaming:
        def ss(it, ctx):
            guard(ctx)
            yield from handler.stream_stream(it, ctx)
        return grpc.stream_stream_rpc_method_handler(ss, **kwargs)
    if handler.request_streaming:
        def su(it, ctx):
            guard(ctx)
            return handler.stream_unary(it, ctx)
        return grpc.stream_unary_rpc_method_handler(su, **kwargs)
    if handler.response_streaming:
        def us(req, ctx):
            guard(ctx)
            yield from handler.unary_stream(req, ctx)
        return grpc.unary_stream_rpc_method_handler(us, **kwargs)

    def uu(req, ctx):
        guard(ctx)
        return handler.unary_unary(req, ctx)
    return grpc.unary_unary_rpc_method_handler(uu, **kwargs)


class AuthInterceptor(grpc.ServerInterceptor):
    """R1 authentication of every ``colosseum.v1`` RPC (module docstring). ``token_secret`` is read on every call
    (a rotated secret applies at once); ``nodes`` is the hub identity's node book."""

    def __init__(self, token_secret: Callable[[], str], nodes: NodeBook) -> None:
        self._token_secret = token_secret
        self._nodes = nodes

    def intercept_service(self, continuation: Callable, handler_call_details: grpc.HandlerCallDetails) -> Any:
        method = handler_call_details.method
        handler = continuation(handler_call_details)
        if handler is None or not method.startswith(SERVICE_PREFIX) or method in OPEN_METHODS:
            return handler
        metadata = handler_call_details.invocation_metadata
        if method.startswith(DATA_SERVICE):
            return _refusing(handler, grpc.StatusCode.PERMISSION_DENIED, "NodeData is served on node data ports only")
        if method in TOKEN_METHODS:
            secret = metadata_value(metadata, MD_TOKEN)
            if secret is None or not hmac.compare_digest(secret.encode(), self._token_secret().encode()):
                return _refusing(handler, grpc.StatusCode.UNAUTHENTICATED,
                                 "a valid token is required (x-colosseum-token); print it with `colosseum hub token` "
                                 "on the hub machine")
            return handler
        node_id, node_secret = metadata_value(metadata, MD_NODE_ID), metadata_value(metadata, MD_NODE_SECRET)
        if node_id is None or node_secret is None:
            return _refusing(handler, grpc.StatusCode.UNAUTHENTICATED,
                             "this call needs a registered node (x-colosseum-node / x-colosseum-node-secret)")
        _, status = self._nodes.authenticate(node_id, node_secret)   # the ban only after a valid secret
        if status == "banned":
            return _refusing(handler, grpc.StatusCode.PERMISSION_DENIED, "this node (or its machine) is banned on "
                             "this hub")
        if status != "ok":
            return _refusing(handler, grpc.StatusCode.UNAUTHENTICATED,
                             "unknown node or wrong node secret: register again with the token (the hub's identity "
                             "may have been rotated)")
        return handler


class DataPortInterceptor(grpc.ServerInterceptor):
    """A node data port (mutual TLS): ``NodeData`` calls only from a client certificate whose CN ``allowed(cn)`` accepts
    (a registered, not banned node of the run); every other ``colosseum.v1`` method is refused."""

    def __init__(self, allowed: Callable[[str], bool]) -> None:
        self._allowed = allowed

    def intercept_service(self, continuation: Callable, handler_call_details: grpc.HandlerCallDetails) -> Any:
        method = handler_call_details.method
        handler = continuation(handler_call_details)
        if handler is None or not method.startswith(SERVICE_PREFIX):
            return handler
        if not method.startswith(DATA_SERVICE):
            return _refusing(handler, grpc.StatusCode.PERMISSION_DENIED, "a node data port serves NodeData only")
        return _checking_peer(handler, self._refusal)

    def _refusal(self, context: grpc.ServicerContext) -> str | None:
        name = peer_common_name(context)
        if name is None:
            return "a node certificate of this hub is required"
        if not self._allowed(name):
            return f"node {name[:12]} is not a node of this run (removed or banned)"
        return None


# ---- blobs in parts (T2.3; spec 4.1 "Большие данные") -----------------------------------------------------------

PART_BYTES = 1 << 20                     # blob part size
BLOB_DIGEST_SIZE = 16                    # blake2b digest bytes (the shm cache's digest, T1.5)


class BlobError(ValueError):
    """A blob stream that is not a whole, consistent blob (``reason``: ``too_large``, ``offset``, ``size``,
    ``digest``, ``empty``)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def blob_digest(data: bytes | memoryview) -> str:
    """blake2b-128 hex of a whole blob (the digest of ``BlobPart`` streams and uploads)."""
    return hashlib.blake2b(data, digest_size=BLOB_DIGEST_SIZE).hexdigest()


def _parts(read: Callable[[], bytes], total: int, digest: str, meta_json: str) -> Iterator[pb.BlobPart]:
    offset = 0
    while True:
        block = read()
        last = offset + len(block) >= total or not block
        yield pb.BlobPart(data=block, offset=offset, total_size=total, last=last,
                          digest=digest if offset == 0 else "", meta_json=meta_json if offset == 0 else "")
        offset += len(block)
        if last:
            return


def send_parts(data: bytes | memoryview | Path, *, meta_json: str = "", digest: str | None = None
               ) -> Iterator[pb.BlobPart]:
    """``data`` (bytes, or a file read part by part) as ``BlobPart``s of at most ``PART_BYTES``: the first part
    carries ``total_size``, ``digest`` (blake2b-128 of the whole blob; computed unless given) and ``meta_json``; the
    last has ``last``; an empty blob is one part. A file is opened once (its digest pass and the parts read the same
    open file; blob files are replaced atomically, never rewritten in place)."""
    if isinstance(data, Path):
        with open(data, "rb") as f:
            total = os.fstat(f.fileno()).st_size
            if digest is None:
                h = hashlib.blake2b(digest_size=BLOB_DIGEST_SIZE)
                for block in iter(lambda: f.read(PART_BYTES), b""):
                    h.update(block)
                digest = h.hexdigest()
                f.seek(0)
            yield from _parts(lambda: f.read(PART_BYTES), total, digest, meta_json)
        return
    view = memoryview(data).cast("B")
    digest = blob_digest(view) if digest is None else digest
    offset = 0

    def read() -> bytes:
        nonlocal offset
        block = bytes(view[offset:offset + PART_BYTES])
        offset += len(block)
        return block

    yield from _parts(read, len(view), digest, meta_json)


def not_modified_part(meta_json: str = "") -> pb.BlobPart:
    """The single part of a ``GetLatest`` answer whose version is not newer (T3.1)."""
    return pb.BlobPart(not_modified=True, last=True, meta_json=meta_json)


def receive_parts(parts: Iterable[pb.BlobPart], *, cap: int, into: Path | None = None
                  ) -> tuple[bytes | Path | None, str]:
    """The blob of a part stream and its first part's ``meta_json``; ``None`` for a ``not_modified`` answer.
    Hostile input: ``total_size`` above ``cap`` is refused at the first part (nothing allocated), parts must be
    contiguous and end exactly at ``total_size``, the digest must match; ``BlobError`` otherwise. ``into``: write to
    that file (atomically: a uniquely named temp file next to it, renamed at the end; removed on any error) and
    return its path. Errors of the part iterator itself (a cut gRPC stream) propagate as they are."""
    it = iter(parts)
    first = next(it, None)
    if first is None:
        raise BlobError("empty", "blob stream without parts")
    if first.not_modified:
        return None, first.meta_json
    total, digest, meta = int(first.total_size), first.digest, first.meta_json
    if total < 0 or total > cap:
        raise BlobError("too_large", f"blob of {total} bytes exceeds the cap of {cap} bytes")
    h = hashlib.blake2b(digest_size=BLOB_DIGEST_SIZE)
    tmp = None if into is None else into.with_name(f".{into.name}.part-{uuid.uuid4().hex}")
    sink = io.BytesIO() if tmp is None else open(tmp, "xb")
    try:
        offset, part, done = 0, first, False
        while True:
            if part.offset != offset or offset + len(part.data) > total:
                raise BlobError("offset", f"part at offset {part.offset} (expected {offset}) of a {total}-byte blob")
            h.update(part.data)
            sink.write(part.data)
            offset += len(part.data)
            if part.last:
                done = True
                break
            part = next(it, None)
            if part is None:
                break
        if not done or offset != total:
            raise BlobError("size", f"blob ended at {offset} of {total} bytes")
        if h.hexdigest() != digest:
            raise BlobError("digest", "blob digest mismatch")
        if tmp is None:
            return sink.getvalue(), meta
        sink.close()
        os.replace(tmp, into)
        return into, meta
    except BaseException:
        sink.close()
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        raise
