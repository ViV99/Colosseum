"""Admin clients of a running hub (SP5 spec 3.3; amendment A5): find the hub, then call one of ``ADMIN_COMMANDS``.

``resolve_hub`` picks the hub as spec 3.3 says: ``--hub ADDR`` (with a token; on the hub machine the local hub
identity supplies it) > ``--run RUN`` (the address in ``runs/<name>/hub.json``, the token and CA of that hub's
identity) > the single running hub of this machine (``hub.registry.running_hubs``); several running hubs are a
``SecurityError`` that lists their run dirs. ``call`` / ``stream`` open one channel per call (TLS pinned to the
hub's CA with the fixed server name, or plain for ``train``'s loopback port and ``--insecure`` hubs) and send the
token secret as metadata; a gRPC failure propagates as ``grpc.RpcError`` (the CLI turns it into a message).
"""

from __future__ import annotations

import ipaddress
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc

from colosseum.hub.registry import HubInfo, read_hub_json, running_hubs
from colosseum.net.grpc_util import channel_options, token_metadata
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc
from colosseum.net.security import (
    HUB_SERVER_NAME,
    TOKEN_ENV,
    HubIdentity,
    SecurityError,
    Token,
    fetch_hub_ca,
    hub_channel_credentials,
    hub_identity_dir,
    resolve_token,
)

__all__ = ["ADMIN_COMMANDS", "CONNECT_TIMEOUT_SEC", "HubTarget", "call", "resolve_hub", "stream"]

ADMIN_COMMANDS = ("hub.status", "nodes.list", "nodes.remove", "tournament.submit", "tournament.status",
                  "tournament.cancel", "nodes.add", "nodes.update_code", "agents.add", "agents.freeze",
                  "agents.remove", "agents.list")
CONNECT_TIMEOUT_SEC = 10.0          # fetching the hub's certificate for --hub ADDR
ADMIN_MAX_MESSAGE_MB = 64           # admin replies are small JSON documents


@dataclass(frozen=True)
class HubTarget:
    """Where and how to call a hub: ``address`` (``host:port``), TLS (``ca_pem`` pins the hub's CA) or plain, and
    the token whose secret authenticates the admin call."""

    address: str
    tls: bool
    ca_pem: bytes | None
    token: Token


def _split(address: str) -> tuple[str, int]:
    host, sep, port = address.rpartition(":")
    if not sep or not port.isdigit():
        raise SecurityError(f"malformed hub address {address!r}: expected HOST:PORT")
    return host.strip("[]") or "127.0.0.1", int(port)


def _local_identity(path: Path) -> HubIdentity | None:
    """The hub identity at ``path`` if one exists there (never created here); SecurityError naming ``path`` when it
    cannot be read (e.g. another user's directory)."""
    try:
        return HubIdentity.load_or_create(path) if (path / "ca.crt").is_file() else None
    except OSError as e:
        raise SecurityError(f"cannot read the hub identity {path} ({type(e).__name__}: {e}); pass --token or "
                            f"--token-file") from None


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _local_token_for(address: str, host: str, *, insecure: bool) -> Token:
    """The token for ``--hub ADDR`` when none is given: the local hub identity's (spec 3.3: on the hub machine).
    With TLS the pinned certificate fetch proves ADDR is that hub. Without TLS nothing would, so the owner's token
    is sent in clear only to a loopback address or to an address of a running hub of this machine with this
    identity; otherwise the "no token" SecurityError."""
    identity = _local_identity(hub_identity_dir())
    if identity is None:
        return resolve_token(None, None)               # raises "no token"
    if insecure and not _plain_local_ok(address, identity):
        raise SecurityError(f"no token for the insecure hub at {address}: this machine's hub token is sent in clear "
                            f"only to a loopback address or to a hub running here; pass --token, --token-file or "
                            f"set {TOKEN_ENV}")
    return identity.token


def _plain_local_ok(address: str, identity: HubIdentity) -> bool:
    """The ``--hub`` rule for a plain address: the owner token goes in clear only to a loopback address or to an
    address of a live hub of this machine with this identity."""
    host, _port = _split(address)
    return _is_loopback(host) or any(info.ca_fingerprint == identity.fingerprint
                                     and address in (info.loopback, *info.addresses)
                                     for info in running_hubs(identity.path))


def _given_token(token: str | None, token_file: str | None) -> Token | None:
    """The token from ``--token`` / ``--token-file`` / ``$COLOSSEUM_TOKEN``, or None when none is given."""
    if token or token_file or os.environ.get(TOKEN_ENV):
        return resolve_token(token, token_file)
    return None


def _from_info(info: HubInfo, token: Token | None, where: str) -> HubTarget:
    """The target of a local hub described by ``info`` (``hub.json`` or a running-registry entry): its loopback
    address (else its first listen address); the token and CA of its identity (else ``token`` and a fetched CA)."""
    if info.loopback:
        address, tls = info.loopback, info.loopback_tls
    elif info.addresses:
        address, tls = info.addresses[0], info.tls
    else:
        raise SecurityError(f"{where}: the hub records no address")
    identity = _local_identity(Path(info.identity_dir))
    if identity is not None and identity.fingerprint == info.ca_fingerprint and (
            tls or _plain_local_ok(address, identity)):
        return HubTarget(address, tls, identity.ca_pem if tls else None, identity.token)
    if identity is not None and identity.fingerprint == info.ca_fingerprint and token is None:
        raise SecurityError(f"{where}: the hub at {address} runs without TLS and is neither on a loopback address nor "
                            f"a hub running here; this machine's hub token is not sent there in clear; pass --token, "
                            f"--token-file or set {TOKEN_ENV}")
    if token is None:
        why = (f"the hub's identity {info.identity_dir} is not on this machine" if identity is None else
               f"the identity {info.identity_dir} now has fingerprint {identity.fingerprint[:16]}..., the hub "
               f"recorded {info.ca_fingerprint[:16]}... (rotated since the hub started?)")
        raise SecurityError(f"{where}: {why}; pass --token or --token-file (print it with `colosseum hub token` on "
                            f"the hub machine)")
    if not tls:
        return HubTarget(address, False, None, token)
    host, port = _split(address)
    return HubTarget(address, True, fetch_hub_ca(host, port, fingerprint=token.ca_fingerprint,
                                                 timeout=CONNECT_TIMEOUT_SEC), token)


def resolve_hub(*, hub: str | None = None, run: str | None = None, token: str | None = None,
                token_file: str | None = None, insecure: bool = False) -> HubTarget:
    """The hub an admin command talks to (module docstring; spec 3.3). ``insecure``: the ``--hub ADDR`` hub runs
    without TLS (``hub.json`` says so itself for ``--run`` and the registry). SecurityError when no hub can be
    chosen or the token cannot be found; HubUnreachableError when ``--hub ADDR`` cannot be reached for its CA."""
    given = _given_token(token, token_file)
    if hub:
        host, port = _split(hub)
        if given is None:                              # on the hub machine: the local identity's token
            given = _local_token_for(hub, host, insecure=insecure)
        if insecure:
            return HubTarget(hub, False, None, given)
        return HubTarget(hub, True, fetch_hub_ca(host, port, fingerprint=given.ca_fingerprint,
                                                 timeout=CONNECT_TIMEOUT_SEC), given)
    if run:
        info = read_hub_json(Path(run))
        if info is None:
            raise SecurityError(f"{run} has no hub.json: it never ran a hub (or it is not a run directory)")
        return _from_info(info, given, run)
    hubs = running_hubs(hub_identity_dir())
    if not hubs:
        raise SecurityError("no running hub on this machine; pass --hub ADDR (with a token) or --run RUN")
    if len(hubs) > 1:
        listing = "; ".join(f"{h.run_dir} (pid {h.pid}, {h.mode})" for h in hubs)
        raise SecurityError(f"{len(hubs)} hubs run on this machine: {listing}; choose one with --run RUN or "
                            f"--hub ADDR")
    return _from_info(hubs[0], given, f"the running hub of {hubs[0].run_dir}")


def _channel(target: HubTarget) -> grpc.Channel:
    if not target.tls:
        return grpc.insecure_channel(target.address, options=channel_options(ADMIN_MAX_MESSAGE_MB))
    if target.ca_pem is None:
        raise SecurityError(f"no CA for the TLS hub at {target.address}")
    return grpc.secure_channel(target.address, hub_channel_credentials(target.ca_pem),
                               options=channel_options(ADMIN_MAX_MESSAGE_MB, server_name=HUB_SERVER_NAME))


def _request(command: str, args: Mapping[str, Any] | None) -> pb.AdminRequest:
    return pb.AdminRequest(command=command, args_json=json.dumps(dict(args or {})))


def _reply(reply: pb.AdminReply) -> dict:
    return {"code": int(reply.code), "text": reply.text, "data": json.loads(reply.data_json) if reply.data_json
            else None}


def call(target: HubTarget, command: str, args: Mapping[str, Any] | None = None, *, timeout: float = 30.0) -> dict:
    """Run one admin command; ``{"code", "text", "data"}`` (code 0 ok, 1 failed, 2 unknown command or bad
    arguments). ``grpc.RpcError`` when the call itself fails (unreachable, refused token, timeout)."""
    channel = _channel(target)
    try:
        reply = pb_grpc.HubStub(channel).Admin(_request(command, args), metadata=token_metadata(target.token),
                                               timeout=timeout)
        return _reply(reply)
    finally:
        channel.close()


def stream(target: HubTarget, command: str, args: Mapping[str, Any] | None = None) -> Iterator[dict]:
    """Run a streaming admin command: yields ``{"code", "text", "data", "done"}`` until a reply is ``done`` (or the
    hub ends the stream). ``grpc.RpcError`` when the stream fails."""
    channel = _channel(target)
    try:
        for reply in pb_grpc.HubStub(channel).AdminStream(_request(command, args),
                                                          metadata=token_metadata(target.token)):
            yield {**_reply(reply), "done": bool(reply.done)}
            if reply.done:
                return
    finally:
        channel.close()
