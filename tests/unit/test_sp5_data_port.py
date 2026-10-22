"""The node data port (SP5 T3.1, spec 3.1-3.3, 4.5): GetLatest serves the cached frame over mutual TLS, answers
"not modified", refuses foreign, missing and banned certificates, needs the token with --insecure; the port moves
to the next free one; announced addresses; probe and fetch checks."""
from __future__ import annotations

import grpc
import numpy as np
import pytest

from colosseum.net.chunks import WeightsMeta, encode_weights
from colosseum.net.grpc_util import channel_options
from colosseum.net.security import MD_TOKEN, NODE_SERVER_NAME, data_channel_credentials, data_server_credentials
from colosseum.node.dataport import (
    DataServer,
    FetchError,
    check_fetched,
    data_addresses,
    fetch_latest,
    frame_digest,
    probe_direct,
)
from colosseum.node.weights import WeightCacheReader, WeightCacheWriter
from net_kit import hub_identity, node_identity, register

MAX_MB = 64


def _frame(version, *, teacher_active=False, agent="main"):
    state = {"w": np.full((750_000,), float(version), np.float32)}          # 3 MB: several parts
    return encode_weights(state, WeightsMeta(agent_id=agent, kind="latest", generation=1, policy_version=version,
                                             teacher_active=teacher_active))


def _cache(tmp_path, versions=(3,), teacher_active=False):
    writer = WeightCacheWriter(tmp_path / "shm", "P")
    for v in versions:
        writer.publish_latest("main", _frame(v, teacher_active=teacher_active), version=v, generation=1,
                              teacher_active=teacher_active)
    return WeightCacheReader(tmp_path / "shm", "P")


def _registered(tmp_path, hub, name):
    """A node identity of ``hub`` with its registration (PF3-11: ``net_kit.node_identity`` + ``register``)."""
    node = node_identity(tmp_path, hub.fingerprint, name)
    register(hub, node, name=name, machine_id=f"m-{name}")
    return node


@pytest.fixture
def pki(tmp_path):
    hub = hub_identity(tmp_path)
    other = hub_identity(tmp_path / "other")                  # another CA
    return {"hub": hub, "other": other, "a": _registered(tmp_path, hub, "a"), "b": _registered(tmp_path, hub, "b"),
            "x": _registered(tmp_path / "other", other, "x")}


def _server(tmp_path, pki, *, banned=(), cache=None, port=0):
    srv = DataServer(cache=cache or _cache(tmp_path), own_agents=lambda: {"main"}, max_message_mb=MAX_MB,
                     insecure=False, token_secret=pki["hub"].token.secret, banned=lambda: set(banned), node=1)
    srv.start(port, credentials=data_server_credentials(pki["a"], pki["hub"].ca_pem), host="127.0.0.1")
    return srv


def _channel(srv, node, ca_pem):
    """A data-port channel with ``node``'s certificate, trusting ``ca_pem`` (the name is always colosseum-node)."""
    creds = data_channel_credentials(node, ca_pem)
    return grpc.secure_channel(f"127.0.0.1:{srv.port}", creds, channel_options(MAX_MB, server_name=NODE_SERVER_NAME))


def test_get_latest_streams_the_cached_frame_and_answers_not_modified(tmp_path, pki):
    srv = _server(tmp_path, pki, cache=_cache(tmp_path, versions=(3, 4), teacher_active=True))
    try:
        ch = _channel(srv, pki["b"], pki["hub"].ca_pem)
        got = fetch_latest(ch, "main", -1, max_message_mb=MAX_MB)
        assert (got.agent_id, got.version, got.generation, got.teacher_active, got.source) == (
            "main", 4, 1, True, "direct")
        assert got.frame == _frame(4, teacher_active=True)
        assert fetch_latest(ch, "main", 4, max_message_mb=MAX_MB) is None
        assert srv.counters["latest_served"] == 1 and srv.counters["latest_not_modified"] == 1
        with pytest.raises(FetchError, match="NOT_FOUND"):
            fetch_latest(ch, "other", -1, max_message_mb=MAX_MB)
        ch.close()
    finally:
        srv.stop()


def test_peers_without_a_certificate_of_the_ca_or_banned_are_refused(tmp_path, pki):
    srv = _server(tmp_path, pki, banned={pki["b"].node_id})
    try:
        with pytest.raises(FetchError, match="PERMISSION_DENIED"):
            fetch_latest(_channel(srv, pki["b"], pki["hub"].ca_pem), "main", -1, max_message_mb=MAX_MB, timeout=10)
        for ch in (_channel(srv, pki["x"], pki["other"].ca_pem),           # a certificate of another CA
                   _channel(srv, pki["x"], pki["hub"].ca_pem)):
            with pytest.raises(FetchError, match="UNAVAILABLE"):           # the handshake fails
                fetch_latest(ch, "main", -1, max_message_mb=MAX_MB, timeout=3)
        bare = grpc.secure_channel(f"127.0.0.1:{srv.port}",
                                   grpc.ssl_channel_credentials(root_certificates=pki["hub"].ca_pem),
                                   channel_options(MAX_MB, server_name=NODE_SERVER_NAME))
        with pytest.raises(FetchError, match="UNAVAILABLE"):               # no client certificate
            fetch_latest(bare, "main", -1, max_message_mb=MAX_MB, timeout=3)
        assert srv.counters["peer_refused"] >= 1 and srv.counters["latest_served"] == 0
    finally:
        srv.stop()


def test_insecure_data_port_needs_the_token_secret(tmp_path):
    srv = DataServer(cache=_cache(tmp_path), own_agents=lambda: {"main"}, max_message_mb=MAX_MB, insecure=True,
                     token_secret="sekret", node=1)
    srv.start(0, credentials=None, host="127.0.0.1")
    try:
        ch = grpc.insecure_channel(f"127.0.0.1:{srv.port}", channel_options(MAX_MB))
        with pytest.raises(FetchError, match="UNAUTHENTICATED"):
            fetch_latest(ch, "main", -1, max_message_mb=MAX_MB)
        with pytest.raises(FetchError, match="UNAUTHENTICATED"):
            fetch_latest(ch, "main", -1, max_message_mb=MAX_MB, metadata=[(MD_TOKEN, "wrong")])
        assert fetch_latest(ch, "main", -1, max_message_mb=MAX_MB, metadata=[(MD_TOKEN, "sekret")]).version == 3
        assert srv.counters["peer_refused"] == 2
        ch.close()
    finally:
        srv.stop()


def test_a_taken_data_port_moves_to_the_next_free_one(tmp_path, pki):
    first = _server(tmp_path, pki)
    second = None
    try:
        second = _server(tmp_path / "2", pki, port=first.port)
        assert second.port > first.port              # SO_REUSEPORT off: the taken port is not shared (PF3-21: > P)
    finally:
        first.stop()
        if second is not None:
            second.stop()


def test_probe_picks_a_live_address_within_the_timeout(tmp_path, pki):
    srv = _server(tmp_path, pki)

    def make_channel(address):
        creds = data_channel_credentials(pki["b"], pki["hub"].ca_pem)
        return grpc.secure_channel(address, creds, channel_options(MAX_MB, server_name=NODE_SERVER_NAME))

    try:
        live = f"127.0.0.1:{srv.port}"
        address, ch = probe_direct(["127.0.0.1:1", live], make_channel, timeout=3.0)
        assert address == live
        assert fetch_latest(ch, "main", -1, max_message_mb=MAX_MB).version == 3
        ch.close()
        assert probe_direct(["127.0.0.1:1"], make_channel, timeout=0.5) is None
        assert probe_direct([], make_channel, timeout=0.5) is None
    finally:
        srv.stop()


def test_data_addresses_start_with_loopback_for_a_loopback_hub():
    assert data_addresses(47102, "127.0.0.1")[0] == "127.0.0.1:47102"
    remote = data_addresses(47102, "10.255.255.1")
    assert not any(a.startswith("127.") for a in remote)
    assert all(a.endswith(":47102") for a in data_addresses(47102, "127.0.0.1") + remote)


def test_a_fetched_frame_is_checked_against_its_metadata():
    f = _frame(5)
    meta = {"version": 5, "generation": 1, "digest": frame_digest(f)}
    assert check_fetched("main", f, meta, "direct").version == 5
    flipped = bytearray(f)
    flipped[-1] ^= 0xFF                                     # a byte of the data region
    with pytest.raises(FetchError) as e:
        check_fetched("main", bytes(flipped), meta, "direct")
    assert e.value.reason == "digest"
    for bad_meta, agent in (({**meta, "version": 6}, "main"), (meta, "other"),
                            ({k: v for k, v in meta.items() if k != "generation"}, "main")):
        with pytest.raises(FetchError) as e:
            check_fetched(agent, f, bad_meta, "direct")
        assert e.value.reason == "invalid"
    garbage = b"not a frame" * 10
    with pytest.raises(FetchError) as e:
        check_fetched("main", garbage, {**meta, "digest": frame_digest(garbage)}, "direct")
    assert e.value.reason == "invalid"


def test_a_tls_data_port_without_credentials_is_refused(tmp_path, pki):
    srv = DataServer(cache=_cache(tmp_path), own_agents=lambda: {"main"}, max_message_mb=MAX_MB, insecure=False,
                     token_secret="sekret", node=1)
    with pytest.raises(ValueError, match="TLS credentials"):
        srv.start(0, credentials=None, host="127.0.0.1")       # would serve plain gRPC without any peer check
    assert srv.port is None


@pytest.mark.parametrize("field, value", [("generation", 2), ("teacher_active", True)])
def test_a_frame_whose_header_disagrees_with_the_metadata_is_invalid(field, value):
    f = _frame(5)                                              # generation 1, teacher_active False
    meta = {"version": 5, "generation": 1, "digest": frame_digest(f), "teacher_active": False, field: value}
    with pytest.raises(FetchError) as e:
        check_fetched("main", f, meta, "direct")
    assert e.value.reason == "invalid"


def test_a_client_accepts_only_the_announced_nodes_certificate(tmp_path, pki):
    """Per-node data-port identity (SP5 T3.2; closes ruling R32's gap): the client's TLS server name names the source
    node the hub announced (``node_server_name``), so a data port presenting another node's certificate of the same
    CA (a banned node posing as the learner's port, say) is refused."""
    from colosseum.net.security import node_server_name
    from colosseum.node.fetch import NodeFetcher

    impostor = DataServer(cache=_cache(tmp_path), own_agents=lambda: {"main"}, max_message_mb=MAX_MB, insecure=False,
                          token_secret=pki["hub"].token.secret, node=pki["b"].number)
    impostor.start(0, credentials=data_server_credentials(pki["b"], pki["hub"].ca_pem), host="127.0.0.1")
    link = NodeFetcher.link(insecure=False, token_secret="", identity=pki["a"], max_message_mb=MAX_MB,
                            connect_timeout_sec=3.0)
    address = f"127.0.0.1:{impostor.port}"
    try:
        assert pki["a"].number != pki["b"].number
        assert node_server_name(pki["a"].number) == f"n{pki['a'].number}.colosseum-node"
        posing = link.make_channel(address, pki["a"].number)            # the hub announced node a at this address
        with pytest.raises(FetchError, match="UNAVAILABLE"):             # the handshake fails: b's certificate
            fetch_latest(posing, "main", -1, max_message_mb=MAX_MB, timeout=5)
        assert probe_direct([address], lambda a: link.make_channel(a, pki["a"].number), timeout=2.0) is None
        honest = link.make_channel(address, pki["b"].number)            # node b's own port
        assert fetch_latest(honest, "main", -1, max_message_mb=MAX_MB, timeout=10).version == 3
        posing.close()
        honest.close()
    finally:
        impostor.stop()


def _hosts_file_on_loopback(monkeypatch, interfaces):
    """Debian / Ubuntu / WSL2: the host name maps to 127.0.1.1; the routes give loopback sources (a loopback hub)."""
    import socket

    from colosseum.node import dataport

    monkeypatch.setattr(dataport, "_route_source", lambda dest: "127.0.0.1")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, 0, 0, "", ("127.0.1.1", 0))])
    monkeypatch.setattr(dataport, "_interface_addresses", lambda: list(interfaces))


def test_a_wildcard_data_port_announces_the_machines_lan_addresses_even_behind_a_loopback_hub(monkeypatch):
    """Review round 1 of T3.2(a): ``train --listen`` (0.0.0.0) has a loopback hub address; its node still announces
    the machine's non-loopback addresses (interfaces), never 127.0.1.1."""
    _hosts_file_on_loopback(monkeypatch, ["127.0.0.1", "10.1.2.3", "172.31.0.9"])
    assert data_addresses(47102, "127.0.0.1") == ["127.0.0.1:47102", "10.1.2.3:47102", "172.31.0.9:47102"]
    assert data_addresses(47102, "10.9.9.9") == ["10.1.2.3:47102", "172.31.0.9:47102"]


@pytest.mark.parametrize("host", ["0.0.0.0", "127.0.0.1"])
def test_open_data_port_announces_only_bound_addresses_and_warns_without_a_lan_address(monkeypatch, caplog, host,
                                                                                         tmp_path):
    import logging

    from colosseum.node.dataport import open_data_port

    _hosts_file_on_loopback(monkeypatch, [])                              # no address but loopback anywhere
    with caplog.at_level(logging.WARNING, logger="colosseum.node.dataport"):
        server, addresses = open_data_port(port=0, hub_host="127.0.0.1", insecure=True, token_secret="s",
                                           identity=None, cache=None, own_agents=set, banned=frozenset,
                                           max_message_mb=MAX_MB, node=0, host=host)
    try:
        assert addresses == (f"127.0.0.1:{server.port}",)
        warned = [r for r in caplog.records if "no non-loopback address" in r.getMessage()]
        if host == "0.0.0.0":
            assert len(warned) == 1 and "NodeOptions.data_addresses" in warned[0].getMessage()
            assert "--listen" in warned[0].getMessage()
        else:
            assert warned == []                                           # a specific host: exactly what is bound
    finally:
        server.stop()


def test_a_certificate_without_its_per_node_name_is_a_warning_with_the_remedy(tmp_path, pki, caplog):
    """Review round 1 of T3.2(a): a node registered before T3.2 lacks ``n<number>.colosseum-node``; its data port
    starts but every peer refuses it, so the node says so at start."""
    import logging

    from colosseum.net.security import MemoryNodeIdentity, mint_node_credentials
    from colosseum.node.dataport import open_data_port

    hub = pki["hub"]
    current = mint_node_credentials(hub, pki["a"].node_id)                 # carries n<a>.colosseum-node
    old = MemoryNodeIdentity(current.node_id, current.key_pem,
                             hub.issue_node_certificate("0" * 32, _public_pem(current.key_pem)), hub.ca_pem)
    for identity, expect in ((current, 0), (old, 1)):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="colosseum.node.dataport"):
            server, _ = open_data_port(port=0, hub_host="127.0.0.1", insecure=False, token_secret="s",
                                       identity=identity, cache=None, own_agents=set, banned=frozenset,
                                       max_message_mb=MAX_MB, node=pki["a"].number, host="127.0.0.1")
        server.stop()
        warned = [r.getMessage() for r in caplog.records if "Re-register this node" in r.getMessage()]
        assert len(warned) == expect


def _public_pem(key_pem):
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_private_key(key_pem, password=None)
    return key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def test_in_memory_credentials_never_show_the_private_key_in_a_repr(tmp_path, pki):
    from colosseum.net.security import Token, mint_node_credentials
    from colosseum.node.agent import NodeOptions

    minted = mint_node_credentials(pki["hub"], pki["a"].node_id)
    options = NodeOptions(hub_address="127.0.0.1:1", token=Token("s" * 43, "0" * 64), local=True,
                          data_credentials=(minted.key_pem, minted.cert_pem, minted.ca_pem))
    assert b"PRIVATE KEY" in minted.key_pem
    assert "PRIVATE KEY" not in repr(minted) and "PRIVATE KEY" not in repr(options)
