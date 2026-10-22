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
