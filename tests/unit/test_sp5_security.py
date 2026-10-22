"""Security of the SP5 wire layer (T1.3, spec blocks 2.1 and 3.2-3.4, rulings R1 and R15; criterion 7 at the library
level): the token, the hub and node identities with private file modes, the node book (numbers, names, secret
hashes, ban, remove), rotation, node certificates, and every refusal on a real loopback gRPC server: no token, a
wrong token, a wrong fingerprint, another CA, plain against TLS, --insecure still needing the token, node secrets,
bans, and mutual TLS on a node data port."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import stat

import grpc
import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from colosseum.net.grpc_util import AuthInterceptor, DataPortInterceptor, node_metadata, token_metadata
from colosseum.net.security import (
    HUB_SERVER_NAME,
    NODE_SERVER_NAME,
    TOKEN_ENV,
    HubIdentity,
    HubUnreachableError,
    NodeIdentity,
    SecurityError,
    Token,
    cache_home,
    curl_pin,
    data_channel_credentials,
    data_home,
    data_server_credentials,
    fetch_hub_ca,
    hub_channel_credentials,
    hub_identity_dir,
    parse_token,
    resolve_token,
    spki_fingerprint,
)
from net_kit import call, call_stream, channel, hub_identity, loopback_server, node_identity, register


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def servers():
    started: list[grpc.Server] = []
    yield started
    for server in started:
        server.stop(None)


def _hub_server(hub: HubIdentity, servers, *, tls: bool = True) -> int:
    server, port = loopback_server(hub.server_credentials() if tls else None,
                                   interceptors=[AuthInterceptor(lambda: hub.token.secret, hub.nodes)])
    servers.append(server)
    return port


def _tls_channel(hub: HubIdentity, port: int) -> grpc.Channel:
    ca = fetch_hub_ca("127.0.0.1", port, fingerprint=hub.fingerprint, timeout=5.0)
    return channel(port, hub_channel_credentials(ca), HUB_SERVER_NAME)


def _code(fn, *args, **kwargs) -> grpc.StatusCode:
    with pytest.raises(grpc.RpcError) as info:
        fn(*args, **kwargs)
    return info.value.code()


# ---------------------------------------------------------------------------
# Directories, token, fingerprints
# ---------------------------------------------------------------------------


def test_identity_dirs_are_redirected_into_tmp_path(tmp_path):
    """tests/conftest.py's autouse fixture: no test touches the real ~/.local/share or ~/.cache."""
    assert data_home().is_relative_to(tmp_path) and cache_home().is_relative_to(tmp_path)
    assert hub_identity_dir() == data_home() / "colosseum" / "hub"
    assert os.environ["COLOSSEUM_SHM_DIR"].startswith(str(tmp_path))


def test_the_token_format_and_its_resolution(tmp_path):
    token = Token("A" * 43, "0f" * 32)
    assert str(token) == f"cst1.{'A' * 43}.{'0f' * 32}" and parse_token(str(token) + "\n") == token
    assert "A" * 43 not in repr(token)                                          # never logged by accident
    for bad in ("cst2." + "A" * 43 + "." + "0" * 64, "cst1." + "A" * 42 + "." + "0" * 64,
                "cst1." + "A" * 43 + "." + "0" * 63, "cst1." + "A" * 43 + "." + "G" * 64, ""):
        with pytest.raises(SecurityError, match="malformed token"):
            parse_token(bad)
    other = Token("B" * 43, "1a" * 32)
    path = tmp_path / "token.txt"
    path.write_text(f"{other}\n")
    env = {TOKEN_ENV: str(Token("C" * 43, "2b" * 32))}
    assert resolve_token(str(token), str(path), env=env) == token                # --token first
    assert resolve_token(None, str(path), env=env) == other                      # then --token-file
    assert resolve_token(None, None, env=env).secret == "C" * 43                 # then COLOSSEUM_TOKEN
    with pytest.raises(SecurityError, match="--token-file"):
        resolve_token(None, None, env={})
    with pytest.raises(SecurityError, match="cannot read --token-file"):
        resolve_token(None, str(tmp_path / "missing"), env={})


def test_the_hub_identity_is_created_once_with_private_modes(tmp_path):
    hub = hub_identity(tmp_path)
    assert _mode(hub.path) == 0o700
    assert _mode(hub.path / "ca.key") == 0o600 and _mode(hub.path / "secret") == 0o600
    again = HubIdentity.load_or_create(hub.path)
    assert again.fingerprint == hub.fingerprint and again.token == hub.token
    assert hub.fingerprint == spki_fingerprint((hub.path / "ca.crt").read_bytes()) and len(hub.fingerprint) == 64
    cert = x509.load_pem_x509_certificate(hub.ca_pem)
    assert cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    assert cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(
        x509.DNSName) == [HUB_SERVER_NAME]
    assert isinstance(cert.public_key(), ec.EllipticCurvePublicKey) and cert.public_key().curve.name == "secp256r1"
    (hub.path / "secret").unlink()
    with pytest.raises(SecurityError, match="incomplete"):
        HubIdentity.load_or_create(hub.path)


def test_the_curl_pin_is_the_base64_of_the_same_digest(tmp_path):
    hub = hub_identity(tmp_path)
    pin = curl_pin(hub.fingerprint)
    assert pin.startswith("sha256//") and base64.b64decode(pin.removeprefix("sha256//")).hex() == hub.fingerprint


# ---------------------------------------------------------------------------
# Node book, bans, rotation, node identities
# ---------------------------------------------------------------------------


def test_node_numbers_are_never_reused_names_are_unique_and_only_secret_hashes_are_kept(tmp_path):
    hub = hub_identity(tmp_path)
    a, secret_a = hub.nodes.register(machine_id="m1", name="laptop", tags=["cpu"], node_id=None)
    b, _ = hub.nodes.register(machine_id="m2", name="laptop", tags=[], node_id=None)
    assert (a.number, b.number) == (1, 2) and (a.name, b.name) == ("laptop", "laptop-2")
    assert len(a.node_id) == 32 and hub.nodes.verify(a.node_id, secret_a).name == "laptop"
    again, secret_again = hub.nodes.register(machine_id="m1", name="renamed", tags=["gpu"], node_id=a.node_id)
    assert (again.number, again.name, again.tags) == (1, "laptop", ["gpu"])     # a known id keeps number and name
    assert hub.nodes.verify(a.node_id, secret_a) is None and hub.nodes.verify(a.node_id, secret_again)   # new secret
    hub.nodes.remove(b.node_id)
    c, _ = hub.nodes.register(machine_id="m3", name="pc", tags=[], node_id=None)
    assert c.number == 3                                                         # 2 is not reused
    text = (hub.path / "nodes.json").read_text()
    assert secret_again not in text and hashlib.sha256(secret_again.encode()).hexdigest() in text
    assert _mode(hub.path / "nodes.json") == 0o600 and json.loads(text)["next_number"] == 4
    assert [r.number for r in HubIdentity.load_or_create(hub.path).nodes.records()] == [1, 3]
    with pytest.raises(SecurityError, match="invalid node name"):
        hub.nodes.register(machine_id="m4", name="bad name", tags=[], node_id=None)


def test_a_ban_refuses_the_node_id_and_the_machine_id(tmp_path):
    hub = hub_identity(tmp_path)
    a, secret = hub.nodes.register(machine_id="m1", name="laptop", tags=[], node_id=None)
    hub.nodes.ban(a.node_id)
    assert hub.nodes.verify(a.node_id, secret) is None and hub.nodes.is_banned(None, "m1")
    with pytest.raises(SecurityError, match="banned"):
        hub.nodes.register(machine_id="m1", name="laptop", tags=[], node_id=None)          # same machine, new id
    with pytest.raises(SecurityError, match="banned"):
        hub.nodes.register(machine_id="m9", name="other", tags=[], node_id=a.node_id)      # same id
    assert hub.nodes.register(machine_id="m2", name="pc", tags=[], node_id=None)[0].number == 2
    assert json.loads((hub.path / "banned.json").read_text())["banned"][0]["machine_id"] == "m1"


def test_rotate_makes_a_new_ca_and_secret_and_revokes_every_node(tmp_path):
    hub = hub_identity(tmp_path)
    book = hub.nodes
    old_fp, old_token = hub.fingerprint, hub.token
    a, secret = hub.nodes.register(machine_id="m1", name="laptop", tags=[], node_id=None)
    hub.nodes.ban(hub.nodes.register(machine_id="m2", name="pc", tags=[], node_id=None)[0].node_id)
    hub.rotate()
    assert hub.fingerprint != old_fp and hub.token.secret != old_token.secret and hub.nodes is book
    assert hub.nodes.records() == [] and hub.nodes.verify(a.node_id, secret) is None
    assert not hub.nodes.is_banned(None, "m2")                                  # the ban list starts over too
    assert spki_fingerprint((hub.path / "rotated" / "1" / "ca.crt").read_bytes()) == old_fp
    assert HubIdentity.load_or_create(hub.path).fingerprint == hub.fingerprint


def test_a_node_identity_keeps_its_key_and_registration(tmp_path):
    hub = hub_identity(tmp_path)
    node = node_identity(tmp_path, hub.fingerprint)
    assert _mode(node.path) == 0o700 and _mode(node.path / "node.key") == 0o600 and not node.registered
    record = register(hub, node, name="laptop")
    again = NodeIdentity.load_or_create(hub.fingerprint, root=tmp_path / "node-identity")
    assert (again.node_id, again.number, again.name, again.secret) == (record.node_id, 1, "laptop", node.secret)
    assert again.registered and _mode(again.path / "node.json") == 0o600
    assert again.public_key_pem() == node.public_key_pem()
    cert = x509.load_pem_x509_certificate(again.cert_pem)
    assert cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == record.node_id
    assert cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(
        x509.DNSName) == [NODE_SERVER_NAME, f"n1.{NODE_SERVER_NAME}"]      # + the per-node name (T3.2)
    cert.verify_directly_issued_by(x509.load_pem_x509_certificate(hub.ca_pem))
    assert hub.nodes.get(record.node_id).cert_serial == format(cert.serial_number, "x")
    again.forget()
    assert not NodeIdentity.load_or_create(hub.fingerprint, root=tmp_path / "node-identity").registered
    with pytest.raises(SecurityError, match="invalid node id"):
        hub.issue_node_certificate("x", node.public_key_pem())
    with pytest.raises(SecurityError, match="PEM"):
        hub.issue_node_certificate("0" * 32, b"not a key")


# ---------------------------------------------------------------------------
# The hub port on a real loopback server (criterion 7)
# ---------------------------------------------------------------------------


def test_the_tls_hub_port_accepts_the_pinned_ca_and_the_token(tmp_path, servers):
    hub = hub_identity(tmp_path)
    port = _hub_server(hub, servers)
    ch = _tls_channel(hub, port)
    assert call(ch, "/colosseum.v1.Bootstrap/Info") == b"ok:x"                    # open
    assert call(ch, "/colosseum.v1.Hub/Register", metadata=token_metadata(hub.token)) == b"ok:x"
    assert call(ch, "/colosseum.v1.Hub/Admin", metadata=token_metadata(hub.token)) == b"ok:x"
    assert list(ch.unary_stream("/colosseum.v1.Bootstrap/Bundle")(b"", metadata=token_metadata(hub.token),
                                                                    timeout=5)) == [b"part1", b"part2"]


def test_calls_without_the_token_or_with_a_wrong_one_are_refused(tmp_path, servers):
    hub = hub_identity(tmp_path)
    port = _hub_server(hub, servers)
    ch = _tls_channel(hub, port)
    wrong = Token("Z" * 43, hub.fingerprint)
    bundle = ch.unary_stream("/colosseum.v1.Bootstrap/Bundle")
    for metadata in ((), token_metadata(wrong)):
        for method in ("/colosseum.v1.Hub/Register", "/colosseum.v1.Hub/Admin"):
            assert _code(call, ch, method, metadata=metadata) == grpc.StatusCode.UNAUTHENTICATED
        assert _code(lambda md: list(bundle(b"", metadata=md, timeout=5)), metadata) == grpc.StatusCode.UNAUTHENTICATED
    with pytest.raises(grpc.RpcError) as info:
        call(ch, "/colosseum.v1.Hub/Register")
    assert "colosseum hub token" in info.value.details()


def test_a_wrong_fingerprint_is_refused_before_any_call(tmp_path, servers):
    hub, other = hub_identity(tmp_path), hub_identity(tmp_path, "other-hub")
    port = _hub_server(hub, servers)
    with pytest.raises(SecurityError, match="fingerprint"):
        fetch_hub_ca("127.0.0.1", port, fingerprint=other.fingerprint, timeout=5.0)


def test_an_unreachable_hub_is_told_apart_from_a_refusal(tmp_path, servers):
    import socket

    hub = hub_identity(tmp_path)
    with socket.socket() as closed:                                  # bound, not listening: the connect is refused
        closed.bind(("127.0.0.1", 0))
        with pytest.raises(HubUnreachableError, match="cannot reach the hub"):
            fetch_hub_ca("127.0.0.1", closed.getsockname()[1], fingerprint=hub.fingerprint, timeout=5.0)
    plain = _hub_server(hub, servers, tls=False)                    # a hub without TLS: no retry would help
    with pytest.raises(SecurityError) as info:
        fetch_hub_ca("127.0.0.1", plain, fingerprint=hub.fingerprint, timeout=5.0)
    assert not isinstance(info.value, HubUnreachableError) and "--insecure" in str(info.value)
    assert issubclass(HubUnreachableError, SecurityError)


def test_a_client_trusting_another_ca_or_no_tls_cannot_connect(tmp_path, servers):
    hub, other = hub_identity(tmp_path), hub_identity(tmp_path, "other-hub")
    port = _hub_server(hub, servers)
    foreign = channel(port, hub_channel_credentials(other.ca_pem), HUB_SERVER_NAME)
    assert _code(lambda: call(foreign, "/colosseum.v1.Bootstrap/Info", timeout=10)) == grpc.StatusCode.UNAVAILABLE
    plain = channel(port)                                                        # no TLS against the TLS port
    assert _code(lambda: call(plain, "/colosseum.v1.Bootstrap/Info", timeout=10)) == grpc.StatusCode.UNAVAILABLE
    ca = fetch_hub_ca("127.0.0.1", port, fingerprint=hub.fingerprint, timeout=5.0)
    unnamed = channel(port, hub_channel_credentials(ca), server_name=None)       # 127.0.0.1 is not colosseum-hub
    assert _code(lambda: call(unnamed, "/colosseum.v1.Bootstrap/Info", timeout=10)) == grpc.StatusCode.UNAVAILABLE


def test_insecure_mode_still_requires_the_token_and_the_node_secret(tmp_path, servers):
    hub = hub_identity(tmp_path)
    port = _hub_server(hub, servers, tls=False)
    ch = channel(port)
    assert _code(lambda: call(ch, "/colosseum.v1.Hub/Register")) == grpc.StatusCode.UNAUTHENTICATED
    assert call(ch, "/colosseum.v1.Hub/Register", metadata=token_metadata(hub.token)) == b"ok:x"
    assert _code(lambda: call_stream(ch, "/colosseum.v1.Hub/Session")) == grpc.StatusCode.UNAUTHENTICATED


def test_node_calls_need_a_registered_not_banned_node_and_its_secret(tmp_path, servers):
    hub = hub_identity(tmp_path)
    node = node_identity(tmp_path, hub.fingerprint)
    record = register(hub, node)
    port = _hub_server(hub, servers)
    ch = _tls_channel(hub, port)
    good = node_metadata(record.node_id, node.secret)
    assert call_stream(ch, "/colosseum.v1.Hub/Session", metadata=good) == [b"ok:a", b"ok:b"]
    assert list(ch.unary_stream("/colosseum.v1.Hub/Download")(b"", metadata=good, timeout=5)) == [b"part1", b"part2"]
    assert ch.stream_unary("/colosseum.v1.Hub/Upload")(iter([b"1", b"2", b"3"]), metadata=good, timeout=5) == b"3"
    for metadata in ((), token_metadata(hub.token), node_metadata(record.node_id, "wrong"),
                     node_metadata("f" * 32, node.secret)):
        assert _code(call_stream, ch, "/colosseum.v1.Hub/Session", metadata=metadata) == grpc.StatusCode.UNAUTHENTICATED
    hub.nodes.ban(record.node_id)
    assert _code(lambda: call_stream(ch, "/colosseum.v1.Hub/Session", metadata=good)) \
        == grpc.StatusCode.PERMISSION_DENIED
    # without the secret a banned id looks like an unknown one (the ban is checked only after a valid secret)
    assert _code(call_stream, ch, "/colosseum.v1.Hub/Session", metadata=node_metadata(record.node_id, "wrong")) \
        == grpc.StatusCode.UNAUTHENTICATED


def test_rotation_revokes_registered_nodes_and_the_old_token_at_once(tmp_path, servers):
    hub = hub_identity(tmp_path)
    node = node_identity(tmp_path, hub.fingerprint)
    record = register(hub, node)
    port = _hub_server(hub, servers, tls=False)                                 # the CA change needs a restart
    ch = channel(port)
    old_token = hub.token
    hub.rotate()
    assert _code(lambda: call_stream(ch, "/colosseum.v1.Hub/Session",
                                     metadata=node_metadata(record.node_id, node.secret))) \
        == grpc.StatusCode.UNAUTHENTICATED
    assert _code(lambda: call(ch, "/colosseum.v1.Hub/Register", metadata=token_metadata(old_token))) \
        == grpc.StatusCode.UNAUTHENTICATED
    assert call(ch, "/colosseum.v1.Hub/Register", metadata=token_metadata(hub.token)) == b"ok:x"


# ---------------------------------------------------------------------------
# A node data port (mutual TLS)
# ---------------------------------------------------------------------------


def _data_server(hub: HubIdentity, owner: NodeIdentity, servers) -> int:
    allowed = lambda cn: hub.nodes.get(cn) is not None and not hub.nodes.is_banned(cn, None)  # noqa: E731
    server, port = loopback_server(data_server_credentials(owner, hub.ca_pem),
                                   interceptors=[DataPortInterceptor(allowed)])
    servers.append(server)
    return port


def test_a_data_port_takes_only_node_certificates_of_the_run(tmp_path, servers):
    hub, other = hub_identity(tmp_path), hub_identity(tmp_path, "other-hub")
    a, b = node_identity(tmp_path, hub.fingerprint, "a"), node_identity(tmp_path, hub.fingerprint, "b")
    register(hub, a, name="a", machine_id="ma")
    record_b = register(hub, b, name="b", machine_id="mb")
    stranger = node_identity(tmp_path, other.fingerprint, "stranger")
    register(other, stranger, name="s", machine_id="ms")
    port = _data_server(hub, a, servers)
    good = channel(port, data_channel_credentials(b, hub.ca_pem), NODE_SERVER_NAME)
    assert good.stream_unary("/colosseum.v1.NodeData/Chunks")(iter([b"f1", b"f2"]), timeout=5) == b"2"
    assert list(good.unary_stream("/colosseum.v1.NodeData/GetLatest")(b"", timeout=5)) == [b"part1", b"part2"]
    no_cert = channel(port, hub_channel_credentials(hub.ca_pem), NODE_SERVER_NAME)
    assert _code(lambda: no_cert.stream_unary("/colosseum.v1.NodeData/Chunks")(iter([b"f"]), timeout=10)) \
        in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNKNOWN)
    foreign = channel(port, grpc.ssl_channel_credentials(root_certificates=hub.ca_pem,
                                                         private_key=stranger.key_pem,
                                                         certificate_chain=stranger.cert_pem), NODE_SERVER_NAME)
    assert _code(lambda: foreign.stream_unary("/colosseum.v1.NodeData/Chunks")(iter([b"f"]), timeout=10)) \
        in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNKNOWN)
    assert _code(lambda: call(good, "/colosseum.v1.Hub/Register", metadata=token_metadata(hub.token))) \
        == grpc.StatusCode.PERMISSION_DENIED                                    # a data port serves NodeData only
    hub.nodes.ban(record_b.node_id)
    assert _code(lambda: good.stream_unary("/colosseum.v1.NodeData/Chunks")(iter([b"f"]), timeout=5)) \
        == grpc.StatusCode.PERMISSION_DENIED


def test_node_data_is_never_served_on_the_hub_port(tmp_path, servers):
    hub = hub_identity(tmp_path)
    node = node_identity(tmp_path, hub.fingerprint)
    record = register(hub, node)
    ch = _tls_channel(hub, _hub_server(hub, servers))
    assert _code(lambda: ch.stream_unary("/colosseum.v1.NodeData/Chunks")(
        iter([b"f"]), metadata=node_metadata(record.node_id, node.secret), timeout=5)) \
        == grpc.StatusCode.PERMISSION_DENIED


# ---------------------------------------------------------------------------
# Amendment A8: the ephemeral local node, concurrent creation, no SO_REUSEPORT
# ---------------------------------------------------------------------------


def test_an_ephemeral_node_is_number_0_and_never_stored(tmp_path, servers):
    hub = hub_identity(tmp_path)
    persistent, _ = hub.nodes.register(machine_id="m1", name="laptop", tags=[], node_id=None)
    local, secret = hub.nodes.add_ephemeral("local", "m1")
    assert (local.number, local.name, local.persistent) == (0, "local", False) and len(local.node_id) == 32
    assert hub.nodes.verify(local.node_id, secret).number == 0 and hub.nodes.verify(local.node_id, "wrong") is None
    assert hub.nodes.get(local.node_id).number == 0 and not hub.nodes.is_banned(local.node_id, None)
    node = node_identity(tmp_path, hub.fingerprint)
    cert = x509.load_pem_x509_certificate(hub.issue_node_certificate(local.node_id, node.public_key_pem()))
    assert hub.nodes.get(local.node_id).cert_serial == format(cert.serial_number, "x")
    assert [r.number for r in hub.nodes.records()] == [1]                       # registered nodes only
    assert local.node_id not in (hub.path / "nodes.json").read_text()
    assert [r.number for r in HubIdentity.load_or_create(hub.path).nodes.records()] == [1]
    assert hub.nodes.register(machine_id="m2", name="local", tags=[], node_id=None)[0].number == 2  # 0 is not taken
    stored = json.loads((hub.path / "nodes.json").read_text())                  # P-PF1-5
    assert local.node_id not in stored["nodes"] and 0 not in [r["number"] for r in stored["nodes"].values()]
    hub.nodes.reload()
    assert hub.nodes.verify(local.node_id, secret).number == 0                  # kept across a reload
    hub.nodes.ban(persistent.node_id)                                           # bans machine m1 too ...
    assert hub.nodes.verify(local.node_id, secret) is not None                  # ... but never the hub's own node
    ch = channel(_hub_server(hub, servers, tls=False))
    assert call_stream(ch, "/colosseum.v1.Hub/Session", metadata=node_metadata(local.node_id, secret)) \
        == [b"ok:a", b"ok:b"]
    with pytest.raises(SecurityError, match="invalid node name"):
        hub.nodes.add_ephemeral("bad name", "m1")


def test_two_processes_creating_the_hub_identity_at_once_agree(tmp_path):
    """A8: two ``train`` hubs started together on a fresh machine create one identity, not two half ones."""
    import multiprocessing as mp

    from net_kit import create_hub_identity_child

    ctx = mp.get_context("spawn")
    path = tmp_path / "fresh-hub"
    barrier, results = ctx.Barrier(2), ctx.Queue()
    procs = [ctx.Process(target=create_hub_identity_child, args=(str(path), barrier, results)) for _ in range(2)]
    for proc in procs:
        proc.start()
    try:
        outcomes = [results.get(timeout=120) for _ in procs]
    finally:
        for proc in procs:
            proc.join(timeout=30)
    assert [o[0] for o in outcomes] == ["ok", "ok"], outcomes
    assert outcomes[0][1:] == outcomes[1][1:]
    mine = HubIdentity.load_or_create(path)
    assert (mine.fingerprint, mine.token.secret) == outcomes[0][1:]
    files = sorted(p.name for p in path.iterdir() if not p.name.startswith("."))
    assert files == ["ca.crt", "ca.key", "nodes.lock", "secret"]


def test_a_second_server_cannot_bind_a_taken_port(tmp_path, servers):
    """A8: ``server_options`` turns grpc's default SO_REUSEPORT off, so a taken port is an error, not a silent
    split of the connections between two servers (two hubs on one machine, the data port's next-free-port rule)."""
    from colosseum.net.grpc_util import server_options

    assert ("grpc.so_reuseport", 0) in server_options(4)
    first, port = loopback_server()
    servers.append(first)
    second, bound = loopback_server(port=port)
    assert bound == 0
    assert call(channel(port), "/colosseum.v1.Bootstrap/Info") == b"ok:x"


# ---------------------------------------------------------------------------
# P-PF1-2: several processes of one identity (two hubs of one machine)
# ---------------------------------------------------------------------------


def test_two_books_of_one_identity_never_lose_bans_or_registrations(tmp_path):
    """Each hub process holds its own NodeBook of the machine's identity: mutations are read-modify-write under the
    file lock and reads see the other book's changes, so a ban is never lifted by another process's save."""
    from colosseum.net.security import NodeBook

    hub = hub_identity(tmp_path)
    one, two = hub.nodes, NodeBook(hub.path)
    a, secret_a = one.register(machine_id="m1", name="laptop", tags=[], node_id=None)
    b, secret_b = two.register(machine_id="m2", name="laptop", tags=[], node_id=None)
    assert (a.number, b.number) == (1, 2) and b.name == "laptop-2"              # numbers and names stay unique
    assert two.verify(a.node_id, secret_a) is not None and one.verify(b.node_id, secret_b) is not None
    one.ban(a.node_id)
    assert two.verify(a.node_id, secret_a) is None and two.is_banned(None, "m1")
    with pytest.raises(SecurityError, match="banned"):
        two.register(machine_id="m1", name="laptop", tags=[], node_id=None)
    c, _ = two.register(machine_id="m3", name="pc", tags=[], node_id=None)      # two's save keeps one's ban ...
    assert c.number == 3 and one.is_banned(a.node_id, None)
    assert [e["node_id"] for e in json.loads((hub.path / "banned.json").read_text())["banned"]] == [a.node_id]
    assert [r.number for r in one.records()] == [2, 3] == [r.number for r in two.records()]   # ... and b and c
    one.remove(c.node_id)
    assert two.get(c.node_id) is None
    two.set_cert_serial(b.node_id, "abc")
    assert one.get(b.node_id).cert_serial == "abc"


def test_concurrent_node_identities_keep_one_key(tmp_path):
    """P-PF1-2: two creators of one node identity (e.g. a join retried while the first still starts) end up with
    the same key; the second never replaces the key the first already uses."""
    import threading

    hub = hub_identity(tmp_path)
    barrier, keys, errors = threading.Barrier(8), [], []

    def create() -> None:
        barrier.wait(timeout=30)
        try:
            keys.append(node_identity(tmp_path, hub.fingerprint, "racing").public_key_pem())
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=create) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors and len(keys) == 8 and len(set(keys)) == 1
    again = node_identity(tmp_path, hub.fingerprint, "racing")
    assert again.public_key_pem() == keys[0] and _mode(again.path / "node.key") == 0o600
    assert not [p.name for p in again.path.iterdir() if p.name.startswith(".")]   # no temp file left


# ---------------------------------------------------------------------------
# Fix round 1: hardening
# ---------------------------------------------------------------------------


def test_names_ids_and_tokens_with_a_trailing_newline_are_refused(tmp_path):
    from colosseum.net.security import NODE_ID_RE, NODE_NAME_RE, TOKEN_RE

    hub = hub_identity(tmp_path)
    node = node_identity(tmp_path, hub.fingerprint)
    assert NODE_NAME_RE.match("laptop\n") is None and NODE_ID_RE.match("0" * 32 + "\n") is None
    assert TOKEN_RE.match(str(hub.token) + "\n") is None and TOKEN_RE.match(str(hub.token))
    with pytest.raises(SecurityError, match="invalid node name"):
        hub.nodes.register(machine_id="m1", name="laptop\n", tags=[], node_id=None)
    with pytest.raises(SecurityError, match="invalid node name"):
        hub.nodes.add_ephemeral("local\n", "m1")
    with pytest.raises(SecurityError, match="invalid node id"):
        hub.issue_node_certificate("0" * 32 + "\n", node.public_key_pem())
    with pytest.raises(SecurityError, match="invalid CA fingerprint"):
        NodeIdentity.load_or_create(hub.fingerprint + "\n", root=tmp_path / "x")


def test_a_ban_survives_a_crash_between_the_two_book_files(tmp_path, monkeypatch):
    """banned.json is replaced before nodes.json: a crash in between never forgets a node without its ban."""
    import colosseum.net.security as security

    hub = hub_identity(tmp_path)
    a, secret = hub.nodes.register(machine_id="m1", name="laptop", tags=[], node_id=None)
    real = security.write_private

    def crashing(path, data, **kwargs):
        if path.name == "nodes.json":
            raise OSError("disk gone")
        return real(path, data, **kwargs)

    monkeypatch.setattr(security, "write_private", crashing)
    with pytest.raises(OSError, match="disk gone"):
        hub.nodes.ban(a.node_id)
    monkeypatch.setattr(security, "write_private", real)
    fresh = security.NodeBook(hub.path)
    assert fresh.get(a.node_id) is not None                                     # still listed ...
    assert fresh.verify(a.node_id, secret) is None and fresh.is_banned(a.node_id, None)   # ... but banned


def test_the_hubs_own_ephemeral_node_cannot_be_banned(tmp_path):
    hub = hub_identity(tmp_path)
    local, secret = hub.nodes.add_ephemeral("local", "hub-machine")
    with pytest.raises(SecurityError, match="cannot be banned"):
        hub.nodes.ban(local.node_id)
    assert hub.nodes.verify(local.node_id, secret) is not None and not hub.nodes.is_banned(None, "hub-machine")


def test_a_node_of_a_banned_machine_is_told_it_is_banned(tmp_path, servers):
    hub = hub_identity(tmp_path)
    first = node_identity(tmp_path, hub.fingerprint, "first")
    second = node_identity(tmp_path, hub.fingerprint, "second")
    record_1 = register(hub, first, name="one", machine_id="shared")
    record_2 = register(hub, second, name="two", machine_id="shared")
    hub.nodes.ban(record_1.node_id)                                             # bans machine "shared"
    assert hub.nodes.authenticate(record_2.node_id, second.secret) == (None, "banned")
    assert hub.nodes.authenticate(record_2.node_id, "wrong") == (None, "unknown")
    ch = channel(_hub_server(hub, servers, tls=False))
    good = node_metadata(record_2.node_id, second.secret)
    assert _code(call_stream, ch, "/colosseum.v1.Hub/Session", metadata=good) == grpc.StatusCode.PERMISSION_DENIED
    assert _code(call_stream, ch, "/colosseum.v1.Hub/Session", metadata=node_metadata(record_2.node_id, "wrong")) \
        == grpc.StatusCode.UNAUTHENTICATED


def test_a_registration_reply_of_another_ca_is_refused(tmp_path):
    hub, other = hub_identity(tmp_path), hub_identity(tmp_path, "other-hub")
    node = node_identity(tmp_path, hub.fingerprint)
    with pytest.raises(SecurityError, match="pinned"):
        register(other, node)                                                   # other's CA in the reply
    assert not node.registered and not (node.path / "node.json").exists() and not (node.path / "ca.crt").exists()
    register(hub, node)
    assert node.registered and _mode(node.path / "node.json") == 0o600
    assert not [p.name for p in node.path.iterdir() if p.name.startswith(".")]   # no temp file left


def test_node_certificates_are_issued_for_p256_keys_only(tmp_path):
    from cryptography.hazmat.primitives import serialization

    hub = hub_identity(tmp_path)
    p384 = ec.generate_private_key(ec.SECP384R1()).public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    with pytest.raises(SecurityError, match="secp256r1"):
        hub.issue_node_certificate("0" * 32, p384)


def test_every_token_and_node_method_refuses_a_caller_without_credentials(tmp_path, servers):
    hub = hub_identity(tmp_path)
    ch = channel(_hub_server(hub, servers, tls=False))
    unauthenticated = grpc.StatusCode.UNAUTHENTICATED
    assert _code(lambda: list(ch.unary_stream("/colosseum.v1.Hub/AdminStream")(b"", timeout=5))) == unauthenticated
    assert list(ch.unary_stream("/colosseum.v1.Hub/AdminStream")(b"", metadata=token_metadata(hub.token),
                                                                 timeout=5)) == [b"part1", b"part2"]
    for metadata in ((), token_metadata(hub.token)):                            # the token is not a node credential
        assert _code(lambda md: ch.stream_unary("/colosseum.v1.Hub/Upload")(iter([b"1"]), metadata=md, timeout=5),
                     metadata) == unauthenticated
        assert _code(lambda md: list(ch.unary_stream("/colosseum.v1.Hub/Download")(b"", metadata=md, timeout=5)),
                     metadata) == unauthenticated
        assert _code(lambda md: ch.stream_unary("/colosseum.v1.Hub/RelaySend")(iter([b"f"]), metadata=md, timeout=5),
                     metadata) == unauthenticated
        assert _code(lambda md: call_stream(ch, "/colosseum.v1.Hub/RelayInbox", metadata=md), metadata) \
            == unauthenticated
