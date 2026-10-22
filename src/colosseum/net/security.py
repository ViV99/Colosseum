"""Identities, tokens, certificates and node authentication (SP5 spec blocks 2.1, 3.2-3.4; rulings R1, R15).

Trust is per hub machine, not per run (spec 2.1): ``<data_home>/colosseum/hub/`` (``0700``) holds the hub's root of
trust — an ECDSA P-256 key ``ca.key`` (``0600``) and its self-signed certificate ``ca.crt`` (CA, SAN
``colosseum-hub``), which is also the hub's TLS server certificate (R15: a node fetches it unverified with
``ssl.get_server_certificate``, checks its SPKI fingerprint against the token and then uses it as the gRPC root; the
curl one-liner pins the same key) — the token secret ``secret`` (``0600``), the node book ``nodes.json`` and the
ban list ``banned.json``. ``rotate()`` moves all of it to ``rotated/<n>/`` and starts a new CA and secret: every
node is revoked.

The token ``cst1.<secret>.<fp>`` = 32 random bytes base64url without padding + the lowercase hex SHA-256 of the CA
certificate's SubjectPublicKeyInfo (DER). It is the owner's full access (registration and admin commands).

Node authentication (R1): grpc-python cannot ask for an optional client certificate, and the hub's one port must also
serve clients without one (the installer, the first registration), so the hub port is server-side TLS only and every
hub RPC after registration carries the node id and a per-node secret in metadata (``x-colosseum-node``,
``x-colosseum-node-secret``); the hub keeps only the secret's SHA-256 (``nodes.json``). Registration and admin RPCs
carry the token secret (``x-colosseum-token``). Node data ports (node <-> node) are full mutual TLS with node
certificates issued by the CA (CN = node id, SANs ``colosseum-node`` and ``n<number>.colosseum-node``: a client checks
the second one, so a data port answers only for the node the hub announced as the source, SP5 T3.2). ``train``'s local
node gets its key and certificate minted in memory by the hub (``mint_node_credentials``; ruling R30). ``ban`` and
``rotate`` revoke both kinds.
``--insecure`` drops TLS but keeps the token and node secrets; loopback (``train`` without ``--listen``) is plain.

A node's identity per hub CA lives in ``<data_home>/colosseum/node/<ca fingerprint>/`` (``0700``): ``node.key``
(``0600``), ``node.crt``, ``ca.crt`` (the pinned CA) and ``node.json`` (``0600``: node id, number, name, secret,
hub addresses).

Every private key and secret is written ``0600`` (created with that mode, then replaced atomically); every
identity directory is ``0700``. Creating, loading and rotating a hub identity hold an exclusive ``flock`` on
``<hub dir>/.lock``, so two processes starting on a fresh machine at once (two ``train`` runs) make one identity.
Refusals raise :class:`SecurityError` with a hint (exit 1 at the CLI).
"""

from __future__ import annotations

import base64
import contextlib
import datetime
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import ssl
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import grpc
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

TOKEN_PREFIX = "cst1"
TOKEN_ENV = "COLOSSEUM_TOKEN"
HUB_SERVER_NAME = "colosseum-hub"
NODE_SERVER_NAME = "colosseum-node"
MD_TOKEN, MD_NODE_ID, MD_NODE_SECRET = "x-colosseum-token", "x-colosseum-node", "x-colosseum-node-secret"
# \A ... \Z (not ^ ... $): a trailing newline never passes, also for callers that use .match
TOKEN_RE = re.compile(r"\Acst1\.([A-Za-z0-9_-]{43})\.([0-9a-f]{64})\Z")
NODE_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")
NODE_NAME_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]{0,62}\Z")
SECRET_RE = re.compile(r"\A[A-Za-z0-9_-]{43}\Z")
FINGERPRINT_RE = re.compile(r"\A[0-9a-f]{64}\Z")
BOOK_VERSION = 1
CERT_BACKDATE = datetime.timedelta(days=7)          # tolerate machine clocks a few days apart
CA_LIFETIME = datetime.timedelta(days=20 * 365)
NODE_LIFETIME = datetime.timedelta(days=10 * 365)


def node_server_name(number: int) -> str:
    """The TLS server name of node ``number``'s data port (a SAN of its certificate; T3.2): a client that connects to
    the source the hub announced overrides the target name with it, so another node's certificate of the same CA
    (a banned node's, say) cannot answer in its place. Node numbers are unique per hub CA (registered nodes never
    reuse one; ``train``'s local nodes are 0)."""
    return f"n{int(number)}.{NODE_SERVER_NAME}"


def certificate_dns_names(cert_pem: bytes) -> list[str]:
    """The DNS names (SAN) of a PEM certificate; ValueError when it cannot be read."""
    try:
        cert = x509.load_pem_x509_certificate(cert_pem)
    except (TypeError, ValueError) as e:
        raise ValueError(f"not a PEM certificate ({e})") from None
    try:
        return list(cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(
            x509.DNSName))
    except x509.ExtensionNotFound:
        return []


class SecurityError(Exception):
    """A refusal or a broken identity (exit 1 at the CLI); the message says what to do."""


class HubUnreachableError(SecurityError):
    """``fetch_hub_ca`` could not reach the hub at all (no connection, a timeout): nothing was refused, so a node may
    retry. A ``SecurityError`` for callers that treat every failure alike."""


# ---------------------------------------------------------------------------
# Directories and files
# ---------------------------------------------------------------------------


def data_home() -> Path:
    """``$XDG_DATA_HOME`` or ``~/.local/share``."""
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def cache_home() -> Path:
    """``$XDG_CACHE_HOME`` or ``~/.cache``."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def hub_identity_dir() -> Path:
    return data_home() / "colosseum" / "hub"


def node_identity_dir(ca_fingerprint: str, root: Path | None = None) -> Path:
    """The node identity for the hub CA ``ca_fingerprint``; ``root`` replaces ``<data_home>/colosseum/node``."""
    return (Path(root) if root is not None else data_home() / "colosseum" / "node") / ca_fingerprint


def private_dir(path: Path) -> Path:
    """``path`` (and missing parents) as a directory with mode ``0700``."""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def write_private(path: Path, data: bytes | str, *, mode: int = 0o600) -> Path:
    """Atomically (re)write ``path`` with ``mode`` (default ``0600``; the temp file is created with it)."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _create_exclusive(path: Path, data: bytes, mode: int = 0o600) -> bool:
    """Create ``path`` with ``data`` unless it exists (complete or not at all: a ``0600`` temp file hard-linked to
    the final name); False when another process created it first (P-PF1-2: two creators keep one key)."""
    tmp = path.with_name(f".{path.name}.new-{os.getpid()}-{threading.get_ident()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        return True
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """An exclusive ``flock`` on ``<directory>/.lock`` (between processes and between threads: every call opens the
    file anew)."""
    fd = os.open(directory / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)                                   # closing releases the lock


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise SecurityError(f"{path} is unreadable ({e}); move it away to start over") from None
    if not isinstance(data, dict):
        raise SecurityError(f"{path} is not a JSON object; move it away to start over")
    return data


# ---------------------------------------------------------------------------
# Token and fingerprints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    secret: str
    ca_fingerprint: str

    def __str__(self) -> str:
        return f"{TOKEN_PREFIX}.{self.secret}.{self.ca_fingerprint}"

    def __repr__(self) -> str:                      # never print the secret by accident
        return f"Token(fingerprint={self.ca_fingerprint[:8]}...)"


def new_secret() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")


def parse_token(text: str) -> Token:
    m = TOKEN_RE.fullmatch(text.strip())
    if m is None:
        raise SecurityError("malformed token: expected cst1.<43 base64url chars>.<64 hex chars> (print it with "
                            "`colosseum hub token` on the hub machine)")
    return Token(m.group(1), m.group(2))


def resolve_token(token: str | None, token_file: str | None, *, env: Mapping[str, str] = os.environ) -> Token:
    """The token from ``--token`` > ``--token-file`` > ``$COLOSSEUM_TOKEN``."""
    if token:
        return parse_token(token)
    if token_file:
        try:
            return parse_token(Path(token_file).read_text())
        except OSError as e:
            raise SecurityError(f"cannot read --token-file {token_file}: {e}") from None
    if env.get(TOKEN_ENV):
        return parse_token(env[TOKEN_ENV])
    raise SecurityError(f"no token: pass --token, --token-file or set {TOKEN_ENV} (print it with `colosseum hub "
                        f"token` on the hub machine)")


def spki_fingerprint(cert_pem: bytes) -> str:
    """Lowercase hex SHA-256 of the certificate's SubjectPublicKeyInfo (DER)."""
    cert = x509.load_pem_x509_certificate(cert_pem)
    spki = cert.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(spki).hexdigest()


def curl_pin(fingerprint: str) -> str:
    """``sha256//<base64>``: the same SPKI digest for ``curl --pinnedpubkey``."""
    return "sha256//" + base64.b64encode(bytes.fromhex(fingerprint)).decode("ascii")


def _key_pem(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _load_key(path: Path) -> ec.EllipticCurvePrivateKey:
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as e:
        raise SecurityError(f"{path} is not a readable private key ({e})") from None
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise SecurityError(f"{path} is not an ECDSA key")
    return key


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _make_ca(key: ec.EllipticCurvePrivateKey) -> x509.Certificate:
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, HUB_SERVER_NAME)])
    now = _now()
    usage = x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                          key_encipherment=False, data_encipherment=False, key_agreement=False, encipher_only=False,
                          decipher_only=False)
    return (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - CERT_BACKDATE)
            .not_valid_after(now + CA_LIFETIME)
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(usage, critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(HUB_SERVER_NAME)]), critical=False)
            .sign(key, hashes.SHA256()))


# ---------------------------------------------------------------------------
# The node book (nodes.json + banned.json)
# ---------------------------------------------------------------------------


@dataclass
class NodeRecord:
    node_id: str
    number: int
    name: str
    machine_id: str
    tags: list[str] = field(default_factory=list)
    secret_sha256: str = ""
    cert_serial: str = ""
    persistent: bool = True
    registered_run: str = ""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NodeBook:
    """Registered and banned nodes of one hub identity (``nodes.json``, ``banned.json``; both ``0600``).

    ``nodes.json``: ``{"version": 1, "next_number", "nodes": {node_id: NodeRecord fields}}`` (node ids are 32 hex
    chars chosen here; numbers >= 1 are never reused; names are unique, ``-2``, ``-3``, ... on a collision);
    ``banned.json``: ``{"version": 1, "banned": [{"node_id", "machine_id", "name"}]}``. Thread-safe (servicer
    threads verify while the main loop registers).

    Several processes may hold books of one identity at once (two hubs of one machine, P-PF1-2): every mutation is
    read-modify-write under an exclusive ``flock`` on ``nodes.lock`` (reload both files, apply, save), and every read
    re-reads the files (under a shared lock) when ``os.stat`` of either changed, so a ban or a registration made by
    another process is seen at the next call and never overwritten.

    Ephemeral nodes (:meth:`add_ephemeral`, A8 / A9 / P-PF1-5: ``train``'s local node) have number 0 and live in this
    object only: never in ``nodes.json``, never taking a number, not in :meth:`records`, never banned (a ban of the
    hub machine's id does not stop its own local node); :meth:`get` and :meth:`verify` see them, :meth:`reload`
    keeps them."""

    LOCK_FILE = "nodes.lock"

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._lock = threading.Lock()
        self._ephemeral: dict[str, NodeRecord] = {}
        self._next, self._nodes, self._banned = 1, {}, []
        self._seen: tuple = ()
        self.reload()

    # -- files ---------------------------------------------------------------

    @contextlib.contextmanager
    def _file_lock(self, exclusive: bool) -> Iterator[None]:
        fd = os.open(self._dir / self.LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            os.close(fd)                               # closing releases the lock

    def _signature(self) -> tuple:
        """What identifies the current contents of both files (``None`` for a missing one)."""
        out = []
        for name in ("nodes.json", "banned.json"):
            try:
                st = os.stat(self._dir / name)
            except FileNotFoundError:
                out.append(None)
            else:
                out.append((st.st_ino, st.st_mtime_ns, st.st_size))
        return tuple(out)

    def _load_files(self) -> None:
        """Read both files into memory (caller holds ``_lock`` and a file lock)."""
        seen = self._signature()
        data = _read_json(self._dir / "nodes.json") or {"version": BOOK_VERSION, "next_number": 1, "nodes": {}}
        banned = _read_json(self._dir / "banned.json") or {"version": BOOK_VERSION, "banned": []}
        if data.get("version") != BOOK_VERSION or banned.get("version") != BOOK_VERSION:
            raise SecurityError(f"{self._dir}: node book of another version; move nodes.json / banned.json away")
        try:
            next_number = int(data["next_number"])
            nodes = {nid: NodeRecord(**rec) for nid, rec in data["nodes"].items()}
            banned_list = [dict(entry) for entry in banned["banned"]]
        except (KeyError, TypeError, ValueError) as e:
            raise SecurityError(f"{self._dir}/nodes.json is malformed ({e}); move it away to start over") from None
        self._next, self._nodes, self._banned, self._seen = next_number, nodes, banned_list, seen

    def _refresh(self) -> None:
        """Re-read the files if another process (or a rotation) changed them (caller holds ``_lock``)."""
        if self._signature() != self._seen:
            with self._file_lock(exclusive=False):
                self._load_files()

    def _save(self) -> None:
        """Write both files (caller holds ``_lock`` and the exclusive file lock)."""
        nodes = {"version": BOOK_VERSION, "next_number": self._next,
                 "nodes": {nid: asdict(rec) for nid, rec in self._nodes.items()}}
        # banned.json first: a crash between the two replaces leaves a ban with its node still listed (refused
        # anyway), never a forgotten node without its ban (fail closed)
        write_private(self._dir / "banned.json",
                      json.dumps({"version": BOOK_VERSION, "banned": self._banned}, indent=2, sort_keys=True))
        write_private(self._dir / "nodes.json", json.dumps(nodes, indent=2, sort_keys=True))
        self._seen = self._signature()

    @contextlib.contextmanager
    def _mutating(self) -> Iterator[None]:
        """Read-modify-write: the in-process lock, the exclusive file lock and the files' current contents."""
        with self._lock, self._file_lock(exclusive=True):
            self._load_files()
            yield

    def reload(self) -> None:
        """Re-read both files (after ``HubIdentity.rotate``; the object stays the one interceptors hold; ephemeral
        nodes stay)."""
        with self._lock, self._file_lock(exclusive=False):
            self._load_files()

    # -- reads ---------------------------------------------------------------

    def _banned_entry(self, node_id: str | None, machine_id: str | None) -> bool:
        return any((node_id and e.get("node_id") == node_id) or (machine_id and e.get("machine_id") == machine_id)
                   for e in self._banned)

    def records(self) -> list[NodeRecord]:
        """The registered (stored) nodes by number; ephemeral nodes are not listed."""
        with self._lock:
            self._refresh()
            return sorted((NodeRecord(**asdict(r)) for r in self._nodes.values()), key=lambda r: r.number)

    def get(self, node_id: str) -> NodeRecord | None:
        with self._lock:
            rec = self._ephemeral.get(node_id)
            if rec is None:
                self._refresh()
                rec = self._nodes.get(node_id)
            return None if rec is None else NodeRecord(**asdict(rec))

    def is_banned(self, node_id: str | None, machine_id: str | None) -> bool:
        with self._lock:
            if node_id and node_id in self._ephemeral:
                return False
            self._refresh()
            return self._banned_entry(node_id, machine_id)

    def authenticate(self, node_id: str, secret: str) -> tuple[NodeRecord | None, str]:
        """``(record, "ok")`` for a node's current secret; ``(None, "banned")`` when the secret is valid but the node
        id or its machine id is banned; ``(None, "unknown")`` otherwise (an unknown id and a wrong secret look the
        same, also for banned ids: a caller without the secret learns nothing)."""
        with self._lock:
            digest = _sha256(secret)
            rec = self._ephemeral.get(node_id)
            if rec is not None:
                return (NodeRecord(**asdict(rec)), "ok") if hmac.compare_digest(rec.secret_sha256, digest) \
                    else (None, "unknown")
            self._refresh()
            rec = self._nodes.get(node_id)
            if rec is not None and rec.secret_sha256 and hmac.compare_digest(rec.secret_sha256, digest):
                if self._banned_entry(node_id, rec.machine_id):
                    return None, "banned"
                return NodeRecord(**asdict(rec)), "ok"
            if any(e.get("node_id") == node_id and e.get("secret_sha256")
                   and hmac.compare_digest(e["secret_sha256"], digest) for e in self._banned):
                return None, "banned"                  # a banned node with its last secret
            return None, "unknown"

    def verify(self, node_id: str, secret: str) -> NodeRecord | None:
        """The node's record if ``secret`` is its current secret and it is not banned; else None."""
        return self.authenticate(node_id, secret)[0]

    # -- mutations -----------------------------------------------------------

    def add_ephemeral(self, name: str, machine_id: str) -> tuple[NodeRecord, str]:
        """An in-memory node with number 0 and a fresh secret (``train``'s local node, A8 / A9); never stored."""
        if not NODE_NAME_RE.fullmatch(name):
            raise SecurityError(f"invalid node name {name!r} (letters, digits, '_', '.', '-'; at most 63 chars)")
        secret = new_secret()
        with self._lock:
            node_id = secrets.token_hex(16)
            rec = NodeRecord(node_id=node_id, number=0, name=name, machine_id=machine_id,
                             secret_sha256=_sha256(secret), persistent=False)
            self._ephemeral[node_id] = rec
            return NodeRecord(**asdict(rec)), secret

    def _unique_name(self, wanted: str, node_id: str) -> str:
        taken = {r.name for nid, r in self._nodes.items() if nid != node_id}
        if wanted not in taken:
            return wanted
        k = 2
        while f"{wanted}-{k}" in taken:
            k += 1
        return f"{wanted}-{k}"

    def register(self, *, machine_id: str, name: str, tags: Sequence[str], node_id: str | None,
                 persistent: bool = True, run_id: str = "") -> tuple[NodeRecord, str]:
        """Register a new node, or re-register a known ``node_id`` (its number and name stay, the old secret stops
        working at once); returns ``(record, new secret)``. A banned node id or machine id is a SecurityError."""
        if not NODE_NAME_RE.fullmatch(name):
            raise SecurityError(f"invalid node name {name!r} (letters, digits, '_', '.', '-'; at most 63 chars)")
        with self._mutating():
            if self._banned_entry(node_id, machine_id):
                raise SecurityError(f"node {name!r} (machine {machine_id[:12]}) is banned on this hub; "
                                    f"`colosseum hub token --rotate` revokes every node and clears the ban list")
            secret = new_secret()
            rec = self._nodes.get(node_id) if node_id else None
            if rec is None:
                node_id = secrets.token_hex(16)
                rec = NodeRecord(node_id=node_id, number=self._next, name=self._unique_name(name, node_id),
                                 machine_id=machine_id)
                self._next += 1
                self._nodes[node_id] = rec
            rec.machine_id, rec.tags = machine_id, list(tags)
            rec.secret_sha256, rec.persistent, rec.registered_run = _sha256(secret), bool(persistent), run_id
            self._save()
            return NodeRecord(**asdict(rec)), secret

    def set_cert_serial(self, node_id: str, serial: str) -> None:
        """Record the serial of the node's newest certificate (a node removed meanwhile is left alone)."""
        with self._lock:
            if node_id in self._ephemeral:
                self._ephemeral[node_id].cert_serial = serial
                return
        with self._mutating():
            rec = self._nodes.get(node_id)
            if rec is not None:
                rec.cert_serial = serial
                self._save()

    def ban(self, node_id: str) -> None:
        """Forget the node and refuse its node id and machine id from now on (the entry keeps the node's secret hash,
        so the node itself is told it is banned). The hub's own ephemeral node cannot be banned (it would ban the
        hub machine's id)."""
        with self._lock:
            if node_id in self._ephemeral:
                raise SecurityError("the hub's own local node cannot be banned")
        with self._mutating():
            rec = self._nodes.pop(node_id, None)
            entry = {"node_id": node_id, "machine_id": rec.machine_id if rec else "", "name": rec.name if rec else ""}
            if rec is not None and rec.secret_sha256:
                entry["secret_sha256"] = rec.secret_sha256
            self._banned.append(entry)
            self._save()

    def remove(self, node_id: str) -> None:
        """Forget the node (it may register again with the token; its number is not reused)."""
        with self._lock:
            if self._ephemeral.pop(node_id, None) is not None:
                return
        with self._mutating():
            if self._nodes.pop(node_id, None) is not None:
                self._save()


# ---------------------------------------------------------------------------
# The hub identity
# ---------------------------------------------------------------------------


class HubIdentity:
    """The hub machine's CA, token secret and node book (module docstring). ``load_or_create`` makes them once."""

    def __init__(self, path: Path, key: ec.EllipticCurvePrivateKey, ca_pem: bytes, secret: str) -> None:
        self.path = path
        self._key = key
        self.ca_pem = ca_pem
        self.fingerprint = spki_fingerprint(ca_pem)
        self.token = Token(secret, self.fingerprint)
        self.nodes = NodeBook(path)

    @classmethod
    def load_or_create(cls, path: Path | None = None) -> HubIdentity:
        """Load the identity at ``path`` (default :func:`hub_identity_dir`), creating it on first use; safe when
        several processes do it at once (the directory lock)."""
        path = private_dir(Path(path) if path is not None else hub_identity_dir())
        with _locked(path):
            return cls._load_or_create_locked(path)

    @classmethod
    def _load_or_create_locked(cls, path: Path) -> HubIdentity:
        key_path, ca_path, secret_path = path / "ca.key", path / "ca.crt", path / "secret"
        present = [p.exists() for p in (key_path, ca_path, secret_path)]
        if any(present) and not all(present):
            missing = [p.name for p, ok in zip((key_path, ca_path, secret_path), present) if not ok]
            raise SecurityError(f"hub identity {path} is incomplete (missing {missing}); restore it, or move the "
                                f"directory away to create a new one (every node must then join again)")
        if not any(present):
            key = ec.generate_private_key(ec.SECP256R1())
            write_private(key_path, _key_pem(key))
            write_private(ca_path, _make_ca(key).public_bytes(serialization.Encoding.PEM), mode=0o644)
            write_private(secret_path, new_secret() + "\n")
        key = _load_key(key_path)
        ca_pem = ca_path.read_bytes()
        secret = secret_path.read_text().strip()
        if not SECRET_RE.fullmatch(secret):
            raise SecurityError(f"{secret_path} is malformed; `colosseum hub token --rotate` makes a new one")
        cert = x509.load_pem_x509_certificate(ca_pem)
        if cert.public_key().public_numbers() != key.public_key().public_numbers():
            raise SecurityError(f"{ca_path} does not belong to {key_path}; restore the pair or rotate")
        return cls(path, key, ca_pem, secret)

    @property
    def key_pem(self) -> bytes:
        return _key_pem(self._key)

    def rotate(self) -> None:
        """A new CA and secret; the old identity (and with it every node and ban) moves to ``rotated/<n>/``."""
        with _locked(self.path):
            with self.nodes._file_lock(exclusive=True):    # no book mutation of another process in between
                rotated = private_dir(self.path / "rotated")
                n = 1 + max((int(p.name) for p in rotated.iterdir() if p.name.isdigit()), default=0)
                target = private_dir(rotated / str(n))
                for name in ("ca.key", "ca.crt", "secret", "nodes.json", "banned.json"):
                    if (self.path / name).exists():
                        shutil.move(str(self.path / name), str(target / name))
            fresh = HubIdentity._load_or_create_locked(self.path)
        self._key, self.ca_pem, self.fingerprint, self.token = fresh._key, fresh.ca_pem, fresh.fingerprint, fresh.token
        self.nodes.reload()                       # the same object: interceptors holding it see the empty book

    def issue_node_certificate(self, node_id: str, public_key_pem: bytes) -> bytes:
        """A client + server certificate for the node's key (CN = node id, SANs ``colosseum-node`` and, for a node of
        the book (ephemeral ones included), ``node_server_name(number)``), signed by the CA."""
        if not NODE_ID_RE.fullmatch(node_id):
            raise SecurityError(f"invalid node id {node_id!r}")
        try:
            public = serialization.load_pem_public_key(public_key_pem)
        except (ValueError, TypeError) as e:
            raise SecurityError(f"the node's public key is not a PEM key ({e})") from None
        if not isinstance(public, ec.EllipticCurvePublicKey) or not isinstance(public.curve, ec.SECP256R1):
            raise SecurityError("the node's key must be an ECDSA P-256 (secp256r1) key")
        ca = x509.load_pem_x509_certificate(self.ca_pem)
        record = self.nodes.get(node_id)
        names = [x509.DNSName(NODE_SERVER_NAME)]
        if record is not None:
            names.append(x509.DNSName(node_server_name(record.number)))
        now = _now()
        cert = (x509.CertificateBuilder()
                .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node_id)]))
                .issuer_name(ca.subject).public_key(public).serial_number(x509.random_serial_number())
                .not_valid_before(now - CERT_BACKDATE).not_valid_after(now + NODE_LIFETIME)
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(x509.SubjectAlternativeName(names), critical=False)
                .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH,
                                                      ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
                .sign(self._key, hashes.SHA256()))
        if record is not None:
            self.nodes.set_cert_serial(node_id, format(cert.serial_number, "x"))
        return cert.public_bytes(serialization.Encoding.PEM)

    def server_credentials(self) -> grpc.ServerCredentials:
        """TLS for the hub port with the CA certificate as the server certificate; no client certificates (R1)."""
        return grpc.ssl_server_credentials([(self.key_pem, self.ca_pem)], require_client_auth=False)


# ---------------------------------------------------------------------------
# The node identity
# ---------------------------------------------------------------------------


class NodeIdentity:
    """A node's key, certificate and registration for one hub CA (module docstring)."""

    def __init__(self, path: Path, key: ec.EllipticCurvePrivateKey, info: dict, ca_fingerprint: str) -> None:
        self.path = path
        self.ca_fingerprint = ca_fingerprint
        self._key = key
        self.node_id: str | None = info.get("node_id")
        self.number: int | None = info.get("number")
        self.name: str | None = info.get("name")
        self.secret: str | None = info.get("secret")
        self.hub_addresses: list[str] = list(info.get("hub_addresses", []))

    @classmethod
    def load_or_create(cls, ca_fingerprint: str, root: Path | None = None) -> NodeIdentity:
        if not FINGERPRINT_RE.fullmatch(ca_fingerprint):
            raise SecurityError(f"invalid CA fingerprint {ca_fingerprint!r}")
        path = private_dir(node_identity_dir(ca_fingerprint, root))
        key_path = path / "node.key"
        if not key_path.exists():
            _create_exclusive(key_path, _key_pem(ec.generate_private_key(ec.SECP256R1())))
        info = _read_json(path / "node.json") or {}
        return cls(path, _load_key(key_path), info, ca_fingerprint)

    def public_key_pem(self) -> bytes:
        return self._key.public_key().public_bytes(serialization.Encoding.PEM,
                                                   serialization.PublicFormat.SubjectPublicKeyInfo)

    @property
    def key_pem(self) -> bytes:
        return _key_pem(self._key)

    @property
    def cert_pem(self) -> bytes | None:
        path = self.path / "node.crt"
        return path.read_bytes() if path.exists() else None

    @property
    def ca_pem(self) -> bytes:
        """The pinned hub CA (``ca.crt``, kept at registration)."""
        return (self.path / "ca.crt").read_bytes()

    @property
    def registered(self) -> bool:
        return bool(self.node_id and self.secret and self.cert_pem)

    def save_registration(self, reply: Any, *, hub_addresses: Sequence[str] = ()) -> None:
        """Keep a ``RegisterReply`` (node id, number, name, certificate, secret, CA) for the next connections. A reply
        whose CA is not the one this identity is pinned to is refused; the files are replaced atomically, ``node.json``
        last (it marks the registration complete)."""
        ca_pem = reply.ca_pem.encode("ascii")
        try:
            actual = spki_fingerprint(ca_pem)
        except ValueError as e:
            raise SecurityError(f"the hub's reply carries no readable CA certificate ({e})") from None
        if not hmac.compare_digest(actual, self.ca_fingerprint):
            raise SecurityError(f"the hub's reply names CA {actual[:16]}..., this node identity is pinned to "
                                f"{self.ca_fingerprint[:16]}...: refusing the registration")
        if not NODE_ID_RE.fullmatch(str(reply.node_id)):
            raise SecurityError(f"the hub's reply carries an invalid node id {reply.node_id!r}")
        write_private(self.path / "node.crt", reply.certificate_pem.encode("ascii"), mode=0o644)
        write_private(self.path / "ca.crt", ca_pem, mode=0o644)
        self.node_id, self.number, self.name, self.secret = reply.node_id, int(reply.number), reply.name, \
            reply.node_secret
        self.hub_addresses = list(dict.fromkeys([*hub_addresses, *self.hub_addresses]))
        write_private(self.path / "node.json", json.dumps({
            "node_id": self.node_id, "number": self.number, "name": self.name, "secret": self.secret,
            "hub_addresses": self.hub_addresses}, indent=2))

    def forget(self) -> None:
        """Drop the registration (after a ban or a rotation); the key stays."""
        for name in ("node.json", "node.crt", "ca.crt"):
            (self.path / name).unlink(missing_ok=True)
        self.node_id = self.number = self.name = self.secret = None


@dataclass(frozen=True)
class MemoryNodeIdentity:
    """A node's data-port credentials held in memory (``train``'s local node, ruling R30; a T1.3 contract amendment of
    SP5 T3.2): what ``data_server_credentials`` / ``data_channel_credentials`` read of a ``NodeIdentity``."""

    node_id: str
    key_pem: bytes = field(repr=False)            # never in a log line or a traceback
    cert_pem: bytes = field(repr=False)
    ca_pem: bytes = field(repr=False)


def mint_node_credentials(hub: HubIdentity, node_id: str) -> MemoryNodeIdentity:
    """A fresh key and its certificate for ``node_id`` (a node of ``hub``'s book, e.g. the ephemeral record of
    ``train``'s local node), never written to disk (ruling R30)."""
    key = ec.generate_private_key(ec.SECP256R1())
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    cert = hub.issue_node_certificate(node_id, public)
    return MemoryNodeIdentity(node_id=node_id, key_pem=_key_pem(key), cert_pem=cert, ca_pem=hub.ca_pem)


# ---------------------------------------------------------------------------
# Fetching the hub CA and gRPC credentials
# ---------------------------------------------------------------------------


def fetch_hub_ca(host: str, port: int, *, fingerprint: str, timeout: float) -> bytes:
    """The hub's certificate (= its CA, R15) from an unverified TLS handshake, accepted only if its SPKI fingerprint
    is the token's. HubUnreachableError when no TLS connection could be made (refused, unreachable, timed out);
    SecurityError otherwise (no TLS on that address, a wrong token, another hub, or interception)."""
    try:
        pem = ssl.get_server_certificate((host, int(port)), timeout=timeout).encode("ascii")
    except ssl.SSLError as e:
        raise SecurityError(f"cannot fetch the hub certificate from {host}:{port} ({type(e).__name__}: {e}); is the "
                            f"hub running with TLS on that address? (a hub started with --insecure needs --insecure "
                            f"here)") from None
    except (OSError, TimeoutError) as e:
        raise HubUnreachableError(f"cannot reach the hub at {host}:{port} to fetch its certificate "
                                  f"({type(e).__name__}: {e}); is it running and is the address right?") from None
    actual = spki_fingerprint(pem)
    if not hmac.compare_digest(actual, fingerprint):
        raise SecurityError(f"the certificate at {host}:{port} has fingerprint {actual[:16]}..., the token names "
                            f"{fingerprint[:16]}...: a wrong token, another hub on that address, or someone in "
                            f"between; check the token with `colosseum hub token` on the hub machine")
    return pem


def hub_channel_credentials(ca_pem: bytes) -> grpc.ChannelCredentials:
    """TLS to the hub port, trusting only the pinned CA (use with ``grpc_util.channel_options(server_name=
    HUB_SERVER_NAME)``)."""
    return grpc.ssl_channel_credentials(root_certificates=ca_pem)


def data_server_credentials(node: NodeIdentity | MemoryNodeIdentity, ca_pem: bytes) -> grpc.ServerCredentials:
    """Mutual TLS for a node's data port: the node's certificate, clients must present a certificate of the CA."""
    if node.cert_pem is None:
        raise SecurityError("the node has no certificate yet: register with the hub first")
    return grpc.ssl_server_credentials([(node.key_pem, node.cert_pem)], root_certificates=ca_pem,
                                       require_client_auth=True)


def data_channel_credentials(node: NodeIdentity | MemoryNodeIdentity, ca_pem: bytes) -> grpc.ChannelCredentials:
    """Mutual TLS to another node's data port (with ``channel_options(server_name=node_server_name(<its number>))``:
    only that node's certificate is accepted; ``NODE_SERVER_NAME`` accepts any node of the CA)."""
    if node.cert_pem is None:
        raise SecurityError("the node has no certificate yet: register with the hub first")
    return grpc.ssl_channel_credentials(root_certificates=ca_pem, private_key=node.key_pem,
                                        certificate_chain=node.cert_pem)
