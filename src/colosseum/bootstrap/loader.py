"""The thin ``join`` loader (SP5 spec 2.3): from a hub address and a token to a running node agent.

Steps: check the hub's certificate against the token's fingerprint (TLS; skipped with ``--insecure``) ->
``Bootstrap/Info`` -> the code bundle by hash (``Bootstrap/Bundle``, token in the metadata; cached in
``<cache>/colosseum/bundles/<hash>/``) -> the node environment (uv: the hub's Python major.minor, the hub's
requirement list, ``torch==<hub version>`` with the build of this machine; cached in
``<cache>/colosseum/envs/<key>/``) -> ``exec`` of ``python -m colosseum.node.agent`` from that environment, with
``PYTHONPATH`` on the bundle and the bundle's project directory as the working directory.

The file is self-contained (standard library + ``grpcio``; no protobuf, no colosseum imports): the installer script
embeds it verbatim next to ``bundle.py`` and ``torch_select.py`` (loaded from this file's directory), and
``colosseum join`` runs it from an installed colosseum of any version. It speaks only the ``Bootstrap`` service,
whose three messages it encodes and decodes itself (field numbers checked against the generated classes by
``tests/unit/test_sp5_join_loader.py``).

Cache entries (``bundles/<hash>/``, ``envs/<key>/``) are complete once their marker file exists. A loader marks the
entry it uses with a shared ``flock`` on its ``.in-use`` file (inherited by the node agent through ``exec``); placing
an entry, marking it in use and pruning old entries all run under the cache directory's lock
(``bundle.cache_lock``), so one loader never removes what another has just checked or is about to use.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

BOOTSTRAP = "/colosseum.v1.Bootstrap/"
MD_TOKEN = "x-colosseum-token"
HUB_SERVER_NAME = "colosseum-hub"
DEFAULT_PORT = 47100
TOKEN_ENV = "COLOSSEUM_TOKEN"
NODE_PYTHON_ENV = "COLOSSEUM_NODE_PYTHON"     # use this interpreter instead of building an environment
UV_ENV = "COLOSSEUM_UV"                       # the uv binary to use
IN_USE_FDS_ENV = "COLOSSEUM_IN_USE_FDS"       # the agent's inherited .in-use lock fds, closed by a re-exec'd loader
PROTOCOL_MAJOR = 1                            # colosseum.net.messages.PROTOCOL_VERSION[0] (checked by a test)
TOKEN_RE = re.compile(r"^cst1\.([A-Za-z0-9_-]{43})\.([0-9a-f]{64})$")
UV_VERSION = "0.13.1"
MIN_UV = (0, 7, 14)                           # --torch-backend without --preview
UV_INSTALL_URL = "https://astral.sh/uv/{version}/install.sh"
CLEAN_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV", "PYTHONSAFEPATH",
             "PYTHONNOUSERSITE")
ENV_MARKER = ".colosseum-env.json"
IN_USE = ".in-use"
KEEP_ENVS = 2
KEEP_BUNDLES = 5
CONNECT_TIMEOUT_SEC = 10.0
USE_ATTEMPTS = 3                              # an entry pruned by another loader before it is in use: make it again
AGENT_MODULE = "colosseum.node.agent"
CHECK_SCRIPT = (
    "import json, sys\n"
    "try:\n"
    "    import torch\n"
    "except Exception as e:\n"
    "    print(json.dumps({'error': repr(e)})); raise SystemExit(0)\n"
    "mps = getattr(torch.backends, 'mps', None)\n"
    "print(json.dumps({'python': '%d.%d' % sys.version_info[:2], 'torch': torch.__version__,\n"
    "                  'cuda': torch.version.cuda, 'cuda_available': bool(torch.cuda.is_available()),\n"
    "                  'mps': bool(mps is not None and mps.is_available())}))\n"
)


class LoaderError(Exception):
    """A refusal or failure of the loader; ``colosseum join`` prints it and exits with ``code``."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


def _say(message: str) -> None:
    print(f"colosseum join: {message}", file=sys.stderr, flush=True)


def _sibling(name: str):
    """``bundle`` / ``torch_select``: from the package when this file is ``colosseum.bootstrap.loader``, else the
    file next to this one (installer copy)."""
    if __package__:
        return importlib.import_module(f"{__package__}.{name}")
    module_name = f"_colosseum_loader_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name(f"{name}.py"))
    if spec is None or spec.loader is None:
        raise LoaderError(f"{name}.py is missing next to {__file__}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------------------------
# minimal protobuf wire codec for BootstrapInfo, BundleRequest, BlobPart
# ---------------------------------------------------------------------------------------------------------------

INFO_FIELDS = {1: ("protocol_major", "int"), 2: ("protocol_minor", "int"), 3: ("code_hash", "str"),
               4: ("requirements_hash", "str"), 5: ("requirements", "strs"), 6: ("python", "str"),
               7: ("torch_version", "str"), 8: ("torch_build", "str"), 9: ("bundle_size", "int")}
PART_FIELDS = {1: ("data", "bytes"), 2: ("offset", "int"), 3: ("total_size", "int"), 4: ("last", "bool"),
               5: ("digest", "str"), 6: ("not_modified", "bool"), 7: ("meta_json", "str")}
BUNDLE_REQUEST_CODE_HASH = 1


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf) or shift > 63:
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _encode_varint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def decode_message(buf: bytes, fields: Mapping[int, tuple[str, str]]) -> dict:
    """Decode the known fields of a protobuf message (unknown fields are skipped: newer hubs may add some)."""
    out: dict = {name: ([] if kind == "strs" else b"" if kind == "bytes" else "" if kind == "str"
                        else False if kind == "bool" else 0) for name, kind in fields.values()}
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 2:
            length, pos = _varint(buf, pos)
            if pos + length > len(buf):
                raise ValueError("truncated field")
            value, pos = buf[pos:pos + length], pos + length
        elif wire == 1:
            value, pos = buf[pos:pos + 8], pos + 8
        elif wire == 5:
            value, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        if number not in fields:
            continue
        name, kind = fields[number]
        if wire != (0 if kind in ("int", "bool") else 2):
            raise ValueError(f"field {number} ({name}) has wire type {wire}")
        if kind == "int":
            out[name] = value - (1 << 64) if value >= 1 << 63 else value
        elif kind == "bool":
            out[name] = bool(value)
        elif kind == "str":
            out[name] = value.decode("utf-8")
        elif kind == "strs":
            out[name].append(value.decode("utf-8"))
        else:
            out[name] = bytes(value)
    return out


def encode_bundle_request(code_hash: str) -> bytes:
    data = code_hash.encode()
    return _encode_varint(BUNDLE_REQUEST_CODE_HASH << 3 | 2) + _encode_varint(len(data)) + data


# ---------------------------------------------------------------------------------------------------------------
# token, address, certificate
# ---------------------------------------------------------------------------------------------------------------

def resolve_token(token: str | None, token_file: str | None, environ: Mapping[str, str]) -> tuple[str, str]:
    """(token, fingerprint): ``--token`` > ``--token-file`` > ``COLOSSEUM_TOKEN``."""
    text = token
    if text is None and token_file is not None:
        try:
            text = Path(token_file).read_text().strip()
        except OSError as e:
            raise LoaderError(f"cannot read the token file {token_file}: {e}", 2) from e
    if text is None:
        text = environ.get(TOKEN_ENV)
    if not text:
        raise LoaderError("no token: pass --token, --token-file or set COLOSSEUM_TOKEN "
                          "(colosseum hub token prints it on the hub machine)", 2)
    match = TOKEN_RE.match(text.strip())
    if match is None:
        raise LoaderError("the token is malformed (expected cst1.<secret>.<fingerprint>; copy it again from "
                          "colosseum hub token)", 2)
    return text.strip(), match.group(2)


def parse_address(address: str) -> tuple[str, int]:
    """``host``, ``host:port``, ``[v6]:port`` -> (host, port); the port defaults to 47100."""
    text = address.strip()
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        port = rest[1:] if rest.startswith(":") else ""
    elif text.count(":") == 1:
        host, port = text.split(":")
    else:
        host, port = text, ""
    if not host or (port and not port.isdigit()) or (port and not 0 < int(port) < 65536):
        raise LoaderError(f"bad hub address {address!r} (expected host:port)", 2)
    return host, int(port) if port else DEFAULT_PORT


def _tlv(buf: bytes, pos: int) -> tuple[int, int, int]:
    """(tag, content start, content end) of the DER element at ``pos``."""
    tag = buf[pos]
    length = buf[pos + 1]
    pos += 2
    if length & 0x80:
        count = length & 0x7F
        if not 0 < count <= 4:
            raise ValueError("bad DER length")
        length = int.from_bytes(buf[pos:pos + count], "big")
        pos += count
    if pos + length > len(buf):
        raise ValueError("truncated DER")
    return tag, pos, pos + length


def spki_sha256(cert_der: bytes) -> str:
    """SHA-256 hex of the certificate's SubjectPublicKeyInfo (DER), as in the token."""
    tag, start, _ = _tlv(cert_der, 0)                  # Certificate
    tag, pos, end = _tlv(cert_der, start)              # tbsCertificate
    if tag != 0x30:
        raise ValueError("not a certificate")
    fields = []
    while pos < end:
        t, s, e = _tlv(cert_der, pos)
        fields.append((t, pos, e))
        pos = e
    if fields and fields[0][0] == 0xA0:                # [0] version
        fields = fields[1:]
    # serialNumber, signature, issuer, validity, subject, subjectPublicKeyInfo
    _, s, e = fields[5]
    return hashlib.sha256(cert_der[s:e]).hexdigest()


def curl_pin(fingerprint: str) -> str:
    return "sha256//" + base64.b64encode(bytes.fromhex(fingerprint)).decode()


def _unreachable(host: str, port: int, error: BaseException) -> LoaderError:
    return LoaderError(f"cannot reach the hub at {host}:{port} ({error}); is it running, is the port open "
                       "(WSL2 hubs: docs/DISTRIBUTED_GUIDE.md, WSL2)?")


def fetch_hub_ca(host: str, port: int, fingerprint: str, *, timeout: float) -> bytes:
    """The hub's certificate (R15: the CA itself) from an unverified handshake, accepted only when its key
    matches the token's fingerprint."""
    try:
        pem = ssl.get_server_certificate((host, port), timeout=timeout)
    except ssl.SSLError as e:
        raise LoaderError(f"TLS handshake with {host}:{port} failed ({e.reason or e}); if the hub runs with "
                          "--insecure, add --insecure here") from e
    except (TimeoutError, OSError) as e:
        raise _unreachable(host, port, e) from e
    actual = spki_sha256(ssl.PEM_cert_to_DER_cert(pem))
    if actual != fingerprint:
        raise LoaderError(f"the hub at {host}:{port} does not have the key of this token (fingerprint "
                          f"{actual[:16]}..., the token names {fingerprint[:16]}...): another hub, a rotated "
                          "token (colosseum hub token) or an interception")
    return pem.encode()


def open_channel(host: str, port: int, *, fingerprint: str, insecure: bool, max_message_mb: int = 64):
    import grpc

    target = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    options = [("grpc.max_receive_message_length", max_message_mb << 20),
               ("grpc.max_send_message_length", max_message_mb << 20)]
    if insecure:
        try:                                           # a hub that is down is not "a TLS hub" (_rpc_error)
            socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_SEC).close()
        except OSError as e:
            raise _unreachable(host, port, e) from e
        return grpc.insecure_channel(target, options=options)
    ca = fetch_hub_ca(host, port, fingerprint, timeout=CONNECT_TIMEOUT_SEC)
    options.append(("grpc.ssl_target_name_override", HUB_SERVER_NAME))
    return grpc.secure_channel(target, grpc.ssl_channel_credentials(root_certificates=ca), options=options)


def _rpc_error(e, what: str, *, insecure: bool = False) -> LoaderError:
    import grpc

    code = e.code() if hasattr(e, "code") else None
    if code == grpc.StatusCode.UNAUTHENTICATED:
        return LoaderError(f"{what}: the hub refused the token ({e.details()})")
    if code == grpc.StatusCode.PERMISSION_DENIED:
        return LoaderError(f"{what}: refused by the hub ({e.details()})")
    if code == grpc.StatusCode.UNIMPLEMENTED:
        return LoaderError(f"{what}: the hub serves no code to nodes (a plain `colosseum train` listens on "
                           "loopback only: start `colosseum hub` or `colosseum train --listen`)")
    if code == grpc.StatusCode.UNAVAILABLE:
        hint = ("the port answers but not plain gRPC: unless the hub runs with --insecure, drop --insecure here"
                if insecure else "is the hub running and its port open?")
        return LoaderError(f"{what}: the hub is not reachable ({e.details()}); {hint}")
    return LoaderError(f"{what} failed: {code} {e.details() if hasattr(e, 'details') else e}")


def fetch_info(channel, *, insecure: bool = False) -> dict:
    import grpc

    call = channel.unary_unary(BOOTSTRAP + "Info", request_serializer=None, response_deserializer=None)
    try:
        raw = call(b"", timeout=CONNECT_TIMEOUT_SEC * 3)
    except grpc.RpcError as e:
        raise _rpc_error(e, "Bootstrap/Info", insecure=insecure) from e
    try:
        return decode_message(raw, INFO_FIELDS)
    except ValueError as e:
        raise LoaderError(f"the hub's Bootstrap/Info cannot be read by this loader ({e}); run the hub's "
                          "one-liner instead") from e


# ---------------------------------------------------------------------------------------------------------------
# cache: bundle, uv, environment
# ---------------------------------------------------------------------------------------------------------------

def cache_home(environ: Mapping[str, str]) -> Path:
    return Path(environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def _hold(path: Path, *, exclusive: bool, blocking: bool = True):
    """An flock on ``path`` (created); returns the open file object (keep it to keep the lock)."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    try:
        fcntl.flock(f, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | (0 if blocking else fcntl.LOCK_NB))
    except OSError:
        f.close()
        raise
    return f


def _last_used(entry: Path) -> float:
    """The mtime of the entry's ``.in-use`` file; the directory's own mtime before its first use (just built)."""
    for path in (entry / IN_USE, entry):
        try:
            return path.stat().st_mtime
        except OSError:
            continue
    return 0.0


def prune(directory: Path, keep: int, current: Path) -> list[str]:
    """Remove all but the ``keep`` most recently used entries of ``directory`` (by their ``.in-use`` file's
    mtime), under the cache lock; an entry another process holds (shared lock on ``.in-use``), an environment
    being built (its ``<name>.lock``) or ``current`` is never removed."""
    removed: list[str] = []
    if not directory.is_dir():
        return removed
    with _sibling("bundle").cache_lock(directory):
        entries = [p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")]
        entries.sort(key=_last_used, reverse=True)
        for entry in entries[keep:]:
            if entry == current:
                continue
            held = []
            try:
                for lock_path in (entry / IN_USE, directory / f"{entry.name}.lock"):
                    if lock_path.exists():
                        held.append(_hold(lock_path, exclusive=True, blocking=False))
            except OSError:
                continue                               # in use by a running node, or being built
            else:
                shutil.rmtree(entry, ignore_errors=True)
                removed.append(entry.name)
            finally:
                for lock in held:
                    lock.close()
    return removed


def _use(entry: Path, marker: str):
    """The shared ``.in-use`` lock of the complete cache entry ``entry`` (its ``marker`` file exists), its mtime
    touched for pruning; None when the entry is incomplete or gone (another loader pruned it meanwhile). Taken under
    the cache lock, so no prune runs between the check and the lock."""
    with _sibling("bundle").cache_lock(entry.parent):
        if not (entry / marker).is_file():
            return None
        lock = _hold(entry / IN_USE, exclusive=False)
        os.utime(entry / IN_USE)
        return lock


def ensure_bundle(channel, info: dict, token: str, cache: Path) -> Path:
    import grpc

    bundle = _sibling("bundle")
    code_hash = info["code_hash"]
    if not re.fullmatch(r"[0-9a-f]{64}", code_hash or ""):
        raise LoaderError(f"the hub announced a malformed code hash {code_hash!r}")
    target = bundle.bundles_dir(cache) / code_hash
    if (target / bundle.COMPLETE_MARKER).is_file():
        return target
    size = int(info["bundle_size"])
    if not 0 < size <= bundle.MAX_ARCHIVE_BYTES:
        raise LoaderError(f"the hub announced a code bundle of {size} bytes")
    _say(f"downloading the code bundle {code_hash[:12]} ({size / 1e6:.1f} MB)")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.parent / f".dl-{code_hash[:16]}-{os.getpid()}.tar.gz"
    call = channel.unary_stream(BOOTSTRAP + "Bundle", request_serializer=None, response_deserializer=None)
    received = 0
    try:
        with open(part, "wb") as out:
            for raw in call(encode_bundle_request(code_hash), metadata=((MD_TOKEN, token.split(".")[1]),),
                            timeout=3600):
                try:
                    piece = decode_message(raw, PART_FIELDS)
                except ValueError as e:
                    raise LoaderError(f"code bundle: a malformed part from the hub ({e})") from e
                if piece["offset"] != received:
                    raise LoaderError(f"code bundle: part at offset {piece['offset']}, expected {received}")
                received += len(piece["data"])
                if received > size:
                    raise LoaderError("code bundle: more data than announced")
                out.write(piece["data"])
                if piece["last"]:
                    break
        if received != size:
            raise LoaderError(f"code bundle: {received} of {size} bytes received")
        try:
            return bundle.unpack_bundle(part, code_hash, cache_root=cache)
        except bundle.BundleError as e:
            raise LoaderError(str(e)) from e
    except grpc.RpcError as e:
        raise _rpc_error(e, "Bootstrap/Bundle") from e
    finally:
        part.unlink(missing_ok=True)


def _child_environ(environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` for third-party subprocesses (uv, its installer, build backends, nvidia-smi, the torch check):
    without the token (only the node agent gets it), the inherited lock fds and the Python path variables."""
    return {k: v for k, v in environ.items() if k not in CLEAN_ENV and k not in (TOKEN_ENV, IN_USE_FDS_ENV)}


def _uv_version(uv: str, environ: Mapping[str, str]) -> tuple[int, ...] | None:
    try:
        out = subprocess.run([uv, "--version"], capture_output=True, text=True, timeout=30,
                             env=_child_environ(environ))
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", out.stdout)
    return tuple(int(x) for x in match.groups()) if out.returncode == 0 and match else None


def ensure_uv(explicit: str | None, environ: Mapping[str, str], cache: Path) -> str:
    """``COLOSSEUM_UV`` > ``--uv`` > ``<cache>/colosseum/uv/uv`` > ``uv`` on PATH (>= 0.7.14) > a private uv
    installed with the pinned astral.sh installer (``UV_UNMANAGED_INSTALL``: no profile edits, no receipt)."""
    private = cache / "colosseum" / "uv" / "uv"
    path = environ.get("PATH")
    for candidate in (environ.get(UV_ENV), explicit, str(private), shutil.which("uv", path=path)):
        if candidate and Path(candidate).exists() and (_uv_version(candidate, environ) or (0,)) >= MIN_UV:
            return candidate
    _say(f"installing uv {UV_VERSION} into {private.parent}")
    url = UV_INSTALL_URL.format(version=UV_VERSION)
    fetch = (f"curl -LsSf {url}" if shutil.which("curl", path=path)
             else f"wget -qO- {url}" if shutil.which("wget", path=path) else None)
    if fetch is None:
        raise LoaderError("uv is not installed and neither curl nor wget is available to install it")
    env = {k: v for k, v in _child_environ(environ).items() if k != "UV_INSTALL_DIR"}
    env.update(UV_UNMANAGED_INSTALL=str(private.parent), UV_NO_MODIFY_PATH="1")
    if subprocess.run(["sh", "-c", f"{fetch} | sh"], env=env).returncode != 0 or not private.exists():
        raise LoaderError(f"installing uv from {url} failed (install uv yourself and retry)")
    return str(private)


def env_key(info: dict, backend: str | None) -> str:
    text = "|".join([info["requirements_hash"], info["python"], info["torch_version"], backend or "default"])
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def _uv_environ(environ: Mapping[str, str], cache: Path, driver: str | None) -> dict[str, str]:
    torch_select = _sibling("torch_select")
    env = _child_environ(torch_select.nvidia_smi_environ(environ))
    env.setdefault("UV_CACHE_DIR", str(cache / "colosseum" / "uv-cache"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(cache / "colosseum" / "python"))
    if driver:
        env["UV_CUDA_DRIVER_VERSION"] = driver
    return env


def check_env(python: str, env: Mapping[str, str]) -> dict:
    out = subprocess.run([python, "-I", "-c", CHECK_SCRIPT], capture_output=True, text=True, env=dict(env),
                         timeout=300)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": (out.stderr or out.stdout).strip()[-500:]}


def ensure_env(info: dict, *, uv: str, cache: Path, environ: Mapping[str, str], choice, gpus) -> Path:
    """The node environment for ``info`` with ``choice`` (a ``torch_select.TorchChoice``): built once per
    (requirements, Python, torch version, backend), under an exclusive lock; the marker file is written last."""
    root = cache / "colosseum" / "envs"
    env_dir = root / env_key(info, choice.backend)
    python = env_dir / "bin" / "python"
    if (env_dir / ENV_MARKER).is_file() and python.exists():
        return env_dir
    root.mkdir(parents=True, exist_ok=True)
    with _hold(root / f"{env_dir.name}.lock", exclusive=True):
        if (env_dir / ENV_MARKER).is_file() and python.exists():
            return env_dir
        shutil.rmtree(env_dir, ignore_errors=True)
        env = _uv_environ(environ, cache, choice.cuda_driver)
        build = choice.backend or "the default build"
        _say(f"building the node environment (Python {info['python']}, {choice.requirement}, {build}); "
             "the first time this downloads torch")
        try:
            if subprocess.run([uv, "venv", "--quiet", "--python", info["python"], str(env_dir)], env=env).returncode:
                raise LoaderError(f"uv could not create a Python {info['python']} environment in {env_dir}")
            req = env_dir / "colosseum-requirements.txt"
            req.write_text("\n".join([*info["requirements"], choice.requirement]) + "\n")
            cmd = [uv, "pip", "install", "--python", str(python), "-r", str(req)]
            if choice.backend:
                cmd += ["--torch-backend", choice.backend]
            proc = subprocess.run(cmd, env=env, stderr=subprocess.PIPE, text=True)
            if proc.returncode:
                tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
                sys.stderr.write(proc.stderr)
                raise LoaderError("installing the hub's requirements failed:\n" + tail + "\nA package without a "
                                  "wheel for this machine can be replaced on the hub with distributed.requirements "
                                  "(a requirements file of your own).")
            if proc.stderr:
                sys.stderr.write(proc.stderr)
            found = check_env(str(python), env)
            if "error" in found:
                raise LoaderError(f"the new environment cannot import torch: {found['error']}")
            if found["torch"].split("+", 1)[0] != info["torch_version"]:
                raise LoaderError(f"uv installed torch {found['torch']}, the hub runs {info['torch_version']}")
            if gpus and not found["cuda"]:
                _say(f"WARNING: this machine has an NVIDIA GPU but the CPU build of torch was installed (driver "
                     f"{choice.cuda_driver}); learners here run on the CPU. Update the driver to use the GPU.")
            elif gpus and not found["cuda_available"]:
                _say("WARNING: the CUDA build of torch was installed but CUDA is not available here")
            (env_dir / ENV_MARKER).write_text(json.dumps({**found, "backend": choice.backend,
                                                          "requirements_hash": info["requirements_hash"]}))
        except BaseException:
            shutil.rmtree(env_dir, ignore_errors=True)
            raise
    return env_dir


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="colosseum join", description="Join this machine to a Colosseum hub.")
    p.add_argument("address", help="the hub's host:port")
    p.add_argument("--token")
    p.add_argument("--token-file")
    p.add_argument("--workers", type=int)
    p.add_argument("--evaluators", type=int)
    p.add_argument("--cpu-learners", type=int)
    p.add_argument("--no-learners", action="store_true")
    p.add_argument("--gpus")
    p.add_argument("--name")
    p.add_argument("--tags")
    p.add_argument("--once", action="store_true")
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--uv", help=argparse.SUPPRESS)
    return p


def agent_args(ns: argparse.Namespace) -> list[str]:
    """The node agent's arguments: the join arguments without the token and the loader-only flags."""
    out = [ns.address]
    for flag in ("workers", "evaluators", "cpu_learners", "gpus", "name", "tags"):
        value = getattr(ns, flag)
        if value is not None:
            out += [f"--{flag.replace('_', '-')}", str(value)]
    for flag in ("no_learners", "once", "insecure"):
        if getattr(ns, flag):
            out.append(f"--{flag.replace('_', '-')}")
    return out


def agent_environ(environ: Mapping[str, str], bundle_dir: Path, token: str,
                  in_use_fds: Sequence[int] = ()) -> dict[str, str]:
    """The node agent's environment: the bundle's ``PYTHONPATH``, the token, and (``COLOSSEUM_IN_USE_FDS``) the
    numbers of the ``.in-use`` lock descriptors it inherits, so a loader it re-execs closes them first."""
    bundle = _sibling("bundle")
    torch_select = _sibling("torch_select")
    env = _child_environ(torch_select.nvidia_smi_environ(environ))
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in bundle.bundle_python_path(bundle_dir))
    env["PYTHONNOUSERSITE"] = "1"
    env[TOKEN_ENV] = token
    if in_use_fds:
        env[IN_USE_FDS_ENV] = ",".join(_fd_entry(fd) for fd in in_use_fds)
    return env


def _fd_entry(fd: int) -> str:
    st = os.fstat(fd)
    return f"{fd}:{st.st_dev}:{st.st_ino}"


def close_inherited_locks(text: str | None) -> None:
    """Close the ``.in-use`` lock descriptors a node agent inherited (``COLOSSEUM_IN_USE_FDS``: ``fd:dev:inode``
    entries) when it re-execs the loader (T6.3): otherwise they pin its old bundle and environment for its lifetime
    (a new descriptor of the same process cannot take the exclusive prune lock against them). A descriptor is closed
    only while it still refers to the recorded file (the variable reaches every descendant of the agent through its
    environment: a stale entry must never close an unrelated descriptor); malformed entries are ignored."""
    for item in (text or "").split(","):
        parts = item.strip().split(":")
        if len(parts) != 3 or not all(re.fullmatch(r"[0-9]{1,20}", p) for p in parts):
            continue
        fd, dev, ino = (int(p) for p in parts)
        if fd <= 2:
            continue
        try:
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) == (dev, ino):
                os.close(fd)
        except OSError:
            continue


def _in_use(make: Callable[[], Path], marker: str, locks: list, what: str) -> Path:
    """``make()`` (an ensured cache entry) marked in use (its lock appended to ``locks``); made again when another
    loader pruned it between the check and the lock."""
    for _ in range(USE_ATTEMPTS):
        entry = make()
        lock = _use(entry, marker)
        if lock is not None:
            locks.append(lock)
            return entry
    raise LoaderError(f"the {what} in {entry} was removed {USE_ATTEMPTS} times while this loader prepared it "
                      "(other loaders pruning the same cache?); retry")


def loader_main(argv: Sequence[str], *, environ: Mapping[str, str] | None = None,
                exec_fn: Callable[[str, list[str], dict[str, str]], object] | None = None) -> int:
    environ = dict(os.environ if environ is None else environ)
    close_inherited_locks(environ.pop(IN_USE_FDS_ENV, None))   # before any new lock is taken
    exec_fn = exec_fn or os.execve
    try:
        ns = build_parser().parse_args(list(argv))
    except SystemExit as e:
        return int(e.code or 0)
    locks: list = []                                   # .in-use locks: inherited by the agent through exec
    try:
        if os.name != "posix":
            raise LoaderError("native Windows is not supported: run colosseum join inside WSL2")
        token, fingerprint = resolve_token(ns.token, ns.token_file, environ)
        host, port = parse_address(ns.address)
        ns.address = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        bundle = _sibling("bundle")
        torch_select = _sibling("torch_select")
        cache = cache_home(environ)
        channel = open_channel(host, port, fingerprint=fingerprint, insecure=ns.insecure)
        with contextlib.closing(channel):
            info = fetch_info(channel, insecure=ns.insecure)
            if info["protocol_major"] != PROTOCOL_MAJOR:
                raise LoaderError(f"the hub speaks protocol {info['protocol_major']}.{info['protocol_minor']}, this "
                                  f"loader {PROTOCOL_MAJOR}.x: run the hub's one-liner (this loader is from another "
                                  "Colosseum version)")
            if not re.fullmatch(r"\d+\.\d+", info["python"]) or not info["torch_version"]:
                raise LoaderError(f"the hub announced Python {info['python']!r} and torch {info['torch_version']!r}: "
                                  "a hub of another colosseum version? run the hub's one-liner instead")
            bundle_dir = _in_use(lambda: ensure_bundle(channel, info, token, cache), bundle.COMPLETE_MARKER, locks,
                                 "code bundle")
        prune(bundle.bundles_dir(cache), KEEP_BUNDLES, bundle_dir)
        python = environ.get(NODE_PYTHON_ENV)
        if not python:
            gpus = torch_select.probe_nvidia(environ=_child_environ(environ))
            try:
                choice = torch_select.torch_spec(info["torch_version"], info["torch_build"],
                                                 platform=torch_select.probe_platform(), gpus=gpus)
            except torch_select.TorchSelectError as e:
                raise LoaderError(str(e)) from e
            if choice.warning:
                _say(f"WARNING: {choice.warning}")
            uv = ensure_uv(ns.uv, environ, cache)
            env_dir = _in_use(lambda: ensure_env(info, uv=uv, cache=cache, environ=environ, choice=choice, gpus=gpus),
                              ENV_MARKER, locks, "node environment")
            prune(env_dir.parent, KEEP_ENVS, env_dir)
            python = str(env_dir / "bin" / "python")
        for lock in locks:
            os.set_inheritable(lock.fileno(), True)    # the agent keeps both in use for its lifetime
        args = [python, "-m", AGENT_MODULE, *agent_args(ns), "--code-hash", info["code_hash"],
                "--bundle", str(bundle_dir)]
        os.chdir(bundle.project_dir(bundle_dir))
        _say(f"starting the node agent (code {info['code_hash'][:12]})")
        exec_fn(python, args, agent_environ(environ, bundle_dir, token, [lock.fileno() for lock in locks]))
        return 0
    except LoaderError as e:
        _say(str(e))
        return e.code
    except OSError as e:                               # disk full, an unwritable cache, a failing exec ...
        _say(f"failed: {e}")
        return 1
    except KeyboardInterrupt:
        _say("interrupted")
        return 130
    finally:                                           # reached only when exec did not replace the process
        for lock in locks:
            lock.close()


if __name__ == "__main__":
    sys.exit(loader_main(sys.argv[1:]))
