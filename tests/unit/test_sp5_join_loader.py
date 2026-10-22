"""The join loader (SP5 T6.2, spec 2.3-2.4) against a real gRPC Bootstrap server on loopback: pinned TLS, Info,
the bundle by hash, the uv environment (fake uv), the exec of the node agent (recorded), refusals."""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

import join_kit
from colosseum.bootstrap import bundle as B
from colosseum.bootstrap import loader as L
from colosseum.bootstrap import torch_select as T
from colosseum.net.messages import PROTOCOL_VERSION
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.security import HubIdentity, spki_fingerprint


@pytest.fixture
def hub(tmp_path):
    identity = HubIdentity.load_or_create(tmp_path / "hub-identity")
    servers = []

    def start(*, tls=True, service="default", **kw):
        svc = join_kit.tiny_service(tmp_path, insecure=not tls, **kw) if service == "default" else service
        server, port = join_kit.bootstrap_server(svc, identity, tls=tls, seen=hub.seen)
        servers.append(server)
        hub.service = svc
        return port
    hub = type("Hub", (), {"identity": identity, "token": str(identity.token), "start": staticmethod(start),
                           "seen": []})
    yield hub
    for s in servers:
        s.stop(0)


def _env(tmp_path, **extra):
    uv, py = join_kit.write_fakes(tmp_path / "fakes")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home"), "XDG_CACHE_HOME": str(tmp_path / "cache"),
           "FAKE_RECORD": str(tmp_path / "record.jsonl"), "FAKE_UV_LOG": str(tmp_path / "uv.jsonl"),
           "COLOSSEUM_UV": str(uv), "PYTHONPATH": "/opt/ros/iron/lib/python3.10/site-packages"}
    env.update(extra)
    return env, py


def _run(argv, env, monkeypatch, gpus=()):
    monkeypatch.setattr(T, "probe_nvidia", lambda **kw: list(gpus))
    monkeypatch.setattr(T, "probe_platform", lambda: T.NodePlatform("Linux", "x86_64"))
    calls = []
    code = L.loader_main(argv, environ=env, exec_fn=lambda path, args, e: calls.append((path, args, e)))
    return code, calls


def _uv_calls(tmp_path):
    p = tmp_path / "uv.jsonl"
    calls = [json.loads(line)["args"] for line in p.read_text().splitlines()] if p.exists() else []
    return [c for c in calls if c != ["--version"]]                          # the loader's uv version probes


# --- codec and certificate --------------------------------------------------------------------------------------

def test_codec_matches_the_generated_messages():
    msg = pb.BootstrapInfo(protocol_major=1, protocol_minor=3, code_hash="c" * 64, requirements_hash="r" * 64,
                           requirements=["a==1", "b==2"], python="3.12", torch_version="2.14.1",
                           torch_build="cu130", bundle_size=5_000_000_000)
    assert L.decode_message(msg.SerializeToString(), L.INFO_FIELDS) == {
        "protocol_major": 1, "protocol_minor": 3, "code_hash": "c" * 64, "requirements_hash": "r" * 64,
        "requirements": ["a==1", "b==2"], "python": "3.12", "torch_version": "2.14.1", "torch_build": "cu130",
        "bundle_size": 5_000_000_000}
    part = pb.BlobPart(data=b"\x00\xff" * 10, offset=3, total_size=-1, last=True, digest="d", meta_json="{}")
    assert L.decode_message(part.SerializeToString(), L.PART_FIELDS) == {
        "data": b"\x00\xff" * 10, "offset": 3, "total_size": -1, "last": True, "digest": "d",
        "not_modified": False, "meta_json": "{}"}
    assert pb.BundleRequest.FromString(L.encode_bundle_request("h" * 64)).code_hash == "h" * 64
    for fields, message in ((L.INFO_FIELDS, pb.BootstrapInfo), (L.PART_FIELDS, pb.BlobPart)):
        assert {n: name for n, (name, _) in fields.items()} == {
            f.number: f.name for f in message.DESCRIPTOR.fields if f.name in {v[0] for v in fields.values()}}
    assert pb.BundleRequest.DESCRIPTOR.fields_by_name["code_hash"].number == L.BUNDLE_REQUEST_CODE_HASH
    # unknown fields of a newer hub are skipped
    extra = (msg.SerializeToString() + L._encode_varint(10 << 3) + b"\x07"
             + L._encode_varint(99 << 3 | 2) + b"\x01A")
    assert L.decode_message(extra, L.INFO_FIELDS)["python"] == "3.12"
    with pytest.raises(ValueError):
        L.decode_message(b"\x1a\x05ab", L.INFO_FIELDS)


def _rsa_cert():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "x" * 60)])     # RSA-2048: multi-byte DER lengths
    import datetime
    now = datetime.datetime.now(datetime.UTC)
    return (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=1)).sign(key, hashes.SHA256()))


def test_spki_fingerprint_matches_security(tmp_path):
    identity = HubIdentity.load_or_create(tmp_path / "hub-identity")
    der = x509.load_pem_x509_certificate(identity.ca_pem).public_bytes(serialization.Encoding.DER)
    assert L.spki_sha256(der) == identity.fingerprint == spki_fingerprint(identity.ca_pem)
    cert = _rsa_cert()
    spki = cert.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    assert L.spki_sha256(cert.public_bytes(serialization.Encoding.DER)) == hashlib.sha256(spki).hexdigest()
    assert L.curl_pin("00" * 32) == "sha256//" + "A" * 43 + "="


def test_token_and_address_parsing(tmp_path):
    token = str(HubIdentity.load_or_create(tmp_path / "a").token)
    f = tmp_path / "tok"
    f.write_text(token + "\n")
    env = {"COLOSSEUM_TOKEN": token}
    assert L.resolve_token(None, None, env) == (token, token.split(".")[2])
    assert L.resolve_token(None, str(f), {"COLOSSEUM_TOKEN": "bad"})[0] == token
    other = str(HubIdentity.load_or_create(tmp_path / "b").token)
    assert L.resolve_token(other, str(f), env)[0] == other
    for bad in ("", "cst1.short.abc", token + "x"):
        with pytest.raises(L.LoaderError) as e:
            L.resolve_token(bad or None, None, {})
        assert e.value.code == 2
    assert L.parse_address("10.0.0.5") == ("10.0.0.5", 47100)
    assert L.parse_address("hub.lan:5000") == ("hub.lan", 5000)
    assert L.parse_address("[::1]:47100") == ("::1", 47100)
    for bad in ("host:x", ":5", "h:70000"):
        with pytest.raises(L.LoaderError):
            L.parse_address(bad)


# --- end to end with exec_fn ------------------------------------------------------------------------------------

def test_join_builds_env_and_execs_the_agent(hub, tmp_path, monkeypatch):
    port = hub.start()
    env, _ = _env(tmp_path)
    code, calls = _run([f"127.0.0.1:{port}", "--workers", "3", "--tags", "gpu,big", "--once"],
                       {**env, "COLOSSEUM_TOKEN": hub.token}, monkeypatch)
    assert code == 0 and len(calls) == 1
    path, args, agent_env = calls[0]
    cache = tmp_path / "cache" / "colosseum"
    bundle_dir = cache / "bundles" / hub.service.bundle.code_hash
    env_dir = Path(path).parents[1]
    assert env_dir.parent == cache / "envs" and (env_dir / L.ENV_MARKER).is_file()
    assert args == [path, "-m", "colosseum.node.agent", f"127.0.0.1:{port}", "--workers", "3", "--tags", "gpu,big",
                    "--once", "--code-hash", hub.service.bundle.code_hash, "--bundle", str(bundle_dir)]
    assert hub.token not in " ".join(args)
    assert agent_env["COLOSSEUM_TOKEN"] == hub.token
    assert agent_env["PYTHONPATH"] == str(bundle_dir / "project")             # the caller's PYTHONPATH is gone
    assert agent_env["PYTHONNOUSERSITE"] == "1"
    assert Path.cwd() == bundle_dir / "project"
    venv_call, pip_call = _uv_calls(tmp_path)
    assert venv_call[:4] == ["venv", "--quiet", "--python", "3.12"]
    assert pip_call[:2] == ["pip", "install"] and pip_call[-2:] == ["--torch-backend", "cpu"]   # no GPU here
    req = (env_dir / "colosseum-requirements.txt").read_text().splitlines()
    assert req == [*join_kit.REQUIREMENTS, "torch==2.14.1"]
    uv_env = [json.loads(line) for line in (tmp_path / "uv.jsonl").read_text().splitlines()
              if json.loads(line)["args"] != ["--version"]][0]["env"]
    assert uv_env["UV_CACHE_DIR"] == str(cache / "uv-cache")
    assert uv_env["UV_PYTHON_INSTALL_DIR"] == str(cache / "python")


def test_second_join_reuses_bundle_and_env(hub, tmp_path, monkeypatch):
    port = hub.start()
    env, _ = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    assert _run([f"127.0.0.1:{port}"], env, monkeypatch)[0] == 0
    first = _uv_calls(tmp_path)
    code, calls = _run([f"127.0.0.1:{port}"], env, monkeypatch)
    assert code == 0 and _uv_calls(tmp_path) == first                           # no uv call the second time
    gpu = T.NvidiaGpu(0, "RTX", 12282, "610.88", "8.9")
    code, calls = _run([f"127.0.0.1:{port}"], env, monkeypatch, gpus=[gpu])    # another build: another env
    assert code == 0 and len(_uv_calls(tmp_path)) == len(first) + 2
    pip = _uv_calls(tmp_path)[-1]
    assert pip[-2:] == ["--torch-backend", "cu130"]
    assert json.loads((tmp_path / "uv.jsonl").read_text().splitlines()[-1])["env"]["UV_CUDA_DRIVER_VERSION"] == "610.88"
    assert len(list((tmp_path / "cache" / "colosseum" / "envs").glob("*/" + L.ENV_MARKER))) == 2


def test_gpu_node_with_cpu_torch_warns(hub, tmp_path, monkeypatch, capsys):
    port = hub.start()
    env, _ = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    gpu = T.NvidiaGpu(0, "RTX", 12282, "610.88", "8.9")
    assert _run([f"127.0.0.1:{port}"], env, monkeypatch, gpus=[gpu])[0] == 0
    assert "CPU build of torch was installed" in capsys.readouterr().err


def test_failed_install_names_the_package_and_removes_the_env(hub, tmp_path, monkeypatch, capsys):
    port = hub.start()
    env, _ = _env(tmp_path, COLOSSEUM_TOKEN=hub.token,
                  FAKE_UV_FAIL="error: Distribution `box2d==2.3.10` can't be installed because it doesn't have a "
                               "wheel for the current platform")
    code, calls = _run([f"127.0.0.1:{port}"], env, monkeypatch)
    err = capsys.readouterr().err
    assert code == 1 and not calls
    assert "box2d==2.3.10" in err and "distributed.requirements" in err
    assert not any((tmp_path / "cache" / "colosseum" / "envs").glob("*/bin"))


def test_wrong_torch_version_in_env_is_refused(hub, tmp_path, monkeypatch, capsys):
    port = hub.start()
    env, _ = _env(tmp_path, COLOSSEUM_TOKEN=hub.token, FAKE_TORCH_JSON=json.dumps(
        {"python": "3.12", "torch": "2.13.0+cpu", "cuda": None, "cuda_available": False, "mps": False}))
    code, _ = _run([f"127.0.0.1:{port}"], env, monkeypatch)
    assert code == 1 and "torch 2.13.0+cpu" in capsys.readouterr().err


def test_node_python_override_skips_the_env(hub, tmp_path, monkeypatch):
    port = hub.start()
    env, py = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    env[L.NODE_PYTHON_ENV] = str(py)
    code, calls = _run([f"127.0.0.1:{port}"], env, monkeypatch)
    assert code == 0 and calls[0][0] == str(py) and _uv_calls(tmp_path) == []


@pytest.mark.parametrize("case, needle", [
    ("fingerprint", "does not have the key of this token"),
    ("secret", "refused the token"),
    ("insecure_hub", "--insecure"),
    ("tls_hub_insecure_node", "drop --insecure"),
    ("no_bootstrap", "colosseum train --listen"),
    ("down", "cannot reach the hub"),
    ("down_insecure", "cannot reach the hub"),
    ("protocol", "this loader is from another Colosseum version"),
])
def test_refusals(hub, tmp_path, monkeypatch, capsys, case, needle):
    env, _ = _env(tmp_path)
    token, argv, closed = hub.token, [], None
    if case == "fingerprint":
        port = hub.start()
        token = str(HubIdentity.load_or_create(tmp_path / "other-identity").token)
    elif case == "secret":
        port = hub.start()
        token = "cst1." + "Z" * 43 + "." + hub.identity.fingerprint
    elif case == "insecure_hub":
        port = hub.start(tls=False)
    elif case == "tls_hub_insecure_node":
        port = hub.start()
        argv = ["--insecure"]
    elif case == "no_bootstrap":                                   # a plain `train` hub: no Bootstrap service
        port = hub.start(tls=False, service=None)
        argv = ["--insecure"]
    elif case == "protocol":
        port = hub.start(service=dataclasses.replace(join_kit.tiny_service(tmp_path), protocol=(2, 0)))
    else:                                   # P-PF4-23: a bound, non-listening socket (ECONNREFUSED, no reuse race)
        closed = socket.socket()
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
        argv = ["--insecure"] if case == "down_insecure" else []
    try:
        code, calls = _run([f"127.0.0.1:{port}", *argv], {**env, "COLOSSEUM_TOKEN": token}, monkeypatch)
    finally:
        if closed is not None:
            closed.close()
    assert code == 1 and not calls
    err = capsys.readouterr().err
    assert needle in err
    if case.startswith("down"):
        assert "WSL2" in err and "drop --insecure" not in err
    if case == "fingerprint":                     # nothing reached the server: the key is checked before any RPC
        assert hub.seen == []
    if case == "protocol":                        # refused at Info: the token secret was never sent
        assert [m for m, _ in hub.seen] == ["/colosseum.v1.Bootstrap/Info"]


def test_prune_keeps_recent_and_in_use(tmp_path):
    root = tmp_path / "envs"
    import os
    for i, name in enumerate(["a", "b", "c", "d"]):
        (root / name).mkdir(parents=True)
        (root / name / L.IN_USE).write_text("")
        os.utime(root / name / L.IN_USE, (100 + i, 100 + i))
    held = L._hold(root / "a" / L.IN_USE, exclusive=False)       # a running node holds the oldest env
    try:
        assert sorted(L.prune(root, 2, root / "d")) == ["b"]
    finally:
        held.close()
    assert sorted(p.name for p in root.iterdir()) == ["a", "c", "d"]


def test_standalone_copy_runs_without_colosseum(hub, tmp_path):
    """The installer's copy: loader.py with bundle.py and torch_select.py, under python -I."""
    port = hub.start()
    boot = tmp_path / "boot"
    boot.mkdir()
    for module in (L, B, T):
        shutil.copy(module.__file__, boot)
    env, py = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    env[L.NODE_PYTHON_ENV] = str(py)
    out = subprocess.run([sys.executable, "-I", str(boot / "loader.py"), f"127.0.0.1:{port}", "--name", "lap"],
                         env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    record = json.loads((tmp_path / "record.jsonl").read_text())
    assert record["argv"][1:4] == ["-m", "colosseum.node.agent", f"127.0.0.1:{port}"]
    assert "--name" in record["argv"] and record["cwd"].endswith(f"{hub.service.bundle.code_hash}/project")


def test_bundle_in_several_parts(hub, tmp_path, monkeypatch):
    """A bundle larger than one blob part (send_parts) arrives whole and verified."""
    port = hub.start(extra_files={"data/noise.bin": os.urandom(3 * (1 << 20))})
    assert hub.service.bundle.size > 2 * (1 << 20)
    env, py = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    env[L.NODE_PYTHON_ENV] = str(py)
    code, calls = _run([f"127.0.0.1:{port}"], env, monkeypatch)
    bundle_dir = tmp_path / "cache" / "colosseum" / "bundles" / hub.service.bundle.code_hash
    assert code == 0 and (bundle_dir / "project" / "data" / "noise.bin").stat().st_size == 3 * (1 << 20)


def test_info_of_the_real_service_decodes(tmp_path):
    service = join_kit.tiny_service(tmp_path)
    info = L.decode_message(service.info().SerializeToString(), L.INFO_FIELDS)
    assert info["code_hash"] == service.bundle.code_hash and info["requirements"] == list(join_kit.REQUIREMENTS)
    assert info["bundle_size"] == service.bundle.size and info["protocol_major"] == PROTOCOL_VERSION[0]


# --- cache locking (T6.1's deferred placement race; prune vs a starting node) -----------------------------------

def _assert_cache_locked(directory: Path) -> None:
    """The cache lock of ``directory`` is held (by someone else's open file description)."""
    with open(B.cache_lock_path(directory), "a+") as f:
        with pytest.raises(BlockingIOError):
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_placement_and_prune_run_under_the_cache_lock(tmp_path, monkeypatch):
    """A stale partial dir is replaced, and old entries are pruned, only under the cache dir's lock: the marker
    check and the rename / removal cannot interleave with another loader's placement or use."""
    project = tmp_path / "proj"
    (project / "examples").mkdir(parents=True)
    (project / "examples" / "game.py").write_text("X = 1\n")
    built = B.build_bundle(project, include=[], exclude=[], max_file_mb=1, out_dir=tmp_path / "out", roots=[])
    cache = tmp_path / "cache"
    root = B.bundles_dir(cache)
    (root / built.code_hash / "project").mkdir(parents=True)          # stale: no completion marker
    renames, removals = [], []
    real_rename, real_rmtree = os.rename, shutil.rmtree

    def rename(src, dst):
        if Path(dst) == root / built.code_hash:
            _assert_cache_locked(root)
            renames.append(dst)
        return real_rename(src, dst)

    def rmtree(path, *a, **kw):
        if Path(path).parent == root and not Path(path).name.startswith("."):
            _assert_cache_locked(root)
            removals.append(Path(path).name)
        return real_rmtree(path, *a, **kw)

    monkeypatch.setattr(B.os, "rename", rename)
    target = B.unpack_bundle(built.path, built.code_hash, cache_root=cache)
    assert renames and (target / B.COMPLETE_MARKER).is_file()
    for i, name in enumerate(["old1", "old2"]):
        (root / name).mkdir()
        (root / name / L.IN_USE).write_text("")
        os.utime(root / name / L.IN_USE, (i, i))
    monkeypatch.setattr(L.shutil, "rmtree", rmtree)
    held = L._use(target, B.COMPLETE_MARKER)
    try:
        assert sorted(L.prune(root, 1, target)) == ["old1", "old2"] == sorted(removals)
    finally:
        held.close()
    assert sorted(p.name for p in root.iterdir()) == [built.code_hash]


def test_use_refuses_an_incomplete_entry(tmp_path):
    entry = tmp_path / "envs" / "k"
    entry.mkdir(parents=True)
    assert L._use(entry, L.ENV_MARKER) is None and not (entry / L.IN_USE).exists()
    (entry / L.ENV_MARKER).write_text("{}")
    lock = L._use(entry, L.ENV_MARKER)
    try:
        assert lock is not None and L.prune(entry.parent, 0, tmp_path / "none") == []   # in use: kept
    finally:
        lock.close()
    assert L.prune(entry.parent, 0, tmp_path / "none") == ["k"]


def test_loader_rechecks_a_bundle_pruned_before_it_is_in_use(hub, tmp_path, monkeypatch):
    """Another loader's prune may remove the bundle between the check and the in-use lock: the loader fetches it
    again instead of starting the agent on a missing directory."""
    port = hub.start()
    env, py = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    env[L.NODE_PYTHON_ENV] = str(py)
    real, seen = L.ensure_bundle, []

    def racing(*args, **kwargs):
        target = real(*args, **kwargs)
        seen.append(target)
        if len(seen) == 1:
            shutil.rmtree(target)
        return target

    monkeypatch.setattr(L, "ensure_bundle", racing)
    code, calls = _run([f"127.0.0.1:{port}"], env, monkeypatch)
    assert code == 0 and len(calls) == 1 and len(seen) == 2
    assert (seen[1] / B.COMPLETE_MARKER).is_file() and (seen[1] / L.IN_USE).is_file()


def test_loader_module_is_standalone():
    """loader.py is embedded by the installer: module-level imports are stdlib only (grpc inside functions), and it
    loads by file path under python -I without colosseum, torch or grpc."""
    for node in ast.parse(Path(L.__file__).read_text()).body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                top = name.split(".")[0]
                assert top == "__future__" or top in sys.stdlib_module_names, name
    out = subprocess.run([sys.executable, "-I", "-c", "import importlib.util, sys; "
                          "s = importlib.util.spec_from_file_location('loader', sys.argv[1]); m = importlib.util."
                          "module_from_spec(s); sys.modules['loader'] = m; s.loader.exec_module(m); "
                          "print(*(n in sys.modules for n in ('torch', 'colosseum', 'grpc')))", L.__file__],
                         capture_output=True, text=True, timeout=60)
    assert out.stdout.split() == ["False", "False", "False"], out.stderr


# --- review round 1 ---------------------------------------------------------------------------------------------

def test_reexec_closes_the_inherited_in_use_locks(hub, tmp_path, monkeypatch):
    """The agent inherits the .in-use locks through exec; when it re-execs the loader (new code hash, T6.3), the
    loader closes them first, so the old bundle becomes prunable (a new descriptor of the same process could not
    take the exclusive prune lock against them)."""
    first_port = hub.start()
    first_hash = hub.service.bundle.code_hash
    env, py = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    env[L.NODE_PYTHON_ENV] = str(py)
    monkeypatch.setattr(T, "probe_nvidia", lambda **kw: [])
    inherited = []

    def exec_keeping_locks(path, args, agent_env):          # what exec does: the descriptions stay locked
        entries = []
        for entry in agent_env[L.IN_USE_FDS_ENV].split(","):
            fd, dev, ino = entry.split(":")
            inherited.append(os.dup(int(fd)))                  # same open file: same device and inode
            entries.append(f"{inherited[-1]}:{dev}:{ino}")
        agent_env[L.IN_USE_FDS_ENV] = ",".join(entries)
        exec_keeping_locks.env = agent_env

    assert L.loader_main([f"127.0.0.1:{first_port}"], environ=env, exec_fn=exec_keeping_locks) == 0
    root = B.bundles_dir(tmp_path / "cache")
    assert L.prune(root, 0, root / "none") == []                            # held by the "agent"
    second_port = hub.start(extra_files={"examples/new.py": b"Y = 2\n"})
    assert hub.service.bundle.code_hash != first_hash
    calls = []
    agent_env = {**env, **{k: v for k, v in exec_keeping_locks.env.items() if k == L.IN_USE_FDS_ENV}}
    code = L.loader_main([f"127.0.0.1:{second_port}"], environ=agent_env,
                         exec_fn=lambda p, a, e: calls.append(e))
    assert code == 0 and calls
    for fd in inherited:                                                   # closed by the re-exec'd loader
        with pytest.raises(OSError):
            os.fstat(fd)
    assert first_hash in L.prune(root, 0, root / "none")


def test_close_inherited_locks_checks_the_file_and_ignores_bad_entries(tmp_path):
    """Only a listed descriptor that still refers to the recorded file (device and inode) is closed: a stale entry
    in a descendant's environment never closes an unrelated descriptor; malformed entries are ignored."""
    lock = open(tmp_path / "a", "w")
    other = open(tmp_path / "b", "w")
    try:
        st = os.fstat(lock.fileno())
        reused = f"{other.fileno()}:{st.st_dev}:{st.st_ino}"            # the number now names another file
        L.close_inherited_locks(",".join([
            "x", "", "0:1:2", "1:1:1", f"{lock.fileno()}", "\u00b2:1:1", f"{lock.fileno()}:\u00b2:1", "-3:1:1",
            f"{lock.fileno()}:{st.st_dev}", "999999:1:1", reused]))
        os.fstat(lock.fileno())                                         # nothing closed so far
        os.fstat(other.fileno())
        L.close_inherited_locks(f"{lock.fileno()}:{st.st_dev}:{st.st_ino}")
        with pytest.raises(OSError):
            os.fstat(lock.fileno())
        os.fstat(other.fileno())
    finally:
        other.close()
        with contextlib.suppress(OSError):
            lock.close()


def test_token_reaches_only_the_agent(hub, tmp_path, monkeypatch):
    """No third-party subprocess (uv, its installer, nvidia-smi, the torch check) sees COLOSSEUM_TOKEN."""
    port = hub.start()
    env, _ = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    seen, real = [], subprocess.run

    def run(cmd, *a, **kw):
        seen.append((cmd, kw.get("env")))
        return real(cmd, *a, **kw)

    monkeypatch.setattr(L.subprocess, "run", run)
    probes = []
    monkeypatch.setattr(T, "probe_nvidia", lambda **kw: probes.append(kw["environ"]) or [])
    monkeypatch.setattr(T, "probe_platform", lambda: T.NodePlatform("Linux", "x86_64"))
    calls = []
    assert L.loader_main([f"127.0.0.1:{port}"], environ=env, exec_fn=lambda p, a, e: calls.append(e)) == 0
    assert len(seen) >= 4 and all(e is not None and L.TOKEN_ENV not in e for _, e in seen), seen
    assert probes and L.TOKEN_ENV not in probes[0]
    assert calls[0][L.TOKEN_ENV] == hub.token


def test_prune_treats_a_fresh_entry_as_recent(tmp_path):
    """An entry without .in-use yet (just built / unpacked) sorts by its own mtime, not as the oldest."""
    root = tmp_path / "envs"
    (root / "old").mkdir(parents=True)
    (root / "old" / L.IN_USE).write_text("")
    os.utime(root / "old" / L.IN_USE, (100, 100))
    (root / "fresh").mkdir()
    assert L.prune(root, 1, root / "none") == ["old"]


def test_unexpected_wire_type_is_a_value_error():
    with pytest.raises(ValueError, match="wire type"):
        L.decode_message(b"\x18\x05", L.INFO_FIELDS)                       # code_hash (3) as a varint
    with pytest.raises(ValueError, match="wire type"):
        L.decode_message(b"\x0a\x01x", L.INFO_FIELDS)                      # protocol_major (1) as bytes
    assert L.PROTOCOL_MAJOR == PROTOCOL_VERSION[0]


def test_os_errors_end_with_a_message(hub, tmp_path, monkeypatch, capsys):
    port = hub.start()
    env, py = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    env[L.NODE_PYTHON_ENV] = str(py)

    def failing_exec(path, args, agent_env):
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(T, "probe_nvidia", lambda **kw: [])
    assert L.loader_main([f"127.0.0.1:{port}"], environ=env, exec_fn=failing_exec) == 1
    assert "colosseum join: failed: [Errno 13] Permission denied" in capsys.readouterr().err
