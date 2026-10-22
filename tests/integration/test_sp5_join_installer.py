"""The installer script and the curl one-liner (SP5 T6.2, spec 2.2; R2): end marker, truncation safety, embedded
files, the uv / loader chain with a fake uv, and the real curl against a real gRPC Bootstrap/Script."""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

import join_kit
from colosseum.bootstrap import bundle as B
from colosseum.bootstrap import installer as I
from colosseum.bootstrap import loader as L
from colosseum.bootstrap import torch_select as T
from colosseum.net.security import HubIdentity

SHELLS = [s for s in ("sh", "dash", "bash") if shutil.which(s)]


def _script(insecure=False) -> bytes:
    return I.installer_script(insecure=insecure, python="3.12", grpcio="1.84.0")


def test_script_shape_and_embedded_files(tmp_path):
    text = _script().decode()
    assert text.startswith("#!/bin/sh\n")
    assert text.endswith('{ _colosseum_install "$@"; } # colosseum-installer-end\n')
    assert text.count("_colosseum_install") == 2                         # the definition and the one call
    assert 'run --no-project --python 3.12 --with "grpcio==1.84.0" --' in text
    assert "https://astral.sh/uv/0.13.1/install.sh" in text and "UV_UNMANAGED_INSTALL" in text
    def tail(script: str) -> str:
        return script.rsplit('exec "$uv" run', 1)[1]
    assert "--insecure" not in tail(text) and "--insecure " in tail(_script(True).decode())
    assert _script() == _script()                                          # deterministic (served bytes)
    for module in (L, B, T):                                               # embedded verbatim (no substitution)
        assert Path(module.__file__).read_text() in text


def _env(tmp_path, port, token, **extra):
    uv, py = join_kit.write_fakes(tmp_path / "fakes")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home"), "XDG_CACHE_HOME": str(tmp_path / "cache"),
           "FAKE_RECORD": str(tmp_path / "record.jsonl"), "COLOSSEUM_UV": str(uv),
           "COLOSSEUM_HUB": f"127.0.0.1:{port}", "COLOSSEUM_TOKEN": token, L.NODE_PYTHON_ENV: str(py)}
    env.update(extra)
    return env


@pytest.fixture
def hub(tmp_path):
    identity = HubIdentity.load_or_create(tmp_path / "hub-identity")
    servers = []

    def start(tls=True):
        server, port = join_kit.bootstrap_server(join_kit.tiny_service(tmp_path, insecure=not tls), identity, tls=tls)
        servers.append(server)
        return port
    yield type("Hub", (), {"identity": identity, "token": str(identity.token), "start": staticmethod(start)})
    for s in servers:
        s.stop(0)


@pytest.mark.parametrize("shell", SHELLS)
def test_script_runs_the_loader_and_passes_join_flags(hub, tmp_path, shell):
    port = hub.start()
    env = _env(tmp_path, port, hub.token)
    out = subprocess.run([shell, "-s", "--", "--workers", "2", "--once"], input=_script(), env=env,
                         capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr.decode()
    record = json.loads((tmp_path / "record.jsonl").read_text())
    assert record["argv"][3:7] == [f"127.0.0.1:{port}", "--workers", "2", "--once"]
    boot = next((tmp_path / "cache" / "colosseum" / "loader").iterdir())
    for module in (L, B, T):
        assert (boot / Path(module.__file__).name).read_bytes() == Path(module.__file__).read_bytes()
    assert not [p.name for p in boot.iterdir() if ".py." in p.name]          # no temp copies left


LAST_LINE = b'{ _colosseum_install "$@"; } # colosseum-installer-end\n'
IN_CALL = {"in-name": b"{ _colosseum_ins", "after-name": b"{ _colosseum_install",
           "after-name-space": b"{ _colosseum_install ", "after-args": b'{ _colosseum_install "$@"',
           "after-semicolon": b'{ _colosseum_install "$@";'}


@pytest.mark.parametrize("cut", [0.05, 0.3, 0.5, 0.8, 0.99, "before-call", *IN_CALL])
def test_truncated_script_does_nothing(tmp_path, cut):
    """Every cut before the end of the call runs nothing: the function body is parsed whole first, and the braces
    make a cut call a syntax error (never a call without the join flags)."""
    script = _script()
    if cut in IN_CALL:
        end = len(script) - len(LAST_LINE) + len(IN_CALL[cut])
    elif cut == "before-call":
        end = len(script) - len(LAST_LINE)
    else:
        end = int(len(script) * cut)
    env = _env(tmp_path, 1, "x")
    for shell in SHELLS:
        subprocess.run([shell], input=script[:end], env=env, capture_output=True, timeout=30)
    assert not (tmp_path / "cache").exists() and not (tmp_path / "record.jsonl").exists()


def test_missing_hub_or_token_is_explained(tmp_path):
    env = _env(tmp_path, 1, "x")
    del env["COLOSSEUM_HUB"]
    out = subprocess.run(["sh"], input=_script(), env=env, capture_output=True, timeout=30)
    assert out.returncode == 2 and b"COLOSSEUM_HUB is not set" in out.stderr


def _curl_http2() -> bool:
    exe = shutil.which("curl")
    return bool(exe) and "HTTP2" in subprocess.run([exe, "--version"], capture_output=True, text=True).stdout


@pytest.mark.skipif(not _curl_http2(), reason="needs curl with HTTP/2")
@pytest.mark.parametrize("tls", [True, False])
def test_one_liner_end_to_end_with_real_curl(hub, tmp_path, tls):
    port = hub.start(tls=tls)
    address = f"127.0.0.1:{port}"
    line = I.one_liner(address, hub.token, insecure=not tls)
    env = _env(tmp_path, port, "unused")
    del env["COLOSSEUM_HUB"], env["COLOSSEUM_TOKEN"]                      # the one-liner sets both
    out = subprocess.run(["sh", "-c", line], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    record = json.loads((tmp_path / "record.jsonl").read_text())
    assert record["argv"][3] == address and record["env"]["COLOSSEUM_TOKEN"] == hub.token
    assert ("--insecure" in record["argv"]) == (not tls)


@pytest.mark.skipif(not _curl_http2(), reason="needs curl with HTTP/2")
def test_one_liner_with_a_wrong_pin_runs_nothing(hub, tmp_path):
    port = hub.start()
    other = HubIdentity.load_or_create(tmp_path / "other-identity")
    token = f"cst1.{hub.identity.token.secret}.{other.fingerprint}"      # pins a key the hub does not have
    line = I.one_liner(f"127.0.0.1:{port}", token, insecure=False)
    out = subprocess.run(["sh", "-c", line], env=_env(tmp_path, port, token), capture_output=True, text=True,
                         timeout=60)
    assert "(90)" in out.stderr                                          # curl: public key does not match
    assert not (tmp_path / "record.jsonl").exists() and not (tmp_path / "cache").exists()


def test_one_liner_and_join_command_text():
    token = "cst1." + "A" * 43 + "." + "0" * 64
    line = I.one_liner("10.0.0.5:47100", token, insecure=False)
    assert line == ("printf '\\0\\0\\0\\0\\0' | curl -fsSg --http2 -k --pinnedpubkey 'sha256//" + "A" * 43 + "=' "
                    "-H 'content-type: application/grpc' -H 'te: trailers' --data-binary @- "
                    "'https://10.0.0.5:47100/colosseum.v1.Bootstrap/Script' | tail -c +6 | "
                    f"COLOSSEUM_HUB='10.0.0.5:47100' COLOSSEUM_TOKEN='{token}' sh")
    assert I.join_command("10.0.0.5:47100", token, insecure=True) == \
        f"colosseum join '10.0.0.5:47100' --token '{token}' --insecure"
    # a bracketed IPv6 address is neither a shell glob nor a curl glob
    v6 = I.one_liner("[::1]:47100", token, insecure=True)
    assert " -fsSg " in v6 and "'http://[::1]:47100/colosseum.v1.Bootstrap/Script'" in v6
    assert I.join_command("[::1]:47100", token, insecure=False) == f"colosseum join '[::1]:47100' --token '{token}'"
    assert shlex.split(I.join_command("[::1]:47100", token, insecure=False))[2] == "[::1]:47100"


FAKE_CURL = """#!/bin/sh
# Fake curl: serves a fake uv install.sh that records its environment and installs the fake uv (no network).
echo "$*" >> "$FAKE_CURL_LOG"
cat <<'FAKE_INSTALL_SH'
mkdir -p "$UV_UNMANAGED_INSTALL"
env | grep '^UV_' > "$FAKE_INSTALL_ENV"
cp "$FAKE_UV_SOURCE" "$UV_UNMANAGED_INSTALL/uv"
FAKE_INSTALL_SH
"""


def _fake_curl(tmp_path) -> tuple[Path, dict[str, str]]:
    uv, _ = join_kit.write_fakes(tmp_path / "fakes")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    (fake_bin / "curl").write_text(FAKE_CURL)
    (fake_bin / "curl").chmod(0o755)
    assert shutil.which("uv", path="/usr/bin:/bin") is None
    return fake_bin, {"PATH": f"{fake_bin}:/usr/bin:/bin", "FAKE_CURL_LOG": str(tmp_path / "curl.log"),
                      "FAKE_INSTALL_ENV": str(tmp_path / "install-env"), "FAKE_UV_SOURCE": str(uv),
                      "UV_INSTALL_DIR": "/nowhere"}


def test_script_installs_a_private_uv_when_none_is_usable(hub, tmp_path):
    """No uv anywhere: the installer fetches the pinned astral.sh script (fake curl) and runs it with
    UV_UNMANAGED_INSTALL (no profile edits, no receipt: never UV_INSTALL_DIR), then runs the loader with it."""
    port = hub.start()
    _, extra = _fake_curl(tmp_path)
    env = _env(tmp_path, port, hub.token, **extra)
    del env["COLOSSEUM_UV"]
    out = subprocess.run(["sh"], input=_script(), env=env, capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr.decode()
    private = tmp_path / "cache" / "colosseum" / "uv"
    assert "https://astral.sh/uv/0.13.1/install.sh" in (tmp_path / "curl.log").read_text()
    installed = dict(line.split("=", 1) for line in (tmp_path / "install-env").read_text().splitlines())
    assert installed["UV_UNMANAGED_INSTALL"] == str(private) and "UV_INSTALL_DIR" not in installed
    assert (private / "uv").is_file() and (tmp_path / "record.jsonl").is_file()


def test_loader_installs_a_private_uv_when_none_is_usable(tmp_path):
    _, extra = _fake_curl(tmp_path)
    env = {**extra, "HOME": str(tmp_path / "home"), "COLOSSEUM_TOKEN": "cst1.secret"}
    cache = tmp_path / "cache"
    found = L.ensure_uv(None, env, cache)
    private = cache / "colosseum" / "uv"
    assert found == str(private / "uv") and Path(found).is_file()
    assert "https://astral.sh/uv/0.13.1/install.sh" in (tmp_path / "curl.log").read_text()
    installed = dict(line.split("=", 1) for line in (tmp_path / "install-env").read_text().splitlines())
    assert installed["UV_UNMANAGED_INSTALL"] == str(private) and "UV_INSTALL_DIR" not in installed
