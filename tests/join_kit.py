"""Support for the join tests (SP5 T6.2, T6.3): fake ``uv`` and fake environment ``python`` scripts, a tiny
``BootstrapService``, a real Bootstrap gRPC server on loopback (``AuthInterceptor`` + ``add_bootstrap``).

The fakes never touch the network or the machine: ``uv venv DIR`` writes ``DIR/bin/python`` (the fake python);
``uv pip install`` records its arguments and fails with ``FAKE_UV_FAIL``'s text when set; ``uv run ... -- python
ARGS`` executes the real interpreter of the test run with ARGS (so the installer really runs the loader); ``uv
--version`` prints 0.13.1. The fake python answers the loader's torch check with ``FAKE_TORCH_JSON`` (default: a CPU
torch 2.14.1) and otherwise records ``argv`` / ``cwd`` / ``env`` as one JSON line in ``FAKE_RECORD`` and exits 0:
that is the "node agent" the loader execs.
"""
from __future__ import annotations

import stat
import sys
from concurrent import futures
from pathlib import Path

import grpc

from colosseum.bootstrap.bundle import build_bundle
from colosseum.bootstrap.installer import BootstrapService, installer_script
from colosseum.hub.server import add_bootstrap
from colosseum.net.grpc_util import AuthInterceptor, server_options
from colosseum.net.messages import PROTOCOL_VERSION

REQUIREMENTS = ("numpy==2.5.4", "lz4==4.4.5")


FAKE_PYTHON = '''#!{python}
"""Fake env python: the torch check prints FAKE_TORCH_JSON; anything else records argv / cwd / env and exits."""
import json, os, sys
if "-c" in sys.argv:
    print(os.environ.get("FAKE_TORCH_JSON", json.dumps({{"python": "3.12", "torch": "2.14.1+cpu", "cuda": None,
          "cuda_available": False, "mps": False}})))
    raise SystemExit(0)
with open(os.environ["FAKE_RECORD"], "a") as f:
    f.write(json.dumps({{"argv": sys.argv, "cwd": os.getcwd(), "env": dict(os.environ)}}) + "\\n")
'''

FAKE_UV = '''#!{python}
"""Fake uv: --version; venv DIR (a fake python); pip install (records, may fail); run ... -- python ARGS (real)."""
import json, os, sys
args = sys.argv[1:]
log = os.environ.get("FAKE_UV_LOG")
if log:
    with open(log, "a") as f:
        uv_env = {{k: v for k, v in os.environ.items() if k.startswith("UV_")}}
        f.write(json.dumps({{"args": args, "env": uv_env}}) + "\\n")
if args == ["--version"]:
    print("uv 0.13.1 (fake)"); raise SystemExit(0)
if args[0] == "venv":
    target = args[-1]
    os.makedirs(os.path.join(target, "bin"), exist_ok=True)
    py = os.path.join(target, "bin", "python")
    with open(py, "w") as f:
        f.write({fake_python!r})
    os.chmod(py, 0o755)
    raise SystemExit(0)
if args[:2] == ["pip", "install"]:
    fail = os.environ.get("FAKE_UV_FAIL")
    if fail:
        print(fail, file=sys.stderr); raise SystemExit(1)
    raise SystemExit(0)
if args[0] == "run":
    rest = args[args.index("--") + 1:]
    assert rest[0] == "python"
    os.execv({python!r}, [{python!r}, *rest[1:]])
raise SystemExit(f"fake uv: unexpected {{args}}")
'''


def write_fakes(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    fake_python = FAKE_PYTHON.format(python=sys.executable)
    uv = directory / "uv"
    uv.write_text(FAKE_UV.format(python=sys.executable, fake_python=fake_python))
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    py = directory / "python"
    py.write_text(fake_python)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    return uv, py


def tiny_service(tmp_path: Path, *, insecure: bool = False, torch_build: str = "cu130",
                 extra_files: dict[str, bytes] | None = None) -> BootstrapService:
    """A BootstrapService over a two-file project (no import roots), the hub's real installer script."""
    project = tmp_path / "hub-project"
    (project / "examples").mkdir(parents=True, exist_ok=True)
    (project / "examples" / "game.py").write_text("X = 1\n")
    for rel, data in (extra_files or {}).items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_bytes(data)
    bundle = build_bundle(project, include=[], exclude=[], max_file_mb=10, out_dir=tmp_path / "hub-bundles", roots=[])
    script = installer_script(insecure=insecure, python="3.12", grpcio="1.84.0")
    return BootstrapService(bundle, REQUIREMENTS, "r" * 64, "3.12", "2.14.1", torch_build, script,
                            PROTOCOL_VERSION, bundle.path.read_bytes())


class _Recorder(grpc.ServerInterceptor):
    def __init__(self, seen: list) -> None:
        self._seen = seen

    def intercept_service(self, continuation, handler_call_details):
        self._seen.append((handler_call_details.method, dict(handler_call_details.invocation_metadata)))
        return continuation(handler_call_details)


def bootstrap_server(service: BootstrapService | None, identity, *, tls: bool = True,
                     seen: list | None = None) -> tuple[grpc.Server, int]:
    """A started gRPC server on 127.0.0.1:<free port> with the hub's AuthInterceptor and (unless ``service`` is
    None) the Bootstrap service; TLS with the hub identity's certificate, or plain. ``seen`` receives
    ``(method, metadata)`` of every RPC that reaches the server."""
    interceptors = [AuthInterceptor(lambda: identity.token.secret, identity.nodes)]
    if seen is not None:
        interceptors.insert(0, _Recorder(seen))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=server_options(64),
                         interceptors=interceptors)
    if service is not None:
        add_bootstrap(server, service)
    if tls:
        port = server.add_secure_port("127.0.0.1:0", identity.server_credentials())
    else:
        port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port
