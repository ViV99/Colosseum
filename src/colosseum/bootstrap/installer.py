"""The installer script and the join commands a hub prints (SP5 spec 2.2).

``installer_script`` is what ``Bootstrap/Script`` serves (raw bytes, R2): a POSIX ``sh`` script whose whole body is
the function ``_colosseum_install``, called by its last line (``{ _colosseum_install "$@"; } #
colosseum-installer-end``: the braces make a cut inside the call a syntax error, never a call without the join
flags), so a script cut anywhere does nothing. It writes the embedded ``loader.py``, ``bundle.py`` and
``torch_select.py`` into ``<cache>/colosseum/loader/<id>/`` (a temp name, then ``mv``), finds or installs uv
(``COLOSSEUM_UV``, the private ``<cache>/colosseum/uv/uv``, ``uv`` on PATH, else the pinned astral.sh installer with
``UV_UNMANAGED_INSTALL``) and runs the loader with the hub's Python and grpcio through ``uv run``. The hub address
and the token come from the environment (``COLOSSEUM_HUB``, ``COLOSSEUM_TOKEN``) set by the one-liner, extra
arguments are ``join`` flags (``sh -s -- --workers 4``), so one script serves every address of the hub and the SSH
mode (T8.1).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from colosseum.bootstrap import bundle as _bundle
from colosseum.bootstrap import loader as _loader
from colosseum.bootstrap import torch_select as _torch_select

END_MARKER = "# colosseum-installer-end"
SCRIPT_PATH = "/colosseum.v1.Bootstrap/Script"
EMBEDDED = (_loader, _bundle, _torch_select)
KEEP_HUB_BUNDLES = 5

_TEMPLATE = r"""#!/bin/sh
# Colosseum node installer, served by the hub (Bootstrap/Script). Run the command the hub printed:
#   ... | COLOSSEUM_HUB=host:port COLOSSEUM_TOKEN=cst1... sh [-s -- join flags]
_colosseum_install() {
    set -eu
    hub="${COLOSSEUM_HUB:-}"
    if [ -z "$hub" ]; then
        echo "colosseum: COLOSSEUM_HUB is not set (copy the whole command the hub printed)" >&2; return 2
    fi
    if [ -z "${COLOSSEUM_TOKEN:-}" ]; then
        echo "colosseum: COLOSSEUM_TOKEN is not set (copy the whole command the hub printed)" >&2; return 2
    fi
    case "$(uname -s)" in
        Linux|Darwin) ;;
        *) echo "colosseum: $(uname -s) is not supported (Linux, macOS, or Windows via WSL2)" >&2; return 1 ;;
    esac
    cache="${XDG_CACHE_HOME:-$HOME/.cache}/colosseum"
    boot="$cache/loader/@ID@"
    mkdir -p "$boot"
@FILES@
    uv="${COLOSSEUM_UV:-}"
    if [ -z "$uv" ] && [ -x "$cache/uv/uv" ]; then uv="$cache/uv/uv"; fi
    if [ -z "$uv" ] && command -v uv >/dev/null 2>&1; then uv="$(command -v uv)"; fi
    if [ -z "$uv" ]; then
        echo "colosseum: installing uv @UV@ into $cache/uv" >&2
        if command -v curl >/dev/null 2>&1; then
            curl -LsSf "https://astral.sh/uv/@UV@/install.sh" \
                | env -u UV_INSTALL_DIR UV_UNMANAGED_INSTALL="$cache/uv" UV_NO_MODIFY_PATH=1 sh
        elif command -v wget >/dev/null 2>&1; then
            wget -qO- "https://astral.sh/uv/@UV@/install.sh" \
                | env -u UV_INSTALL_DIR UV_UNMANAGED_INSTALL="$cache/uv" UV_NO_MODIFY_PATH=1 sh
        else
            echo "colosseum: neither curl nor wget is available to install uv" >&2; return 1
        fi
        uv="$cache/uv/uv"
    fi
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$cache/uv-cache}"
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$cache/python}"
    unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV
    exec "$uv" run --no-project --python @PYTHON@ --with "grpcio==@GRPCIO@" -- \
        python -I "$boot/loader.py" --uv "$uv" @INSECURE@"$hub" "$@"
}
{ _colosseum_install "$@"; } @END@
"""


def _embedded_files() -> list[tuple[str, str]]:
    return [(Path(m.__file__).name, Path(m.__file__).read_text(encoding="utf-8")) for m in EMBEDDED]


def installer_script(*, insecure: bool, python: str, grpcio: str, uv_version: str = _loader.UV_VERSION) -> bytes:
    """The installer script for a hub with Python ``python`` (major.minor) and grpcio ``grpcio``."""
    files = _embedded_files()
    ident = hashlib.sha256("\0".join(f"{n}\0{t}" for n, t in files).encode()).hexdigest()[:16]
    blocks = []
    for name, text in files:
        tag = "COLOSSEUM_EOF_" + hashlib.sha256(text.encode()).hexdigest()[:12].upper()
        if tag in text or not text.endswith("\n"):
            raise ValueError(f"{name} cannot be embedded in the installer (heredoc tag inside, or no final newline)")
        # a temp name + mv: another installer may be running python on the same (content-addressed) copy
        blocks.append(f"    cat > \"$boot/{name}.$$\" <<'{tag}'\n{text}{tag}\n"
                      f"    mv -f \"$boot/{name}.$$\" \"$boot/{name}\"\n")
    script = (_TEMPLATE.replace("@ID@", ident).replace("@UV@", uv_version).replace("@PYTHON@", python)
              .replace("@GRPCIO@", grpcio).replace("@INSECURE@", "--insecure " if insecure else "")
              .replace("@END@", END_MARKER)
              .replace("@FILES@", "".join(blocks).rstrip("\n")))     # last: the embedded text is never substituted
    return script.encode()


def _sq(text: str) -> str:
    """Single-quoted for sh; the values (address, pin, token) never contain a quote (checked)."""
    if "'" in text:
        raise ValueError(f"cannot quote {text!r}")
    return f"'{text}'"


curl_pin = _loader.curl_pin          # curl's --pinnedpubkey value of a token's fingerprint


def one_liner(address: str, token: str, *, insecure: bool) -> str:
    """The spec 2.2 command for a machine without colosseum (``address`` = host:port as the node reaches it)."""
    fingerprint = str(token).split(".")[2]
    if insecure:
        transport, url = "--http2-prior-knowledge", f"http://{address}{SCRIPT_PATH}"
    else:
        transport, url = f"--http2 -k --pinnedpubkey {_sq(curl_pin(fingerprint))}", f"https://{address}{SCRIPT_PATH}"
    # -g: no curl URL globbing (a bracketed IPv6 address); the URL quoted (no shell globbing either)
    return (f"printf '\\0\\0\\0\\0\\0' | curl -fsSg {transport} -H 'content-type: application/grpc' "
            f"-H 'te: trailers' --data-binary @- {_sq(url)} | tail -c +6 | "
            f"COLOSSEUM_HUB={_sq(address)} COLOSSEUM_TOKEN={_sq(str(token))} sh")


def join_command(address: str, token: str, *, insecure: bool) -> str:
    return f"colosseum join {_sq(address)} --token {_sq(str(token))}" + (" --insecure" if insecure else "")


# ---------------------------------------------------------------------------------------------------------------
# what the hub serves (built once by the hub's main thread at start; immutable)
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapService:
    bundle: _bundle.Bundle
    requirements: tuple[str, ...]
    requirements_hash: str
    python: str                   # "3.12"
    torch_version: str            # "2.14.1"
    torch_build: str              # "cu130"
    script: bytes
    protocol: tuple[int, int]
    archive: bytes                # the bundle's bytes (served from memory: pruning the file never breaks a hub)

    def info(self):
        from colosseum.net.proto import colosseum_v1_pb2 as pb

        return pb.BootstrapInfo(protocol_major=self.protocol[0], protocol_minor=self.protocol[1],
                                code_hash=self.bundle.code_hash, requirements_hash=self.requirements_hash,
                                requirements=list(self.requirements), python=self.python,
                                torch_version=self.torch_version, torch_build=self.torch_build,
                                bundle_size=self.bundle.size)


def make_bootstrap_service(*, project_dir: Path, include: Sequence[str], exclude: Sequence[str],
                           max_file_mb: float, requirements: Path | None, run_dir: Path, insecure: bool,
                           out_dir: Path, protocol: tuple[int, int]) -> BootstrapService:
    """Bundle the project, freeze the requirements, write the installer (raises BundleError / ValueError /
    OSError; the hub turns them into a ConfigError)."""
    import importlib.metadata
    import sys

    from colosseum.bootstrap.requirements import node_requirements
    from colosseum.net.security import cache_home, data_home
    from colosseum.node.weights import shm_dir

    project = Path(project_dir).resolve()
    extra = []
    # Never shipped when inside the project: the run dir (unless runs/ already leaves it out), and (P-PF4-24) the
    # hub and node identities (CA key, token secret, node keys), the colosseum cache and the shared-memory weight
    # segments.
    for path in (run_dir, data_home() / "colosseum", cache_home() / "colosseum", shm_dir()):
        try:
            rel = Path(path).resolve().relative_to(project).as_posix()
        except ValueError:
            continue
        if rel not in (".", "runs") and not rel.startswith("runs/"):
            extra.append(rel)
    bundle = _bundle.build_bundle(project, include=include, exclude=[*exclude, *extra], max_file_mb=max_file_mb,
                                  out_dir=out_dir)
    reqs, reqs_hash = node_requirements(override=requirements)
    torch_version, torch_build = _torch_select.torch_identity()
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    script = installer_script(insecure=insecure, python=python, grpcio=importlib.metadata.version("grpcio"))
    archive = bundle.path.read_bytes()
    prune_archives(Path(out_dir), keep=KEEP_HUB_BUNDLES, current=bundle.path)
    return BootstrapService(bundle, tuple(reqs), reqs_hash, python, torch_version, torch_build, script, protocol,
                            archive)


def prune_archives(out_dir: Path, *, keep: int, current: Path) -> None:
    """Keep the ``keep`` newest hub archives (by mtime) and ``current``."""
    archives = sorted(out_dir.glob("*" + _bundle.ARCHIVE_SUFFIX), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in archives[keep:]:
        if path != current:
            path.unlink(missing_ok=True)
