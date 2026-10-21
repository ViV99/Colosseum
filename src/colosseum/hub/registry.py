"""Where a hub can be found (SP5 spec 1, 3.3): ``runs/<name>/hub.json`` (written at start, kept after the stop) and
the machine's running-hubs registry ``<hub identity>/running/<pid>-<run_id>.json`` (one entry per hub, so several hubs
of one process, e.g. in-process test clusters, never share one (ruling R25); removed at the stop; entries of dead
pids are removed by ``running_hubs``). ``addresses`` are the hub port's listen addresses (``host:port``, TLS unless
``tls`` is False); ``loopback`` is the address a client on this machine uses: ``train``'s plain loopback port, or
``127.0.0.1:<port>`` of a ``hub`` listening on all interfaces (TLS like ``addresses``), None when the hub listens only
on another address."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from colosseum.core.run_state import RunState, process_alive, process_start_ticks
from colosseum.utils.fs import write_text_atomic

HUB_JSON = "hub.json"
RUNNING_DIR = "running"
_SAFE_RUN_ID = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")


@dataclass(frozen=True)
class HubInfo:
    run_id: str
    run_dir: str
    pid: int
    proc_start: int | None
    addresses: tuple[str, ...]
    loopback: str | None
    tls: bool
    ca_fingerprint: str
    identity_dir: str
    mode: str                      # "train" | "hub" | "idle"

    @property
    def loopback_tls(self) -> bool:
        """``train``'s loopback port is plain; a ``hub``'s loopback address is its TLS port (unless insecure)."""
        return self.tls and self.mode != "train"

    def to_json(self) -> str:
        d = asdict(self)
        d["addresses"] = list(self.addresses)
        return json.dumps(d, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> HubInfo:
        return cls(run_id=str(d["run_id"]), run_dir=str(d["run_dir"]), pid=int(d["pid"]),
                   proc_start=None if d.get("proc_start") is None else int(d["proc_start"]),
                   addresses=tuple(d.get("addresses", ())), loopback=d.get("loopback"), tls=bool(d["tls"]),
                   ca_fingerprint=str(d["ca_fingerprint"]), identity_dir=str(d["identity_dir"]), mode=str(d["mode"]))


def _read(path: Path) -> HubInfo | None:
    try:
        return HubInfo.from_dict(json.loads(path.read_text()))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _root(run_dir) -> Path:
    """A ``RunDir``'s root, or a path (``Path.root`` is "/": never ``getattr(path, "root")``)."""
    return Path(run_dir) if isinstance(run_dir, (str, os.PathLike)) else Path(run_dir.root)


def write_hub_json(run_dir, info: HubInfo) -> Path:
    return write_text_atomic(_root(run_dir) / HUB_JSON, info.to_json())


def read_hub_json(run_dir) -> HubInfo | None:
    """The run's ``hub.json`` (None when missing or malformed)."""
    return _read(_root(run_dir) / HUB_JSON)


def alive(info: HubInfo) -> bool:
    return process_alive(RunState(status="running", pid=info.pid, proc_start=info.proc_start))


def running_entry_name(info: HubInfo) -> str:
    """``<pid>-<run_id>.json`` (a run id that is not a plain file-name token is replaced by its SHA-256 prefix)."""
    run = info.run_id if _SAFE_RUN_ID.fullmatch(info.run_id) else hashlib.sha256(
        info.run_id.encode("utf-8")).hexdigest()[:32]
    return f"{int(info.pid)}-{run}.json"


def register_running(identity_dir: Path, info: HubInfo) -> Path:
    d = Path(identity_dir) / RUNNING_DIR
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return write_text_atomic(d / running_entry_name(info), info.to_json())


def unregister_running(identity_dir: Path, info: HubInfo) -> None:
    """Remove the caller's own entry (the one ``register_running(identity_dir, info)`` wrote); other hubs of the
    same process keep theirs."""
    (Path(identity_dir) / RUNNING_DIR / running_entry_name(info)).unlink(missing_ok=True)


def running_hubs(identity_dir: Path) -> list[HubInfo]:
    """The live hubs of this machine (by pid and process start); stale and malformed entries are removed."""
    d = Path(identity_dir) / RUNNING_DIR
    out = []
    for path in sorted(d.glob("*.json")) if d.is_dir() else []:
        info = _read(path)
        if info is None or not alive(info):
            path.unlink(missing_ok=True)
            continue
        out.append(info)
    return out


def this_process_info(**fields) -> HubInfo:
    """A ``HubInfo`` of the calling process (pid and start ticks filled in)."""
    pid = os.getpid()
    return HubInfo(pid=pid, proc_start=process_start_ticks(pid), **fields)
