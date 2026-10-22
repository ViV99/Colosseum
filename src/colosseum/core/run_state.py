"""``runs/<name>/run_state.json``: whether a training run is running, finished or failed (SP4 spec block 5).

The hub (``colosseum.hub.hub.run_training`` / ``run_hub``) writes ``running`` with its pid when the run dir exists,
``finished`` when ``launch`` returns 0 / 130 / 143 (budget reached, SIGINT, SIGTERM: final checkpoints were saved),
``stopped`` with exit code 130 for a Ctrl-C (``KeyboardInterrupt``) that reaches ``run_training`` (during the
start, before the hub's signal handlers: nothing was saved) and ``failed`` for any other exit code or an
exception. Readers (report, dashboard) decide whether a ``running`` run is alive by its pid (``os.kill(pid, 0)``)
and by the process start time recorded with it (Linux ``/proc/<pid>/stat``, macOS ``ps -o lstart=``; SP5 spec 7.8:
a reused pid is not the run) — never by wall-clock timestamps.
Other keys of the file are preserved on every rewrite. The run id is not here: it lives in ``run.json``
(``colosseum.core.run_dir.RUN_INFO_FILE``, contract amendment A2). Only ``run_training`` / ``run_hub`` write this
file (not the benchmark's in-process ``Hub`` nor the distributed roles: their run dirs read as "unknown").
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colosseum.utils.fs import write_text_atomic

RUN_STATE_FILE = "run_state.json"
STATUSES = ("running", "finished", "failed", "stopped")
_OWN_KEYS = ("status", "pid", "exit_code", "proc_start", "updated_ts")


@dataclass(frozen=True)
class RunState:
    status: str                     # one of STATUSES
    pid: int | None
    exit_code: int | None = None
    proc_start: int | None = None   # start time of ``pid`` in clock ticks since boot (Linux), pid-reuse guard
    extra: dict[str, Any] = field(default_factory=dict)


def process_start_ticks(pid: int) -> int | None:
    """The start time of ``pid``, an integer compared only with itself on the same machine (a reused pid is not the
    same process): on Linux field 22 of ``/proc/<pid>/stat`` (clock ticks after boot); on macOS
    ``ps -o lstart= -p <pid>`` in seconds since the epoch (SP5 spec 7.8); None elsewhere or when the process is
    gone."""
    if sys.platform.startswith("linux"):
        try:
            text = Path(f"/proc/{int(pid)}/stat").read_text()
            return int(text.rsplit(")", 1)[1].split()[19])
        except (OSError, ValueError, IndexError):
            return None
    if sys.platform == "darwin":
        return _ps_start_seconds(pid)
    return None


def _ps_start_seconds(pid: int) -> int | None:
    """macOS: ``ps -o lstart=`` ("Mon Oct 18 12:00:00 2026", C locale) as local epoch seconds; None if gone."""
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(int(pid))], capture_output=True, text=True,
                             timeout=5.0, env={**os.environ, "LC_ALL": "C"})
    except (OSError, subprocess.SubprocessError):
        return None
    text = " ".join(out.stdout.split())
    if out.returncode != 0 or not text:
        return None
    try:
        return int(time.mktime(time.strptime(text, "%a %b %d %H:%M:%S %Y")))
    except (ValueError, OverflowError):
        return None


def _read_raw(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_run_state(run_dir: str | Path, status: str, pid: int, *, exit_code: int | None = None) -> Path:
    """Atomically (re)write ``run_state.json`` keeping keys this module does not own."""
    if status not in STATUSES:
        raise ValueError(f"run state must be one of {STATUSES}, got {status!r}")
    path = Path(run_dir) / RUN_STATE_FILE
    data = {k: v for k, v in _read_raw(path).items() if k not in _OWN_KEYS}
    data.update({"status": status, "pid": int(pid), "exit_code": exit_code,
                 "proc_start": process_start_ticks(pid), "updated_ts": time.time()})   # ts: informational only
    return write_text_atomic(path, json.dumps(data, indent=2, sort_keys=True))


def read_run_state(run_dir: str | Path) -> RunState | None:
    """The run's state, or None when the file is missing or malformed (SP3 run dirs have none)."""
    data = _read_raw(Path(run_dir) / RUN_STATE_FILE)
    if data.get("status") not in STATUSES:
        return None
    pid = data.get("pid")
    return RunState(status=data["status"], pid=int(pid) if isinstance(pid, int) else None,
                    exit_code=data.get("exit_code") if isinstance(data.get("exit_code"), int) else None,
                    proc_start=data.get("proc_start") if isinstance(data.get("proc_start"), int) else None,
                    extra={k: v for k, v in data.items() if k not in _OWN_KEYS})


def process_alive(state: RunState) -> bool:
    """Whether the process that wrote ``state`` still runs (pid exists and, where known, has the same start)."""
    if state.pid is None or state.pid <= 0:
        return False
    try:
        os.kill(state.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass                                   # exists, owned by another user
    if state.proc_start is not None:
        now = process_start_ticks(state.pid)
        if now is not None and now != state.proc_start:
            return False                       # the pid was reused by another process
    return True


def effective_status(state: RunState | None) -> tuple[str, str]:
    """``(status, detail)`` for display: running (alive), stopped (``running`` but the process is gone, or a
    Ctrl-C during the start), finished, failed, unknown (no file)."""
    if state is None:
        return "unknown", "no run_state.json"
    if state.status == "running":
        if process_alive(state):
            return "running", f"pid {state.pid}"
        return "stopped", f"pid {state.pid} is gone without a final state (killed?)"
    code = "" if state.exit_code is None else f"exit code {state.exit_code}"
    if state.status == "finished" and state.exit_code in (130, 143):
        code += " (stopped by a signal; final checkpoints saved)"
    if state.status == "stopped":
        code += " (interrupted during the start; no final checkpoints)"
    return state.status, code
