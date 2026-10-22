"""Portable process helpers (SP5 T4.4, spec 7.8): the parent-death watch (PR_SET_PDEATHSIG on Linux, a watchdog
thread elsewhere), process start times on Linux and macOS (``ps``), process liveness by pid + start time; and the
clock audit of spec 7.6 (durations, timeouts, leases and cadences of the SP5 packages use the monotonic clock)."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from colosseum.core import run_state
from colosseum.core.run_state import RunState, process_alive, process_start_ticks
from colosseum.node import procs
from colosseum.utils import process as proc


def test_watch_parent_uses_pdeathsig_on_linux_and_the_watchdog_elsewhere(monkeypatch):
    started = []
    monkeypatch.setattr(proc, "start_parent_watchdog", lambda *a, **k: started.append(1))
    monkeypatch.setattr(proc, "set_parent_death_signal", lambda sig: True)
    assert proc.watch_parent() == "pdeathsig" and started == []
    monkeypatch.setattr(proc, "set_parent_death_signal", lambda sig: False)    # macOS: no prctl
    assert proc.watch_parent() == "watchdog" and started == [1]
    assert procs.watch_parent is proc.watch_parent                          # the contract's name (node.procs)


def test_the_watchdog_fires_once_the_parent_pid_changes():
    parent = [1234]
    died = threading.Event()
    thread = proc.start_parent_watchdog(0.01, getppid=lambda: parent[0], on_death=died.set)
    time.sleep(0.05)
    assert not died.is_set() and thread.is_alive()
    parent[0] = 1                                                           # re-parented: the parent is gone
    assert died.wait(2.0)
    thread.join(2.0)
    assert not thread.is_alive()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc")
def test_linux_start_ticks_identify_this_process():
    ticks = process_start_ticks(os.getpid())
    assert isinstance(ticks, int) and ticks == process_start_ticks(os.getpid())
    assert process_start_ticks(2**22 + 12345) is None


def test_macos_start_time_from_ps(monkeypatch):
    monkeypatch.setattr(run_state.sys, "platform", "darwin")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs["env"]["LC_ALL"]))
        pid = cmd[-1]
        out = "Sun Oct 18 02:15:25 2026\n" if pid == "42" else ""
        return subprocess.CompletedProcess(cmd, 0 if out else 1, stdout=out, stderr="")

    monkeypatch.setattr(run_state.subprocess, "run", fake_run)
    first = process_start_ticks(42)
    assert isinstance(first, int) and first == process_start_ticks(42)
    assert calls[0] == (["ps", "-o", "lstart=", "-p", "42"], "C")
    assert process_start_ticks(7) is None                                  # gone: ps prints nothing, exit 1
    monkeypatch.setattr(run_state.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("ps")))
    assert process_start_ticks(42) is None


def test_other_platforms_have_no_start_time(monkeypatch):
    monkeypatch.setattr(run_state.sys, "platform", "win32")
    assert process_start_ticks(os.getpid()) is None


def test_liveness_by_pid_and_start_time(monkeypatch):
    me = os.getpid()
    assert process_alive(RunState("running", me, proc_start=process_start_ticks(me)))
    assert not process_alive(RunState("running", 2**22 + 12345))
    if sys.platform.startswith("linux"):
        assert not process_alive(RunState("running", me, proc_start=process_start_ticks(me) + 1))   # pid reused
    monkeypatch.setattr(run_state, "process_start_ticks", lambda pid: None)   # no start time known: pid only
    assert process_alive(RunState("running", me, proc_start=5))


SRC = Path(__file__).resolve().parents[2] / "src" / "colosseum"
_WALL_CLOCK = re.compile(r"\btime\.time\(\)|\bdatetime\.(?:datetime\.)?now\(")


def test_sp5_packages_measure_time_with_the_monotonic_clock():
    """Spec 7.6: durations, timeouts, leases and cadences use time.monotonic(); a wall-clock read in the SP5
    packages is allowed only as an informational timestamp marked ``# ts: informational``."""
    offenders = []
    for package in ("hub", "node", "net", "bootstrap"):
        for path in sorted((SRC / package).rglob("*.py")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if _WALL_CLOCK.search(line) and "ts: informational" not in line:
                    offenders.append(f"{path.relative_to(SRC)}:{number}: {line.strip()}")
    assert offenders == [], "use time.monotonic() for durations:\n" + "\n".join(offenders)
