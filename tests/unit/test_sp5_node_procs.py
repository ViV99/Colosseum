"""Node children (SP5 T2.4; spec 1, 7.1): incarnations, exits reported once with their signal, restart limits per
role in a sliding window, a bounded stop that kills stragglers."""
from __future__ import annotations

import os
import signal
import time

from colosseum.node import procs
from colosseum.node.procs import ChildSpec, ChildSupervisor


def _sleeper(*, seconds: float, incarnation: int) -> None:
    from colosseum.utils.process import init_child_process
    init_child_process()
    time.sleep(seconds)


def _stubborn(*, incarnation: int) -> None:
    from colosseum.utils.process import init_child_process
    init_child_process()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)


def _wait(pred, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_incarnations_grow_and_exits_are_reported_once():
    sup = ChildSupervisor()
    assert sup.start(ChildSpec("worker-0", "worker", _sleeper, {"seconds": 0.0})) == 1
    assert _wait(lambda: sup.get("worker-0").exitcode is not None)
    assert sup.exited() == [("worker-0", 0)] and sup.exited() == []
    assert sup.restart("worker-0") == 2 and sup.incarnation("worker-0") == 2
    sup.stop_all(5.0)


def test_restart_limits_per_role_in_a_sliding_window(monkeypatch):
    monkeypatch.setattr(procs, "WORKER_RESTARTS", (2, 10.0))
    t = [0.0]
    sup = ChildSupervisor(clock=lambda: t[0])
    sup.start(ChildSpec("worker-0", "worker", _sleeper, {"seconds": 0.0}))
    sup.start(ChildSpec("learner-a", "learner", _sleeper, {"seconds": 0.0}))
    assert sup.may_restart("worker-0") and sup.may_restart("worker-0") and not sup.may_restart("worker-0")
    t[0] = 10.5
    assert sup.may_restart("worker-0")
    assert all(sup.may_restart("learner-a") for _ in range(3)) and not sup.may_restart("learner-a")
    assert not sup.may_restart("unknown")


def test_stop_all_kills_stragglers_within_its_bound():
    sup = ChildSupervisor()
    sup.start(ChildSpec("worker-0", "worker", _stubborn))
    assert _wait(lambda: sup.get("worker-0").is_alive())
    time.sleep(1.0)
    start = time.monotonic()
    assert sup.stop_all(0.5) == ["worker-0"]
    assert time.monotonic() - start < 4.0 and sup.get("worker-0").exitcode == -signal.SIGKILL


def test_a_killed_child_is_reported_with_its_signal():
    sup = ChildSupervisor()
    sup.start(ChildSpec("worker-0", "worker", _sleeper, {"seconds": 30.0}))
    assert _wait(lambda: sup.get("worker-0").pid is not None)
    os.kill(sup.get("worker-0").pid, signal.SIGKILL)
    assert _wait(lambda: sup.exited() == [("worker-0", -signal.SIGKILL)])
