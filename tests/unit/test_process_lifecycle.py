"""ProcessSupervisor, child-process init and failure messages (T6.5)."""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import pytest

import colosseum.utils.process as process_module
from colosseum.utils.process import ChildFailure, ProcessSupervisor, init_child_process


def spawn(target, *args, name="child"):
    proc = mp.get_context("spawn").Process(target=target, args=args, name=name, daemon=True)
    proc.start()
    return proc


def test_child_failure_message_points_to_log(tmp_path):
    failure = ChildFailure("worker-0", -9, tmp_path / "logs" / "worker-0.log")
    assert failure.message() == f"worker-0 died (exit -9), see {tmp_path / 'logs' / 'worker-0.log'}"


def test_supervisor_reports_dead_child(tmp_path):
    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop, log_dir=tmp_path / "logs")
    sup.add("worker-1", spawn(os._exit, 3, name="worker-1"))
    sup.add("learner-a", spawn(time.sleep, 30, name="learner-a"))
    deadline = time.monotonic() + 20
    while sup.first_failure() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    failure = sup.first_failure()
    assert failure is not None and failure.name == "worker-1" and failure.exitcode == 3
    assert failure.log_path == tmp_path / "logs" / "worker-1.log"
    assert sup.alive() == ["learner-a"]
    assert sup.kill_remaining() == ["learner-a"]
    assert sup.alive() == []


def test_first_failure_nonzero_only_ignores_clean_exit():
    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop)
    proc = spawn(os._exit, 0, name="worker-0")
    sup.add("worker-0", proc)
    proc.join(20)
    assert sup.first_failure(nonzero_only=True) is None
    assert sup.first_failure().exitcode == 0


def test_wait_all_polls_until_children_exit():
    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop)
    sup.add("sleeper", spawn(time.sleep, 0.5, name="sleeper"))
    polls = []
    assert sup.wait_all(20.0, poll=lambda: polls.append(1))
    assert polls


def test_signal_handlers_set_stop_event_and_record_signal():
    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop)
    before = signal.getsignal(signal.SIGTERM)
    sup.install_signal_handlers()
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        deadline = time.monotonic() + 5
        while not stop.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert stop.is_set() and sup.received_signal == signal.SIGTERM
    finally:
        sup.restore_signal_handlers()
    assert signal.getsignal(signal.SIGTERM) == before


def test_init_child_process_ignores_sigint_and_sets_death_signal(monkeypatch):
    calls = []
    monkeypatch.setattr(process_module, "set_parent_death_signal", lambda sig=signal.SIGTERM: calls.append(sig) or True)
    old = signal.getsignal(signal.SIGINT)
    try:
        init_child_process()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
        assert calls == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGINT, old)


# ---------------------------------------------------------------------------
# Carried rulings: shutdown-grace constant, parent death (T5.3 review items)
# ---------------------------------------------------------------------------


def test_final_checkpoint_put_timeout_derives_from_the_shutdown_grace():
    """The learner enqueues its final snapshot before the main process stops waiting."""
    from colosseum.learner.learner import FINAL_CHECKPOINT_TIMEOUT_SEC

    assert process_module.SHUTDOWN_GRACE_SEC == 7.0
    assert FINAL_CHECKPOINT_TIMEOUT_SEC == process_module.SHUTDOWN_GRACE_SEC - 2.0


def _wait_for_file(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    return path.exists()


def _pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat[stat.rfind(")") + 2:].split()[0] != "Z"


def _unread_queue_sender(marker_dir: str) -> None:
    """Grandchild: put a payload larger than the pipe buffer on a queue nobody reads, then flush it.

    Without a parent-death check the flush (join_thread) would block forever.
    No PR_SET_PDEATHSIG here: the bounded flush itself must notice the dead parent.
    """
    from colosseum.utils.process import flush_queue

    q = mp.get_context("spawn").Queue()
    q.put(b"x" * (4 << 20))
    Path(marker_dir, "pid").write_text(str(os.getpid()))
    flushed = flush_queue(q, timeout=120.0)
    Path(marker_dir, "returned").write_text(str(flushed))


def _short_lived_parent(marker_dir: str) -> None:
    """Child: start the sender, wait until it is blocked on the unread payload, then die."""
    proc = mp.get_context("spawn").Process(target=_unread_queue_sender, args=(marker_dir,))
    proc.start()
    _wait_for_file(Path(marker_dir, "pid"), 60)
    time.sleep(0.5)
    os._exit(0)


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")
def test_flush_queue_gives_up_when_the_parent_dies(tmp_path):
    # Non-daemonic: it starts a child of its own.
    parent = mp.get_context("spawn").Process(target=_short_lived_parent, args=(str(tmp_path),), name="parent")
    parent.start()
    parent.join(60)
    assert parent.exitcode == 0
    pid = int((tmp_path / "pid").read_text())
    try:
        assert _wait_for_file(tmp_path / "returned", 15), "flush_queue blocked after the parent died"
        assert (tmp_path / "returned").read_text() == "False"
        deadline = time.monotonic() + 15
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_alive(pid), "the sender did not exit after giving up on the flush"
    finally:
        if _pid_alive(pid):
            os.kill(pid, signal.SIGKILL)


def _read_one(q, out_dir: str) -> None:
    item = q.get(timeout=60)
    Path(out_dir, "read").write_text(str(len(item["payload"])))


def test_flush_queue_returns_true_once_the_payload_is_read(tmp_path):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    q.put({"payload": b"y" * (1 << 20)})  # larger than the pipe buffer: the feeder waits for the reader
    reader = spawn(_read_one, q, str(tmp_path), name="reader")
    assert process_module.flush_queue(q, timeout=60.0)
    reader.join(60)
    assert reader.exitcode == 0 and (tmp_path / "read").read_text() == str(1 << 20)
