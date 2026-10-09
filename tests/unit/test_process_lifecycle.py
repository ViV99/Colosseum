"""ProcessSupervisor, child-process init and failure messages (T6.5)."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import pytest

import colosseum.utils.process as process_module
from colosseum.utils.process import ChildFailure, ProcessSupervisor, init_child_process


@pytest.fixture(autouse=True)
def _no_lost_interrupt_leaks():
    """The lost-interrupt record is process-global: a failing test must not leave a phantom
    SIGINT for later in-process tests (FIX-2)."""
    process_module.take_lost_interrupt()
    yield
    process_module.take_lost_interrupt()


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
    No PR_SET_PDEATHSIG here: the flush itself must notice the dead parent (its warn_after
    elapses long before, and must not make it give up).
    """
    from colosseum.utils.process import flush_queue

    q = mp.get_context("spawn").Queue()
    q.put(b"x" * (4 << 20))
    Path(marker_dir, "pid").write_text(str(os.getpid()))
    flushed = flush_queue(q, warn_after=0.5)
    Path(marker_dir, "returned").write_text(str(flushed))


def _short_lived_parent(marker_dir: str) -> None:
    """Child: start the sender, wait until its payload is queued, then die (whether the sender
    is already inside flush_queue or enters it later, it must notice the dead parent)."""
    proc = mp.get_context("spawn").Process(target=_unread_queue_sender, args=(marker_dir,))
    proc.start()
    _wait_for_file(Path(marker_dir, "pid"), 60)
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
    assert process_module.flush_queue(q, warn_after=60.0)
    reader.join(60)
    assert reader.exitcode == 0 and (tmp_path / "read").read_text() == str(1 << 20)
    q.close()
    q.join_thread()  # flushed: the feeder exits now, not at the next GC


def test_flush_queue_never_abandons_items_while_the_parent_lives(tmp_path, caplog):
    """Past ``warn_after`` the flush warns and keeps waiting: the main process still reads the
    final checkpoint (or terminates this process); only parent death abandons it."""
    import threading

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    q.put({"payload": b"z" * (1 << 20)})  # the feeder blocks until someone reads
    result: list[bool] = []
    flusher = threading.Thread(target=lambda: result.append(process_module.flush_queue(q, warn_after=0.2)),
                               daemon=True)
    with caplog.at_level(logging.WARNING, logger="colosseum.utils.process"):
        flusher.start()
        deadline = time.monotonic() + 30
        while "still unread" not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.05)
        assert "still unread after" in caplog.text
        assert flusher.is_alive() and not q._joincancelled  # warned, not cancelled
        reader = spawn(_read_one, q, str(tmp_path), name="reader")
        flusher.join(60)
    assert result == [True]
    reader.join(60)
    assert reader.exitcode == 0 and (tmp_path / "read").read_text() == str(1 << 20)
    q.close()
    q.join_thread()


def test_init_child_process_exits_if_the_parent_already_died(monkeypatch):
    exits = []

    class _Exit(Exception):
        pass

    def fake_exit(code):
        exits.append(code)
        raise _Exit

    monkeypatch.setattr(process_module, "set_parent_death_signal", lambda sig=signal.SIGTERM: True)
    monkeypatch.setattr(process_module, "parent_alive", lambda: False)
    monkeypatch.setattr(process_module.os, "_exit", fake_exit)
    old = signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(_Exit):
            init_child_process()
    finally:
        signal.signal(signal.SIGINT, old)
    assert exits == [1]


def _sigint_blocked() -> bool:
    return signal.SIGINT in signal.pthread_sigmask(signal.SIG_BLOCK, [])


def _report_sigint_state(out_dir: str) -> None:
    """The child's SIGINT state at its first statement and after ``init_child_process``."""
    import json

    first = {"blocked": _sigint_blocked()}
    init_child_process()
    after = {"blocked": _sigint_blocked(), "ignored": signal.getsignal(signal.SIGINT) is signal.SIG_IGN}
    Path(out_dir, "sigint.json").write_text(json.dumps({"first": first, "after": after}))


def test_start_process_children_block_sigint_until_init(tmp_path):
    """Blocked from the child's first instruction; after init ignored with a normal mask."""
    import json

    before_handler, before_blocked = signal.getsignal(signal.SIGINT), _sigint_blocked()
    proc = mp.get_context("spawn").Process(target=_report_sigint_state, args=(str(tmp_path),), daemon=True)
    process_module.start_process(proc)
    assert signal.getsignal(signal.SIGINT) is before_handler and _sigint_blocked() == before_blocked
    proc.join(60)
    assert proc.exitcode == 0
    state = json.loads((tmp_path / "sigint.json").read_text())
    assert state == {"first": {"blocked": True}, "after": {"blocked": False, "ignored": True}}


def test_sigint_during_start_reaches_the_parent_handler_afterwards():
    """A Ctrl-C arriving while a child starts is delivered once the start is over, not lost."""
    received = []

    class SignallingProcess:
        def start(self):
            os.kill(os.getpid(), signal.SIGINT)

    old = signal.signal(signal.SIGINT, lambda signum, frame: received.append(signum))
    try:
        process_module.start_process(SignallingProcess())
        deadline = time.monotonic() + 5
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        signal.signal(signal.SIGINT, old)
    assert received == [signal.SIGINT]


def _ignore_sigterm_until_killed(ready) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    time.sleep(120)


def test_kill_remaining_stops_many_stragglers_in_shared_time():
    ctx = mp.get_context("spawn")
    sup = ProcessSupervisor(ctx.Event())
    readies = []
    for i in range(3):
        ready = ctx.Event()
        readies.append(ready)
        proc = ctx.Process(target=_ignore_sigterm_until_killed, args=(ready,), daemon=True)
        proc.start()
        sup.add(f"worker-{i}", proc)
    assert all(r.wait(60) for r in readies)
    start = time.monotonic()
    assert sup.kill_remaining() == ["worker-0", "worker-1", "worker-2"]
    assert time.monotonic() - start < 2.5  # one shared terminate wait, then kill: not 1 s per child
    assert sup.alive() == []


def test_install_signal_handlers_twice_is_a_noop():
    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop)
    before = signal.getsignal(signal.SIGTERM)
    sup.install_signal_handlers()
    try:
        pipe, watcher = sup._pipe, sup._watcher
        sup.install_signal_handlers()
        assert (sup._pipe, sup._watcher) == (pipe, watcher)
    finally:
        sup.restore_signal_handlers()
    assert signal.getsignal(signal.SIGTERM) == before  # one restore undoes the single install


def test_restore_keeps_the_pipe_open_if_the_watcher_does_not_exit(caplog):
    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop)
    sup.install_signal_handlers()
    read_fd = sup._pipe[0]
    real_watcher = sup._watcher

    class StuckWatcher:
        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True

    sup._watcher = StuckWatcher()
    with caplog.at_level(logging.WARNING, logger="colosseum.utils.process"):
        sup.restore_signal_handlers()
    try:
        os.fstat(read_fd)  # still open: never closed under a (possibly) reading thread
        assert "did not exit" in caplog.text
    finally:
        real_watcher.join(5)  # the real one saw EOF
        os.close(read_fd)


def test_signal_handler_cannot_deadlock_on_the_stop_event_lock():
    """A signal landing while the main thread holds the stop event's non-reentrant lock (e.g.
    inside ``stop_event.set()`` / ``is_set()``) must not deadlock: the handler only records it."""
    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop)
    sup.install_signal_handlers()
    try:
        with stop._cond:
            os.kill(os.getpid(), signal.SIGINT)
            sum(range(10))  # bytecode boundary: the Python-level handler runs here
            assert sup.received_signal == signal.SIGINT
        assert stop.wait(5)
    finally:
        sup.restore_signal_handlers()


def test_install_signal_handlers_off_the_main_thread_is_a_noop():
    import threading

    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop)
    before = signal.getsignal(signal.SIGTERM)
    errors = []

    def install():
        try:
            sup.install_signal_handlers()
            sup.restore_signal_handlers()
        except Exception as e:  # noqa: BLE001 - asserted below
            errors.append(e)

    thread = threading.Thread(target=install)
    thread.start()
    thread.join(10)
    assert errors == [] and signal.getsignal(signal.SIGTERM) == before


def test_sigint_while_the_watcher_thread_starts_is_recorded_not_raised(monkeypatch):
    """The handlers are in place before the watcher starts: a Ctrl-C landing in
    ``Thread.start`` is recorded like any other, never a KeyboardInterrupt that leaves an
    unstarted watcher for ``restore_signal_handlers`` to join (FIX-2)."""
    import threading

    real_start = threading.Thread.start

    def start_with_ctrl_c(self):
        signal.raise_signal(signal.SIGINT)
        sum(range(10))  # bytecode boundary: the Python-level handler runs here
        real_start(self)

    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop)
    before = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(threading.Thread, "start", start_with_ctrl_c)
    try:
        try:
            sup.install_signal_handlers()
        except KeyboardInterrupt:
            pytest.fail("a Ctrl-C during the watcher start escaped as KeyboardInterrupt")
        monkeypatch.undo()
        assert stop.wait(5) and sup.received_signal == signal.SIGINT
    finally:
        monkeypatch.undo()
        sup.restore_signal_handlers()
    assert signal.getsignal(signal.SIGINT) is before


def _lose_a_keyboard_interrupt() -> None:
    """A KeyboardInterrupt raised where Python cannot propagate it (a weakref callback, as in
    importlib's module-lock bookkeeping during imports): it is reported as unraisable."""
    import weakref

    class Target:
        pass

    def callback(_ref):
        raise KeyboardInterrupt

    target = Target()
    ref = weakref.ref(target, callback)
    del target
    assert ref() is None


def test_interrupt_lost_in_an_unraisable_context_reaches_the_supervisor(capsys):
    """A Ctrl-C lost during startup (catch_lost_interrupts) is taken over by the run's signal
    handling once installed: the run stops with SIGINT, nothing is printed (FIX-2)."""
    with process_module.catch_lost_interrupts():
        _lose_a_keyboard_interrupt()
    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop)
    sup.install_signal_handlers()
    try:
        assert stop.wait(5) and sup.received_signal == signal.SIGINT
    finally:
        sup.restore_signal_handlers()
    assert not process_module.take_lost_interrupt()  # consumed by the supervisor
    assert capsys.readouterr().err == ""


def test_catch_lost_interrupts_passes_other_unraisables_on_and_restores_the_hook(monkeypatch):
    import sys

    seen = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    with process_module.catch_lost_interrupts():
        assert sys.unraisablehook is not seen.append
        _lose_a_keyboard_interrupt()
        sys.unraisablehook(type("U", (), {"exc_type": ValueError})())
    assert sys.unraisablehook == seen.append
    assert [u.exc_type for u in seen] == [ValueError]
    assert process_module.take_lost_interrupt() and not process_module.take_lost_interrupt()
