"""Same-machine connections (SP5 T2.2; spec 4.7, 7.8; R3): listeners and connections with the node's authkey;
the learner's intake reads every connection, applies backpressure, and drops by reason (invalid, stale generation,
disconnect) without ever letting one sender's death block another; the control connection carries env steps, stop
and the intake report; a dead learner never blocks a worker's send; a stuck or dying client never stops later
connections (P-PF2-6); long agent ids keep socket names short (P-PF2-18)."""
from __future__ import annotations

import math
import multiprocessing as mp
import multiprocessing.connection as mpc
import os
import queue
import secrets
import signal
import threading
import time
from collections import Counter

import pytest

from colosseum.core.types import TrajectoryChunk
from colosseum.net import local
from colosseum.net.chunks import ChunkValidator, encode_chunk
from colosseum.net.local import (
    ChildHello,
    ConnectionIntake,
    ControlChannel,
    FrameSender,
    IntakeReport,
    LearnerEnvSteps,
    Stop,
    address,
    connect,
    learner_proc_name,
    listener,
)
from game_helpers import chunk_v2_payload, learner_role

KEY = b"k" * 32


def _name(proc="x"):
    return address("abcdef12", secrets.token_hex(4), 1, proc)


def _chunk(version=0, agent="a"):
    return TrajectoryChunk.from_payload(chunk_v2_payload(version=version, agent_id=agent))


def _frame(version=0, agent="a", generation=0, nan=False):
    chunk = _chunk(version, agent)
    if nan:
        chunk.behavior_logp[0] = math.nan
    return encode_chunk(chunk, generation=generation, node=1, worker=0)


def _hello(conn):
    conn.send(ChildHello("worker", "w", 1, os.getpid(), "chunks"))
    return conn


def _wait(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class _Learner:
    """An in-process ConnectionIntake with its control pair."""

    def __init__(self, capacity=2, cap=1 << 26, generation=1, floor=0):
        self.name = _name("learner-a-i1")
        self.lst = listener(self.name, KEY)
        self.node_end, learner_end = mp.Pipe()
        self.control = ControlChannel(learner_end)
        validator = ChunkValidator(learner_role(), agent_id="a", teacher=False)
        self.intake = ConnectionIntake(self.lst, self.control, validator, agent_id="a", generation=generation,
                                       version_floor=floor, capacity=capacity, cap_bytes=cap)

    def close(self):
        self.intake.close()


@pytest.fixture
def learner():
    made = []

    def make(**kw):
        made.append(_Learner(**kw))
        return made[-1]
    yield make
    for lrn in made:
        lrn.close()


def test_a_wrong_authkey_is_refused_and_connect_times_out_without_a_listener():
    name = _name()
    lst = listener(name, KEY)
    errors = []

    def accept():
        try:
            lst.accept()
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    t = threading.Thread(target=accept, daemon=True)
    t.start()
    with pytest.raises(mpc.AuthenticationError):
        connect(name, b"z" * 32, timeout=2.0)
    t.join(5)
    assert errors and isinstance(errors[0], mpc.AuthenticationError)
    lst.close()
    start = time.monotonic()
    with pytest.raises(ConnectionError):
        connect(_name(), KEY, timeout=0.3)
    assert time.monotonic() - start < 2.0
    with pytest.raises(ValueError):
        listener("x" * 200, KEY)


def test_connect_waits_for_a_listener_that_comes_up_later():
    name = _name()
    box = {}

    def late():
        time.sleep(0.3)
        box["lst"] = lst = listener(name, KEY)
        box["conn"] = lst.accept()
    t = threading.Thread(target=late, daemon=True)
    t.start()
    conn = connect(name, KEY, timeout=5.0)
    conn.send("hi")
    t.join(5)
    assert box["conn"].recv() == "hi"
    box["lst"].close()


def test_long_agent_ids_are_hashed_in_learner_socket_names():
    assert learner_proc_name("main", 3) == "learner-main-i3"
    long_id = "a" * 100
    proc = learner_proc_name(long_id, 1)
    assert long_id not in proc and len(proc) <= len("learner-") + 16 + len("-i1")
    assert proc != learner_proc_name("a" * 99 + "b", 1)
    name = address("abcdef12", secrets.token_hex(4), 12345, proc)
    lst = listener(name, KEY)                          # the full name stays within the socket limits
    lst.close()


def test_the_intake_reads_every_connection_and_applies_backpressure(learner):
    lrn = learner(capacity=2)
    dropped = Counter()
    senders = [FrameSender(f"s{i}", lambda: _hello(connect(lrn.name, KEY, timeout=5)), dropped) for i in range(2)]
    for v in range(3):
        for s in senders:
            assert s.put(_frame(v), timeout=5.0)
    assert _wait(lambda: lrn.intake.depth == 2)
    time.sleep(0.3)
    assert lrn.intake.depth == 2                       # full: the reader stops reading the sockets
    got = [lrn.intake.get(timeout=5.0).policy_version for _ in range(6)]
    assert sorted(got) == [0, 0, 1, 1, 2, 2] and dropped == Counter()
    with pytest.raises(queue.Empty):
        lrn.intake.get(timeout=0.2)
    for s in senders:
        s.close()


def _big_sender(name, nbytes, ready):
    conn = _hello(connect(name, KEY, timeout=5))
    data = b"\0" * nbytes                              # allocated before the test is told to go on (P-PF2-16)
    ready.set()
    conn.send_bytes(data)                              # blocks: the learner does not read while full


def test_a_sender_killed_mid_frame_closes_only_its_connection(learner):
    lrn = learner(capacity=1)
    first = _hello(connect(lrn.name, KEY, timeout=5))
    first.send_bytes(_frame(0))
    assert _wait(lambda: lrn.intake.depth == 1)        # now full
    ready = mp.Event()
    proc = mp.Process(target=_big_sender, args=(lrn.name, 64 << 20, ready), daemon=True)
    proc.start()
    assert ready.wait(30)
    time.sleep(0.5)                                    # an upper bound: send_bytes has started and blocks
    os.kill(proc.pid, signal.SIGKILL)
    proc.join(10)
    assert lrn.intake.get(timeout=5.0).policy_version == 0      # frees the deque: the cut frame is read
    first.send_bytes(_frame(7))
    assert lrn.intake.get(timeout=10.0).policy_version == 7     # the other sender is unaffected
    assert _wait(lambda: lrn.intake.dropped.get("disconnect") == 1)


def test_a_clean_close_loses_nothing_and_an_oversized_frame_is_invalid(learner):
    lrn = learner(capacity=8, cap=200_000)
    conn = _hello(connect(lrn.name, KEY, timeout=5))
    conn.send_bytes(_frame(1))
    conn.close()
    assert lrn.intake.get(timeout=5.0).policy_version == 1
    big = _hello(connect(lrn.name, KEY, timeout=5))
    with pytest.raises(BrokenPipeError):               # the learner closes it after reading the length
        big.send_bytes(b"x" * 4_000_000)
    other = _hello(connect(lrn.name, KEY, timeout=5))
    other.send_bytes(_frame(2))
    assert lrn.intake.get(timeout=5.0).policy_version == 2
    assert _wait(lambda: lrn.intake.dropped == {"invalid": 1})


def test_misrouted_garbage_and_stale_generation_chunks_are_dropped_by_reason(learner):
    lrn = learner(capacity=16, generation=2, floor=10)
    conn = _hello(connect(lrn.name, KEY, timeout=5))
    for frame in (_frame(3, agent="b"), b"garbage", _frame(6, nan=True), _frame(12, generation=1),
                  _frame(4, generation=1), _frame(5, generation=3), _frame(11, generation=2), _frame(13, generation=0)):
        conn.send_bytes(frame)
    got = [lrn.intake.get(timeout=5.0).policy_version for _ in range(3)]
    assert got == [4, 11, 13]                          # gen 1 below the floor is kept (R16)
    assert lrn.intake.dropped == {"invalid": 3, "stale_generation": 2}


def test_the_control_connection_carries_env_steps_and_stop_and_reports_the_intake(learner):
    lrn = learner()
    lrn.node_end.send(LearnerEnvSteps(1234))
    assert _wait(lambda: lrn.intake.value == 1234)
    assert lrn.node_end.poll(3.0)
    report = lrn.node_end.recv()
    assert isinstance(report, IntakeReport) and report.depth == 0
    assert not lrn.intake.stop_flag.is_set()
    lrn.node_end.send(Stop("budget"))
    assert lrn.intake.stop_flag.wait(5)


def test_the_learner_stops_when_its_node_is_gone(learner):
    lrn = learner()
    lrn.node_end.close()
    assert lrn.intake.stop_flag.wait(5)


def test_a_stuck_or_dying_client_never_stops_later_connections(learner):
    lrn = learner(capacity=4)
    raw = local._socket_address(lrn.name)
    no_hello = connect(lrn.name, KEY, timeout=5)       # authenticated, but never says hello
    silent = mpc.Client(raw, family="AF_UNIX")         # connects, never answers the challenge
    dying = mpc.Client(raw, family="AF_UNIX")
    if dying.poll(3.0):
        dying.recv_bytes()                             # reads the challenge, then goes away
    dying.close()
    dropped = Counter()
    good = FrameSender("good", lambda: _hello(connect(lrn.name, KEY, timeout=5)), dropped)
    assert good.put(_frame(3), timeout=1.0)
    assert lrn.intake.get(timeout=3.0).policy_version == 3      # within the stuck client's handshake bound
    assert lrn.intake.dropped == {} and dropped == Counter()
    good.close()
    for conn in (silent, no_hello):
        conn.close()


def test_a_sender_whose_connection_never_opens_is_broken_after_the_bound(monkeypatch):
    monkeypatch.setattr(local, "OPEN_TIMEOUT_SEC", 0.3)
    release = threading.Event()
    a, b = mp.Pipe()

    def opener():
        release.wait(30)
        return a
    dropped = Counter()
    sender = FrameSender("stuck", opener, dropped)
    assert sender.put(b"x", timeout=1.0)              # into the slot: the thread is still in its opener
    assert not sender.broken
    time.sleep(0.4)
    start = time.monotonic()
    assert sender.put(b"y", timeout=1.0)              # past the bound: broken, both frames counted
    assert time.monotonic() - start < 0.5
    assert sender.broken and dropped["disconnect"] == 2
    release.set()
    sender.close()
    assert b.poll(5.0)
    with pytest.raises(EOFError):                      # the late connection is closed unused
        b.recv_bytes()
    b.close()


def _serve_then_die(name, ready):
    lst = listener(name, KEY)
    ready.set()
    conn = lst.accept()
    conn.recv()
    time.sleep(60)


def test_a_dead_learner_never_blocks_a_worker_send():
    name = _name("learner-a-i1")
    ready = mp.Event()
    proc = mp.Process(target=_serve_then_die, args=(name, ready), daemon=True)
    proc.start()
    assert ready.wait(30)
    dropped = Counter()
    sender = FrameSender("a", lambda: _hello(connect(name, KEY, timeout=5)), dropped)
    big = b"\0" * (8 << 20)
    assert sender.put(big, timeout=5.0)                # the thread blocks in send_bytes (nobody reads)
    os.kill(proc.pid, signal.SIGKILL)
    proc.join(10)
    assert _wait(lambda: sender.broken)
    start = time.monotonic()
    for _ in range(5):
        assert sender.put(b"y", timeout=0.5)            # consumed at once: dropped and counted
    assert time.monotonic() - start < 1.0
    assert dropped["disconnect"] == 6
    sender.close()
