"""Node internals of the stop and of a child's end (SP5 T4.4): the hub's ``stop_run`` grace (A1) sets the children's
window, a child's last ``ChildHello`` + ``ChildError`` before its exit is a reported error and not a death (carried
item of T2.4 (a)), and a checkpoint frame is read off the main loop (a learner stopped in the middle of one never
stalls the node; carried item of T2.4 (b)). ``NodeAgent`` objects are driven without ``run`` (no hub, no process)."""
from __future__ import annotations

import multiprocessing.connection as mpc
import time
from types import SimpleNamespace

import pytest

from colosseum.net.local import CheckpointHeader, ChildError, ChildHello
from colosseum.net.security import Token, new_secret
from colosseum.node import agent as agent_mod
from colosseum.node.agent import NodeAgent, NodeOptions, _grace_of


def _agent(**options) -> NodeAgent:
    return NodeAgent(NodeOptions(hub_address="127.0.0.1:1", token=Token(new_secret(), "ab" * 32), **options))


@pytest.fixture
def make_agent():
    made = []

    def make(**options):
        made.append(_agent(**options))
        return made[-1]

    yield make
    for agent in made:
        agent._close()


def test_the_stop_run_grace_is_read_from_the_command():
    assert _grace_of("grace=5") == 5.0 and _grace_of("grace=12.5") == 12.5
    assert _grace_of("") is None and _grace_of("grace=x") is None and _grace_of("grace=0") is None
    assert _grace_of("stop_run") is None


def test_the_childrens_window_follows_the_hubs_grace(make_agent):
    run = SimpleNamespace(grace=3.0)
    local = make_agent(local=True)
    assert local._children_window(run, 7.0) == agent_mod.LOCAL_CHILD_GRACE_SEC     # 7 s - 2 s: within the 10 s bound
    remote = make_agent()
    assert remote._children_window(run, 20.0) == 20.0 - agent_mod.REMOTE_GRACE_MARGIN_SEC
    assert remote._children_window(run, None) == 3.0                               # no stop_run: the run's default
    assert make_agent(shutdown_grace_sec=1.0)._children_window(run, 20.0) == 3.0   # an explicit option wins


class _Supervisor:
    """``ChildSupervisor`` stand-in: one worker that exited with code 1."""

    def __init__(self) -> None:
        self.restarts: list[str] = []
        self.proc = SimpleNamespace(exitcode=1, pid=4242)

    def get(self, name):
        return self.proc

    def incarnation(self, name):
        return 1

    def role(self, name):
        return "worker"

    def may_restart(self, name):
        return True

    def restart(self, name):
        self.restarts.append(name)
        return 2

    def alive(self):
        return []

    def exited(self):
        return []

    def sentinels(self):
        return {}

    def names(self):
        return []


def test_a_hello_and_error_sent_just_before_the_exit_is_a_reported_error(make_agent, tmp_path, monkeypatch):
    agent = make_agent()
    agent.supervisor = _Supervisor()
    agent._run = SimpleNamespace(log_dir=tmp_path, run_id="r" * 32, envs=1, workers=1)
    node_end, child_end = mpc.Pipe()
    child_end.send(ChildError("player", "bot failure", "worker 0, env 0, seat 1"))
    child_end.close()                                            # the child exited right after
    agent._on_exit("worker-0", 1)                                # its exit is seen before its hello arrived
    assert "worker-0" in agent._exit_waits and not agent.supervisor.restarts
    agent._on_child_hello(node_end, ChildHello("worker", "worker-0", 1, 4242, "control"))
    agent._check_exit_waits(time.monotonic())
    assert not agent.supervisor.restarts and "worker-0" in agent._reported
    (event,) = list(agent._events)
    assert event.kind == "child_error" and event.error_kind == "player"
    agent._run = None


def test_an_exit_without_a_hello_is_classified_after_the_drain_bound(make_agent, tmp_path):
    agent = make_agent()
    agent.supervisor = _Supervisor()
    agent._run = SimpleNamespace(log_dir=tmp_path, run_id="r" * 32, envs=1, workers=1)
    agent._on_exit("worker-0", -9)
    agent._check_exit_waits(time.monotonic())
    assert not agent.supervisor.restarts                         # still waiting for a hello
    agent._check_exit_waits(time.monotonic() + agent_mod.EXIT_DRAIN_SEC)
    assert agent.supervisor.restarts == ["worker-0"] and not agent._exit_waits
    agent._run = None


def test_a_checkpoint_frame_is_read_off_the_loop(make_agent, monkeypatch):
    agent = make_agent()
    seen = []
    monkeypatch.setattr(agent, "_on_checkpoint", lambda name, header, frame: seen.append((name, frame)))
    node_end, learner_end = mpc.Pipe()
    agent._child_conns[node_end] = "learner-a"
    agent._control["learner-a"] = node_end
    learner_end.send(CheckpointHeader("periodic", "a", 1, 3, 5))
    started = time.monotonic()
    agent._read_child(node_end)                                  # the frame has not been sent: no blocking read
    assert time.monotonic() - started < 1.0 and node_end in agent._frame_reads
    learner_end.send_bytes(b"frame")
    deadline = time.monotonic() + 10
    while not seen and time.monotonic() < deadline:
        agent._drain_inbox()
        time.sleep(0.01)
    assert seen == [("learner-a", b"frame")] and node_end not in agent._frame_reads
    learner_end.close()
    node_end.close()


def test_a_learner_stuck_mid_frame_is_killed(make_agent, monkeypatch):
    agent = make_agent()
    killed = []
    stuck = SimpleNamespace(exitcode=None, kill=lambda: killed.append(1))
    agent.supervisor = SimpleNamespace(get=lambda name: stuck, alive=lambda: [])
    node_end, learner_end = mpc.Pipe()
    agent._child_conns[node_end] = "learner-a"
    learner_end.send(CheckpointHeader("final", "a", 1, 3, 5))
    agent._read_child(node_end)
    agent._check_frame_reads(time.monotonic())
    assert not killed
    agent._check_frame_reads(time.monotonic() + agent_mod.FRAME_READ_TIMEOUT_SEC)
    assert killed == [1]
    learner_end.close()                                          # the reader ends (EOF)
    deadline = time.monotonic() + 10
    while node_end in agent._frame_reads and time.monotonic() < deadline:
        agent._drain_inbox()
        time.sleep(0.01)
    assert node_end not in agent._frame_reads and node_end not in agent._child_conns   # EOF: the connection ends
