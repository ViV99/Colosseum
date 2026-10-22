"""A node's restart policy (SP5 T4.2, spec 7.1): a killed learner is reported with its generation and log tail and
never blocks its siblings; the hub's new assignment brings it back from the recovery point; the learner restart
limit; a final snapshot behind periodic ones.

A ``NodeAgent`` in a thread (``node_kit.NodeThread``) against ``node_kit.FakeHub`` (the real gRPC server) with the
toy game ``turns``: one worker and one learner child (2 loaded processes). The FakeHub never restarts a learner by
itself: the tests send its ``start_learner`` as the hub would. (Worker restarts and disabled slots: T4.2 (b).)"""
from __future__ import annotations

import logging
import os
import signal
import time

import pytest

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node import procs as node_procs
from colosseum.node import session as node_session
from game_helpers import make_test_config, scripted_agent
from node_kit import FakeHub, run_node_in_thread, toy_node_options, wait_for

VS_RANDOM = Lineup("2p", [SeatAssignment("a", "latest", True), SeatAssignment("random", "fixed", False)])


def _config(checkpoint_interval=10_000, **learner):
    return make_test_config("turns", agents={"a": {}, "random": scripted_agent()},
                            rollout={"envs_per_worker": 2, "chunk_length": 8},
                            learner={"batch_chunks": 2, "weight_push_interval": 1, "recovery_interval_sec": 0.5,
                                     **learner},
                            checkpoint={"interval": checkpoint_interval},
                            distributed={"heartbeat_sec": 0.25, "shutdown_grace_sec": 2.0})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    monkeypatch.setattr(node_session, "RECONNECT_MAX_SEC", 0.2)
    hubs, nodes = [], []

    def start(config=None):
        hub = FakeHub(tmp_path, config or _config())
        hubs.append(hub)
        hub.grant(lambda worker, env: VS_RANDOM if hub.versions("a") else None)
        node = run_node_in_thread(toy_node_options(tmp_path, hub, cpu_learners=1))
        nodes.append(node)
        wait_for(lambda: node.agent.cache_prefix is not None)
        return hub, node

    yield start, tmp_path / "node-logs"
    for node in nodes:
        if node.thread.is_alive():
            node.stop()
    for hub in hubs:
        hub.stop()


def _start(hub, generation, floor=0, blob=""):
    hub.send_command("start_learner", learner=pb.LearnerAssignment(
        agent_id="a", generation=generation, version_floor=floor, device="cpu", weights_blob=blob))


def _of(hub, generation):
    return [v for v in hub.versions("a") if v.generation == generation]


def _kill_learner(node):
    proc = node.agent.supervisor.get("learner-a")
    os.kill(proc.pid, signal.SIGKILL)
    return proc.pid


def _train_steps(hub, generation=None):
    return [t.train_step for m in hub.messages("metrics") for t in m.train
            if t.agent_id == "a" and (generation is None or t.generation == generation)]


def test_a_killed_learner_is_reported_and_its_siblings_keep_running(setup):
    start, logs = setup
    hub, node = start()
    _start(hub, 1)
    wait_for(lambda: len(_of(hub, 1)) >= 2 and hub.uploads_of("recovery"), timeout=60)
    worker = node.agent.supervisor.get("worker-0").pid
    killed = _kill_learner(node)
    died = wait_for(lambda: hub.node_events("proc_died"), timeout=30)[0]
    log = logs / "learner-a.log"
    assert (died.proc, died.generation, died.exit_code, died.log_path) == ("learner-a", 1, -signal.SIGKILL, str(log))
    assert died.log_tail and log.read_bytes().endswith(died.log_tail)
    assert hub.node_events("learner_failed") == []

    announced = len(hub.versions("a"))
    beats = len(hub.heartbeats())
    wait_for(lambda: len(hub.heartbeats()) >= beats + 1)
    steps = hub.heartbeats()[-1].env_steps
    wait_for(lambda: len(hub.heartbeats()) >= beats + 3 and hub.heartbeats()[-1].env_steps > steps, timeout=30)
    assert node.agent.supervisor.get("worker-0").pid == worker                  # the sibling keeps playing
    assert len(hub.versions("a")) == announced and node.agent.learners.agents() == []

    header, frame = [(h, d) for h, d in hub.uploads if h.kind == "recovery"][-1]
    assert header.generation == 1
    hub.serve_blob("start", "a", "recovery", frame)
    floor = max(v.version for v in _of(hub, 1)) + 1
    _start(hub, 2, floor=floor, blob="start:recovery")
    first = wait_for(lambda: _of(hub, 2), timeout=60)[0]
    assert first.version >= floor
    assert ("start", "a", "recovery") in set(hub.blobs.requests())
    assert node.agent.supervisor.get("learner-a").pid != killed
    assert node.agent.learners.of("learner-a").listener_name.endswith("-i2")   # a new intake address
    assert f"Resumed from recovery of the hub at policy_version {header.version}" in log.read_text()
    wait_for(lambda: _train_steps(hub, 2), timeout=60)                         # it trains again


def test_the_learner_restart_limit_reports_learner_failed(setup, monkeypatch):
    monkeypatch.setattr(node_procs, "LEARNER_RESTARTS", (1, 1800.0))          # the node is in-process
    start, _logs = setup
    hub, node = start()
    _start(hub, 1)
    wait_for(lambda: _of(hub, 1), timeout=60)
    _kill_learner(node)
    died = wait_for(lambda: hub.node_events("proc_died"), timeout=30)[0]
    assert (died.proc, died.generation) == ("learner-a", 1)
    _start(hub, 2, floor=max(v.version for v in _of(hub, 1)) + 1)
    wait_for(lambda: _of(hub, 2), timeout=60)
    _kill_learner(node)
    failed = wait_for(lambda: hub.node_events("learner_failed"), timeout=30)[0]
    assert (failed.proc, failed.generation, failed.exit_code) == ("learner-a", 2, -signal.SIGKILL)
    assert "1 restarts within 1800 s" in failed.message and failed.log_tail
    assert len(hub.node_events("proc_died")) == 1
    assert node.thread.is_alive() and node.agent.supervisor.get("worker-0").exitcode is None


def test_a_final_snapshot_is_never_lost_behind_periodic_ones(setup, caplog):
    """Spec 7.4 / A29: with a hub slower than the learner's checkpoint.interval the node's upload queue keeps one
    waiting periodic snapshot (the rest are skipped, one WARNING), and the final snapshot goes first once it exists."""
    def slow(header, data):
        time.sleep(0.3)
        return pb.UploadReply(accepted=True, checkpoint_id=f"ckpt_v{header.version}")

    start, _logs = setup
    hub, node = start(_config(checkpoint_interval=1, recovery_interval_sec=3600.0))
    hub.upload_reply = slow
    with caplog.at_level(logging.WARNING, logger="colosseum.node.agent"):
        _start(hub, 1)
        wait_for(lambda: max(_train_steps(hub), default=0) >= 20, timeout=120)
        hub.send_command("stop_learner", learner=pb.LearnerAssignment(agent_id="a", generation=1), reason="stop_run")
        finals = wait_for(lambda: hub.uploads_of("final"), timeout=60)
        wait_for(lambda: not node.agent.learners.agents(), timeout=30)
    hub.sync()
    assert len(finals) == 1 and finals[0].version >= 20
    periodic = hub.uploads_of("checkpoint")
    assert 0 < len(periodic) < 20
    assert all(h.version <= finals[0].version for h in periodic)
    warnings = [r for r in caplog.records if "slower than its checkpoint.interval" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert [e.message for e in hub.node_events("warning") if e.proc == "learner-a"] == [
        "checkpoint uploads of a are slower than its checkpoint.interval; older periodic snapshots are skipped"]
