"""The distributed stop and the exit codes (SP5 T4.4, spec 7.4, 7.5).

Hub-side tests run an in-process hub (``cluster_kit.start_hub``, mode ``hub``) with ``NodeAgent``s in threads; the
node's signal and exit-code tests run the node agent as its own process (``node_kit.start_node_main``) against
``node_kit.FakeHub``; the hub's exit codes run ``colosseum hub`` processes (``cluster_kit.hub_process``, slow). At
most two loaded processes per test (a worker and a learner); Linux (``/proc``) for the process checks."""
from __future__ import annotations

import json
import logging
import os
import signal
from pathlib import Path

import pytest

from cli_runner import TTT_CONFIG
from cluster_kit import cluster_config, hub_process, read_hub_state, start_hub, start_node, wait_until
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.security import parse_token
from colosseum.node.agent import NodeOptions
from game_helpers import make_test_config, scripted_agent
from hub_kit import lineup_hub, open_node
from node_kit import FakeHub, pid_alive, spawned_children, start_node_main, toy_node_options, wait_for

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")

A_VS_RANDOM = Lineup("2p", [SeatAssignment("a", "latest", True), SeatAssignment("random", "fixed", False)])


def _config(tmp_path, **overrides):
    return cluster_config(tmp_path, {"distributed.heartbeat_sec": 0.1, "distributed.shutdown_grace_sec": 5.0,
                                     **overrides})


def _pids(node, names=("worker-0", "learner-agent_0")) -> list[int] | None:
    out = []
    for name in names:
        proc = node.agent.supervisor.get(name)
        if proc is None or not proc.pid or proc.exitcode is not None:
            return None
        out.append(proc.pid)
    return out


def _finals(root: Path, agent: str = "agent_0") -> list[dict]:
    metas = [json.loads((d / "meta.json").read_text()) for d in (root / "checkpoints" / agent).glob("ckpt_v*")
             if (d / "meta.json").exists()]
    return [m for m in metas if m.get("final") is True]


def _messages(caplog, needle: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


def test_budget_stop_collects_final_snapshots_and_the_node_stays(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="colosseum.node")
    caplog.set_level(logging.DEBUG, logger="colosseum.hub")       # last: the capture handler's level is the last set
    hub = start_hub(tmp_path, _config(tmp_path))
    node = start_node(tmp_path, hub, "pc", workers=1, cpu_learners=1, once=False)
    try:
        pids = wait_for(lambda: _pids(node), timeout=120)
        hub.thread.join(240)
        assert not hub.thread.is_alive() and hub.exit_codes == [0]
        assert _finals(hub.root), "no final snapshot"
        assert wait_until(lambda: not any(pid_alive(p) for p in pids), 30)
        assert wait_until(lambda: _messages(caplog, "run finished; waiting for the next run of this hub"), 30)
        assert node.thread.is_alive() and node.agent.run_id is None
        assert _messages(caplog, f"node {node.agent.number} stopped")             # node_stopped before the end
        assert _messages(caplog, "stop grace 5 s (remote nodes)")
        assert not _messages(caplog, "final snapshots missing")
        worker_log = (tmp_path / "logs" / "pc" / "worker-0.log").read_text()
        assert "receiver is gone" not in worker_log                     # learners stop after the workers (T4.4)
    finally:
        assert node.stop() == 0


def test_a_once_node_exits_0_after_the_run(tmp_path):
    hub = start_hub(tmp_path, _config(tmp_path))
    node = start_node(tmp_path, hub, "pc", workers=1, cpu_learners=1, once=True)
    try:
        hub.thread.join(240)
        assert hub.exit_codes == [0]
        node.thread.join(60)
        assert not node.thread.is_alive() and node.exit_codes == [0]
    finally:
        if node.thread.is_alive():
            node.stop()


def test_the_local_grace_applies_without_remote_nodes(tmp_path, caplog):
    """``train``'s local node alone: ``LOCAL_SHUTDOWN_GRACE_SEC`` (the 10 s guarantee); a remote node too:
    ``distributed.shutdown_grace_sec`` (its final snapshots travel over the network). The stop grace reaches the
    nodes in ``stop_run`` (A1)."""
    import uuid

    caplog.set_level(logging.DEBUG, logger="colosseum.hub")
    hub = lineup_hub(make_test_config("turns", agents={"a": {}, "random": scripted_agent()},
                                      distributed={"shutdown_grace_sec": 12.0}), tmp_path)
    record, _secret = hub.identity.nodes.add_ephemeral("local", uuid.uuid4().hex)
    local = open_node(hub, name="local", node_id=record.node_id)
    assert hub._stop_nodes() == 7.0
    assert _messages(caplog, "stop grace 7 s (local nodes only)")
    assert [c.reason for c in local.of("command") if c.kind == "stop_run"] == ["grace=7"]

    hub2 = lineup_hub(make_test_config("turns", agents={"a": {}, "random": scripted_agent()},
                                       distributed={"shutdown_grace_sec": 12.0}), tmp_path / "two", name="two")
    record2, _secret = hub2.identity.nodes.add_ephemeral("local", uuid.uuid4().hex)
    open_node(hub2, name="local", node_id=record2.node_id)
    remote = open_node(hub2, name="pc")
    assert hub2._stop_nodes() == 12.0
    assert _messages(caplog, "stop grace 12 s (remote nodes)")
    assert [c.reason for c in remote.of("command") if c.kind == "stop_run"] == ["grace=12"]
    hub2._place_learners()                                       # the run stops: nothing is placed any more
    assert not [c for c in remote.of("command") if c.kind == "start_learner"]


def test_node_leaving_relocates_at_once(tmp_path, caplog):
    """W04-24 (P24): a node that stops on its own is lost once its stop is done, and its learner moves at once,
    without waiting ``learner_relocate_after_sec``."""
    caplog.set_level(logging.INFO, logger="colosseum.hub")
    config = _config(tmp_path, **{"training.total_timesteps": 10**9, "distributed.node_lost_after_sec": 60.0,
                                  "distributed.learner_relocate_after_sec": 120.0})
    hub = start_hub(tmp_path, config)
    first = start_node(tmp_path, hub, "first", cpu_learners=1, data_port=None)
    second = None
    try:
        learner = lambda: ((read_hub_state(hub).get("learners") or {}).get("agents") or {}).get("agent_0") or {}  # noqa: E731
        assert wait_until(lambda: learner().get("node") == first.agent.number, 120), learner()
        second = start_node(tmp_path, hub, "second", cpu_learners=1, workers=1, data_port=None)
        assert wait_until(lambda: second.agent.run_id is not None, 60)
        assert first.stop() == 0
        assert wait_until(lambda: learner().get("node") == second.agent.number, 60), learner()
        assert learner()["generation"] == 2
        assert _messages(caplog, "is leaving; its cells are dropped")
        assert _messages(caplog, f"node {first.agent.number} (first) left; relocating it now")
        assert wait_until(lambda: read_hub_state(hub).get("nodes_lost", 0) == 1, 30)
    finally:
        hub.stop()
        if second is not None:
            second.stop()
        if first.thread.is_alive():
            first.stop()


def test_a_player_error_on_a_remote_node_stops_the_run_with_its_context(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="colosseum.hub")
    config = make_test_config("turns", agents={"a": {}, "bad": scripted_agent("game_helpers.WorkerPlayerErrorBot")},
                              matchmaking={"opponents": {"latest": 0.0, "snapshots": 0.0, "rivals": 0.0,
                                                         "anchors": 1.0}},
                              training={"total_timesteps": 10**9},
                              distributed={"heartbeat_sec": 0.1, "shutdown_grace_sec": 5.0})
    hub = start_hub(tmp_path, config)
    node = start_node(tmp_path, hub, "pc", workers=1, cpu_learners=1)
    try:
        hub.thread.join(240)
        assert not hub.thread.is_alive() and hub.exit_codes == [1]
    finally:
        if node.thread.is_alive():
            node.stop()
    (failed,) = _messages(caplog, "Run failed: worker-0 failed (player)")
    number = node.agent.number
    for part in (f"node {number} (pc), worker 0, env ", "seat ", "episode step ", "layout ", "agent "):
        assert part in failed, failed
    tail = hub.root / "logs" / "nodes" / str(number) / "worker-0.log.tail"
    assert f"see {tail} (node log {tmp_path / 'logs' / 'pc' / 'worker-0.log'})" in failed
    assert tail.read_bytes()


# ---------------------------------------------------------------------------------------------------------------
# The node agent's exit codes (its own process against a FakeHub)
# ---------------------------------------------------------------------------------------------------------------


def _node_config():
    return make_test_config("turns", agents={"a": {}, "random": scripted_agent()},
                            rollout={"envs_per_worker": 2, "chunk_length": 8},
                            learner={"batch_chunks": 2, "weight_push_interval": 1},
                            distributed={"heartbeat_sec": 0.25, "shutdown_grace_sec": 5.0})


def _node_process(tmp_path, *, once=False):
    hub = FakeHub(tmp_path, _node_config())
    hub.welcome(learners=[pb.LearnerAssignment(agent_id="a", generation=1, version_floor=0, device="cpu")])
    hub.grant(lambda worker, env: A_VS_RANDOM if hub.versions("a") else None)
    options = toy_node_options(tmp_path, hub, cpu_learners=1, once=once)
    proc = start_node_main(options, tmp_path / "node.stderr")
    return hub, proc


@pytest.mark.parametrize("sig, code", [pytest.param(signal.SIGINT, 130, id="sigint"),
                                        pytest.param(signal.SIGTERM, 143, id="sigterm", marks=pytest.mark.slow)])
def test_node_exit_codes_on_signals(tmp_path, sig, code):
    hub, proc = _node_process(tmp_path)
    try:
        wait_for(lambda: hub.results() and hub.versions("a"), timeout=120)
        children = spawned_children(proc.pid)
        assert len(children) >= 2
        os.kill(proc.pid, sig)
        assert proc.wait(30) == code, (tmp_path / "node.stderr").read_text()[-3000:]
        assert not any(pid_alive(p) for p in children)                     # its children stopped before it exited
        hub.sync()
        assert hub.node_events("node_leaving")
        assert hub.uploads_of("final")                                      # the learner's final reached the hub
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        hub.stop()


@pytest.mark.slow
def test_node_exit_code_after_a_release(tmp_path):
    hub, proc = _node_process(tmp_path, once=True)
    try:
        wait_for(lambda: hub.results() and hub.versions("a"), timeout=120)
        hub.send_command("release", reason="nodes remove")
        assert proc.wait(30) == 0, (tmp_path / "node.stderr").read_text()[-3000:]
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        hub.stop()


# ---------------------------------------------------------------------------------------------------------------
# The hub's exit codes (colosseum hub processes)
# ---------------------------------------------------------------------------------------------------------------


def _hub_and_node(tmp_path, name, overrides, cpu_learners=1):
    hub = hub_process(TTT_CONFIG, tmp_path, name, overrides={"learner.device": "cpu",
                                                              "distributed.heartbeat_sec": "0.25",
                                                              "distributed.shutdown_grace_sec": "8.0", **overrides},
                      insecure=True)
    options = NodeOptions(hub_address=hub.address, token=parse_token(hub.token_file.read_text()), insecure=True,
                          workers=1, cpu_learners=cpu_learners, name="pc", identity_root=tmp_path / f"{name}-node-id",
                          log_dir=tmp_path / f"{name}-node-logs")
    return hub, start_node_main(options, tmp_path / f"{name}-node.stderr")


@pytest.mark.parametrize("how, code", [("budget", 0), ("sigint", 130),
                                       pytest.param("sigterm", 143, marks=pytest.mark.slow)])
def test_hub_exit_codes(tmp_path, how, code):
    budget = {"training.total_timesteps": "3000" if how == "budget" else "1000000000"}
    hub, node = _hub_and_node(tmp_path, how, budget)
    try:
        if how != "budget":
            assert wait_until(lambda: _finals(hub.root) == [] and any((hub.root / "checkpoints").glob("*/ckpt_v*")),
                              180), hub.tail()
            if how == "sigint":
                hub.interrupt()
            else:
                os.killpg(hub.proc.pid, signal.SIGTERM)
        assert hub.wait(120) == code, hub.tail()
        assert _finals(hub.root), "no final snapshot"
        state = json.loads((hub.root / "run_state.json").read_text())
        assert state["status"] == "finished" and state["exit_code"] == code
    finally:
        for proc in (node, hub.proc):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


@pytest.mark.slow
def test_hub_exit_code_with_a_failed_agent(tmp_path):
    """Two learners and a worker (3 loaded processes: slow, P-PF3-20): one agent's learner keeps crashing (T4.2's
    ``CrashingAPPO``), the other trains on to the budget; the run ends with exit code 1."""
    overrides = {"training.total_timesteps": "300000", "agents.healthy": "{}",
                 "agents.agent_0": "{algorithm: {algorithm_class: game_helpers.CrashingAPPO}}"}
    hub, node = _hub_and_node(tmp_path, "failed", overrides, cpu_learners=2)
    try:
        assert hub.wait(300) == 1, hub.tail()
        main = (hub.root / "logs" / "main.log").read_text()
        assert "Learner of agent_0 failed" in main and "Training budget reached" in main
    finally:
        for proc in (node, hub.proc):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def test_a_persistent_node_starts_the_next_run_as_a_new_budget_session(tmp_path, monkeypatch):
    """Review round 1 of T4.4 (b): after a ``stop_run`` a persistent node serves the next run with a new session id
    and its env steps and counters from 0 (a new run's budget never starts with the old run's steps)."""
    import uuid

    from colosseum.hub.server import SessionOpened
    from colosseum.node import session as node_session
    from node_kit import run_node_in_thread

    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    monkeypatch.setattr(node_session, "RECONNECT_MAX_SEC", 0.2)
    fixed = Lineup("2p", [SeatAssignment("random", "fixed", False), SeatAssignment("bot2", "fixed", False)])
    hub = FakeHub(tmp_path, make_test_config("turns", agents={"a": {}, "random": scripted_agent(),
                                                              "bot2": scripted_agent()},
                                             rollout={"envs_per_worker": 2}, distributed={"heartbeat_sec": 0.1}))
    hub.grant(lambda worker, env: fixed)
    node = run_node_in_thread(toy_node_options(tmp_path, hub))
    try:
        wait_for(lambda: hub.heartbeats() and hub.heartbeats()[-1].env_steps > 0, timeout=60)
        first = hub.events(SessionOpened)[0].hello.session_id
        hub.send_command("stop_run", reason="grace=3")
        wait_for(lambda: hub.node_events("node_stopped") and node.agent.run_id is None, timeout=30)
        old_run = hub.run_id
        old_steps = max(b.env_steps for b in hub.heartbeats())
        hub.call(lambda: setattr(hub, "run_id", uuid.uuid4().hex))
        opened = len(hub.events(SessionOpened))
        hub.close_session()
        again = wait_for(lambda: hub.events(SessionOpened)[opened:], timeout=30)[0].hello
        assert again.session_id != first and again.env_steps == 0 and again.last_acked_seq == 0
        beats = lambda: [e.message.heartbeat for e in hub.node_messages("heartbeat")  # noqa: E731
                         if e.session_id == again.session_id]
        wait_for(lambda: beats() and beats()[-1].env_steps > 0, timeout=60)        # it plays the new run
        assert beats()[0].env_steps < old_steps                                     # counted from 0 again
        assert beats()[0].counters["dropped_results_node_buffer"] == 0
        assert wait_for(lambda: [r for r in hub.results() if r.uid.startswith(f"{hub.run_id}/")], timeout=60)
        assert node.agent.run_id == hub.run_id != old_run
    finally:
        assert node.stop() == 0
        hub.stop()
