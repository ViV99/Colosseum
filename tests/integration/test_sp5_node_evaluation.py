"""The node agent's evaluation section (SP5 T5.1, spec 5.3, 7.1, 4.6): a ``NodeAgent`` in a thread with
``node_kit.FakeHub``; its evaluator is the only loaded process. Blocks are toy-game gauntlet blocks
(``eval_kit.gauntlet_tasks``) whose neural player's weights the fake hub serves as an eval blob
(``BlobStore.register_weights``, ``Download(kind="eval")``)."""
from __future__ import annotations

import os
import shutil
import signal
import time

import pytest

from colosseum.core.registry import env_spec
from colosseum.hub.evaluation import check_block_results
from colosseum.hub.server import SessionOpened
from colosseum.net.messages import result_from_proto
from colosseum.node import evaluators as node_evaluators
from colosseum.node import session as node_session
from colosseum.node.leases import EVAL_BLOBS_KEPT
from colosseum.node.weights import shm_dir
from colosseum.utils.process import SHUTDOWN_GRACE_SEC
from eval_kit import blob_id_of, fixed_players, gauntlet_tasks, snapshot_player
from game_helpers import make_test_config, scripted_agent
from node_kit import FakeHub, pid_alive, run_node_in_thread, toy_node_options, wait_for

AGENTS = {"main": {}, "rnd": scripted_agent(), "ones": scripted_agent("game_helpers.ConstantBot", value=1),
          "bad": scripted_agent("game_helpers.PulseFailBot")}
LONG = 2000            # gauntlet blocks per composition of a lease that is still playing when the test acts on it


@pytest.fixture(autouse=True)
def _fast_reconnect(monkeypatch):
    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    monkeypatch.setattr(node_session, "RECONNECT_MAX_SEC", 0.2)
    monkeypatch.setattr(node_evaluators, "REASK_SEC", 1e9)      # request counts below are exact (one test lowers it)


def _setup(tmp_path, *, evaluators=1):
    cfg = make_test_config("turns", agents=AGENTS)
    hub = FakeHub(tmp_path, cfg)
    hub.welcome(evaluators=evaluators)
    node = run_node_in_thread(toy_node_options(tmp_path, hub, workers=0, evaluators=1))
    return cfg, hub, node


def _tasks(cfg, hub, tmp_path, *, version=3, blocks=1, owner="tournament:t", field=("rnd", "ones"), seed=0):
    root = tmp_path / f"snap-{owner.split(':')[1]}-{version}"
    if root.exists():                                            # the same snapshot again (the same blob)
        shutil.rmtree(root)
    cand = snapshot_player(cfg, root, "main", version, seed=version)
    return gauntlet_tasks(cfg, cand, fixed_players(cfg, field), blocks=blocks, owner=owner, blobs=hub.blobs,
                          seed=seed)


def _evaluator_pid(node, name="evaluator-0"):
    proc = node.agent.supervisor.get(name)
    return None if proc is None or proc.exitcode is not None else proc.pid


def _procs(hub):
    beats = hub.heartbeats()
    return {p.name: (p.state, p.restarts) for p in beats[-1].procs} if beats else {}


def _received_index(hub, kind, predicate=lambda m: True):
    """Positions (in arrival order) of the node messages of ``kind`` that match."""
    out = []
    for i, ev in enumerate(list(hub.received)):
        message = getattr(ev, "message", None)
        if message is not None and message.WhichOneof("m") == kind and predicate(getattr(message, kind)):
            out.append(i)
    return out


def _stop(hub, node):
    try:
        node.stop()
    finally:
        hub.stop()


def test_a_lease_is_fetched_once_played_on_an_evaluator_and_acked(tmp_path):
    cfg, hub, node = _setup(tmp_path)
    try:
        wait_for(lambda: _procs(hub).get("evaluator-0", ("", 0))[0] == "alive", 60)
        wait_for(lambda: hub.lease_requests() == [1], 60)
        tasks = _tasks(cfg, hub, tmp_path)
        hub.lease("L1", tasks)
        results = wait_for(lambda: (lambda rs: rs if len(rs) == len(tasks) else None)(hub.block_results("L1")), 120)
        by_block = {r.block_id: r for r in results}
        spec = env_spec(cfg)
        for task in tasks:
            got = by_block[task.block_id]
            assert not got.failed and check_block_results(
                task, [result_from_proto(p) for p in got.results], spec=spec) is None
        blob_id = next(blob_id_of(p.blob) for t in tasks for p in t.players if p.blob is not None)
        assert len(hub.eval_downloads(blob_id)) == 1
        wait_for(lambda: len(hub.lease_requests()) == 2, 30)                       # the evaluator is idle again
        last_result = max(_received_index(hub, "block_result", lambda r: r.lease_id == "L1"))
        assert _received_index(hub, "lease_request")[1] > last_result              # results before the request
        hub.lease("L2", _tasks(cfg, hub, tmp_path, seed=1))                      # the same blob again
        wait_for(lambda: len(hub.block_results("L2")) == len(tasks), 120)
        assert len(hub.eval_downloads(blob_id)) == 1                             # one fetch per machine
        assert not hub.errors
    finally:
        _stop(hub, node)


def test_unacked_block_results_are_resent_after_a_reconnect_and_hello_names_the_leases(tmp_path):
    cfg, hub, node = _setup(tmp_path)
    try:
        hub.block_acks = False
        wait_for(lambda: hub.lease_requests(), 60)
        tasks = _tasks(cfg, hub, tmp_path)
        hub.lease("L1", tasks)
        first = wait_for(lambda: (lambda rs: rs if len(rs) == len(tasks) else None)(hub.block_results("L1")), 120)
        wait_for(lambda: node.agent._hello_state[6] == ("L1",), 10)
        sessions = len(hub.sessions_opened())
        hub.close_session()
        hello = wait_for(lambda: (lambda hs: hs[-1] if len(hs) > sessions else None)(hub.sessions_opened()), 30)
        assert list(hello.leases) == ["L1"]
        again = wait_for(lambda: (lambda rs: rs[len(first):] if len(rs) >= 2 * len(tasks) else None)(
            hub.block_results("L1")), 30)
        assert sorted((r.lease_id, r.block_id, r.SerializeToString(deterministic=True)) for r in again) == \
            sorted((r.lease_id, r.block_id, r.SerializeToString(deterministic=True)) for r in first)
        opened = max(i for i, ev in enumerate(list(hub.received)) if isinstance(ev, SessionOpened))
        request = wait_for(lambda: [i for i in _received_index(hub, "lease_request") if i > opened], 30)[0]
        assert max(_received_index(hub, "block_result")) < request              # resends before a new request
        hub.ack_blocks()
        wait_for(lambda: node.agent._hello_state[6] == (), 10)
        sessions = len(hub.sessions_opened())
        hub.close_session()
        hello = wait_for(lambda: (lambda hs: hs[-1] if len(hs) > sessions else None)(hub.sessions_opened()), 30)
        assert list(hello.leases) == []
    finally:
        _stop(hub, node)


def test_set_evaluators_starts_and_stops_evaluators_and_releases_their_leases(tmp_path):
    cfg, hub, node = _setup(tmp_path, evaluators=0)
    try:
        wait_for(lambda: len(hub.heartbeats()) >= 3, 30)
        assert "evaluator-0" not in _procs(hub) and hub.lease_requests() == []
        hub.send_command("set_evaluators", evaluators=1)
        pid = wait_for(lambda: _evaluator_pid(node), 30)
        wait_for(lambda: hub.lease_requests() == [1], 60)
        tasks = _tasks(cfg, hub, tmp_path, blocks=LONG)
        hub.lease("L1", tasks)
        wait_for(lambda: node.agent.evaluators.leases.idle() == [] and node.agent.evaluators.leases.up() == [0], 60)
        hub.send_command("set_evaluators", evaluators=0)
        start = time.monotonic()
        results = wait_for(lambda: (lambda rs: rs if len(rs) == len(tasks) else None)(hub.block_results("L1")), 30)
        assert {(r.failed, r.error_kind) for r in results} == {(True, "abandoned")}
        wait_for(lambda: not pid_alive(pid), SHUTDOWN_GRACE_SEC + 1.0)
        assert time.monotonic() - start <= SHUTDOWN_GRACE_SEC + 1.0
        beats = len(hub.heartbeats())
        wait_for(lambda: len(hub.heartbeats()) >= beats + 2, 30)
        assert "evaluator-0" not in _procs(hub)                                 # forgotten, not dead
        assert node.agent.counters()["worker_restarts"] == 0                     # P-PF4-9: no restart, no death
        assert hub.node_events("proc_died") == [] and hub.node_events("slot_disabled") == []
    finally:
        _stop(hub, node)


def test_cancel_leases_stops_playing_and_reports_nothing(tmp_path):
    cfg, hub, node = _setup(tmp_path)
    try:
        wait_for(lambda: hub.lease_requests() == [1], 60)
        hub.lease("L1", _tasks(cfg, hub, tmp_path, blocks=LONG))
        wait_for(lambda: node.agent.evaluators.leases.idle() == [] and node.agent.evaluators.leases.up() == [0], 60)
        hub.send_command("cancel_leases", lease_ids=["L1"])
        wait_for(lambda: len(hub.lease_requests()) == 2, 10)                    # the evaluator is idle again
        time.sleep(1.0)
        assert hub.block_results("L1") == []
        assert node.agent.evaluators.leases.held() == []
    finally:
        _stop(hub, node)


def test_an_evaluator_killed_mid_lease_is_restarted_and_its_blocks_reported_as_crash(tmp_path):
    cfg, hub, node = _setup(tmp_path)
    try:
        wait_for(lambda: hub.lease_requests() == [1], 60)
        pid = _evaluator_pid(node)
        tasks = _tasks(cfg, hub, tmp_path, blocks=LONG)
        hub.lease("L1", tasks)
        wait_for(lambda: node.agent.evaluators.leases.idle() == [] and node.agent.evaluators.leases.up() == [0], 60)
        beats = len(hub.heartbeats())
        os.kill(pid, signal.SIGKILL)
        results = wait_for(lambda: (lambda rs: rs if len(rs) == len(tasks) else None)(hub.block_results("L1")), 30)
        assert {(r.failed, r.error_kind) for r in results} == {(True, "crash")}
        (event,) = wait_for(lambda: hub.node_events("proc_died"), 30)
        assert event.proc == "evaluator-0" and event.log_path.endswith("evaluator-0.log") and event.log_tail
        wait_for(lambda: _procs(hub).get("evaluator-0") == ("alive", 1), 60)    # incarnation 2: one restart
        wait_for(lambda: len(hub.lease_requests()) == 2, 60)
        assert len(hub.heartbeats()) > beats                                     # the main loop never blocked
        assert node.agent.counters()["worker_restarts"] == 1
    finally:
        _stop(hub, node)


def test_a_failed_blob_fetch_releases_the_lease_as_missing_weights(tmp_path):
    cfg, hub, node = _setup(tmp_path)
    try:
        hub.refuse_downloads()
        wait_for(lambda: hub.lease_requests() == [1], 60)
        tasks = _tasks(cfg, hub, tmp_path)
        hub.lease("L1", tasks)
        results = wait_for(lambda: (lambda rs: rs if len(rs) == len(tasks) else None)(hub.block_results("L1")), 30)
        assert {(r.failed, r.error_kind) for r in results} == {(True, "missing_weights")}
        log = node.agent._run.log_dir / "evaluator-0.log"
        assert "lease" not in log.read_text()                                    # the evaluator never got it
    finally:
        _stop(hub, node)


def test_a_player_error_is_reported_with_its_kind_and_the_evaluator_keeps_running(tmp_path):
    cfg, hub, node = _setup(tmp_path)
    try:
        wait_for(lambda: hub.lease_requests() == [1], 60)
        pid = _evaluator_pid(node)
        bad = _tasks(cfg, hub, tmp_path, field=("bad",))
        hub.lease("L1", bad)
        results = wait_for(lambda: (lambda rs: rs if len(rs) == len(bad) else None)(hub.block_results("L1")), 60)
        number = node.agent.number
        for r in results:
            assert r.failed and r.error_kind == "player"
            assert "pulse failure on purpose" in r.error and f"node {number}, evaluator 0" in r.error
            assert str(node.agent._run.log_dir / "evaluator-0.log") in r.error
        wait_for(lambda: len(hub.lease_requests()) == 2, 30)
        good = _tasks(cfg, hub, tmp_path, owner="tournament:u")
        hub.lease("L2", good)
        results = wait_for(lambda: (lambda rs: rs if len(rs) == len(good) else None)(hub.block_results("L2")), 60)
        assert not any(r.failed for r in results) and _evaluator_pid(node) == pid
        assert hub.node_events("child_error") == []                              # never a ChildError of the run
    finally:
        _stop(hub, node)


def test_unused_evaluation_blobs_are_dropped_from_the_cache(tmp_path):
    cfg, hub, node = _setup(tmp_path)
    try:
        blob_ids = []
        for version in range(1, 11):
            wait_for(lambda version=version: len(hub.lease_requests()) == version, 60)
            tasks = _tasks(cfg, hub, tmp_path, version=version, field=("rnd",))
            blob_ids.append(next(p.blob.ident for t in tasks for p in t.players if p.blob is not None))
            hub.lease(f"L{version}", tasks)
            n = len(tasks)
            wait_for(lambda version=version, n=n: len(hub.block_results(f"L{version}")) == n, 60)

        def eval_segments():
            return sorted(name.split("-blob-", 1)[1] for name in os.listdir(shm_dir()) if "-blob-eval-" in name)

        kept = wait_for(lambda: (lambda segs: segs if len(segs) <= EVAL_BLOBS_KEPT else None)(eval_segments()), 30)
        assert set(kept) <= set(blob_ids[-EVAL_BLOBS_KEPT:]) and blob_ids[-1] in kept
    finally:
        _stop(hub, node)


def test_an_unanswered_request_is_asked_again(tmp_path, monkeypatch):
    """The hub caps a node's held plus waiting leases at its slots, so asking again never over-grants: asked slots
    without a lease within ``REASK_SEC`` are asked again (an evaluator never idles for good)."""
    monkeypatch.setattr(node_evaluators, "REASK_SEC", 0.5)
    _cfg, hub, node = _setup(tmp_path)
    try:
        wait_for(lambda: len(hub.lease_requests()) >= 3, 60)
        assert set(hub.lease_requests()) == {1}
    finally:
        _stop(hub, node)
