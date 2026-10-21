"""The node agent against a fake hub (SP5 T2.4; spec 1, 2.6, 4.1-4.4, 4.6, 4.7, 7.1; rulings R11, R13): registration
and identity, workers started on the Welcome asking lineups for every env, results batched / acked / resent after a
reconnect / kept while the hub is away, cumulative env steps across a worker restart, the restart limit, a reported
contract error, lineups after their weights are cached (snapshots, failed fetches, evictions, frozen players' blobs).

The node runs in a thread of the test process (``node_kit.NodeThread``) against ``node_kit.FakeHub`` (the real gRPC
server); one worker with two envs of the toy game ``turns`` (one loaded process per test)."""
from __future__ import annotations

import logging
import os
import re
import signal
import time

import pytest

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.server import SessionOpened
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.security import NodeIdentity, Token, new_secret
from colosseum.node import agent as agent_mod
from colosseum.node import procs
from colosseum.node import session as node_session
from colosseum.node.weights import shm_dir
from game_helpers import frozen_agent, make_test_config, scripted_agent, write_ckpt_dir
from node_kit import FakeHub, run_node_in_thread, toy_node_options, wait_for

FIXED = Lineup("2p", [SeatAssignment("random", "fixed", False), SeatAssignment("bot2", "fixed", False)])
SNAP5 = Lineup("2p", [SeatAssignment("a", "ckpt_v5", False), SeatAssignment("random", "fixed", False)])
SNAP9 = Lineup("2p", [SeatAssignment("a", "ckpt_v9", False), SeatAssignment("random", "fixed", False)])


def _config(*, agents=None, distributed=None, **sections):
    agents = agents or {"a": {}, "random": scripted_agent(), "bot2": scripted_agent()}
    rollout = {"envs_per_worker": 2, "chunk_length": 8, **sections.pop("rollout", {})}
    return make_test_config("turns", agents=agents, rollout=rollout,
                            distributed={"heartbeat_sec": 0.25, **(distributed or {})}, **sections)


@pytest.fixture
def hub_factory(tmp_path):
    hubs = []

    def make(config, **kw):
        hub = FakeHub(tmp_path, config, **kw)
        hubs.append(hub)
        return hub

    yield make
    for hub in hubs:
        hub.stop()


@pytest.fixture
def node_factory(tmp_path):
    nodes = []

    def make(hub, **kw):
        node = run_node_in_thread(toy_node_options(tmp_path, hub, **kw))
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        if node.thread.is_alive():
            node.stop()


def _uid_re(hub, number):
    return re.compile(rf"{hub.run_id}/train/n{number}/w0_e(\d)_g(\d+)_ep(\d+)")


def _results_with(hub, network):
    return [r for r in hub.results() if any(s.network_id == network for s in r.seats)]


def _requests(hub):
    return [[(c.worker, c.env, c.epoch, c.incarnation) for c in m.cells] for m in hub.messages("lineup_request")]


def _worker_pid(node, incarnation):
    def pid():
        proc = node.agent.supervisor.get("worker-0")
        if node.agent.supervisor.incarnation("worker-0") == incarnation and proc is not None and proc.pid:
            return proc.pid
        return None
    return wait_for(pid)


# ---------------------------------------------------------------------------------------------------------------
# Identity, workers, lineups
# ---------------------------------------------------------------------------------------------------------------


def test_the_node_registers_with_the_token_and_keeps_its_identity(tmp_path, hub_factory, node_factory, caplog):
    hub = hub_factory(_config())
    node = node_factory(hub, workers=0)
    opened = wait_for(lambda: hub.events(SessionOpened))
    identity = NodeIdentity.load_or_create(hub.token.ca_fingerprint, tmp_path / "node-id")
    assert identity.registered and identity.number == 1 and opened[0].hello.node_id == identity.node_id
    assert node.stop() == 0 and len(hub.registrations) == 1

    again = node_factory(hub, workers=0)                            # the same identity root: no new Register
    wait_for(lambda: len(hub.events(SessionOpened)) >= 2)
    assert hub.events(SessionOpened)[1].hello.node_id == identity.node_id and len(hub.registrations) == 1
    assert again.stop() == 0

    wrong = Token(new_secret(), hub.token.ca_fingerprint)
    with caplog.at_level(logging.ERROR):
        refused = agent_mod.NodeAgent(toy_node_options(tmp_path, hub, token=wrong, identity_root=tmp_path / "other"))
        assert refused.run() == 1
    assert any("refused the registration" in r.getMessage() for r in caplog.records)


def test_workers_start_on_welcome_and_ask_lineups_for_every_env(hub_factory, node_factory):
    hub = hub_factory(_config())
    hub.grant(lambda worker, env: FIXED)
    node = node_factory(hub)
    wait_for(lambda: hub.messages("lineup_request"))
    assert _requests(hub)[0] == [(0, 0, 0, 1), (0, 1, 0, 1)]
    wait_for(lambda: len(hub.results()) >= 20)
    pattern = _uid_re(hub, node.agent.number)
    for result in hub.results():
        m = pattern.fullmatch(result.uid)
        assert m is not None, result.uid
        assert (m.group(1), m.group(2)) in {("0", "5"), ("1", "6")} and result.node == node.agent.number
    wait_for(lambda: any(sorted(c[2] for c in cells) == [5, 6] for cells in _requests(hub)[1:]), timeout=10)
    assert node.stop() == 0


def test_results_are_batched_and_acked(hub_factory, node_factory):
    hub = hub_factory(_config(distributed={"results_batch": 4}))
    hub.grant(lambda worker, env: FIXED)
    node = node_factory(hub)
    wait_for(lambda: len(hub.messages("results")) >= 30)
    assert node.stop() == 0
    hub.sync()
    batches = hub.messages("results")
    seqs = [b.seq for b in batches]
    assert all(0 < len(b.results) <= 4 for b in batches)
    assert seqs == list(range(1, len(seqs) + 1))                     # acked batches are never sent again
    assert sorted(hub.acked) == seqs
    assert hub.node_events("run_stopped")


def test_a_refused_batch_is_resent_on_the_live_session(hub_factory, node_factory, monkeypatch):
    monkeypatch.setattr(agent_mod, "RESULT_RESEND_SEC", 0.5)        # P-PF2-4 (a)
    hub = hub_factory(_config(distributed={"results_batch": 4}))
    hub.grant(lambda worker, env: FIXED)
    hub.refuse_once = {2}
    node = node_factory(hub)
    wait_for(lambda: [b.seq for b in hub.messages("results")].count(2) == 2)
    wait_for(lambda: 2 in hub.acked)
    assert node.stop() == 0
    hub.sync()
    assert len({e.serial for e in hub.node_messages("results")}) == 1                 # no reconnect
    assert len(hub.accounted) == len(set(hub.accounted)) and node.agent.results.dropped == 0


def test_results_the_hub_never_acks_are_counted_at_the_stop(hub_factory, node_factory, caplog):
    hub = hub_factory(_config(distributed={"results_batch": 4}))     # P-PF2-4 (b)
    hub.grant(lambda worker, env: FIXED)
    hub.hold_acks = True
    with caplog.at_level(logging.WARNING):
        node = node_factory(hub, shutdown_grace_sec=1.0)
        wait_for(lambda: len(hub.messages("results")) >= 5)
        assert node.stop() == 0
    hub.sync()
    assert node.agent.results.dropped == len(set(hub.accounted)) > 0
    assert hub.heartbeats()[-1].counters["dropped_results_node_buffer"] == node.agent.results.dropped
    assert len([r for r in caplog.records if "gave up" in r.getMessage()]) == 1


def test_unacked_results_are_resent_after_a_reconnect_and_accounted_once(hub_factory, node_factory, monkeypatch):
    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    monkeypatch.setattr(node_session, "RECONNECT_MAX_SEC", 0.2)
    hub = hub_factory(_config(distributed={"results_batch": 16}))
    hub.grant(lambda worker, env: FIXED)
    hub.hold_acks = True
    node = node_factory(hub)
    wait_for(lambda: len(hub.messages("results")) >= 5)
    first_serial = hub.node_messages("results")[0].serial
    hub.close_session()
    opened = wait_for(lambda: (lambda o: o[1:] if len(o) >= 2 else None)(hub.events(SessionOpened)))
    assert opened[0].hello.session_id == hub.events(SessionOpened)[0].hello.session_id
    assert opened[0].hello.last_acked_seq == 0
    wait_for(lambda: len([e for e in hub.node_messages("results") if e.serial != first_serial]) >= 5)
    resent = [e.message.results.seq for e in hub.node_messages("results") if e.serial != first_serial]
    assert resent[:3] == [1, 2, 3]                                   # every unacked batch again, in seq order
    hub.hold_acks = False
    hub.release_acks()
    assert node.stop() == 0
    hub.sync()
    finals = {}
    for m in hub.messages("metrics"):
        for w in m.workers:
            finals[(w.worker, w.incarnation)] = dict(w.counters)
    episodes = sum(c["episodes"] for c in finals.values())
    dropped = sum(c["dropped_results"] for c in finals.values())
    assert len(hub.accounted) == len(set(hub.accounted)) > 0
    assert len(hub.accounted) + dropped == episodes and node.agent.results.dropped == 0


def test_results_wait_in_the_buffer_while_the_hub_is_away(hub_factory, node_factory, monkeypatch, caplog):
    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    monkeypatch.setattr(node_session, "RECONNECT_MAX_SEC", 0.2)
    hub = hub_factory(_config(distributed={"results_batch": 4, "node_result_buffer_mb": 0.002}))
    hub.grant(lambda worker, env: FIXED)
    with caplog.at_level(logging.WARNING):
        node = node_factory(hub)
        wait_for(lambda: len(hub.messages("results")) >= 5)
        hub.pause()
        wait_for(lambda: node.agent.results.dropped > 0)
        sent_before = max(b.seq for b in hub.messages("results"))
        hub.resume()
        wait_for(lambda: len(hub.events(SessionOpened)) >= 2)
        wait_for(lambda: max(b.seq for b in hub.messages("results")) > sent_before + 2)
        assert node.stop() == 0
    hub.sync()
    assert len([r for r in caplog.records if "node result buffer full" in r.getMessage()]) == 1
    seqs = sorted({b.seq for b in hub.messages("results")})
    assert seqs == list(range(1, seqs[-1] + 1))                      # the kept batches arrived after the return
    last = hub.heartbeats()[-1]
    assert last.dropped_results == last.counters["dropped_results_node_buffer"] == node.agent.results.dropped > 0


# ---------------------------------------------------------------------------------------------------------------
# Worker deaths and errors
# ---------------------------------------------------------------------------------------------------------------


def test_heartbeats_carry_cumulative_env_steps_across_a_worker_restart(hub_factory, node_factory):
    hub = hub_factory(_config())
    hub.grant(lambda worker, env: FIXED)
    node = node_factory(hub)
    wait_for(lambda: any(h.env_steps > 0 for h in hub.heartbeats()))
    os.kill(_worker_pid(node, 1), signal.SIGKILL)
    died = wait_for(lambda: hub.node_events("proc_died"))
    assert died[0].proc == "worker-0" and died[0].exit_code == -signal.SIGKILL
    before = max(h.env_steps for h in hub.heartbeats())
    wait_for(lambda: any((0, 0, 0, 2) in cells for cells in _requests(hub)))
    wait_for(lambda: hub.heartbeats()[-1].env_steps > before)
    assert node.stop() == 0
    hub.sync()
    steps = [h.env_steps for h in hub.heartbeats()]
    assert steps == sorted(steps)
    assert hub.heartbeats()[-1].counters["worker_restarts"] == 1


def test_a_worker_beyond_its_restart_limit_disables_its_slot(hub_factory, node_factory, monkeypatch):
    monkeypatch.setattr(procs, "WORKER_RESTARTS", (1, 600.0))
    hub = hub_factory(_config())
    hub.grant(lambda worker, env: FIXED)
    node = node_factory(hub)
    os.kill(_worker_pid(node, 1), signal.SIGKILL)
    os.kill(_worker_pid(node, 2), signal.SIGKILL)
    disabled = wait_for(lambda: hub.node_events("slot_disabled"))
    assert disabled[0].proc == "worker-0"
    wait_for(lambda: any(pb.ProcState(name="worker-0", state="disabled", restarts=1) in h.procs
                         for h in hub.heartbeats()))
    time.sleep(0.5)
    assert node.agent.supervisor.incarnation("worker-0") == 2 and len(hub.node_events("proc_died")) == 2
    assert node.stop() == 0


def test_a_contract_error_is_reported_with_its_context_and_not_restarted(hub_factory, node_factory):
    hub = hub_factory(_config(env={"env_class": "game_helpers.ContractBreakingGame"}))
    hub.grant(lambda worker, env: FIXED)
    node = node_factory(hub)
    error = wait_for(lambda: hub.node_events("child_error"))[0]
    assert error.proc == "worker-0" and error.error_kind == "contract"
    assert error.context.startswith(f"node {node.agent.number} (")
    assert "worker 0, env" in error.context and "episode step" in error.context and "layout 2p" in error.context
    assert error.log_path.endswith("worker-0.log")
    wait_for(lambda: node.agent.supervisor.get("worker-0").exitcode is not None)
    time.sleep(0.5)
    assert node.agent.supervisor.incarnation("worker-0") == 1 and not hub.node_events("proc_died")
    assert node.stop() == 0


# ---------------------------------------------------------------------------------------------------------------
# Lineups after their weights (R11)
# ---------------------------------------------------------------------------------------------------------------


def _serve_snapshot(hub, config, version, gate=None):
    path = write_ckpt_dir(hub.run_dir.checkpoints / "a", config, "a", policy_version=version)
    if gate is not None:
        hub.blobs.gates[("snapshot", "a", path.name)] = gate
    return path


def test_a_lineup_waits_for_its_snapshot_and_a_failed_fetch_keeps_the_old_one(hub_factory, node_factory):
    import threading

    config = _config()
    hub = hub_factory(config)
    gate = threading.Event()
    _serve_snapshot(hub, config, 5, gate)
    hub.not_found("snapshot", "a", "ckpt_v9")
    hub.grant(lambda worker, env: FIXED)
    node = node_factory(hub)
    wait_for(lambda: len(hub.results()) >= 10)
    hub.grant(lambda worker, env: SNAP5 if env == 0 else FIXED)
    wait_for(lambda: ("snapshot", "a", "ckpt_v5") in hub.blobs.requests())
    time.sleep(1.5)
    assert not _results_with(hub, "ckpt_v5")                        # the old lineup plays while the fetch waits
    gate.set()
    wait_for(lambda: _results_with(hub, "ckpt_v5"))

    once = []

    def lineup_for(worker, env):
        if env == 0 and not once:
            once.append(True)
            return SNAP9
        return SNAP5 if env == 0 else FIXED

    hub.grant(lineup_for)
    wait_for(lambda: ("snapshot", "a", "ckpt_v9") in hub.blobs.requests())
    wait_for(lambda: any(h.counters["lineup_fetch_failed"] == 1 for h in hub.heartbeats()))
    played = len(_results_with(hub, "ckpt_v5"))
    asked = len(hub.messages("lineup_request"))
    wait_for(lambda: len(_results_with(hub, "ckpt_v5")) > played)   # the old lineup keeps playing
    wait_for(lambda: any((0, 0) == c[:2] for cells in _requests(hub)[asked:] for c in cells))
    assert node.stop() == 0
    hub.sync()
    assert hub.heartbeats()[-1].counters["lineup_fetch_failed"] == 1


def test_evictions_reach_the_workers_and_leave_the_cache(hub_factory, node_factory, monkeypatch, tmp_path):
    monkeypatch.setattr(agent_mod, "CACHE_DROP_DELAY_SEC", 0.5)
    config = _config()
    hub = hub_factory(config)
    _serve_snapshot(hub, config, 5)
    hub.grant(lambda worker, env: SNAP5 if env == 0 else FIXED)
    node = node_factory(hub)
    wait_for(lambda: _results_with(hub, "ckpt_v5"))
    segment = shm_dir() / f"{node.agent.cache_prefix}-a-ckpt_v5"
    assert segment.exists()
    hub.grant(lambda worker, env: FIXED)
    hub.announce(evictions=[pb.Eviction(agent_id="a", checkpoint_id="ckpt_v5")])
    log = tmp_path / "node-logs" / "worker-0.log"
    wait_for(lambda: log.exists() and "unloaded evicted snapshot ckpt_v5" in log.read_text())
    wait_for(lambda: not segment.exists())
    assert node.stop() == 0


def test_frozen_players_come_from_the_hubs_blobs(hub_factory, node_factory, tmp_path):
    base = _config()
    path = write_ckpt_dir(tmp_path / "frozen", base, "a", policy_version=3)
    config = _config(agents={"a": {}, "random": scripted_agent(), "froz": frozen_agent(path)})
    hub = hub_factory(config)
    blob_id = hub.serve_frozen("froz", path)
    hub.grant(lambda worker, env: Lineup("2p", [SeatAssignment("froz", "fixed", False),
                                                SeatAssignment("random", "fixed", False)]))
    node = node_factory(hub)
    wait_for(lambda: [r for r in hub.results() if r.seats[0].agent_id == "froz"])
    order = [kind for kind, _ in hub.timeline]
    assert hub.timeline[0] == ("download", ("fixed", "froz", blob_id)) and order.index("lineup_request") > 0
    assert all(r.seats[0].network_id == "fixed" for r in hub.results())
    assert node.stop() == 0
    assert not list(shm_dir().glob(f"{node.agent.cache_prefix}*"))  # the node removed its cache files
