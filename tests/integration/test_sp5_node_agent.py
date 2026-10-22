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

import numpy as np
import pytest

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.server import SessionOpened
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.security import NodeIdentity, Token, new_secret
from colosseum.node import agent as agent_mod
from colosseum.node import lineups as lineups_mod
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


def test_an_unreachable_hub_is_retried_and_a_foreign_hub_certificate_is_refused(tmp_path, hub_factory, caplog):
    import socket

    from hub_kit import serve

    hub = hub_factory(_config())                                     # its token: the node's credentials
    with caplog.at_level(logging.WARNING), socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))                                # bound, not listening: nobody answers
        node = run_node_in_thread(toy_node_options(tmp_path, hub, insecure=False,
                                                   hub_address=f"127.0.0.1:{closed.getsockname()[1]}"))
        try:
            wait_for(lambda: len([r for r in caplog.records if "cannot reach the hub" in r.getMessage()]) >= 2)
            assert node.thread.is_alive()                            # retried, not refused
        finally:
            assert node.stop() == 0
    foreign = serve(tmp_path / "foreign")                            # TLS with another CA than the token's
    try:
        with caplog.at_level(logging.ERROR):
            refused = agent_mod.NodeAgent(toy_node_options(tmp_path, hub, insecure=False, hub_address=foreign.address,
                                                           identity_root=tmp_path / "foreign-id"))
            assert refused.run() == 1
        assert any("fingerprint" in r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    finally:
        foreign.stop()


def test_a_run_that_fails_to_start_releases_its_lock_listener_and_acceptor(tmp_path, hub_factory, monkeypatch):
    import threading

    from colosseum.net.local import address, listener
    from colosseum.node.weights import cache_prefix

    def broken(**kwargs):
        raise RuntimeError("a failure after the lock, the listener and the acceptor")

    monkeypatch.setattr(agent_mod, "_Run", broken)
    hub = hub_factory(_config())
    node = run_node_in_thread(toy_node_options(tmp_path, hub))
    node.thread.join(30)
    assert node.exit_code == 1
    prefix = cache_prefix(hub.token.ca_fingerprint, hub.run_id, 1)
    assert not list(shm_dir().glob(f"{prefix}*")) and not list(shm_dir().glob(f".{prefix}*"))
    assert not [t for t in threading.enumerate() if t.name.startswith("accept-node-") and t.is_alive()]
    listener(address(hub.token.ca_fingerprint[:8], hub.run_id[:8], 1, "node"), b"k" * 32).close()   # free again


def test_workers_start_on_welcome_and_ask_lineups_for_every_env(hub_factory, node_factory):
    hub = hub_factory(_config())
    hub.grant(lambda worker, env: FIXED)
    node = node_factory(hub)
    wait_for(lambda: hub.messages("lineup_request"))
    assert _requests(hub)[0] == [(0, 0, 0, 1), (0, 1, 0, 1)]
    wait_for(lambda: len(hub.results()) >= 20)
    pattern = _uid_re(hub, node.agent.number)
    played: dict[int, set[int]] = {}
    for result in hub.results():
        m = pattern.fullmatch(result.uid)
        assert m is not None and result.node == node.agent.number, result.uid
        played.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    # Each env plays one epoch the hub granted it (a retried first request may make the hub grant another, which
    # the node refuses); the two envs' epochs differ; the renewal after the TTL carries them.
    assert sorted(played) == [0, 1] and all(len(epochs) == 1 for epochs in played.values())
    epochs = {env: next(iter(e)) for env, e in played.items()}
    assert epochs[0] != epochs[1] and all(e >= 5 for e in epochs.values())
    assert all((0, env, epoch) in hub.grants_sent for env, epoch in epochs.items())
    wait_for(lambda: any(sorted(c[2] for c in cells) == sorted(epochs.values()) for cells in _requests(hub)[1:]),
             timeout=10)
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
    wait_for(lambda: [b.seq for b in hub.messages("results")].count(2) >= 2)
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
    assert node.agent.supervisor.incarnation("worker-0") == 2
    assert len(hub.node_events("proc_died")) == 1          # T4.2: a restart; the last death is slot_disabled
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

    asked = []                    # requests received when ckpt_v9 was granted (the grant's own request included)

    def lineup_for(worker, env):
        if env == 0 and not asked:
            asked.append(len(hub.messages("lineup_request")))
            return SNAP9
        return SNAP5 if env == 0 else FIXED

    hub.ttl_sec = 60.0            # from now on a renewal comes only after 60 s: a request for (0, 0) is a retry
    hub.grant(lineup_for)
    wait_for(lambda: ("snapshot", "a", "ckpt_v9") in hub.blobs.requests())
    wait_for(lambda: any(h.counters["lineup_fetch_failed"] == 1 for h in hub.heartbeats()))
    played = len(_results_with(hub, "ckpt_v5"))
    wait_for(lambda: len(_results_with(hub, "ckpt_v5")) > played)   # the old lineup keeps playing
    wait_for(lambda: any((0, 0) == c[:2] for cells in _requests(hub)[asked[0]:] for c in cells))
    assert node.stop() == 0
    hub.sync()
    assert hub.heartbeats()[-1].counters["lineup_fetch_failed"] == 1


def test_evictions_reach_the_workers_and_leave_the_cache(hub_factory, node_factory, monkeypatch, tmp_path):
    monkeypatch.setattr(lineups_mod, "CACHE_DROP_DELAY_SEC", 0.5)
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


def test_evictions_in_a_welcome_reach_the_workers(hub_factory, node_factory, tmp_path):
    """SP5 T3.1 (carried item of T2.5 (a2)): the hub keeps evictions while a node is away and hands them out with its
    next ``Welcome``; the node applies them like an ``Announce``'s."""
    config = _config()
    hub = hub_factory(config)
    _serve_snapshot(hub, config, 5)
    hub.grant(lambda worker, env: SNAP5 if env == 0 else FIXED)
    node = node_factory(hub)
    wait_for(lambda: _results_with(hub, "ckpt_v5"))
    hub.grant(lambda worker, env: FIXED)
    hub.welcome(evictions=[pb.Eviction(agent_id="a", checkpoint_id="ckpt_v5")])
    hub.close_session()                                         # the node comes back; its Welcome carries them
    log = tmp_path / "node-logs" / "worker-0.log"
    wait_for(lambda: log.exists() and "unloaded evicted snapshot ckpt_v5" in log.read_text())
    assert node.stop() == 0


def test_frozen_players_come_from_the_hubs_blobs(hub_factory, node_factory, tmp_path):
    import threading

    base = _config()
    path = write_ckpt_dir(tmp_path / "frozen", base, "a", policy_version=3)
    config = _config(agents={"a": {}, "random": scripted_agent(), "froz": frozen_agent(path)})
    hub = hub_factory(config)
    gate = threading.Event()
    blob_id = hub.serve_frozen("froz", path, gate=gate)
    hub.grant(lambda worker, env: Lineup("2p", [SeatAssignment("froz", "fixed", False),
                                                SeatAssignment("random", "fixed", False)]))
    node = node_factory(hub)
    wait_for(lambda: len(hub.heartbeats()) >= 2)
    assert all(not h.cached_keys for h in hub.heartbeats())        # not fetched yet: not reported as cached
    gate.set()
    wait_for(lambda: any(f"blob:froz:fixed-{blob_id}" in h.cached_keys for h in hub.heartbeats()))
    wait_for(lambda: [r for r in hub.results() if r.seats[0].agent_id == "froz"])
    order = [kind for kind, _ in hub.timeline]
    assert hub.timeline[0] == ("download", ("fixed", "froz", blob_id)) and order.index("lineup_request") > 0
    assert all(r.seats[0].network_id == "fixed" for r in hub.results())
    assert node.stop() == 0
    assert not list(shm_dir().glob(f"{node.agent.cache_prefix}*"))  # the node removed its cache files


def test_a_download_in_flight_at_the_stop_leaves_no_segment_behind(hub_factory, node_factory):
    import threading

    config = _config()
    hub = hub_factory(config)
    gate = threading.Event()
    _serve_snapshot(hub, config, 5, gate)
    hub.grant(lambda worker, env: SNAP5 if env == 0 else FIXED)
    node = node_factory(hub)
    wait_for(lambda: ("snapshot", "a", "ckpt_v5") in hub.blobs.requests())
    prefix = node.agent.cache_prefix
    assert node.stop() == 0                                         # the download is still held by the hub
    gate.set()
    time.sleep(0.5)
    assert not list(shm_dir().glob(f"{prefix}*")) and not list(shm_dir().glob(f".{prefix}*"))


def test_a_grant_naming_an_unknown_player_is_refused_with_one_warning(hub_factory, node_factory, caplog):
    ghost = Lineup("2p", [SeatAssignment("ghost", "fixed", False), SeatAssignment("random", "fixed", False)])
    hub = hub_factory(_config())
    hub.grant(lambda worker, env: ghost if env == 0 else FIXED)
    with caplog.at_level(logging.WARNING):
        node = node_factory(hub)
        wait_for(lambda: sum((0, 0) == c[:2] for cells in _requests(hub) for c in cells) >= 4)   # asked again
        assert node.stop() == 0
    refused = [r for r in caplog.records if "refused a lineup grant" in r.getMessage()]
    assert len(refused) == 1 and "ghost" in refused[0].getMessage()


# ---------------------------------------------------------------------------------------------------------------
# Learners on the node (T2.4 b)
# ---------------------------------------------------------------------------------------------------------------

SELF_PLAY = Lineup("2p", [SeatAssignment("a", "latest", True), SeatAssignment("a", "latest", False)])


def _learner_config(**sections):
    return _config(checkpoint={"interval": 3}, learner={"batch_chunks": 2, "weight_push_interval": 1},
                   **sections)


def _learner_hub(hub_factory, **kw):
    """A hub that assigns ``a``'s learner to the node and grants ``a`` vs ``a`` only once ``a``'s first version is
    announced."""
    hub = hub_factory(_learner_config(), **kw)
    hub.welcome(learners=[pb.LearnerAssignment(agent_id="a", generation=1, version_floor=0, device="cpu",
                                               env_steps=0)])
    hub.grant(lambda worker, env: SELF_PLAY if hub.versions("a") else None)
    return hub


def _train_steps(hub):
    return [t.train_step for m in hub.messages("metrics") for t in m.train if t.agent_id == "a"]


def test_a_learner_trains_on_the_node_and_uploads_its_snapshots(hub_factory, node_factory, monkeypatch):
    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    monkeypatch.setattr(node_session, "RECONNECT_MAX_SEC", 0.2)
    hub = _learner_hub(hub_factory)
    node = node_factory(hub)
    first = wait_for(lambda: hub.versions("a"), timeout=60)[0]
    assert (first.generation, first.version, first.node) == (1, 0, node.agent.number) and first.size > 0
    wait_for(lambda: max(_train_steps(hub), default=0) >= 3, timeout=60)
    wait_for(lambda: hub.uploads_of("checkpoint"), timeout=60)
    versions = [v.version for v in hub.versions("a")]
    assert versions == sorted(versions) and versions[-1] > 0
    wait_for(lambda: any(m.intake for m in hub.messages("metrics")))     # IntakeReport: every 1 s
    hub.announce(env_steps=1000)                                    # the hub's global counter: LR progress
    wait_for(lambda: any(t.values["progress"] > 0 for m in hub.messages("metrics") for t in m.train), timeout=60)
    hub.close_session()                                             # a reconnect: the Hello names the learner,
    wait_for(lambda: len(hub.events(SessionOpened)) >= 2)           # the second Welcome starts nothing new
    (running,) = hub.events(SessionOpened)[1].hello.learners
    assert (running.agent_id, running.generation) == ("a", 1) and running.version >= 0
    time.sleep(0.5)
    assert node.agent.supervisor.incarnation("learner-a") == 1 and node.agent.supervisor.get("learner-a").is_alive()
    prefix = node.agent.cache_prefix
    start = time.monotonic()
    hub.send_command("stop_run", reason="grace=5")
    node.thread.join(30)
    assert node.exit_code == 0 and time.monotonic() - start < 30
    hub.sync()
    (final,) = hub.uploads_of("final")
    assert final.agent_id == "a" and final.generation == 1 and final.version >= versions[-1]
    assert hub.node_events("run_stopped")
    assert not node.agent.supervisor.alive()
    assert not list(shm_dir().glob(f"{prefix}*")) and not list(shm_dir().glob(f".{prefix}*"))


def test_between_announces_the_learners_progress_grows_with_the_nodes_own_steps(hub_factory, node_factory):
    """Ruling R28 (a): one Announce of 1000 env steps, then none; the learner's LR progress keeps growing past
    1000 / total with this node's own env steps (forwarded at every WorkerProgress)."""
    hub = _learner_hub(hub_factory)
    node = node_factory(hub)
    wait_for(lambda: max(_train_steps(hub), default=0) >= 1, timeout=60)
    hub.announce(env_steps=1000)
    floor = 1000 / hub.config.training.total_timesteps
    wait_for(lambda: any(t.values["progress"] > floor for m in hub.messages("metrics") for t in m.train
                         if t.agent_id == "a"), timeout=60)
    assert node.stop() == 0


def test_before_any_announce_the_learners_progress_grows_with_the_nodes_own_steps(hub_factory, node_factory):
    """FX3: the Welcome's count (0) is the base: with no Announce at all the learner's LR progress still moves with
    this node's own env steps (the hub's first Announce with steps comes one or two heartbeats late)."""
    hub = _learner_hub(hub_factory)
    node = node_factory(hub)
    wait_for(lambda: any(t.values["progress"] > 0 for m in hub.messages("metrics") for t in m.train
                         if t.agent_id == "a"), timeout=60)
    assert node.stop() == 0


def test_a_killed_learner_blocks_neither_the_worker_nor_the_node(hub_factory, node_factory):
    hub = _learner_hub(hub_factory)
    node = node_factory(hub)
    wait_for(lambda: max(_train_steps(hub), default=0) >= 2, timeout=60)
    learner = node.agent.supervisor.get("learner-a")
    os.kill(learner.pid, signal.SIGKILL)
    died = wait_for(lambda: hub.node_events("proc_died"))[0]               # T4.2: the hub decides the restart
    assert died.proc == "learner-a" and died.exit_code == -signal.SIGKILL and died.generation == 1
    results, beats = len(hub.results()), len(hub.heartbeats())
    wait_for(lambda: len(hub.results()) > results + 10 and len(hub.heartbeats()) > beats + 2)

    def drops():
        finals = {}
        for m in hub.messages("metrics"):
            for w in m.workers:
                finals[(w.worker, w.incarnation)] = dict(w.counters)
        return sum(c.get("dropped_chunks_no_learner", 0) + c.get("dropped_chunks_disconnect", 0)
                   for c in finals.values())

    wait_for(lambda: drops() > 0)                                  # its chunks are dropped and counted
    assert node.agent.supervisor.get("worker-0").exitcode is None
    assert node.stop() == 0


def test_children_die_with_their_node(tmp_path, hub_factory):
    from node_kit import pid_alive, spawned_children, start_node_process

    hub = _learner_hub(hub_factory)
    proc = start_node_process(toy_node_options(tmp_path, hub), tmp_path / "node-process-logs")
    try:
        wait_for(lambda: hub.versions("a") and hub.results(), timeout=60)
        children = wait_for(lambda: (lambda c: c if len(c) >= 2 else None)(spawned_children(proc.pid)))
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(10)
        wait_for(lambda: not any(pid_alive(pid) for pid in children), timeout=5)
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(5)


def test_stop_finishes_within_the_local_grace(hub_factory, node_factory):
    hub = _learner_hub(hub_factory)
    node = node_factory(hub, shutdown_grace_sec=agent_mod.LOCAL_CHILD_GRACE_SEC)
    wait_for(lambda: max(_train_steps(hub), default=0) >= 2, timeout=60)
    start = time.monotonic()
    hub.send_command("stop_run", reason="grace=5")
    node.thread.join(30)
    assert time.monotonic() - start < agent_mod.LOCAL_CHILD_GRACE_SEC + 2.0 and node.exit_code == 0
    hub.sync()
    assert hub.uploads_of("final")


def test_stop_learner_uploads_its_final_and_release_keeps_the_session(hub_factory, node_factory, caplog):
    hub = _learner_hub(hub_factory)
    hub.upload_reply = lambda header, data: pb.UploadReply(accepted=header.kind != "final", reason="test refusal")
    with caplog.at_level(logging.WARNING):
        node = node_factory(hub)
        wait_for(lambda: max(_train_steps(hub), default=0) >= 2, timeout=60)
        hub.send_command("stop_learner", learner=pb.LearnerAssignment(agent_id="a", generation=1), reason="remove")
        wait_for(lambda: hub.uploads_of("final"), timeout=30)
        wait_for(lambda: node.agent.supervisor.get("learner-a").exitcode is not None)
        wait_for(lambda: any("did not take the final snapshot" in r.getMessage() and r.levelno == logging.ERROR
                             for r in caplog.records))
        assert not hub.node_events("learner_failed")
        worker = node.agent.supervisor.get("worker-0")
        prefix = node.agent.cache_prefix
        hub.send_command("release", reason="test")
        wait_for(lambda: hub.node_events("run_stopped"))
        wait_for(lambda: node.agent.run_id is None)
        assert node.thread.is_alive() and worker.exitcode is not None
        assert not list(shm_dir().glob(f"{prefix}*"))
        sessions = len(hub.events(SessionOpened))
        beats = len(hub.heartbeats())
        wait_for(lambda: len(hub.heartbeats()) > beats + 1)          # the session stays, with heartbeats
        assert len(hub.events(SessionOpened)) == sessions
        assert node.stop() == 0
    hub.sync()
    assert not hub.node_events("learner_failed")                  # also after the node handled every exit



def test_an_oversized_checkpoint_fails_the_learner_with_its_size_and_the_setting(hub_factory, node_factory,
                                                                                 monkeypatch):
    monkeypatch.setattr(agent_mod.NodeAgent, "_frame_cap", lambda self: 2048)   # the node's cap only (b)
    hub = _learner_hub(hub_factory)
    node_factory(hub)
    died = wait_for(lambda: hub.node_events("proc_died"), timeout=60)[0]
    assert "max_message_mb" in died.message and "checkpoint v" in died.message and died.proc == "learner-a"
    (warning,) = [e for e in hub.node_events("warning") if e.proc == "learner-a"]
    assert warning.message == died.message


def test_a_learner_that_fails_to_start_reports_why_and_is_never_routed_to(hub_factory, node_factory):
    from colosseum.net.chunks import WeightsMeta, encode_weights

    hub = _learner_hub(hub_factory)
    frame = encode_weights({"bogus": np.zeros(3, np.float32)}, WeightsMeta(agent_id="a", kind="init", generation=0,
                                                                          policy_version=0))
    hub.serve_blob("init", "a", "bad-init", frame)
    hub.welcome(fixed=[pb.FixedBlob(agent_id="a", blob_id="bad-init", kind="init")])
    node = node_factory(hub)
    error = wait_for(lambda: hub.node_events("child_error"), timeout=60)[0]
    assert error.proc == "learner-a" and error.error_kind == "crash" and "learner a (generation 1)" in error.context
    died = wait_for(lambda: hub.node_events("proc_died"))[0]
    assert died.proc == "learner-a" and died.message == f"crash: {error.message}"      # its death names the reason
    assert node.agent._learner_routes.get("a") is None                # no worker was sent to its listener
    assert node.stop() == 0


def _start_point(hub, config, version):
    """The start point ``start:resume`` of ``a`` at policy version ``version`` (A28: one checkpoint frame with the
    weights and the trainer-state bytes)."""
    from colosseum.core.registry import env_spec
    from colosseum.core.roles import agent_role_spec, resolve_agent_roles
    from colosseum.learner.factory import build_algorithm
    from colosseum.learner.learner import make_checkpoint_payload, set_policy_version
    from colosseum.net.chunks import encode_checkpoint

    spec = env_spec(config)
    role = agent_role_spec(spec, resolve_agent_roles(config, spec)["a"])
    algo = build_algorithm(config.get_agent_config("a"), role, spec, device="cpu", teacher=None)
    set_policy_version(algo, version)
    hub.serve_blob("start", "a", "resume", encode_checkpoint(make_checkpoint_payload("a", algo), kind="recovery",
                                                             generation=0))


def test_a_learner_resumes_from_its_start_blobs_and_takes_its_version_floor(hub_factory, node_factory):
    hub = hub_factory(_learner_config())
    _start_point(hub, hub.config, 7)
    hub.welcome(learners=[pb.LearnerAssignment(agent_id="a", generation=1, version_floor=0, device="cpu",
                                               weights_blob="start:resume")])
    hub.grant(lambda worker, env: SELF_PLAY if hub.versions("a") else None)
    node = node_factory(hub)
    assert wait_for(lambda: hub.versions("a"), timeout=60)[0].version == 7      # the resumed version
    assert ("start", "a", "resume") in set(hub.blobs.requests())
    hub.send_command("stop_learner", learner=pb.LearnerAssignment(agent_id="a", generation=1), reason="test")
    wait_for(lambda: node.agent.supervisor.get("learner-a").exitcode is not None and not node.agent.learners.agents())
    announced = len(hub.versions("a"))
    hub.send_command("start_learner", learner=pb.LearnerAssignment(
        agent_id="a", generation=1, version_floor=12, device="cpu", weights_blob="start:resume"))
    wait_for(lambda: len(hub.versions("a")) > announced, timeout=60)
    assert hub.versions("a")[announced].version == 12                         # the floor lifts the resumed 7
    assert node.stop() == 0


def test_a_start_state_that_cannot_be_fetched_is_retried_and_never_started(hub_factory, node_factory, monkeypatch):
    """T4.1 (Step 7.1): a start point the hub does not serve is fetched again at later passes; the learner never
    starts without it and no learner_failed is reported."""
    from colosseum.node import learners as node_learners

    monkeypatch.setattr(node_learners, "START_RETRY_SEC", 0.1)
    hub = hub_factory(_learner_config())
    hub.welcome(learners=[pb.LearnerAssignment(agent_id="a", generation=1, device="cpu",
                                               weights_blob="start:recovery")])                # never served
    node = node_factory(hub)
    wait_for(lambda: hub.blobs.requests().count(("start", "a", "recovery")) >= 3)
    assert node.agent.supervisor.get("learner-a") is None                     # never spawned
    assert hub.node_events("learner_failed") == []
    assert node.stop() == 0


def test_a_final_not_uploaded_before_the_deadline_is_a_warning_event(hub_factory, node_factory):
    hub = _learner_hub(hub_factory)

    def slow_final(header, data):
        if header.kind == "final":
            time.sleep(4.0)                                         # the hub answers after the node's deadline
        return pb.UploadReply(accepted=True, checkpoint_id=f"ckpt_v{header.version}")

    hub.upload_reply = slow_final
    node = node_factory(hub, shutdown_grace_sec=1.0)
    wait_for(lambda: max(_train_steps(hub), default=0) >= 2, timeout=60)
    assert node.stop() == 0
    hub.sync()
    warnings = [e for e in hub.node_events("warning") if e.proc == "learner-a"]
    assert len(warnings) == 1 and "not uploaded before the stop deadline" in warnings[0].message
    kinds = [e.kind for e in hub.messages("event")]
    assert kinds.index("warning") < kinds.index("run_stopped")


def test_a_release_counts_given_up_results_once(hub_factory, node_factory):
    hub = hub_factory(_config(distributed={"heartbeat_sec": 0.02}))       # a heartbeat due in every pass
    hub.grant(lambda worker, env: FIXED)
    node = node_factory(hub, shutdown_grace_sec=1.0)
    wait_for(lambda: len(hub.results()) >= 20)
    hub.hold_acks = True                                            # what is unacked at the release is given up
    wait_for(lambda: len(hub.messages("results")) >= 3 + len(hub.acked))
    hub.send_command("release", reason="test")
    wait_for(lambda: hub.node_events("run_stopped"))
    beats = len(hub.heartbeats())
    wait_for(lambda: len(hub.heartbeats()) > beats + 5)
    assert node.stop() == 0
    hub.sync()
    counts = [h.counters["dropped_results_node_buffer"] for h in hub.heartbeats()]
    assert counts == sorted(counts) and counts[-1] > 0                # cumulative: never down, never twice


def test_a_download_in_flight_at_a_release_leaves_no_segment_behind(hub_factory, node_factory):
    import threading

    config = _config()
    hub = hub_factory(config)
    gate = threading.Event()
    _serve_snapshot(hub, config, 5, gate)
    hub.grant(lambda worker, env: SNAP5 if env == 0 else FIXED)
    node = node_factory(hub, shutdown_grace_sec=1.0)
    wait_for(lambda: ("snapshot", "a", "ckpt_v5") in hub.blobs.requests())
    prefix = node.agent.cache_prefix
    hub.send_command("release", reason="test")
    wait_for(lambda: node.agent.run_id is None)
    gate.set()                                                      # the download completes after the release
    time.sleep(1.0)
    assert not list(shm_dir().glob(f"{prefix}*")) and not list(shm_dir().glob(f".{prefix}*"))
    assert node.stop() == 0


def test_a_child_killed_with_an_unread_message_is_an_exit_not_a_read_failure(hub_factory, node_factory, caplog):
    hub = _learner_hub(hub_factory)
    with caplog.at_level(logging.WARNING):
        node = node_factory(hub)
        wait_for(lambda: max(_train_steps(hub), default=0) >= 2, timeout=60)
        learner, worker = node.agent.supervisor.get("learner-a"), node.agent.supervisor.get("worker-0")
        for proc in (learner, worker):
            os.kill(proc.pid, signal.SIGSTOP)                       # it reads nothing from now on
        hub.announce(env_steps=500, evictions=[pb.Eviction(agent_id="a", checkpoint_id="ckpt_v1")])
        time.sleep(1.0)                                             # LearnerEnvSteps / WorkerEvict stay unread
        for proc in (learner, worker):
            os.kill(proc.pid, signal.SIGKILL)                       # (Linux: the node then reads ECONNRESET)
        failed = wait_for(lambda: [e for e in hub.node_events("proc_died") if e.proc == "learner-a"])[0]
        wait_for(lambda: [e for e in hub.node_events("proc_died") if e.proc == "worker-0"])
        assert node.stop() == 0
    assert "could not read" not in failed.message
    assert not [r for r in caplog.records if "failed (" in r.getMessage() and "reading from" in r.getMessage()]
