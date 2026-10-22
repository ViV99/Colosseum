"""A node while its hub is away (SP5 T4.4, spec 4.5, 7.1): lineups outlive their TTL, results wait and arrive once,
a relay-only learner stops getting data after hub_lost_collect_sec, processes stop after hub_lost_timeout_sec and
come back with the hub, a full cache stops new work.

A ``NodeAgent`` in a thread (``node_kit.NodeThread``) reaches ``node_kit.FakeHub`` (the real gRPC server) through a
``fault_proxy.TcpFaultProxy`` (the hub "goes away" by ``proxy.cut``); the toy game ``turns`` with one worker of two
envs, lineup TTL 0.5 s, heartbeats every 0.1 s, ``hub_lost_collect_sec`` 1 s, ``shutdown_grace_sec`` 3 s. The node's
reconnect backoff is shrunk (pre-flight ruling P-PF3-8) and so is the slow retry after the hub-lost timeout
(``HUB_LOST_RETRY_SEC``); ``hub_lost_timeout_sec`` is 3-3.5 s where a cut must not reach it (each window at least
the cut plus 1 s, P-PF3-8) and 2 s where it must. At most two loaded processes per test (the worker, the learner of
``a``)."""
from __future__ import annotations

import logging

import pytest

from colosseum.core.registry import build_model
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.server import NodeMessageIn, SessionOpened
from colosseum.net.chunks import WeightsMeta, encode_weights
from colosseum.net.messages import VersionInfo, version_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node import session as node_session
from colosseum.node import weights
from fault_proxy import TcpFaultProxy
from game_helpers import agent_role_of, make_test_config, scripted_agent, write_ckpt_dir
from node_kit import FakeHub, pid_alive, run_node_in_thread, toy_node_options, wait_for

FIXED = Lineup("2p", [SeatAssignment("random", "fixed", False), SeatAssignment("bot2", "fixed", False)])
SNAP5 = Lineup("2p", [SeatAssignment("a", "ckpt_v5", False), SeatAssignment("random", "fixed", False)])
A_VS_RANDOM = Lineup("2p", [SeatAssignment("a", "latest", True), SeatAssignment("random", "fixed", False)])
B_VS_RANDOM = Lineup("2p", [SeatAssignment("b", "latest", True), SeatAssignment("random", "fixed", False)])
TTL_SEC = 0.5


def _config(*, timeout=3.0, agents=None, distributed=None):
    agents = agents or {"a": {}, "random": scripted_agent(), "bot2": scripted_agent()}
    return make_test_config("turns", agents=agents, rollout={"envs_per_worker": 2, "chunk_length": 8},
                            learner={"batch_chunks": 2, "weight_push_interval": 1},
                            distributed={"heartbeat_sec": 0.1, "hub_lost_collect_sec": 1.0,
                                         "hub_lost_timeout_sec": timeout, "shutdown_grace_sec": 3.0,
                                         **(distributed or {})})


@pytest.fixture
def away(tmp_path, monkeypatch):
    """``start(config, grant, relay=False, **node_options) -> (hub, proxy, node)``: the FakeHub, the proxy in front of
    it and a node connected through the proxy; everything is stopped at the end."""
    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    monkeypatch.setattr(node_session, "RECONNECT_MAX_SEC", 0.2)
    monkeypatch.setattr(node_session, "HUB_LOST_RETRY_SEC", (0.2, 0.4))
    made = []

    def start(config, grant, *, relay=False, setup=None, **node_options):
        hub = FakeHub(tmp_path, config, ttl_sec=TTL_SEC, relay=relay)
        proxy = TcpFaultProxy(("127.0.0.1", hub.port))
        made.append((hub, proxy, None))
        if setup is not None:
            setup(hub)
        hub.grant(grant)
        node = run_node_in_thread(toy_node_options(tmp_path, hub, hub_address=proxy.address, **node_options))
        made[-1] = (hub, proxy, node)
        return hub, proxy, node

    yield start
    for hub, proxy, node in made:
        proxy.restore()
        if node is not None and node.thread.is_alive():
            node.stop()
        proxy.close()
        hub.stop()


def _away(node) -> float:
    session = node.agent._session
    return 0.0 if session is None else session.away_for()


def _wait_reconnect(hub, opened: int) -> SessionOpened:
    return wait_for(lambda: hub.events(SessionOpened)[opened:], timeout=30)[0]


def test_lineups_outlive_their_ttl_while_the_hub_is_away(away):
    hub, proxy, node = away(_config(), lambda worker, env: FIXED)
    wait_for(lambda: len(hub.grants_sent) >= 4 and node.agent.env_steps > 0, timeout=60)   # TTLs expire and renew
    opened = len(hub.events(SessionOpened))
    channel = node.agent._channel
    proxy.cut(2.0)
    wait_for(lambda: _away(node) > 0, timeout=10)
    hub.sync()
    requests = len(hub.messages("lineup_request"))
    wait_for(lambda: _away(node) >= TTL_SEC, timeout=10)                  # every lineup is past its TTL now
    steps = node.agent.env_steps
    wait_for(lambda: node.agent.env_steps > steps, timeout=10)            # and plays on
    assert _away(node) > 0                                                # still within the cut
    hub.sync()
    assert len(hub.messages("lineup_request")) == requests                # nothing reached the hub
    again = _wait_reconnect(hub, opened)
    assert again.hello.session_id == hub.events(SessionOpened)[0].hello.session_id
    first = wait_for(lambda: hub.messages("lineup_request")[requests:], timeout=30)[0]
    assert sorted((c.worker, c.env) for c in first.cells) == [(0, 0), (0, 1)]   # the expired cells in one request
    renewed = node.agent._channel        # uploads / downloads: a new channel, no gRPC backoff left from the outage
    assert renewed is not channel and node.agent.fetches._hub_channel is renewed
    assert node.stop() == 0


def test_results_wait_in_the_buffer_and_arrive_once(away):
    hub, proxy, node = away(_config(distributed={"results_batch": 4}), lambda worker, env: FIXED)
    wait_for(lambda: len(hub.acked) >= 3, timeout=60)
    hub.call(lambda: setattr(hub, "hold_acks", True))
    held_from = len(hub.messages("results"))
    wait_for(lambda: len(hub.messages("results")) >= held_from + 3, timeout=60)   # sent, never acked
    opened = len(hub.events(SessionOpened))
    first_serial = hub.node_messages("results")[-1].serial
    proxy.cut(1.0)
    _wait_reconnect(hub, opened)
    wait_for(lambda: len([e for e in hub.node_messages("results") if e.serial != first_serial]) >= 3, timeout=30)

    def release():
        hub.hold_acks = False
    hub.call(release)
    hub.release_acks()
    assert node.stop() == 0
    hub.sync()
    uids: dict[int, list[str]] = {}
    sends: dict[int, int] = {}
    for message in hub.messages("results"):
        batch = [r.uid for r in message.results]
        assert uids.setdefault(message.seq, batch) == batch                # a resend carries the same results
        sends[message.seq] = sends.get(message.seq, 0) + 1
    seqs = sorted(uids)
    assert seqs == list(range(1, seqs[-1] + 1))
    assert any(n > 1 for n in sends.values())                             # unacked batches were sent again
    assert len(hub.accounted) == len(set(hub.accounted))                  # every uid accounted (and acked) once
    assert set(hub.accounted) == {u for batch in uids.values() for u in batch}
    assert sorted(hub.acked) == seqs
    assert node.agent.results.unacked() == [] and node.agent.results.dropped == 0


def _relay_only_b(config):
    """Agent ``b``'s learner is on node 7, reachable only through the relay: its version announced relay-only with
    no data address, its weights served by the hub (parked)."""
    _roles, role = agent_role_of(config, "b")
    model = build_model(config.get_agent_config("b"), role)
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    frame = encode_weights(state, WeightsMeta(agent_id="b", kind="latest", generation=1, policy_version=1))

    def setup(hub):
        hub.blobs.write_parked("b", frame)
        hub.welcome(versions=[version_to_proto(VersionInfo(agent_id="b", generation=1, version=1, size=len(frame),
                                                           digest="", node=7, relay_only=True))])
    return setup


def _no_learner(hub) -> int:
    return sum(w.counters.get("dropped_chunks_no_learner", 0) for m in hub.messages("metrics") for w in m.workers
               if w.worker == 0)


def test_a_relay_only_learner_stops_collecting_after_hub_lost_collect_sec(away, caplog):
    config = _config(timeout=3.5, agents={"b": {}, "random": scripted_agent()})
    with caplog.at_level(logging.INFO, logger="colosseum.node"):
        hub, proxy, node = away(config, lambda worker, env: B_VS_RANDOM, relay=True, setup=_relay_only_b(config))
        wait_for(lambda: node.agent.paths() == {7: "relay"}, timeout=60)
        wait_for(lambda: node.agent.counters().get("chunks_sent.relay", 0) > 0, timeout=60)
        dropped = _no_learner(hub)
        opened = len(hub.events(SessionOpened))
        proxy.cut(2.5)
        wait_for(lambda: node.agent._learner_routes["b"].via_node is False, timeout=10)
        assert _away(node) >= 1.0
        _wait_reconnect(hub, opened)
        wait_for(lambda: _no_learner(hub) > dropped, timeout=30)          # counted while the hub was away
        wait_for(lambda: node.agent._learner_routes["b"].via_node is True, timeout=30)   # the route is back
        sent = node.agent.counters().get("chunks_sent.relay", 0)
        wait_for(lambda: node.agent.counters().get("chunks_sent.relay", 0) > sent, timeout=30)
        assert node.stop() == 0
    said = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "hub is away" in r.getMessage()]
    assert said == ["node: the hub is away 1 s: chunks for b (its learner is reachable only through the hub) are not "
                    "collected"]


def _learner_a(hub):
    hub.welcome(learners=[pb.LearnerAssignment(agent_id="a", generation=1, version_floor=0, device="cpu")])


def _pids(node) -> dict[str, int]:
    out = {}
    for name in ("worker-0", "learner-a"):
        proc = node.agent.supervisor.get(name)
        if proc is not None and proc.pid and proc.exitcode is None:
            out[name] = proc.pid
    return out


def test_the_hub_lost_timeout_stops_the_processes_and_the_hub_brings_them_back(away, caplog):
    with caplog.at_level(logging.WARNING, logger="colosseum.node"):
        hub, proxy, node = away(_config(timeout=2.0), lambda worker, env: A_VS_RANDOM if hub.versions("a") else None,
                                setup=_learner_a, cpu_learners=1)
        before = wait_for(lambda: (lambda p: p if len(p) == 2 else None)(_pids(node)), timeout=60)
        wait_for(lambda: hub.versions("a"), timeout=60)
        opened = len(hub.events(SessionOpened))
        proxy.cut(4.0)
        wait_for(lambda: not any(pid_alive(pid) for pid in before.values()), timeout=30)
        assert _away(node) < 4.0 and node.thread.is_alive()                # stopped within the cut; the agent runs
        said = [r.getMessage() for r in caplog.records if "processes stopped" in r.getMessage()]
        assert said == ["node: the hub is away 2 s: processes stopped; reconnecting every 0.2–0.4 s (Ctrl+C to stop)"]
        announced = len(hub.versions("a"))
        _wait_reconnect(hub, opened)
        after = wait_for(lambda: (lambda p: p if len(p) == 2 else None)(_pids(node)), timeout=60)
        assert set(after.values()).isdisjoint(before.values())             # new processes
        wait_for(lambda: len(hub.versions("a")) > announced, timeout=60)   # the learner announces again
        assert node.stop() == 0


def test_a_once_node_exits_1_after_the_hub_lost_timeout(away):
    hub, proxy, node = away(_config(timeout=2.0), lambda worker, env: A_VS_RANDOM if hub.versions("a") else None,
                            setup=_learner_a, cpu_learners=1, once=True)
    before = wait_for(lambda: (lambda p: p if len(p) == 2 else None)(_pids(node)), timeout=60)
    wait_for(lambda: hub.versions("a"), timeout=60)
    proxy.cut(10.0)
    node.thread.join(30)
    assert not node.thread.is_alive() and node.exit_code == 1             # ruling P17
    assert not any(pid_alive(pid) for pid in before.values())


def test_a_full_cache_stops_taking_work_until_space_returns(away, monkeypatch, caplog):
    config = _config()
    real = weights.free_bytes
    full = {"on": False}
    monkeypatch.setattr(weights, "free_bytes", lambda path: 2**20 if full["on"] else real(path))
    lineup = {"now": FIXED}
    with caplog.at_level(logging.INFO, logger="colosseum.node"):
        hub, proxy, node = away(config, lambda worker, env: lineup["now"])
        write_ckpt_dir(hub.run_dir.checkpoints / "a", config, "a", policy_version=5)
        wait_for(lambda: len(hub.grants_sent) >= 4 and node.agent.env_steps > 0, timeout=60)
        full["on"] = True
        lineup["now"] = SNAP5                                    # the next grants need a snapshot: a cache write
        event = wait_for(lambda: [e for e in hub.node_events("warning") if e.error_kind == "cache_full"], timeout=30)
        assert "--shm-size" in event[0].message and "emptyDir" in event[0].message
        hub.sync()
        messages = hub.events(NodeMessageIn)
        at_event = next(i for i, e in enumerate(messages) if e.message.WhichOneof("m") == "event"
                        and e.message.event.error_kind == "cache_full")
        requests = sum(1 for e in messages[:at_event] if e.message.WhichOneof("m") == "lineup_request")
        steps = node.agent.env_steps
        beats = len(hub.heartbeats())
        wait_for(lambda: len(hub.heartbeats()) >= beats + 3, timeout=30)
        hub.sync()
        assert len(hub.messages("lineup_request")) == requests          # no work taken for three heartbeats
        wait_for(lambda: node.agent.env_steps > steps, timeout=30)       # the cells keep playing their lineups
        full["on"] = False
        wait_for(lambda: len(hub.messages("lineup_request")) > requests, timeout=30)
        wait_for(lambda: any(s.network_id == "ckpt_v5" for r in hub.results() for s in r.seats), timeout=60)
        assert len([e for e in hub.node_events("warning") if e.error_kind == "cache_full"]) == 1
        assert node.stop() == 0
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR and "--shm-size" in r.getMessage()]
    assert len(errors) == 1 and "emptyDir" in errors[0]
    assert any("space is back" in r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
