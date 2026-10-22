"""The hub's failure handling with fake nodes (SP5 T4.2, spec 5.2, 7.1, 7.2): recovery points stored and fenced,
learner deaths re-assigned from the recovery point or the start source, failed agents, version floors. (Machine loss,
relocation and disabled slots: T4.2 (b).)

An in-process hub (``cluster_kit.start_hub``: mode ``hub``, TLS on loopback, the test identity under ``tmp_path``) on
the toy game ``turns``; ``hub_kit.FakeNode``s register, open sessions with the ``Hello`` fields each test names and
heartbeat until the test ends. Reads: the commands each fake node received, ``hub_state.json`` (``learners``, the
counter book), the run dir's ``recovery/``, ``metrics.jsonl`` and the hub's log records (``caplog``)."""
from __future__ import annotations

import json
import logging
import uuid

import pytest
import torch

from cluster_kit import hub_counter, read_hub_state, start_hub
from colosseum.net.chunks import encode_checkpoint
from colosseum.net.grpc_util import blob_digest
from colosseum.net.messages import VersionInfo, lineup_from_proto, version_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from game_helpers import agent_role_of, make_test_config
from hub_kit import FakeNode, poll_until

HB = 0.1
DIST = {"heartbeat_sec": HB, "node_lost_after_sec": 2.0, "learner_relocate_after_sec": 2.0,
        "shutdown_grace_sec": 1.0}
MIXED = {"opponents": {"latest": 0.5, "snapshots": 0.0, "rivals": 0.5, "anchors": 0.0}}


def _config(agents=None, **sections):
    agents = agents if agents is not None else {"a": {}}
    sections.setdefault("metrics", {"console_interval_sec": 0.2})
    return make_test_config("turns", agents=agents, distributed=DIST, **sections)


@pytest.fixture
def hubs(tmp_path):
    started = []

    def start(config, name="failures"):
        hub = start_hub(tmp_path, config, name=name)
        started.append(hub)
        return hub

    yield start
    for hub in started:
        hub.stop()


def _node(hub, name, *, cpu=1, workers=1):
    """A registered fake node with a session (``cpu`` CPU learner slots); waits for its ``Welcome``."""
    node = FakeNode(hub.address, token=hub.token, ca_pem=hub.hub.identity.ca_pem)
    node.register(machine_id=uuid.uuid4().hex, name=name)
    node.open(uuid.uuid4().hex, heartbeat_sec=HB, hello_fields=dict(
        slots=pb.Slots(workers=workers, cpu_learners=cpu, envs_per_worker=4), name=name))
    node.wait_for("welcome")
    return node


def _starts(node, n=1):
    """The node's first ``n`` ``start_learner`` assignments (a start on a node being welcomed follows its
    ``Welcome``)."""
    return poll_until(lambda: (lambda c: c[:n] if len(c) >= n else None)(
        [c.learner for c in node.messages("command") if c.kind == "start_learner"]))


def _learner(hub, agent):
    return ((read_hub_state(hub).get("learners") or {}).get("agents") or {}).get(agent) or {}


def _announce(node, agent, generation, *versions):
    for version in versions:
        info = VersionInfo(agent_id=agent, generation=generation, version=version, size=10, digest="d" * 32,
                           node=node.number)
        node.send(version=pb.VersionAnnounce(info=version_to_proto(info)))


def _frame(config, agent, version, generation, kind):
    from colosseum.core.registry import env_spec
    from colosseum.learner.factory import build_algorithm
    from colosseum.learner.learner import make_checkpoint_payload, set_policy_version

    algo = build_algorithm(config.get_agent_config(agent), agent_role_of(config, agent)[1], env_spec(config),
                           device="cpu", teacher=None)
    set_policy_version(algo, version)
    return encode_checkpoint(make_checkpoint_payload(agent, algo, final=kind == "final"), kind=kind,
                             generation=generation)


def _upload(node, agent, version, generation, data, kind="recovery"):
    return node.upload(pb.UploadHeader(kind=kind, agent_id=agent, generation=generation, version=version,
                                       total_size=len(data), digest=blob_digest(data)), data)


def _event(node, kind, generation, agent="a", **fields):
    node.send(event=pb.Event(kind=kind, proc=f"learner-{agent}", generation=generation, **fields))


def _records(hub, kind):
    path = hub.root / "metrics.jsonl"
    if not path.exists():
        return []
    return [r for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
            if r["kind"] == kind]


def _with_recovery(hub, config, *, steps=0):
    """``a`` on ``n1`` (generation 1) announced 0..4 and uploaded its recovery point v4; returns (n1, frame)."""
    n1 = _node(hub, "n1")
    _starts(n1)
    n1.env_steps = steps
    poll_until(lambda: hub.hub.env_steps_done == steps)
    _announce(n1, "a", 1, *range(5))
    poll_until(lambda: hub.hub._versions.version_floor("a") == 5)
    frame = _frame(config, "a", 4, 1, "recovery")
    reply = _upload(n1, "a", 4, 1, frame)
    assert (reply.accepted, reply.reason, reply.checkpoint_id) == (True, "", "recovery")
    return n1, frame


# ---- recovery points and restarts ------------------------------------------------------------------------------


def test_a_learner_death_restarts_it_from_the_recovery_point(hubs):
    config = _config()
    hub = hubs(config)
    n1, frame = _with_recovery(hub, config, steps=123)
    point = hub.root / "recovery" / "a"
    assert (point / "recovery.frame").read_bytes() == frame
    assert json.loads((point / "meta.json").read_text()) == {"generation": 1, "version": 4, "env_steps": 123}

    _event(n1, "proc_died", 1, exit_code=-9)
    (_first, la) = _starts(n1, 2)
    assert (la.agent_id, la.generation, la.device, la.weights_blob, la.version_floor) == \
        ("a", 2, "cpu", "start:recovery", 5)
    assert n1.download("start", "a", "recovery")[0] == frame
    poll_until(lambda: hub_counter(hub, "learner_restarts") == 1)
    assert _learner(hub, "a")["generation"] == 2

    _event(n1, "proc_died", 1, exit_code=-9)                             # the same report again: an old generation
    _announce(n1, "a", 2, 5)                                             # (same stream: handled after the event)
    poll_until(lambda: hub.hub._versions.version_floor("a") == 6)
    assert len(_starts(n1, 2)) == 2 and len([c for c in n1.messages("command") if c.kind == "start_learner"]) == 2
    assert hub.hub._versions.generation("a") == 2 and hub.hub._book.get("learner_restarts") == 1


def test_a_restart_before_any_recovery_point_uses_the_start_source(hubs, tmp_path):
    from colosseum.core.registry import build_model

    base = _config()
    pt = tmp_path / "init.pt"
    torch.save(build_model(base.get_agent_config("a"), agent_role_of(base, "a")[1]).state_dict(), pt)
    hub = hubs(_config({"a": {"init": {"from": str(pt)}}}))
    n1 = _node(hub, "n1")
    assert _starts(n1)[0].weights_blob == "init"
    _announce(n1, "a", 1, *range(4))
    poll_until(lambda: hub.hub._versions.version_floor("a") == 4)
    _event(n1, "proc_died", 1, exit_code=1)
    la = _starts(n1, 2)[1]
    assert (la.generation, la.weights_blob, la.version_floor) == (2, "init", 4)
    assert not (hub.root / "recovery").exists()


def test_a_recovery_point_of_a_stale_generation_is_refused(hubs):
    config = _config()
    hub = hubs(config)
    n1, frame = _with_recovery(hub, config)
    _event(n1, "proc_died", 1, exit_code=-9)
    assert _starts(n1, 2)[1].generation == 2
    refusals = [
        (n1, 9, 1, _frame(config, "a", 9, 1, "recovery"), "stale_generation"),       # the dead generation
        (_node(hub, "n2"), 9, 2, _frame(config, "a", 9, 2, "recovery"), "stale_generation"),   # another node
        (n1, 3, 2, _frame(config, "a", 3, 2, "recovery"), "not_newer"),             # older than the stored v4
        (n1, 9, 2, b"\0" * 64, "invalid"),                                          # not a frame
    ]
    for node, version, generation, data, reason in refusals:
        reply = _upload(node, "a", version, generation, data)
        assert (reply.accepted, reply.reason) == (False, reason)
    assert (hub.root / "recovery" / "a" / "recovery.frame").read_bytes() == frame
    poll_until(lambda: hub_counter(hub, "dropped_uploads.not_newer") == 1)
    assert (hub_counter(hub, "dropped_uploads.stale_generation"), hub_counter(hub, "dropped_uploads.invalid")) == \
        (2, 1)


def test_a_failed_learner_stops_its_agent_but_not_the_run(hubs, caplog):
    config = _config({"a": {}, "b": {}}, matchmaking=MIXED)
    hub = hubs(config)
    with caplog.at_level(logging.INFO, logger="colosseum.hub.hub"):
        n1 = _node(hub, "n1", cpu=2, workers=10)
        assert sorted(la.agent_id for la in _starts(n1, 2)) == ["a", "b"]
        _announce(n1, "a", 1, *range(5))
        _announce(n1, "b", 1, 0)
        frame = _frame(config, "a", 4, 1, "recovery")
        assert _upload(n1, "a", 4, 1, frame).accepted
        _event(n1, "learner_failed", 1, exit_code=1, message="3 restarts within 1800 s on this node; last: crash: x")
        poll_until(lambda: _learner(hub, "a").get("status") == "failed")
        poll_until(lambda: any(r.get("failed_agents") == ["a"] for r in _records(hub, "system")))
    assert "Learner of a failed (3 restarts within 1800 s on this node; last: crash: x" in caplog.text
    parked = [v for m in n1.messages("announce") for v in m.versions if v.agent_id == "a" and v.node == -1]
    assert [(v.generation, v.version, v.relay_only) for v in parked] == [(0, 4, True)]
    assert any(s.agent_id == "a" and s.status == "failed" for m in n1.messages("announce") for s in m.agents)
    assert _learner(hub, "b")["status"] in ("starting", "running") and hub.hub._failure is None
    n1.send(lineup_request=pb.LineupRequest(cells=[pb.CellRequest(worker=w, env=e) for w in range(10)
                                                   for e in range(4)]))
    grants = [g for batch in n1.wait_for("grants") for g in batch.grants]
    lineups = [lineup_from_proto(g.lineup) for g in grants]
    assert lineups and all(s.agent_id == "b" for lu in lineups for s in lu.seats if s.source == "owner")
    a_seats = [s for lu in lineups for s in lu.seats if s.agent_id == "a"]
    assert a_seats and not any(s.collect for s in a_seats)
    assert hub.stop() == 1                                               # a failed agent: exit 1 at the end


def test_every_failed_agent_stops_the_run_with_exit_1(hubs, caplog):
    hub = hubs(_config())
    with caplog.at_level(logging.ERROR, logger="colosseum.hub.hub"):
        n1 = _node(hub, "n1")
        _starts(n1)
        _event(n1, "learner_failed", 1, exit_code=1)
        hub.thread.join(60)
    assert not hub.thread.is_alive() and hub.exit_codes == [1]
    assert "every trainable agent failed" in caplog.text


def test_a_restarted_learner_never_reuses_a_version(hubs):
    config = _config()
    hub = hubs(config)
    n1 = _node(hub, "n1")
    _starts(n1)
    _announce(n1, "a", 1, *range(13))
    poll_until(lambda: hub.hub._versions.version_floor("a") == 13)
    # versions above the last announce are possible: a learner publishes every weight_push_interval steps
    assert _upload(n1, "a", 20, 1, _frame(config, "a", 20, 1, "periodic"), kind="checkpoint").accepted
    assert _upload(n1, "a", 25, 1, _frame(config, "a", 25, 1, "recovery")).accepted
    _event(n1, "proc_died", 1, exit_code=-9)
    la = _starts(n1, 2)[1]
    assert (la.generation, la.weights_blob, la.version_floor) == (2, "start:recovery", 26)
