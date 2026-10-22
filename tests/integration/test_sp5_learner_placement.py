"""Learner placement in the hub (SP5 T4.1) with fake nodes: devices and slots, pending agents and parked weights,
reconciliation, generation fencing, version floors, metrics stamps.

An in-process hub (``cluster_kit.start_hub``: mode ``hub``, TLS on loopback, the test identity under ``tmp_path``) on
the toy game ``turns``; ``hub_kit.FakeNode``s register, open sessions with the ``Hello`` fields each test names and
heartbeat until the test ends. Reads: the commands each fake node received, ``hub_state.json`` (``learners``,
``versions``, the counter book), ``metrics.jsonl`` and the hub's log records (``caplog``)."""
from __future__ import annotations

import json
import logging
import uuid

import numpy as np
import pytest
import torch

from cli_runner import TTT_CONFIG, run_train
from cluster_kit import hub_counter, journal_rows, read_hub_state, start_hub
from colosseum.net.chunks import decode_checkpoint, decode_weights, encode_checkpoint
from colosseum.net.grpc_util import blob_digest
from colosseum.net.messages import VersionInfo, lineup_from_proto, results_to_batch, version_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from game_helpers import agent_role_of, make_match_result, make_test_config, scripted_agent
from hub_kit import FakeNode, poll_until

HB = 0.1
DIST = {"heartbeat_sec": HB, "node_lost_after_sec": 2.0, "learner_relocate_after_sec": 2.0,
        "shutdown_grace_sec": 1.0}
MIXED = {"opponents": {"latest": 0.5, "snapshots": 0.0, "rivals": 0.5, "anchors": 0.0}}


def _config(agents=None, **sections):
    agents = agents if agents is not None else {"a": {}, "b": {}}
    sections.setdefault("metrics", {"console_interval_sec": 0.2})
    return make_test_config("turns", agents=agents, distributed=DIST, **sections)


@pytest.fixture
def hubs(tmp_path):
    started = []

    def start(config, name="placement"):
        hub = start_hub(tmp_path, config, name=name)
        started.append(hub)
        return hub

    yield start
    for hub in started:
        hub.stop()


def _hello(name, *, cpu=0, gpus=(), kind="cuda", learners=(), no_learners=False, workers=1, tags=()):
    return dict(slots=pb.Slots(workers=workers, cpu_learners=cpu, gpus=list(gpus), no_learners=no_learners,
                               envs_per_worker=4),
                hardware=pb.Hardware(gpus=[pb.Gpu(index=i, kind=kind) for i in gpus]),
                learners=[pb.RunningLearner(agent_id=a, generation=g, version=v) for a, g, v in learners],
                name=name, tags=list(tags))


def _node(hub, name, *, session=None, node=None, **hello):
    """A registered fake node (or ``node`` again) with a new session; waits for its ``Welcome``."""
    if node is None:
        node = FakeNode(hub.address, token=hub.token, ca_pem=hub.hub.identity.ca_pem)
        node.register(machine_id=uuid.uuid4().hex, name=name)
    else:
        node.close()
    node.open(session or uuid.uuid4().hex, hello_fields=_hello(name, **hello), heartbeat_sec=HB)
    node.wait_for("welcome")
    return node


def _commands(node, kind):
    return [c for c in node.messages("command") if c.kind == kind]


def _started(node):
    """The node's ``start_learner`` commands once one arrived (a start on a node being welcomed follows its
    ``Welcome``)."""
    return poll_until(lambda: _commands(node, "start_learner"))


def _learner(hub, agent):
    return ((read_hub_state(hub).get("learners") or {}).get("agents") or {}).get(agent) or {}


def _announce(node, agent, generation, version):
    info = VersionInfo(agent_id=agent, generation=generation, version=version, size=10, digest="d" * 32,
                       node=node.number)
    node.send(version=pb.VersionAnnounce(info=version_to_proto(info)))


def _records(hub, kind):
    path = hub.root / "metrics.jsonl"
    if not path.exists():
        return []
    return [r for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
            if r["kind"] == kind]


def _upload(node, config, agent, version, generation, kind="checkpoint"):
    from colosseum.core.registry import env_spec
    from colosseum.learner.factory import build_algorithm
    from colosseum.learner.learner import make_checkpoint_payload, set_policy_version

    _roles, role = agent_role_of(config, agent)
    algo = build_algorithm(config.get_agent_config(agent), role, env_spec(config), device="cpu", teacher=None)
    set_policy_version(algo, version)
    data = encode_checkpoint(make_checkpoint_payload(agent, algo), kind="final" if kind == "final" else "periodic",
                             generation=generation)
    return node.upload(pb.UploadHeader(kind=kind, agent_id=agent, generation=generation, version=version,
                                       total_size=len(data), digest=blob_digest(data)), data)


def _result(hub, node, k, version_end, generation=1):
    seats = [(0, "a", "latest", "player", 0, version_end), (1, "rnd", "fixed", "player")]
    match_id = f"w0_e0_g1_ep{k}"
    result = make_match_result(seats, uid=f"{hub.hub.run_id}/train/n{node.number}/{match_id}", match_id=match_id)
    result.node = node.number
    result.seats[0].generation = generation
    return result


# ---- devices, slots, pending ----------------------------------------------------------------------------------


def test_devices_and_slots_decide_where_learners_start(hubs):
    hub = hubs(_config({"a": {"learner": {"device": "cuda"}}, "b": {"learner": {"device": "cpu"}}}))
    n1 = _node(hub, "n1", gpus=(0,))
    n2 = _node(hub, "n2", cpu=1)
    poll_until(lambda: _learner(hub, "b").get("status") == "starting")
    (s1,), (s2,) = _started(n1), _started(n2)
    la = s1.learner
    assert (la.agent_id, la.generation, la.device, la.version_floor, la.weights_blob, la.env_steps) == \
        ("a", 1, "cuda:0", 0, "", 0)
    assert (s2.learner.agent_id, s2.learner.generation, s2.learner.device) == ("b", 1, "cpu")
    assert [x.agent_id for x in n1.messages("welcome")[0].learners] == ["a"]       # its desired learner set
    assert _learner(hub, "a") == {"status": "starting", "node": n1.number, "device": "cuda:0", "generation": 1,
                                  "reason": "", "local": False}
    assert (_learner(hub, "b")["node"], _learner(hub, "b")["device"]) == (n2.number, "cpu")


def test_an_agent_without_a_slot_is_pending_then_starts_when_a_slot_appears(hubs, caplog):
    hub = hubs(_config())
    with caplog.at_level(logging.INFO, logger="colosseum.hub.hub"):
        n1 = _node(hub, "n1", cpu=1)
        poll_until(lambda: any(r.get("pending_agents") == ["b"] for r in _records(hub, "system")))
        assert "Learner of b: pending (no free CPU learner slot (learner.device cpu))" in caplog.text
        assert [c.learner.agent_id for c in _started(n1)] == ["a"]
        seen = len(_records(hub, "system"))
        n2 = _node(hub, "n2", cpu=1)
        (start,) = _started(n2)
        assert (start.learner.agent_id, start.learner.generation) == ("b", 1)
        poll_until(lambda: any(r.get("pending_agents") == [] for r in _records(hub, "system")[seen:]))
        assert any(s.agent_id == "b" and s.status == "active" for m in n2.messages("welcome") for s in m.agents)


def test_a_pending_agent_plays_with_parked_weights(hubs, tmp_path):
    from colosseum.core.registry import build_model

    base = _config(matchmaking=MIXED)
    model = build_model(base.get_agent_config("b"), agent_role_of(base, "b")[1])
    pt = tmp_path / "init.pt"
    torch.save(model.state_dict(), pt)
    hub = hubs(_config({"a": {}, "b": {"init": {"from": str(pt)}}}, matchmaking=MIXED))
    n1 = _node(hub, "n1", cpu=1)
    infos = [v for m in n1.messages("welcome") + n1.messages("announce") for v in m.versions if v.agent_id == "b"]
    assert [(v.generation, v.version, v.node, v.relay_only) for v in infos] == [(0, 0, -1, True)]
    frame, meta = n1.download("latest", "b", "", since_version=-1)
    state, wmeta = decode_weights(frame, cap=len(frame))
    assert (wmeta.kind, wmeta.policy_version, wmeta.generation) == ("latest", 0, 0)
    assert json.loads(meta)["digest"] == infos[0].digest == blob_digest(frame)
    expected = model.state_dict()
    assert set(state) == set(expected)
    assert all(np.array_equal(state[k], expected[k].numpy()) for k in expected)
    assert n1.download("latest", "b", "", since_version=0)[0] is None              # not modified
    n1.send(lineup_request=pb.LineupRequest(cells=[pb.CellRequest(worker=0, env=e) for e in range(4)]))
    grants = [g for batch in n1.wait_for("grants") for g in batch.grants]
    lineups = [lineup_from_proto(g.lineup) for g in grants]
    assert lineups and all(s.source != "owner" for lu in lineups for s in lu.seats if s.agent_id == "b")
    b_seats = [s for lu in lineups for s in lu.seats if s.agent_id == "b"]
    assert all(not s.collect for s in b_seats)
    n1.send(lineup_request=pb.LineupRequest(cells=[pb.CellRequest(worker=0, env=e, epoch=g.epoch)
                                                   for e, g in enumerate(grants)]))
    n1.wait_for("grants", n=2)
    seated = [s for batch in n1.messages("grants") for g in batch.grants for s in lineup_from_proto(g.lineup).seats
              if s.agent_id == "b"]
    assert seated and all(not s.collect and s.network_id == "latest" for s in seated)


def test_a_pending_agent_without_weights_is_not_drawn(hubs, caplog):
    hub = hubs(_config(matchmaking=MIXED))
    n1 = _node(hub, "n1", cpu=1, workers=10)
    n1.send(lineup_request=pb.LineupRequest(cells=[pb.CellRequest(worker=w, env=e) for w in range(10)
                                                   for e in range(4)]))
    grants = [g for batch in n1.wait_for("grants") for g in batch.grants]
    assert len(grants) == 40
    assert all(s.agent_id != "b" for g in grants for s in lineup_from_proto(g.lineup).seats)
    assert hub.hub._book.totals().get("lineup_redraw_failed", 0) == 0
    hub.stop()

    rivals = {"opponents": {"latest": 0.0, "snapshots": 0.0, "rivals": 1.0, "anchors": 0.0}}
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.hub"):
        hub = hubs(_config(matchmaking=rivals), name="rivals")
        n1 = _node(hub, "n1", cpu=1)
        n1.send(lineup_request=pb.LineupRequest(cells=[pb.CellRequest(worker=0, env=e) for e in range(4)]))
        poll_until(lambda: hub_counter(hub, "lineup_redraw_failed") > 0)
        assert n1.messages("grants") == []
    assert "draws in a row named agents without weights (b)" in caplog.text


# ---- generations, reconciliation, fencing ---------------------------------------------------------------------


def test_the_first_accepted_announcement_makes_a_learner_running(hubs, caplog):
    hub = hubs(_config({"a": {}}))
    with caplog.at_level(logging.INFO, logger="colosseum.hub.hub"):
        session = uuid.uuid4().hex
        n1 = _node(hub, "n1", cpu=1, session=session)
        _announce(n1, "a", 1, 0)
        poll_until(lambda: _learner(hub, "a").get("status") == "running")
    assert f"Learner of a: running on node {n1.number}" in caplog.text
    n1 = _node(hub, "n1", node=n1, session=session, cpu=1, learners=[("a", 1, 0)])
    assert [(x.agent_id, x.generation) for x in n1.messages("welcome")[0].learners] == [("a", 1)]


def test_a_reconnecting_node_keeps_its_current_learner(hubs):
    hub = hubs(_config({"a": {}}))
    session = uuid.uuid4().hex
    n1 = _node(hub, "n1", cpu=1, session=session)
    _announce(n1, "a", 1, 3)
    poll_until(lambda: _learner(hub, "a").get("status") == "running")
    n1 = _node(hub, "n1", node=n1, session=session, cpu=1, learners=[("a", 1, 3)])
    assert _commands(n1, "stop_learner") == [] and _commands(n1, "start_learner") == []
    assert [(x.agent_id, x.generation) for x in n1.messages("welcome")[0].learners] == [("a", 1)]
    assert hub.hub._versions.generation("a") == 1 and _learner(hub, "a")["generation"] == 1


def _replaced_on_n1(hub):
    """``a`` runs on ``n1`` (generation 1, versions 0..5), then ``n1`` comes back without it: generation 2."""
    session = uuid.uuid4().hex
    n1 = _node(hub, "n1", cpu=1, session=session)
    for v in range(6):
        _announce(n1, "a", 1, v)
    poll_until(lambda: hub.hub._versions.version_floor("a") == 6)
    n1 = _node(hub, "n1", node=n1, session=session, cpu=1)                # its learner process restarted
    return n1, session


def test_a_late_announcement_of_an_old_generation_is_fenced(hubs):
    hub = hubs(_config({"a": {}}))
    n1, session = _replaced_on_n1(hub)
    (start,) = _started(n1)
    assert (start.learner.generation, start.learner.version_floor) == (2, 6)
    assert [(x.agent_id, x.generation) for x in n1.messages("welcome")[0].learners] == [("a", 2)]
    _announce(n1, "a", 1, 9)
    (stop,) = n1.wait_for("command", n=2)[1:]
    assert (stop.kind, stop.learner.agent_id, stop.learner.generation, stop.reason) == \
        ("stop_learner", "a", 1, "superseded")
    poll_until(lambda: hub_counter(hub, "dropped_announcements.stale_generation") == 1)
    assert hub.hub._versions.latest("a").version == 5
    assert all(v.version != 9 for m in n1.messages("announce") for v in m.versions)
    n1 = _node(hub, "n1", node=n1, session=session, cpu=1, learners=[("a", 7, 30)])   # an unknown generation
    stops = poll_until(lambda: _commands(n1, "stop_learner"))
    assert [(c.learner.agent_id, c.learner.generation) for c in stops] == [("a", 7)]
    (start,) = _started(n1)                            # generation 2 is gone as well
    assert start.learner.generation == 3 and start.learner.version_floor >= 31      # P-PF3-3: Hello's version


def test_stale_generation_checkpoints_and_results_are_refused(hubs):
    config = _config({"a": {}, "rnd": scripted_agent()})
    hub = hubs(config)
    n1, _session = _replaced_on_n1(hub)
    reply = _upload(n1, config, "a", 10, 1)
    assert (reply.accepted, reply.reason) == (False, "stale_generation")
    assert not (hub.root / "checkpoints" / "a" / "ckpt_v10").exists()
    n1.send(results=results_to_batch(1, [_result(hub, n1, 0, version_end=9)]))       # above cap(1) = 5
    n1.send(results=results_to_batch(2, [_result(hub, n1, 1, version_end=5)]))       # within it
    poll_until(lambda: len(journal_rows(hub)) == 1, timeout=20)
    poll_until(lambda: hub_counter(hub, "dropped_match_results_by_reason.stale_generation") == 1)
    assert journal_rows(hub)[0].uid.endswith("ep1")
    n2 = _node(hub, "n2", cpu=1)
    reply = _upload(n2, config, "a", 11, 2)                              # the current generation, another node
    assert (reply.accepted, reply.reason) == (False, "stale_generation")
    assert not (hub.root / "checkpoints" / "a" / "ckpt_v11").exists()


def test_version_floor_continues_above_announced_and_stored_versions(hubs):
    config = _config({"a": {}})
    hub = hubs(config)
    session = uuid.uuid4().hex
    n1 = _node(hub, "n1", cpu=1, session=session)
    for v in range(13):
        _announce(n1, "a", 1, v)
    poll_until(lambda: hub.hub._versions.version_floor("a") == 13)
    reply = _upload(n1, config, "a", 20, 1)
    assert reply.accepted, reply.reason
    assert (hub.root / "checkpoints" / "a" / "ckpt_v20").is_dir()
    n1 = _node(hub, "n1", node=n1, session=session, cpu=1)
    (start,) = _started(n1)
    assert (start.learner.generation, start.learner.version_floor) == (2, 21)


def test_a_generation_is_durable_before_its_start_learner_is_sent(hubs):
    """P-PF3-2: hub_state.json names the new generation (fsync) before the start_learner command leaves."""
    hub = hubs(_config({"a": {}}, metrics={"console_interval_sec": 3600.0}))
    n1 = _node(hub, "n1", cpu=1)
    _started(n1)
    state = read_hub_state(hub)
    assert state["versions"]["a"]["generation"] == 1
    assert state["learners"]["agents"]["a"]["status"] == "starting"


# ---- metrics, start points, train --------------------------------------------------------------------------------


def test_train_records_carry_learner_node_and_generation(hubs):
    hub = hubs(_config({"a": {}}))
    n1 = _node(hub, "n1", cpu=1)
    n1.send(metrics=pb.MetricsBatch(train=[pb.TrainMetrics(agent_id="a", generation=1, train_step=3,
                                                            values={"total_loss": 0.5})]))
    n1.send(metrics=pb.MetricsBatch(train=[pb.TrainMetrics(agent_id="a", generation=0, train_step=4,
                                                            values={"total_loss": 0.5})]))
    n1.send(metrics=pb.MetricsBatch(train=[pb.TrainMetrics(agent_id="a", generation=1, train_step=5,
                                                            values={"total_loss": 0.5})]))
    records = poll_until(lambda: [r for r in _records(hub, "train") if r["train_step"] == 5] and
                         _records(hub, "train"))
    assert [r["train_step"] for r in records] == [3, 5]                 # generation 0: not this learner's
    assert all(r["learner_node"] == n1.number and r["generation"] == 1 for r in records)


def test_a_resume_state_is_the_start_point_start_resume(hubs, tmp_path):
    """A28: a ``training.resume_from`` state is one checkpoint frame served as ``Download(kind="start")``."""
    from colosseum.core.registry import build_model

    base = _config({"a": {}})
    model = build_model(base.get_agent_config("a"), agent_role_of(base, "a")[1])
    pt = tmp_path / "resume.pt"
    torch.save(model.state_dict(), pt)
    hub = hubs(_config({"a": {}}, training={"resume_from": str(pt)}))
    n1 = _node(hub, "n1", cpu=1)
    (start,) = _started(n1)
    assert (start.learner.weights_blob, start.learner.trainer_state_blob) == ("start:resume", "")
    frame, _meta = n1.download("start", "a", "resume")
    payload, meta = decode_checkpoint(bytearray(frame), cap=len(frame))
    assert (meta["kind"], meta["generation"], payload["policy_version"], payload["trainer_state_bytes"]) == \
        ("recovery", 0, 0, None)
    expected = model.state_dict()
    assert all(np.array_equal(payload["model_state"][k], expected[k].numpy()) for k in expected)
    for kind, ident in (("start", "ckpt_v1"), ("recovery", "x"), ("trainer_state", "x")):
        with pytest.raises(Exception, match="NOT_FOUND"):
            n1.download(kind, "a", ident)


def test_train_without_listen_fails_fast_on_an_impossible_device(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, "nogpu", overrides={"learner.device": "cuda:7"}, timeout=120)
    assert run.returncode == 1, run.stderr[-3000:]
    assert "cannot run on this machine" in run.stderr and "slot on GPU 7" in run.stderr
    assert not (run.root / "logs" / "worker-0.log").exists()
