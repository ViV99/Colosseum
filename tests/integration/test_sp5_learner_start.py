"""The node runs the learners it is assigned (SP5 T4.1): device, generation and version floor; the start point of a
``start:*`` blob; the floor above the machine's cache; superseded and stale stops; the desired set of
``Welcome.learners``.

A ``NodeAgent`` in a thread (``node_kit.NodeThread``) against ``node_kit.FakeHub`` (the real gRPC server) with the
toy game ``turns``: one worker and one learner child (2 loaded processes)."""
from __future__ import annotations

import numpy as np
import pytest

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.server import SessionOpened
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node import session as node_session
from colosseum.node.weights import WeightCacheReader, WeightCacheWriter
from game_helpers import agent_role_of, make_test_config, scripted_agent
from node_kit import FakeHub, run_node_in_thread, toy_node_options, wait_for

SELF_PLAY = Lineup("2p", [SeatAssignment("a", "latest", True), SeatAssignment("a", "latest", False)])


def _config():
    return make_test_config("turns", agents={"a": {}, "random": scripted_agent()},
                            rollout={"envs_per_worker": 2, "chunk_length": 8},
                            learner={"batch_chunks": 2, "weight_push_interval": 1}, checkpoint={"interval": 3},
                            distributed={"heartbeat_sec": 0.25})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    monkeypatch.setattr(node_session, "RECONNECT_MAX_SEC", 0.2)
    hub = FakeHub(tmp_path, _config())
    nodes = []

    def start(**kw):
        node = run_node_in_thread(toy_node_options(tmp_path, hub, cpu_learners=1, **kw))
        nodes.append(node)
        wait_for(lambda: node.agent.cache_prefix is not None)
        return node

    yield hub, start, tmp_path / "node-logs"
    for node in nodes:
        if node.thread.is_alive():
            node.stop()
    hub.stop()


def _start(hub, generation, floor=0, blob=""):
    hub.send_command("start_learner", learner=pb.LearnerAssignment(
        agent_id="a", generation=generation, version_floor=floor, device="cpu", weights_blob=blob))


def _stop(hub, generation, reason="superseded"):
    hub.send_command("stop_learner", learner=pb.LearnerAssignment(agent_id="a", generation=generation), reason=reason)


def _of(hub, generation):
    return [v for v in hub.versions("a") if v.generation == generation]


def _hellos(hub):
    return [e.hello for e in hub.events(SessionOpened)]


def _learner_pid(node):
    proc = node.agent.supervisor.get("learner-a")
    return None if proc is None or proc.exitcode is not None else proc.pid


def _gone(node):
    return not node.agent.learners.agents()


def _weights(config, version):
    from colosseum.core.registry import build_model
    from colosseum.net.chunks import WeightsMeta, encode_weights

    model = build_model(config.get_agent_config("a"), agent_role_of(config, "a")[1])
    state = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    return encode_weights(state, WeightsMeta("a", "latest", generation=1, policy_version=version))


def test_start_learner_spawns_the_learner_with_device_generation_and_floor(setup):
    hub, start, logs = setup
    start()
    _start(hub, 3, floor=17)
    first = wait_for(lambda: _of(hub, 3), timeout=60)[0]
    assert first.version >= 17
    hub.close_session()
    hello = wait_for(lambda: _hellos(hub)[1:], timeout=30)[0]
    assert [(x.agent_id, x.generation) for x in hello.learners] == [("a", 3)] and hello.learners[0].version >= 17
    assert "generation 3, device cpu, versions from 17" in (logs / "learner-a.log").read_text()


def test_the_floor_is_above_the_cached_pointer(setup):
    hub, start, _logs = setup
    node = start()
    run = node.agent._run
    WeightCacheWriter(run.root, run.prefix).publish_latest("a", _weights(hub.config, 40), version=40, generation=1,
                                                           teacher_active=False)
    _start(hub, 2, floor=0)
    assert wait_for(lambda: _of(hub, 2), timeout=60)[0].version >= 41


def test_a_start_frame_restores_weights_and_trainer_state(setup):
    from colosseum.core.registry import env_spec
    from colosseum.learner.factory import build_algorithm
    from colosseum.learner.learner import make_checkpoint_payload, set_policy_version
    from colosseum.net.chunks import encode_checkpoint

    hub, start, logs = setup
    config = hub.config
    algo = build_algorithm(config.get_agent_config("a"), agent_role_of(config, "a")[1], env_spec(config),
                           device="cpu", teacher=None)
    set_policy_version(algo, 9)
    payload = make_checkpoint_payload("a", algo)
    hub.serve_blob("start", "a", "resume", encode_checkpoint(payload, kind="recovery", generation=0))
    node = start()
    _start(hub, 1, floor=5, blob="start:resume")
    assert wait_for(lambda: _of(hub, 1), timeout=60)[0].version == 9
    run = node.agent._run
    cached = WeightCacheReader(run.root, run.prefix).read_latest("a")
    assert cached.meta.policy_version == 9
    assert set(cached.state_dict) == set(payload["model_state"])
    assert all(np.array_equal(cached.state_dict[k], payload["model_state"][k]) for k in payload["model_state"])
    assert "Resumed from resume of the hub at policy_version 9" in (logs / "learner-a.log").read_text()


def test_stop_learner_of_another_generation_is_ignored(setup):
    hub, start, _logs = setup
    node = start()
    _start(hub, 3)
    wait_for(lambda: _of(hub, 3), timeout=60)
    pid = _learner_pid(node)
    _start(hub, 3)                                                       # the same assignment again: one start
    _stop(hub, 2)
    beats = len(hub.heartbeats())
    wait_for(lambda: len(hub.heartbeats()) >= beats + 2)                # the node's loop handled both commands
    assert _learner_pid(node) == pid and node.agent.learners.generations() == {"a": 3}
    _stop(hub, 3)
    wait_for(lambda: _gone(node), timeout=30)
    hub.close_session()
    hello = wait_for(lambda: _hellos(hub)[1:], timeout=30)[0]
    assert list(hello.learners) == []
    hub.sync()
    assert hub.uploads_of("final") == []                                  # superseded: no final snapshot


def test_a_newer_start_replaces_the_running_learner(setup):
    hub, start, _logs = setup
    hub.grant(lambda worker, env: SELF_PLAY if hub.versions("a") else None)
    node = start()
    _start(hub, 3)
    wait_for(lambda: _of(hub, 3), timeout=60)
    first = node.agent.learners.of("learner-a").listener_name
    _start(hub, 4)
    wait_for(lambda: _of(hub, 4), timeout=60)
    assert node.agent.learners.generations() == {"a": 4}
    assert (first[-3:], node.agent.learners.of("learner-a").listener_name[-3:]) == ("-i1", "-i2")   # a new intake
    assert min(v.version for v in _of(hub, 4)) > max(v.version for v in _of(hub, 3))
    wait_for(lambda: [t for m in hub.messages("metrics") for t in m.train if t.generation == 4], timeout=60)
    hub.sync()
    assert hub.uploads_of("final") == []                                  # generation 3 left without a final


def test_welcome_is_the_desired_learner_set(setup):
    hub, start, _logs = setup
    hub.welcome(learners=[pb.LearnerAssignment(agent_id="a", generation=3, device="cpu")])
    node = start()
    wait_for(lambda: _of(hub, 3), timeout=60)
    hub.welcome(learners=[])
    hub.close_session()
    wait_for(lambda: _gone(node), timeout=30)                            # not in the desired set: stopped
    hub.sync()
    assert hub.uploads_of("final") == []
    hub.welcome(learners=[pb.LearnerAssignment(agent_id="a", generation=5, device="cpu")])
    hub.close_session()
    wait_for(lambda: _of(hub, 5), timeout=60)
    pid = _learner_pid(node)
    sessions = len(_hellos(hub))
    hub.close_session()                                                  # the same generation listed again
    wait_for(lambda: len(_hellos(hub)) > sessions, timeout=30)
    hub.sync()
    assert _learner_pid(node) == pid and node.agent.learners.generations() == {"a": 5}
