"""A full weight cache on a node (SP5 T4.4, spec 7.1; review round 1 of part (a)): only frames that go into the cache
set the free space a node waits for (a learner's checkpoint frames, weights plus trainer state, never do), and a
learner whose publish does not fit logs one ERROR per cache-full episode."""
from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import numpy as np

from colosseum.core.types import WeightPayload
from colosseum.net.local import CheckpointHeader, NodeOutputs
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.security import Token, new_secret
from colosseum.node import weights
from colosseum.node.agent import NodeAgent, NodeOptions
from colosseum.node.learners import LearnerEntry
from colosseum.node.weights import SPACE_MARGIN_BYTES, CacheFullError


def test_a_checkpoint_frame_does_not_raise_the_free_space_threshold(tmp_path, monkeypatch):
    agent = NodeAgent(NodeOptions(hub_address="127.0.0.1:1", token=Token(new_secret(), "ab" * 32)))
    try:
        monkeypatch.setattr(agent.learners, "of", lambda name: LearnerEntry("a", 1, pb.LearnerAssignment(), name))
        agent._on_checkpoint("learner-a", CheckpointHeader("periodic", "a", 1, 3, 10**6), b"x" * 10**6)
        assert agent._largest_frame == 0                       # uploaded to the hub, never put into the cache
        agent._on_cache_full(CacheFullError("no room", 4096))
        assert agent._cache_full and agent._largest_frame == 4096
        run = SimpleNamespace(root=tmp_path, heartbeat_sec=0.1)
        monkeypatch.setattr(weights, "free_bytes", lambda path: 2 * 4096 + SPACE_MARGIN_BYTES - 1)
        agent._check_space(run, time.monotonic())
        assert agent._cache_full                                # one byte short of the cache's own rule
        monkeypatch.setattr(weights, "free_bytes", lambda path: 2 * 4096 + SPACE_MARGIN_BYTES)
        agent._check_space(run, time.monotonic() + 1.0)
        assert not agent._cache_full                            # what a cache write of that frame needs is enough
    finally:
        agent._close()


class _Control:
    def __init__(self) -> None:
        self.sent: list = []

    def send(self, message) -> None:
        self.sent.append(message)


class _Cache:
    def __init__(self) -> None:
        self.full = True

    def publish_latest(self, agent_id, frame, *, version, generation, teacher_active):
        if self.full:
            raise CacheFullError("no room: docker run --shm-size=1g", len(frame))
        return f"seg-{version}", "d" * 32


def test_a_publish_that_does_not_fit_logs_one_error_per_episode(caplog):
    control, cache = _Control(), _Cache()
    outputs = NodeOutputs(control, cache, agent_id="a", generation=1)

    def publish(version: int) -> None:
        outputs.publish_weights(WeightPayload("a", version, {"w": np.zeros(3, np.float32)}))

    with caplog.at_level(logging.ERROR, logger="colosseum.net.local"):
        for version in (1, 2, 3):                               # one episode
            publish(version)
        cache.full = False
        publish(4)                                              # space again: published
        cache.full = True
        publish(5)                                              # a new episode
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 2 and "v1 not published" in errors[0] and "v5 not published" in errors[1]
    assert [m.version for m in control.sent] == [4]
