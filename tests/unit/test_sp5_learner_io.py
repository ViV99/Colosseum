"""The learner on a node (SP5 T2.2; spec 4.6, 4.7, R4, R9, R10): ``learner_process`` trains from frames through a
``ConnectionIntake`` and publishes through ``NodeOutputs``: weights into the shared-memory cache with a
``VersionPublished``, train metrics, the final snapshot as a header and one frame on the control connection; the
version floor lifts the first published version; SP4's queue path (``QueueIntake`` / ``QueueOutputs``) is unchanged."""
from __future__ import annotations

import multiprocessing as mp
import secrets
import threading
import time
from collections import Counter

from colosseum.core.config import LearnerConfig
from colosseum.core.types import TrajectoryChunk
from colosseum.learner.learner import learner_process, set_policy_version
from colosseum.net.chunks import ChunkValidator, decode_checkpoint, encode_chunk
from colosseum.net.frame import payload_byte_cap
from colosseum.net.local import (
    CheckpointHeader,
    ChildHello,
    ConnectionIntake,
    ControlChannel,
    FrameSender,
    LearnerMetrics,
    NodeOutputs,
    Stop,
    VersionPublished,
    address,
    connect,
    listener,
)
from colosseum.node.weights import WeightCacheReader, WeightCacheWriter, cache_prefix, shm_dir
from game_helpers import chunk_v2_payload, learner_appo, learner_role

KEY = b"k" * 32


class NodeEnd:
    """The node's side of a learner's control connection: collects every message (and the frame after a header)."""

    def __init__(self, conn):
        self.conn = conn
        self.messages: list = []
        self.frames: list[bytes] = []
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        while True:
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                return
            self.messages.append(msg)
            if isinstance(msg, CheckpointHeader):
                self.frames.append(self.conn.recv_bytes())

    def of(self, kind):
        return [m for m in self.messages if isinstance(m, kind)]


def _wait(pred, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _start(tmp_path, *, version_floor=0, batches=3):
    run8 = secrets.token_hex(4)
    name = address("abcdef12", run8, 1, "learner-a-i1")
    node_conn, learner_conn = mp.Pipe()
    control = ControlChannel(learner_conn)
    intake = ConnectionIntake(listener(name, KEY), control, ChunkValidator(learner_role(), agent_id="a", teacher=False),
                              authkey=KEY, agent_id="a", generation=1, version_floor=version_floor, capacity=8,
                              cap_bytes=payload_byte_cap(64))
    prefix = cache_prefix("abcdef12" * 8, run8 * 4, 1)
    outputs = NodeOutputs(control, WeightCacheWriter(shm_dir(), prefix), agent_id="a", generation=1)
    node = NodeEnd(node_conn)
    algo = learner_appo()
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, intake=intake, outputs=outputs,
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"), stop_event=intake.stop_flag,
        progress_counter=intake, total_timesteps=1000, version_floor=version_floor))
    thread.start()
    sender = FrameSender("w0", lambda: _hello(connect(name, KEY, timeout=5)), Counter())
    for v in range(2 * batches):
        chunk = TrajectoryChunk.from_payload(chunk_v2_payload(version=version_floor + v // 2, agent_id="a"))
        assert sender.put(encode_chunk(chunk, generation=1, node=1, worker=0), timeout=30.0)
    return thread, intake, node, prefix, sender, algo


def _hello(conn):
    conn.send(ChildHello("worker", "worker-0", 1, 0, "chunks"))
    return conn


def test_the_learner_trains_from_frames_through_its_intake(tmp_path):
    thread, intake, node, prefix, sender, algo = _start(tmp_path)
    try:
        assert _wait(lambda: len(node.of(LearnerMetrics)) >= 3)
        published = node.of(VersionPublished)
        assert published[0].version == 0 and [p.version for p in published] == sorted(p.version for p in published)
        assert node.of(LearnerMetrics)[-1].metrics["train_step"] >= 3
        latest = WeightCacheReader(shm_dir(), prefix).read_latest("a")
        assert latest is not None and latest.meta.policy_version == published[-1].version
        assert latest.meta.generation == 1
    finally:
        node.conn.send(Stop("test"))
        thread.join(30)
        sender.close()
        intake.close()
    assert not thread.is_alive()
    (header,) = node.of(CheckpointHeader)
    assert (header.kind, header.agent_id, header.generation) == ("final", "a", 1)
    payload, meta = decode_checkpoint(node.frames[0], cap=payload_byte_cap(64))
    assert payload["agent_id"] == "a" and payload["final"] is True and payload["policy_version"] == algo.policy_version
    assert isinstance(payload["trainer_state_bytes"], bytes) and meta["kind"] == "final"


def test_the_version_floor_lifts_the_first_published_version(tmp_path):
    thread, intake, node, _, sender, algo = _start(tmp_path, version_floor=7, batches=1)
    try:
        assert _wait(lambda: node.of(LearnerMetrics))
        assert node.of(VersionPublished)[0].version == 7
        assert node.of(LearnerMetrics)[0].metrics["train_step"] == 8
    finally:
        node.conn.send(Stop("test"))
        thread.join(30)
        sender.close()
        intake.close()


def test_set_policy_version_keeps_the_rest_of_the_state():
    algo = learner_appo()
    before = {k: v for k, v in algo.state_dict().items() if k != "policy_version"}
    set_policy_version(algo, 12)
    after = algo.state_dict()
    assert algo.policy_version == 12 and after["consumed_samples"] == before["consumed_samples"]
