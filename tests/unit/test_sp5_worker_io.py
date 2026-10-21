"""WorkerIO (SP5 T2.2, spec 4.7, R10, R11): a node worker's LoopIO over its control / chunk connections and the
shared-memory cache: chunks to the local learner (or dropped and counted without one), lineups merged per env with
each snapshot loaded once, weights from the cache only when newer, results stamped with the node and the seat
generations and batched, cumulative progress, stop and first-weights signals over the control connection."""
from __future__ import annotations

import multiprocessing as mp
import secrets
import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest

from colosseum.core.types import Lineup, SeatAssignment, TrajectoryChunk
from colosseum.net.chunks import ChunkValidator, WeightsMeta, decode_chunk
from colosseum.net.local import (
    PROGRESS_INTERVAL_SEC,
    CellLineup,
    ChildHello,
    ConnectionIntake,
    ControlChannel,
    LearnerRoute,
    Stop,
    WeightsReady,
    WorkerEvict,
    WorkerIO,
    WorkerLineups,
    WorkerProgress,
    WorkerResults,
    address,
    listener,
)
from game_helpers import chunk_v2_payload, learner_role, make_match_result

KEY = b"k" * 32


@dataclass(frozen=True)
class Key:                                   # stands for node.weights.CacheKey in these tests
    kind: str
    agent_id: str = ""
    ident: str = ""


@dataclass
class Cached:
    state_dict: dict
    meta: WeightsMeta


class FakeCache:
    """A WeightCacheReader stand-in: ``latest[agent] = (version, generation)``; ``blobs[key] = state dict``."""

    def __init__(self):
        self.latest: dict[str, tuple[int, int]] = {}
        self.blobs: dict = {}
        self.reads: list = []

    def read_latest(self, agent_id, *, newer_than=-1):
        if agent_id not in self.latest or self.latest[agent_id][0] <= newer_than:
            return None
        version, generation = self.latest[agent_id]
        return Cached({"w": np.full(2, version, np.float32)},
                      WeightsMeta(agent_id=agent_id, kind="latest", generation=generation, policy_version=version))

    def get(self, key):
        self.reads.append(key)
        sd = self.blobs.get(key)
        return None if sd is None else Cached(sd, WeightsMeta(agent_id=key.agent_id, kind="snapshot", generation=0,
                                                              policy_version=0))


def _wait(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _io(cache=None, routes=None, **kw):
    node_end, worker_end = mp.Pipe()
    io = WorkerIO(node=3, worker=1, incarnation=2, control=worker_end, route=None, cache=cache or FakeCache(),
                  routes=routes or {}, authkey=KEY, **kw)
    return io, node_end


def _messages(node_end, kind, timeout=5.0):
    out = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        while node_end.poll(0.05):
            msg = node_end.recv()
            if isinstance(msg, kind):
                out.append(msg)
        if out:
            return out
    return out


def test_chunks_without_a_learner_are_dropped_and_counted():
    io, _ = _io(routes={"a": LearnerRoute("a", None, False, 0)})
    io.send_chunk(TrajectoryChunk.from_payload(chunk_v2_payload(agent_id="a")))
    io.send_chunk(TrajectoryChunk.from_payload(chunk_v2_payload(agent_id="b")))     # no route at all
    assert io.dropped["no_learner"] == 2
    assert io.counters()["dropped_chunks_no_learner"] == 2
    io.close()


def test_chunks_go_to_the_local_learner_as_frames_with_the_seat_generation():
    name = address("abcdef12", secrets.token_hex(4), 3, "learner-a-i1")
    lst = listener(name, KEY)
    received = {}

    def learner_side():
        conn = lst.accept()
        received["hello"] = conn.recv()
        received["frames"] = [conn.recv_bytes() for _ in range(2)]
    thread = threading.Thread(target=learner_side, daemon=True)
    thread.start()
    cache = FakeCache()
    cache.latest["a"] = (4, 1)
    io, _ = _io(cache, routes={"a": LearnerRoute("a", name, False, 1)})
    assert io.poll_weights("a").generation == 1
    io.send_chunk(TrajectoryChunk.from_payload(chunk_v2_payload(version=4, agent_id="a")))
    io.send_chunk(TrajectoryChunk.from_payload(chunk_v2_payload(version=2, agent_id="a")))   # never loaded here
    thread.join(10)
    hello = received["hello"]
    assert isinstance(hello, ChildHello) and (hello.role, hello.channel, hello.incarnation) == ("worker", "chunks", 2)
    metas = [decode_chunk(f, cap=1 << 24)[1] for f in received["frames"]]
    assert [(m.policy_version, m.generation, m.node, m.worker) for m in metas] == [(4, 1, 3, 1), (2, 0, 3, 1)]
    io.close()
    lst.close()


def test_a_chunk_of_a_known_generation_passes_a_newer_learners_floor_rule():
    name = address("abcdef12", secrets.token_hex(4), 3, "learner-a-i2")
    node_end, learner_end = mp.Pipe()
    intake = ConnectionIntake(listener(name, KEY), ControlChannel(learner_end),
                              ChunkValidator(learner_role(), agent_id="a", teacher=False),
                              authkey=KEY, agent_id="a", generation=2, version_floor=10, capacity=8, cap_bytes=1 << 24)
    cache = FakeCache()
    cache.latest["a"] = (4, 1)
    io, _ = _io(cache, routes={"a": LearnerRoute("a", name, False, 2)})
    io.poll_weights("a")
    io.send_chunk(TrajectoryChunk.from_payload(chunk_v2_payload(version=4, agent_id="a")))
    chunk = intake.get(timeout=10.0)
    assert chunk.policy_version == 4 and intake.dropped == {}     # generation 1, below the floor: kept (R16)
    io.close()
    intake.close()


def test_weights_come_from_the_cache_only_when_newer():
    cache = FakeCache()
    io, _ = _io(cache)
    assert io.poll_weights("a") is None
    cache.latest["a"] = (3, 1)
    payload = io.poll_weights("a")
    assert (payload.policy_version, payload.generation) == (3, 1) and payload.state_dict["w"][0] == 3
    assert io.poll_weights("a") is None                                   # nothing newer
    cache.latest["a"] = (5, 2)
    assert io.poll_weights("a").policy_version == 5


def _lineup(net="ckpt_v5"):
    return Lineup("2p", [SeatAssignment("a", "latest", True, "owner"), SeatAssignment("a", net, False, "snapshots")])


def test_lineups_are_merged_per_env_and_each_snapshot_is_loaded_once():
    cache = FakeCache()
    snap = Key("snapshot", "a", "ckpt_v5")
    cache.blobs[snap] = {"w": np.zeros(2, np.float32)}
    io, node_end = _io(cache)
    node_end.send(WorkerLineups([CellLineup(0, _lineup(), {0: Key("latest", "a"), 1: snap}),
                                 CellLineup(2, _lineup(), {0: Key("latest", "a"), 1: snap})]))
    node_end.send(WorkerLineups([CellLineup(0, _lineup("ckpt_v5"), {0: Key("latest", "a"), 1: snap})]))
    assert _wait(lambda: (io.pump(), len(io._lineups) == 2)[1])
    cmd = io.poll_command()
    assert [lu is not None for lu in cmd.lineups] == [True, False, True]
    assert list(cmd.new_checkpoints) == ["a"] and list(cmd.new_checkpoints["a"]) == ["ckpt_v5"]
    assert cache.reads == [snap] and io.poll_command() is None
    node_end.send(WorkerEvict([snap]))
    node_end.send(WorkerLineups([CellLineup(1, _lineup(), {0: Key("latest", "a"), 1: snap})]))
    assert _wait(lambda: (io.pump(), bool(io._lineups))[1])
    cmd = io.poll_command()
    assert cmd.evict == {"a": ["ckpt_v5"]} and "ckpt_v5" in cmd.new_checkpoints["a"]   # reloaded after the evict


def test_a_snapshot_the_worker_body_loaded_itself_is_not_read_again():
    cache = FakeCache()
    snap = Key("snapshot", "a", "ckpt_v5")
    cache.blobs[snap] = {"w": np.zeros(2, np.float32)}
    io, node_end = _io(cache)
    io.note_snapshot("a", "ckpt_v5")                                  # from the first lineups (T2.4 review)
    node_end.send(WorkerLineups([CellLineup(0, _lineup(), {0: Key("latest", "a"), 1: snap})]))
    assert _wait(lambda: (io.pump(), bool(io._lineups))[1])
    cmd = io.poll_command()
    assert cmd.lineups[0] is not None and cmd.new_checkpoints == {} and cache.reads == []


def test_a_snapshot_missing_from_the_cache_is_an_error():
    io, node_end = _io(FakeCache())
    node_end.send(WorkerLineups([CellLineup(0, _lineup(), {1: Key("snapshot", "a", "ckpt_v5")})]))
    assert _wait(lambda: (io.pump(), bool(io._lineups))[1])
    with pytest.raises(RuntimeError, match="not in the weight cache"):
        io.poll_command()


def test_results_are_stamped_with_the_node_and_seat_generations_and_batched():
    cache = FakeCache()
    cache.latest["a"] = (6, 2)
    io, node_end = _io(cache, results_batch=2, results_flush_sec=10.0)
    io.poll_weights("a")
    for k in range(2):
        io.report_result(make_match_result([(0, "a", "latest", "player", 6, 6), (1, "random", "fixed")],
                                           uid=f"R/train/n3/w1_e0_g4_ep{k}"))
    (batch,) = _messages(node_end, WorkerResults)
    assert len(batch.results) == 2
    for result in batch.results:
        assert result.node == 3
        assert result.seats[0].generation == 2 and result.seats[1].generation is None


def test_progress_is_cumulative_and_carries_the_loop_and_drop_counters():
    now = [100.0]
    io, node_end = _io(clock=lambda: now[0])
    io.stats_fn = lambda: {"episodes": 4, "dropped_results": 1}
    io.add_env_steps(10)
    assert not node_end.poll(0.1)                                        # not due yet
    now[0] += PROGRESS_INTERVAL_SEC
    io.add_env_steps(5)
    (progress, *_) = _messages(node_end, WorkerProgress)
    assert progress.incarnation == 2 and progress.env_steps == 15
    assert progress.counters["episodes"] == 4 and progress.counters["dropped_chunks_no_learner"] == 0
    io.close()
    final = _messages(node_end, WorkerProgress)
    assert final[-1].env_steps == 15                                     # close sends the final counters


def test_stop_and_first_weights_come_over_the_control_connection():
    io, node_end = _io()
    assert not io.ready() and not io.stopped()
    node_end.send(WeightsReady(["a"]))
    assert _wait(io.ready)
    node_end.send(Stop("budget"))
    assert _wait(io.stopped)


def test_a_closed_node_connection_stops_the_worker():
    io, node_end = _io()
    node_end.close()
    assert _wait(io.stopped)


def test_close_stops_every_sender_within_one_shared_deadline():
    routes = {aid: LearnerRoute(aid, address("abcdef12", secrets.token_hex(4), 3, f"learner-{aid}-i1"), False, 1)
              for aid in ("a", "b")}
    io, _ = _io(routes=routes)
    for aid in ("a", "b"):                    # no listener: both senders stay in their openers (connect retries)
        io.send_chunk(TrajectoryChunk.from_payload(chunk_v2_payload(agent_id=aid)))
    start = time.monotonic()
    io.close(timeout=0.5)
    assert time.monotonic() - start < 0.9     # one deadline for both, not 0.5 s each
    assert io.counters()["dropped_chunks_disconnect"] == 2               # the frames left in their slots
