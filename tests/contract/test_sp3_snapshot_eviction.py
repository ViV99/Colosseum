"""Snapshot eviction (SP3 T2.2, spec blocks 3-4): coordinator -> launcher -> worker, unload when unused."""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue

import numpy as np

from colosseum.core.types import SeatAssignment, WorkerCommand, state_dict_to_numpy
from colosseum.launcher import Launcher
from colosseum.worker.rollout_worker import _drain_commands
from game_harness import GameFactory, lineup, make_loop, run_steps
from game_helpers import Tick, TickGame, make_coordinator, make_test_config, make_test_model

ROLE2 = TickGame([Tick(acting={0})], 2).spec.roles["player"]


def _episode(length):
    return [Tick(acting={0, 1}, rewards={0: 1.0, 1: 1.0}) for _ in range(length)] + [
        Tick(over=True, rewards={0: 1.0, 1: -1.0})]


def sd(value: float) -> dict[str, np.ndarray]:
    return {"w": np.full((2, 3), value, np.float32)}


def test_the_coordinator_hands_out_each_eviction_once(tmp_path):
    cfg = make_test_config("turns", checkpoint={"keep_last": 2, "keep_every": 0},
                           matchmaking={"mode": "self_play", "latest_prob": 0.0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    for version in (10, 20, 30, 40):
        coord.checkpoint_manager.save("agent_0", version, sd(version))
    assert coord.take_evictions() == {"agent_0": ["ckpt_v10", "ckpt_v20"]}
    assert coord.take_evictions() == {}
    used = {s.network_id for lu in coord.generate_lineups(32, 0) for s in lu.seats}
    assert used <= {"latest", "ckpt_v30", "ckpt_v40"}          # evicted snapshots never enter new lineups


def test_without_match_refresh_the_coordinator_keeps_no_evictions(tmp_path):
    cfg = make_test_config("turns", checkpoint={"keep_last": 1, "keep_every": 0},
                           rollout={"match_refresh_interval_sec": 0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    for version in (10, 20, 30):
        coord.checkpoint_manager.save("agent_0", version, sd(version))
    assert coord.take_evictions() == {}                          # nobody drains them: nothing is buffered
    assert [c.checkpoint_id for c in coord.checkpoint_manager.list_checkpoints("agent_0")] == ["ckpt_v30"]


def test_drain_commands_merges_evictions():
    q = queue.Queue()
    q.put(WorkerCommand(lineups=[None], evict={"a": ["ckpt_v1"]}))
    q.put(WorkerCommand(lineups=[None], evict={"a": ["ckpt_v2", "ckpt_v1"], "b": ["ckpt_v3"]}))
    cmd = _drain_commands(q)
    assert cmd.evict == {"a": ["ckpt_v1", "ckpt_v2"], "b": ["ckpt_v3"]}
    assert WorkerCommand(lineups=[]).evict == {}


def test_the_worker_unloads_an_evicted_snapshot_once_no_lineup_uses_it(caplog):
    created = []

    def factory():
        created.append(make_test_model(ROLE2))
        return created[-1]

    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": factory},
                          [lineup("2p", "a", "a"), lineup("2p", "a", "a")])
    ckpt = state_dict_to_numpy(make_test_model(ROLE2).state_dict())

    def snap(ckpt_id):
        return SeatAssignment("a", ckpt_id, collect=False)

    try:
        col.commands.append(WorkerCommand(
            lineups=[lineup("2p", "a", snap("ckpt_v1")), lineup("2p", "a", snap("ckpt_v2"))],
            new_checkpoints={"a": {"ckpt_v1": ckpt, "ckpt_v2": ckpt}}))
        run_steps(loop, 2)              # episode 0 of both envs ends at step 2: the snapshots are seated
        assert loop.get("a", "ckpt_v1") is not None and loop.get("a", "ckpt_v2") is not None
        col.commands.append(WorkerCommand(lineups=[None, lineup("2p", "a", "a")],
                                          evict={"a": ["ckpt_v1", "ckpt_v2", "ckpt_v9"]}))
        run_steps(loop, 1)              # polled; env 0 plays ckpt_v1, env 1 plays ckpt_v2 (latest-only staged)
        assert loop.get("a", "ckpt_v1") is not None and loop.get("a", "ckpt_v2") is not None
        run_steps(loop, 1)              # episode 1 ends: env 1 applies (a, a) -> ckpt_v2 is unused
        assert loop.get("a", "ckpt_v2") is None and loop.get("a", "ckpt_v1") is not None
        col.commands.append(WorkerCommand(lineups=[lineup("2p", "a", "a"), None]))
        run_steps(loop, 2)
        assert loop.get("a", "ckpt_v1") is None and len(created) == 3
        col.commands.append(WorkerCommand(lineups=[lineup("2p", "a", snap("ckpt_v1")), None]))
        with caplog.at_level(logging.WARNING):
            run_steps(loop, 4)          # a lineup naming an unloaded snapshot: latest, collecting (SP1 rule)
    finally:
        loop.close()
    assert "network 'ckpt_v1' of 'a' is not loaded" in caplog.text
    by_episode = [(r.match_id, [s.network_id for s in r.seats]) for r in col.results]
    assert ("w0_e0_ep1", ["latest", "ckpt_v1"]) in by_episode and ("w0_e1_ep1", ["latest", "ckpt_v2"]) in by_episode
    assert ("w0_e0_ep4", ["latest", "latest"]) in by_episode


def test_the_launcher_sends_evictions_to_every_worker_and_retries_a_full_queue(tmp_path):
    cfg = make_test_config("turns", checkpoint={"keep_last": 1, "keep_every": 0},
                           matchmaking={"mode": "self_play", "latest_prob": 0.0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.checkpoint_manager.save("agent_0", 10, sd(10))
    coord.checkpoint_manager.save("agent_0", 20, sd(20))           # evicts ckpt_v10
    launcher = Launcher.__new__(Launcher)                          # _refresh_worker_matches needs only _config
    launcher._config = cfg
    ctx = mp.get_context("spawn")
    q0, q1 = ctx.Queue(maxsize=1), ctx.Queue(maxsize=1)
    q1.put("occupied")
    sent = [{"agent_0": {"ckpt_v10"}}, {"agent_0": {"ckpt_v10"}}]
    pending: list[dict[str, set[str]]] = [{}, {}]
    launcher._refresh_worker_matches(coord, ["agent_0"], [q0, q1], sent, pending)
    first = q0.get(timeout=5)
    assert first.evict == {"agent_0": ["ckpt_v10"]} and sent[0]["agent_0"] == {"ckpt_v20"}
    assert all(s.network_id != "ckpt_v10" for lu in first.lineups for s in lu.seats)
    assert q1.get(timeout=5) == "occupied" and pending[1] == {"agent_0": {"ckpt_v10"}}   # kept for the retry
    launcher._refresh_worker_matches(coord, ["agent_0"], [q0, q1], sent, pending)
    assert q1.get(timeout=5).evict == {"agent_0": ["ckpt_v10"]} and pending[1] == {"agent_0": set()}
    assert q0.get(timeout=5).evict == {}
