"""Snapshot eviction (SP3 T2.2, spec blocks 3-4): coordinator -> hub -> every node (SP5 T2.5 b: ``Announce.evictions``;
the node's ``WorkerEvict`` is ``test_sp5_node_agent.py``'s) -> worker, unload when unused."""
from __future__ import annotations

import logging
import queue

import numpy as np

from colosseum.core.types import SeatAssignment, WorkerCommand, state_dict_to_numpy
from colosseum.hub.server import SessionClosed
from colosseum.net.messages import lineup_from_proto
from colosseum.worker.rollout_worker import _drain_commands
from game_harness import GameFactory, lineup, make_loop, run_steps
from game_helpers import Tick, TickGame, make_coordinator, make_test_config, make_test_model
from hub_kit import accept_version, lineup_hub, open_node, request_cells

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


def test_the_hub_sends_evictions_to_every_node_and_keeps_them_for_one_whose_stream_is_down(tmp_path):
    cfg = make_test_config("turns", checkpoint={"keep_last": 1, "keep_every": 0},
                           matchmaking={"mode": "self_play", "latest_prob": 0.0})
    hub = lineup_hub(cfg, tmp_path)
    up = open_node(hub, name="up")
    down = open_node(hub, name="down", cpu_learners=0)
    hub._on_session_closed(SessionClosed(down.node_id, down.session_id, "stream cut", down))
    accept_version(hub, "agent_0")
    hub._coordinator.checkpoint_manager.save("agent_0", 10, sd(10))
    hub._coordinator.checkpoint_manager.save("agent_0", 20, sd(20))           # evicts ckpt_v10
    hub._broadcast_announce()
    first = up.of("announce")[-1]
    assert [(e.agent_id, e.checkpoint_id) for e in first.evictions] == [("agent_0", "ckpt_v10")]
    grants = request_cells(hub, up, [(w, e, 0, 0) for w in range(2) for e in range(cfg.rollout.envs_per_worker)])
    assert grants and all(s.network_id != "ckpt_v10" for g in grants for s in lineup_from_proto(g.lineup).seats)
    assert hub._pending_evictions[down.node_id] == [("agent_0", "ckpt_v10")]    # kept for the node's return
    back = open_node(hub, node_id=down.node_id, session_id=down.session_id, cpu_learners=0)
    assert [(e.agent_id, e.checkpoint_id) for e in back.of("welcome")[0].evictions] == [("agent_0", "ckpt_v10")]
    hub._broadcast_announce()
    assert not up.of("announce")[-1].evictions and not back.of("announce")[-1].evictions   # handed out once
