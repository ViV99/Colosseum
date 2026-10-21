"""Epoch-aware match ids and cell seeds (SP5 T2.2; spec 4.3, 7.6; ruling R17): a lineup with an epoch gives match ids
``w<W>_e<E>_g<epoch>_ep<K>`` and episode seeds ``SeedSequence([cell seed, K])`` with the cell seed
``cell_seed(training.seed, epoch, node, worker, env)``; a node worker's uids carry ``n<node>/``; without an epoch and
without the new keywords the loop behaves exactly as in SP4; a node worker never replaces a missing snapshot; a
lineup naming an agent whose weights the loop has not loaded yet loads them at once (P-PF2-5); an eviction followed by
a new delivery of the same snapshot keeps it loaded."""
from __future__ import annotations

import dataclasses
import queue

import numpy as np
import pytest

from colosseum.core.types import SeatAssignment, WeightPayload, WorkerCommand, state_dict_to_numpy
from colosseum.utils.seeding import cell_seed, derive_seed, worker_seed
from colosseum.worker.rollout_worker import _drain_commands
from game_harness import Collected, GameFactory, lineup, make_loop, run_steps
from game_helpers import Tick, TickGame, make_test_model

ROLE2 = TickGame([Tick(acting={0})], 2).spec.roles["player"]


def _episode(length: int) -> list[Tick]:
    return [Tick(acting={0, 1}, rewards={0: 1.0, 1: 1.0}) for _ in range(length)] + [
        Tick(over=True, rewards={0: 1.0, 1: -1.0})]


def _models():
    model = make_test_model(ROLE2)
    return {"a": lambda: model}


def _epoch(lu, epoch):
    return dataclasses.replace(lu, epoch=epoch)


def _seed(cell: int, k: int) -> int:
    return int(np.random.SeedSequence([cell, k]).generate_state(1)[0])


def test_without_an_epoch_ids_and_seeds_are_unchanged():
    loop, col = make_loop(GameFactory((_episode(2), 2)), _models(), [lineup("2p", "a", "a")], seed=5, run_id="R")
    run_steps(loop, 20)
    first = col.results[0]
    assert (first.match_id, first.uid) == ("w0_e0_ep0", "R/train/w0_e0_ep0")
    assert first.seed == int(np.random.SeedSequence([5, 0, 0]).generate_state(1)[0])


def test_an_epoch_goes_into_the_match_id_and_the_node_uid_prefix():
    loop, col = make_loop(GameFactory((_episode(2), 2)), _models(), [_epoch(lineup("2p", "a", "a"), 7)], seed=5,
                          run_id="R", uid_prefix="R/train/n3/", cell_seed_fn=lambda env, epoch: cell_seed(5, epoch, 3,
                                                                                                        0, env))
    run_steps(loop, 20)
    assert [r.match_id for r in col.results[:2]] == ["w0_e0_g7_ep0", "w0_e0_g7_ep1"]
    assert col.results[0].uid == "R/train/n3/w0_e0_g7_ep0"


def test_episode_seeds_follow_the_cell_seed_and_the_episode_index():
    loop, col = make_loop(GameFactory((_episode(2), 2), (_episode(2), 2)), _models(),
                          [_epoch(lineup("2p", "a", "a"), 7), _epoch(lineup("2p", "a", "a"), 7)], seed=5,
                          cell_seed_fn=lambda env, epoch: cell_seed(5, epoch, 3, 0, env))
    run_steps(loop, 30)
    for result in col.results:
        env = int(result.match_id.split("_e", 1)[1].split("_", 1)[0])
        k = int(result.match_id.rsplit("_ep", 1)[1])
        assert result.seed == _seed(cell_seed(5, 7, 3, 0, env), k)
    assert len({r.seed for r in col.results}) == len(col.results)


def test_a_new_epoch_changes_the_ids_and_seeds_of_the_same_env():
    col = Collected()
    col.commands.append(WorkerCommand(lineups=[_epoch(lineup("2p", "a", "a"), 8)]))
    loop, col = make_loop(GameFactory((_episode(2), 2)), _models(), [_epoch(lineup("2p", "a", "a"), 7)], seed=5,
                          collected=col, cell_seed_fn=lambda env, epoch: cell_seed(5, epoch, 3, 0, env))
    run_steps(loop, 20)
    first, second = col.results[0], col.results[1]
    assert first.match_id == "w0_e0_g7_ep0" and second.match_id == "w0_e0_g8_ep1"
    assert second.seed == _seed(cell_seed(5, 8, 3, 0, 0), 1)


def test_without_a_training_seed_episode_seeds_stay_none():
    loop, col = make_loop(GameFactory((_episode(2), 2)), _models(), [_epoch(lineup("2p", "a", "a"), 7)], seed=0,
                          cell_seed_fn=lambda env, epoch: cell_seed(None, epoch, 3, 0, env))
    run_steps(loop, 10)
    assert col.results and all(r.seed is None for r in col.results)


def test_cell_seeds_are_distinct_per_epoch_node_worker_env():
    seeds = {cell_seed(0, g, n, w, e) for g in range(1, 4) for n in range(3) for w in range(2) for e in range(3)}
    assert len(seeds) == 54 and all(0 <= s < 2**32 for s in seeds)
    assert cell_seed(0, 1, 2, 3, 4) == cell_seed(0, 1, 2, 3, 4) and cell_seed(1, 1, 2, 3, 4) != cell_seed(0, 1, 2, 3, 4)
    assert cell_seed(None, 1, 0, 0, 0) is None and worker_seed(None, 1, 0, 1) is None
    assert len({worker_seed(0, n, w, i) for n in range(2) for w in range(2) for i in range(1, 3)}) == 8
    assert cell_seed(0, 1, 0, 0, 0) != derive_seed(0, "cell", 1)


def test_node_workers_never_replace_a_missing_snapshot():
    snapshot_seat = lineup("2p", "a", SeatAssignment("a", "ckpt_v9", collect=False))
    with pytest.raises(ValueError, match="no model for agent 'a'"):
        make_loop(GameFactory((_episode(2), 2)), _models(), [snapshot_seat], snapshot_fallback=False)
    loop, col = make_loop(GameFactory((_episode(2), 2)), _models(), [snapshot_seat])      # SP4's in-process default
    assert [s.network_id for s in loop._runner.lineup(0).seats] == ["latest", "latest"]


def test_a_lineup_naming_an_agent_without_weights_loads_them_before_its_first_episode():
    col = Collected()
    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": _models()["a"], "b": lambda: make_test_model(ROLE2)},
                          [lineup("2p", "a", "a")], collected=col, weight_sync_interval=1e9)
    col.weights["b"] = [WeightPayload.from_model("b", 3, make_test_model(ROLE2))]
    col.commands.append(WorkerCommand(lineups=[lineup("2p", "b", "a")]))
    loop.step()
    assert col.weights["b"] == []                      # polled with the lineup, not at the next periodic sync
    run_steps(loop, 20)
    first_b = next(r for r in col.results if r.seats[0].agent_id == "b")
    assert first_b.seats[0].version_start == 3


def test_a_snapshot_evicted_and_delivered_again_in_one_command_stays_loaded():
    sd = state_dict_to_numpy(make_test_model(ROLE2).state_dict())
    snap = lineup("2p", "a", SeatAssignment("a", "ckpt_v5", collect=False))
    col = Collected()
    col.commands.append(WorkerCommand(lineups=[snap], new_checkpoints={"a": {"ckpt_v5": sd}}))
    loop, col = make_loop(GameFactory((_episode(2), 2)), _models(), [lineup("2p", "a", "a")], collected=col,
                          snapshot_fallback=False)
    run_steps(loop, 8)
    assert loop._runner.lineup(0).seats[1].network_id == "ckpt_v5"
    col.commands.append(WorkerCommand(lineups=[snap], new_checkpoints={"a": {"ckpt_v5": sd}},
                                      evict={"a": ["ckpt_v5"]}))
    col.commands.append(WorkerCommand(lineups=[lineup("2p", "a", "a")]))
    run_steps(loop, 20)
    assert loop._runner.lineup(0).seats[1].network_id == "latest"
    assert loop.get("a", "ckpt_v5") is not None        # the new delivery cancelled the eviction


def test_merged_commands_keep_the_order_of_evictions_and_deliveries():
    sd = {"w": np.zeros(1, np.float32)}
    q = queue.Queue()
    q.put(WorkerCommand(lineups=[], new_checkpoints={"a": {"ckpt_v1": sd, "ckpt_v2": sd}}, evict={"a": ["ckpt_v3"]}))
    q.put(WorkerCommand(lineups=[], new_checkpoints={"a": {"ckpt_v3": sd}}, evict={"a": ["ckpt_v1"]}))
    cmd = _drain_commands(q)
    assert cmd.new_checkpoints == {"a": {"ckpt_v2": sd, "ckpt_v3": sd}}     # v1 evicted after its delivery
    assert cmd.evict == {"a": ["ckpt_v1"]}                                 # v3 delivered after its eviction
