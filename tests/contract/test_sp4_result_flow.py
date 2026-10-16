"""Result flow on current (SP3) behaviour (SP4 T0.1, spec section 6 and §7 P0), the baseline the SP4 journal must
keep: every finished episode is reported exactly once with a unique match id, the launcher feeds each drained
result to the coordinator and the hub once, and a full results queue never blocks a worker. T1.1 extends this
file with the SP4 record fields, versions, uids and the counted drops."""
from __future__ import annotations

import queue
import threading

import pytest
import torch

from colosseum.core.types import Lineup, MatchResult, SeatAssignment, SeatResult, TeamResult, WorkerCommand
from colosseum.launcher import Launcher
from colosseum.worker.rollout_worker import rollout_worker_process
from game_harness import GameFactory, lineup, make_loop, run_steps
from game_helpers import NumpyOnlyQueue, Tick, TickGame, make_test_config, make_test_model, make_test_run_dir

ROLE2 = TickGame([Tick(acting={0})], 2).spec.roles["player"]
SOLO = [Tick(acting={0}), Tick(acting={0}, rewards={0: 1.0}), Tick(over=True, rewards={0: 1.0})]
SOLO_ROLE = TickGame(SOLO, 1).spec.roles["player"]


def _solo_env() -> TickGame:
    return TickGame(SOLO, 1)


def _solo_model():
    return make_test_model(SOLO_ROLE)


def _episode(length: int) -> list[Tick]:
    return [Tick(acting={0, 1}, rewards={0: 1.0, 1: 1.0}) for _ in range(length)] + [
        Tick(over=True, rewards={0: 1.0, 1: -1.0})]


@pytest.fixture
def restore_torch_threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def test_every_finished_episode_is_reported_once_with_a_unique_match_id():
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((_episode(2), 2), (_episode(3), 2), (_episode(5), 2)),
                          {"a": lambda: model, "b": lambda: model},
                          [lineup("2p", "a", "b"), lineup("2p", "b", "a"), lineup("2p", "a", "a")])
    run_steps(loop, 60)
    assert len(col.results) == loop.stats["episodes"] > 0
    ids = [r.match_id for r in col.results]
    assert len(set(ids)) == len(ids)
    for env in range(3):
        episodes = [int(m.rsplit("_ep", 1)[1]) for m in ids if m.startswith(f"w0_e{env}_")]
        assert episodes == list(range(len(episodes)))          # consecutive per env, none skipped or repeated


def test_lineup_changes_keep_one_result_per_episode_with_the_lineup_that_played_it():
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model, "b": lambda: model},
                          [lineup("2p", "a", "a")])
    for i in range(40):
        if i % 5 == 0:
            col.commands.append(WorkerCommand(lineups=[lineup("2p", *("ab" if i % 10 == 0 else "ba"))]))
        loop.step()
    assert len(col.results) == loop.stats["episodes"]
    assert {tuple(s.agent_id for s in r.seats) for r in col.results} <= {("a", "a"), ("a", "b"), ("b", "a")}


def _result(i: int) -> MatchResult:
    return MatchResult(f"w0_e0_ep{i}", "2p", "wdl",
                       [SeatResult(0, "player", 0, "agent_0", "latest", 1.0),
                        SeatResult(1, "player", 1, "agent_0", "latest", -1.0)],
                       [TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], 4)


class _Recorder:
    def __init__(self) -> None:
        self.results: list[MatchResult] = []

    def report_match_result(self, result: MatchResult) -> None:
        self.results.append(result)

    def on_match_result(self, result: MatchResult) -> None:
        self.results.append(result)


def test_the_launcher_feeds_each_drained_result_to_the_coordinator_and_the_hub_once(tmp_path):
    config = make_test_config("turns")
    launcher = Launcher(config, make_test_run_dir(config, tmp_path))
    results_queue: queue.Queue = queue.Queue()
    sent = [_result(i) for i in range(5)]
    for item in sent:
        results_queue.put(item)
    coordinator, hub = _Recorder(), _Recorder()
    launcher._results_queue, launcher._coordinator, launcher._hub = results_queue, coordinator, hub
    launcher._drain_results()
    launcher._drain_results()                                   # nothing new: nothing fed twice
    assert [r.match_id for r in coordinator.results] == [r.match_id for r in sent]
    assert [r.match_id for r in hub.results] == [r.match_id for r in sent]


def test_a_full_results_queue_never_blocks_the_worker(restore_torch_threads):
    results = NumpyOnlyQueue(maxsize=1)
    rollout_worker_process(
        worker_id=0, env_fn=_solo_env, num_envs=2, chunk_length=3, agent_ids=["a"], agent_roles={"a": ["player"]},
        model_factories={"a": _solo_model}, trajectory_queues={"a": NumpyOnlyQueue()},
        weight_queues={"a": NumpyOnlyQueue()}, stop_event=threading.Event(), max_env_steps=40,
        lineups=[Lineup("solo", [SeatAssignment("a")])] * 2, results_queue=results, seed=0,
    )
    assert results.qsize() == 1                                  # the first result; the others were dropped
