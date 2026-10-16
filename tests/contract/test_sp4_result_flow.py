"""Result flow (SP4 T0.1 on SP3 behaviour, extended by T1.1; spec section 6, §7 P0 and block 1): every finished
episode is reported exactly once with a unique match id and uid, with the weights versions of every seat at the
episode's start and end; the launcher stamps env_steps and feeds each drained result to the coordinator and the
hub once; a full results queue never blocks a worker, and every drop is counted (a WARNING on the first)."""
from __future__ import annotations

import logging
import queue
import threading

import pytest
import torch

from colosseum.core.types import (
    Lineup,
    MatchResult,
    SeatAssignment,
    SeatResult,
    TeamResult,
    WeightPayload,
    WorkerCommand,
    state_dict_to_numpy,
)
from colosseum.launcher import Launcher
from colosseum.worker.rollout_worker import rollout_worker_process
from game_harness import Collected, GameFactory, lineup, make_loop, run_steps
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
    """A command's lineup is staged and applied at the env's next episode end (``MatchRunner.set_next_lineup``).
    The expected lineup of each episode is derived from the test's own timeline (the step each command was
    polled at, the step each episode ended at), not from the runner's state."""
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model, "b": lambda: model},
                          [lineup("2p", "a", "a")])
    commands: list[tuple[int, tuple[str, str]]] = []           # (step the command was polled at, agents)
    ended: list[int] = []                                      # the step each reported episode ended at
    for i in range(40):
        if i % 5 == 0:
            agents = ("a", "b") if i % 10 == 0 else ("b", "a")
            col.commands.append(WorkerCommand(lineups=[lineup("2p", *agents)]))
            commands.append((i, agents))
        before = len(col.results)
        loop.step()                                            # polls the command first, then steps the env
        ended += [i] * (len(col.results) - before)
    assert len(col.results) == loop.stats["episodes"] == len(ended) > 4
    for k, result in enumerate(col.results):
        # Episode k started after episode k-1 ended; the last command polled at or before that end step is its
        # lineup (an earlier one staged in the same episode is overwritten), else the initial lineup.
        start_after = ended[k - 1] if k > 0 else -1
        expected = ("a", "a")
        for polled, agents in commands:
            if polled <= start_after:
                expected = agents
        assert tuple(s.agent_id for s in result.seats) == expected, (k, ended, commands)
    played = {tuple(s.agent_id for s in r.seats) for r in col.results}
    assert {("a", "b"), ("b", "a")} <= played


def _result(i: int) -> MatchResult:
    return MatchResult(f"w0_e0_ep{i}", "2p", "wdl",
                       [SeatResult(0, "player", 0, "agent_0", "latest", 1.0),
                        SeatResult(1, "player", 1, "agent_0", "latest", -1.0)],
                       [TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], 4)


class _CoordinatorRecorder:
    """Only the coordinator's method: a call meant for the hub fails with AttributeError."""

    def __init__(self) -> None:
        self.results: list[MatchResult] = []

    def report_match_result(self, result: MatchResult) -> None:
        self.results.append(result)


class _HubRecorder:
    """Only the hub's method: a call meant for the coordinator fails with AttributeError."""

    def __init__(self) -> None:
        self.results: list[MatchResult] = []

    def on_match_result(self, result: MatchResult) -> None:
        self.results.append(result)


def test_the_launcher_feeds_each_drained_result_to_the_coordinator_and_the_hub_once(tmp_path):
    config = make_test_config("turns")
    launcher = Launcher(config, make_test_run_dir(config, tmp_path))
    results_queue: queue.Queue = queue.Queue()
    sent = [_result(i) for i in range(5)]
    sent[4].env_steps = 77                                      # a pulse result arrives with its snapshot's
    for item in sent:
        results_queue.put(item)
    coordinator, hub = _CoordinatorRecorder(), _HubRecorder()
    launcher._results_queue, launcher._coordinator, launcher._hub = results_queue, coordinator, hub
    launcher._env_step_counter.add(1234)
    launcher._drain_results()
    launcher._drain_results()                                   # nothing new: nothing fed twice
    assert [r.match_id for r in coordinator.results] == [r.match_id for r in sent]
    assert [r.match_id for r in hub.results] == [r.match_id for r in sent]
    assert [r.env_steps for r in coordinator.results] == [1234] * 4 + [77]     # stamped at arrival when unset


def test_a_full_results_queue_never_blocks_the_worker_and_every_drop_is_counted(restore_torch_threads, caplog):
    results, stats = NumpyOnlyQueue(maxsize=1), NumpyOnlyQueue()
    with caplog.at_level(logging.WARNING, logger="colosseum.worker.rollout_worker"):
        rollout_worker_process(
            worker_id=0, env_fn=_solo_env, num_envs=2, chunk_length=3, agent_ids=["a"],
            agent_roles={"a": ["player"]}, model_factories={"a": _solo_model},
            trajectory_queues={"a": NumpyOnlyQueue()}, weight_queues={"a": NumpyOnlyQueue()},
            stop_event=threading.Event(), max_env_steps=40, lineups=[Lineup("solo", [SeatAssignment("a")])] * 2,
            results_queue=results, stats_queue=stats, stats_interval_sec=3600.0, seed=0, run_id="RUN",
        )
    assert results.qsize() == 1                                  # the first result; the others were dropped
    reports = [stats.get_nowait() for _ in range(stats.qsize())]
    final = reports[-1]                                          # the final report on exit
    assert final["episodes"] > 1 and final["dropped_results"] == final["episodes"] - 1
    assert final["dropped_results"] > 0
    assert results.get_nowait().uid == "RUN/train/w0_e0_ep0"
    drops = [r for r in caplog.records if "results queue is full" in r.getMessage()]
    assert len(drops) == 1 and drops[0].levelno == logging.WARNING   # the first drop only


def test_latest_seat_versions_follow_weight_syncs_in_the_middle_of_an_episode():
    model = make_test_model(ROLE2)
    weights = state_dict_to_numpy(model.state_dict())
    collected = Collected(weights={"a": [WeightPayload("a", v, weights) for v in range(1, 30)]})
    # weight_sync_interval 0: one payload at construction (version 1), then one after every step; episodes take
    # two steps, so episode 0 starts at version 1 and ends at version 2 (the sync after its first step)
    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model, "b": lambda: model},
                          [lineup("2p", "a", SeatAssignment("b", collect=False))], collected=collected,
                          run_id="RUN")
    run_steps(loop, 4)
    first, second = col.results[:2]
    assert [(s.version_start, s.version_end) for s in first.seats] == [(1, 2), (0, 0)]   # b: never synced
    assert [(s.version_start, s.version_end) for s in second.seats] == [(2, 4), (0, 0)]
    assert (first.uid, second.uid) == ("RUN/train/w0_e0_ep0", "RUN/train/w0_e0_ep1")
    assert loop.stats["dropped_results"] == 0


def test_a_report_result_that_returns_false_is_counted_as_dropped():
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model}, [lineup("2p", "a", "a")])
    delivered: list[MatchResult] = []
    loop._io.report_result = lambda r: delivered.append(r) or len(delivered) % 2 == 1   # every second one fails
    run_steps(loop, 12)                                           # six episodes of two steps
    assert loop.stats["episodes"] == len(delivered) == 6 and loop.stats["dropped_results"] == 3
