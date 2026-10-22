"""Result flow (SP4 T0.1 on SP3 behaviour, extended by T1.1 and FX1; spec section 6, §7 P0 and block 1; SP5 T2.5 b):
every finished episode is reported exactly once with a unique match id and uid, with the weights versions of every
seat at the episode's start and end; workers put their results in batches (lists); the hub (SP5's evolved launcher,
R5) takes a node's batches in arrival order, stamps env_steps at intake and feeds each result to the coordinator, the
metrics hub and the journal writer once, after its commit (R8); a batch that cannot be decoded is counted, acked and
feeds nothing; a full results queue never blocks a worker, and every drop is counted (a WARNING on the first), also
the drops of the last flush on exit. The worker-side tests run on ``rollout_worker_process`` (the SP4 queue worker,
kept until T7.2)."""
from __future__ import annotations

import logging
import queue
import threading
import time
from functools import partial

import pytest
import torch

from colosseum.arena.matchlog import Commit
from colosseum.core.types import (
    Lineup,
    MatchResult,
    SeatAssignment,
    WeightPayload,
    WorkerCommand,
    state_dict_to_numpy,
)
from colosseum.worker import rollout_worker
from colosseum.worker.rollout_worker import ResultBatcher, rollout_worker_process
from game_harness import Collected, GameFactory, lineup, make_loop, run_steps
from game_helpers import NumpyOnlyQueue, Tick, TickGame, make_test_config, make_test_model, scripted_agent
from hub_kit import hub_results, lineup_hub, open_node, result_event

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


class _CoordinatorRecorder:
    """Only the coordinator's method: a call meant for the metrics hub fails with AttributeError."""

    def __init__(self) -> None:
        self.results: list[MatchResult] = []

    def report_match_result(self, result: MatchResult) -> None:
        self.results.append(result)


class _HubRecorder:
    """Only the metrics hub's method: a call meant for the coordinator fails with AttributeError."""

    def __init__(self) -> None:
        self.results: list[MatchResult] = []

    def on_match_result(self, result: MatchResult) -> None:
        self.results.append(result)


class _WriterRecorder:
    """Only the journal writer's tokened ``submit`` and ``take_commits`` (R8): each submit is committed whole."""

    dead = False

    def __init__(self) -> None:
        self.results: list[MatchResult] = []
        self._commits: list[Commit] = []

    def submit(self, results: list[MatchResult], *, token=None) -> int:
        self.results.extend(results)
        self._commits.append(Commit(token, list(results), 0, len(self.results)))
        return 0

    def take_commits(self) -> list[Commit]:
        out, self._commits = self._commits, []
        return out


def _hub_with_recorders(tmp_path):
    """A hub with one node session (number 1, ``a``'s learner) whose coordinator, metrics hub and journal writer are
    recorders."""
    config = make_test_config("turns", agents={"a": {}, "random": scripted_agent()})
    hub = lineup_hub(config, tmp_path)
    node = open_node(hub)
    coordinator, metrics, writer = _CoordinatorRecorder(), _HubRecorder(), _WriterRecorder()
    hub._coordinator, hub._hub, hub._match_log_writer = coordinator, metrics, writer
    return hub, node, coordinator, metrics, writer


def test_the_hub_accounts_each_result_once_after_its_commit(tmp_path):
    hub, node, coordinator, metrics, writer = _hub_with_recorders(tmp_path)
    sent = hub_results(hub, 1, 7)
    hub._budget.add(1234)
    for seq, batch in enumerate((sent[:3], sent[3:4], sent[4:7]), 1):     # a node's batches, in arrival order
        hub._on_message(result_event(hub, batch, seq=seq))
    assert coordinator.results == metrics.results == []                    # not before the commit (R8)
    hub._drain_commits()
    late = hub_results(hub, 1, 2, first=7)
    hub._budget.add(1000)
    hub._on_message(result_event(hub, late, seq=4))                        # a batch arriving at a later pass
    hub._drain_commits()
    hub._drain_commits()                                                   # nothing new: nothing fed twice
    expected = [r.uid for r in sent + late]
    for recorder in (coordinator, metrics, writer):
        assert [r.uid for r in recorder.results] == expected
    # stamped with the hub's env steps at intake (spec 4.3; node results always, P-PF2-13)
    assert [r.env_steps for r in coordinator.results] == [1234] * 7 + [2234] * 2
    assert hub._results_received == len(expected)
    assert node.acks() == [1, 2, 3, 4]


def test_an_undecodable_result_batch_is_counted_acked_and_feeds_nothing(tmp_path):
    hub, node, coordinator, metrics, writer = _hub_with_recorders(tmp_path)
    good = hub_results(hub, 1, 1, first=5)
    hub._on_message(result_event(hub, good, seq=1))
    broken = result_event(hub, hub_results(hub, 1, 2), seq=2)
    broken.results, broken.error = None, "ValueError: malformed seat"      # as the servicer's conversion fails
    hub._on_message(broken)
    hub._drain_commits()
    assert [r.uid for r in coordinator.results] == [r.uid for r in good]   # nothing fed from the broken batch
    assert [r.uid for r in metrics.results] == [r.uid for r in writer.results] == [r.uid for r in good]
    assert hub._book.get("dropped_match_results_by_reason.invalid") == 2
    assert sorted(node.acks()) == [1, 2]                                    # acked: no poison resend


def test_a_full_results_queue_never_blocks_the_worker_and_every_drop_is_counted(restore_torch_threads, caplog,
                                                                              monkeypatch):
    """Batches of 3 and no due-time flush: the first batch fills the queue, every later batch is dropped, the last
    one by the flush on exit, which the final stats report (and the "finished" log line) must count."""
    monkeypatch.setattr(rollout_worker, "ResultBatcher", partial(ResultBatcher, max_results=3, max_age_sec=3600.0))
    results, stats = NumpyOnlyQueue(maxsize=1), NumpyOnlyQueue()
    with caplog.at_level(logging.INFO, logger="colosseum.worker.rollout_worker"):
        rollout_worker_process(
            worker_id=0, env_fn=_solo_env, num_envs=2, chunk_length=3, agent_ids=["a"],
            agent_roles={"a": ["player"]}, model_factories={"a": _solo_model},
            trajectory_queues={"a": NumpyOnlyQueue()}, weight_queues={"a": NumpyOnlyQueue()},
            stop_event=threading.Event(), max_env_steps=40, lineups=[Lineup("solo", [SeatAssignment("a")])] * 2,
            results_queue=results, stats_queue=stats, stats_interval_sec=3600.0, seed=0, run_id="RUN",
        )
    assert results.qsize() == 1                                  # the first batch; the others were dropped
    first = results.get_nowait()
    assert [r.uid for r in first[:2]] == ["RUN/train/w0_e0_ep0", "RUN/train/w0_e1_ep0"] and len(first) == 3
    reports = [stats.get_nowait() for _ in range(stats.qsize())]
    final = reports[-1]                                          # the final report on exit
    assert final["episodes"] % 3 != 0                            # the flush on exit had results to drop
    assert final["dropped_results"] == final["episodes"] - 3 > 0
    drops = [r for r in caplog.records if "results queue is full" in r.getMessage()]
    assert len(drops) == 1 and drops[0].levelno == logging.WARNING   # the first drop only
    finished = [r.getMessage() for r in caplog.records if "finished." in r.getMessage()]
    assert len(finished) == 1 and f"'dropped_results': {final['dropped_results']}" in finished[0]


def test_a_running_worker_puts_due_batches_below_the_size_cap(restore_torch_threads, monkeypatch):
    """The due-time flush is on the worker's loop path (``should_stop``): with an unreachable size cap, batches still
    arrive while the worker runs (the flush on exit is only one)."""
    monkeypatch.setattr(rollout_worker, "ResultBatcher", partial(ResultBatcher, max_results=10**9))
    results, stop = NumpyOnlyQueue(), threading.Event()
    thread = threading.Thread(target=rollout_worker_process, kwargs=dict(
        worker_id=0, env_fn=_solo_env, num_envs=1, chunk_length=3, agent_ids=["a"],
        agent_roles={"a": ["player"]}, model_factories={"a": _solo_model},
        trajectory_queues={"a": NumpyOnlyQueue()}, weight_queues={"a": NumpyOnlyQueue()}, stop_event=stop,
        lineups=[Lineup("solo", [SeatAssignment("a")])], results_queue=results, seed=0,
    ))
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while results.qsize() < 2 and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        running = thread.is_alive()
    finally:
        stop.set()
        thread.join(timeout=10)
    assert running and not thread.is_alive()
    batches = [results.get_nowait() for _ in range(results.qsize())]
    assert len(batches) >= 2 and all(isinstance(b, list) and b for b in batches)     # due ones (+ the exit flush)
    ids = [r.match_id for b in batches for r in b]
    assert ids == [f"w0_e0_ep{i}" for i in range(len(ids))]


class _CountingFullQueue(NumpyOnlyQueue):
    """A one-slot trajectory queue nobody reads: counts the puts that gave up on Full (the worker's retries)."""

    def __init__(self) -> None:
        super().__init__(maxsize=1)
        self.full_waits = 0

    def put(self, item, block=True, timeout=None):
        try:
            super().put(item, block, timeout)
        except queue.Full:
            self.full_waits += 1
            raise


class _ManualClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_a_due_batch_is_put_while_a_chunk_waits_for_a_full_trajectory_queue(restore_torch_threads, monkeypatch):
    """The second chunk never fits (nobody reads the trajectory queue), so the loop stops stepping inside
    ``send_chunk``'s retries; a batch that falls due then is still put, from the retry branch. The batcher's clock
    is the test's: nothing is due until the worker is known to be retrying."""
    clock = _ManualClock()
    monkeypatch.setattr(rollout_worker, "ResultBatcher", partial(ResultBatcher, max_results=10**9, clock=clock))
    trajectories, results, stop = _CountingFullQueue(), NumpyOnlyQueue(), threading.Event()
    thread = threading.Thread(target=rollout_worker_process, kwargs=dict(
        worker_id=0, env_fn=_solo_env, num_envs=1, chunk_length=3, agent_ids=["a"],
        agent_roles={"a": ["player"]}, model_factories={"a": _solo_model},
        trajectory_queues={"a": trajectories}, weight_queues={"a": NumpyOnlyQueue()}, stop_event=stop,
        lineups=[Lineup("solo", [SeatAssignment("a")])], results_queue=results, seed=0,
    ))
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while trajectories.full_waits < 1 and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert trajectories.full_waits >= 1 and results.empty()         # retrying; nothing was due so far
        clock.t = 1.0                                                    # now the pending results are due
        while results.empty() and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        delivered_while_retrying = not results.empty() and thread.is_alive()
    finally:
        stop.set()
        thread.join(timeout=10)
    assert delivered_while_retrying and not thread.is_alive()
    batch = results.get_nowait()
    assert isinstance(batch, list) and [r.match_id for r in batch] == [f"w0_e0_ep{i}" for i in range(len(batch))]
    assert batch and results.empty()                  # the exit flush had nothing left: no episode ended meanwhile


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
