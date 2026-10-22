"""Startup race (SP4 final fix wave, ruling A-M3): a training worker never plays a trainable agent's latest seats
before that agent's first weights arrived from its learner. ``RolloutLoop(wait_for_first_weights=stop)`` waits for
them before any env is reset (in short slices, stopping on the predicate, one INFO line after a few seconds); the
default (None) keeps the single non-blocking poll for in-process loops and tests that never push weights. The worker
processes (the node's, which ``train`` runs since SP5, and the distributed one) turn the wait on."""
from __future__ import annotations

import logging
import threading
import time

import pytest
import torch

from colosseum.core.ipc import SharedCounter
from colosseum.core.types import Lineup, SeatAssignment, WeightPayload, state_dict_to_numpy
from colosseum.worker import rollout_loop
from colosseum.worker.rollout_worker import rollout_worker_process
from game_harness import Collected, GameFactory, lineup, make_loop, run_steps
from game_helpers import NumpyOnlyQueue, Tick, TickGame, make_test_config, make_test_model

ROLE2 = TickGame([Tick(acting={0})], 2).spec.roles["player"]
SOLO = [Tick(acting={0}), Tick(acting={0}, rewards={0: 1.0}), Tick(over=True, rewards={0: 1.0})]
SOLO_ROLE = TickGame(SOLO, 1).spec.roles["player"]


def _episode(length: int) -> list[Tick]:
    return [Tick(acting={0, 1}, rewards={0: 1.0, 1: 1.0}) for _ in range(length)] + [
        Tick(over=True, rewards={0: 1.0, 1: -1.0})]


def _solo_env() -> TickGame:
    return TickGame(SOLO, 1)


def _solo_model():
    return make_test_model(SOLO_ROLE)


@pytest.fixture
def restore_torch_threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


class _LateWeights(Collected):
    """In-memory I/O whose weights arrive late: agent ``a``'s payload after ``late`` polls of it, agent ``b``'s
    right away. Every poll records how many env resets the games had seen by then."""

    def __init__(self, factory: GameFactory, payloads: dict[str, WeightPayload], late: int) -> None:
        super().__init__()
        self.factory, self.payloads, self.late = factory, dict(payloads), late
        self.polls: dict[str, int] = {}
        self.resets_at_poll: list[int] = []

    def io(self):
        io = super().io()

        def poll_weights(agent_id: str):
            self.resets_at_poll.append(sum(1 for g in self.factory.created for entry in g.log if entry[3] is None))
            self.polls[agent_id] = self.polls.get(agent_id, 0) + 1
            if agent_id == "a" and self.polls[agent_id] <= self.late:
                return None
            return self.payloads.pop(agent_id, None)

        io.poll_weights = poll_weights
        return io


def _trained_payload(agent_id: str, version: int) -> WeightPayload:
    model = make_test_model(ROLE2)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.01 * version)
    return WeightPayload(agent_id, version, state_dict_to_numpy(model.state_dict()))


def test_without_the_option_the_loop_starts_at_once_on_its_factory_weights_as_version_0():
    """The default for in-process loops and tests that never push weights: one non-blocking poll."""
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model}, [lineup("2p", "a", "a")])
    run_steps(loop, 6)
    assert col.results and all(s.version_start == 0 for r in col.results for s in r.seats)


def test_a_waiting_loop_resets_no_env_before_every_agent_has_its_first_weights(monkeypatch):
    monkeypatch.setattr(rollout_loop, "FIRST_WEIGHTS_POLL_SEC", 0.001)
    factory = GameFactory((_episode(2), 2))
    payloads = {"a": _trained_payload("a", 7), "b": _trained_payload("b", 3)}
    expected_a = {k: v.copy() for k, v in payloads["a"].state_dict.items()}
    col = _LateWeights(factory, payloads, late=5)
    loop, _ = make_loop(factory, {"a": lambda: make_test_model(ROLE2), "b": lambda: make_test_model(ROLE2)},
                        [lineup("2p", "a", "b"), lineup("2p", "b", "a")], collected=col,
                        weight_sync_interval=3600.0, wait_for_first_weights=lambda: False)
    assert col.polls["a"] == 6                              # five empty polls, then the payload
    assert set(col.resets_at_poll) == {0}                   # no env was reset while the loop waited
    assert sum(1 for g in factory.created for entry in g.log if entry[3] is None) == 2   # reset once it had them
    latest = loop.get("a", "latest").state_dict()
    assert all(torch.equal(latest[k], torch.as_tensor(v)) for k, v in expected_a.items())
    run_steps(loop, 9)
    assert len(col.results) >= 4
    versions = {(s.agent_id, s.version_start) for r in col.results for s in r.seats}
    assert versions == {("a", 7), ("b", 3)}                 # no match counted for version 0


def test_a_stop_during_the_wait_ends_it_and_the_loop_then_plays_nothing(monkeypatch):
    monkeypatch.setattr(rollout_loop, "FIRST_WEIGHTS_POLL_SEC", 0.001)
    factory = GameFactory((_episode(2), 2))
    col = _LateWeights(factory, {"a": _trained_payload("a", 7)}, late=10**9)    # a's learner never pushes
    stop_after = iter(range(3))

    def stopped() -> bool:
        return next(stop_after, None) is None

    loop, _ = make_loop(factory, {"a": lambda: make_test_model(ROLE2)}, [lineup("2p", "a", "a")], collected=col,
                        wait_for_first_weights=stopped)
    loop.run(should_stop=stopped)                           # the stop is sticky: run returns before any step
    assert loop.stats["env_steps"] == 0 and not col.results and not col.chunks


def test_a_long_wait_is_logged_once_at_info(monkeypatch, caplog):
    monkeypatch.setattr(rollout_loop, "FIRST_WEIGHTS_POLL_SEC", 0.001)
    monkeypatch.setattr(rollout_loop, "FIRST_WEIGHTS_INFO_SEC", 0.0)
    factory = GameFactory((_episode(2), 2))
    col = _LateWeights(factory, {"a": _trained_payload("a", 2)}, late=20)
    with caplog.at_level(logging.INFO, logger="colosseum.worker.rollout_loop"):
        make_loop(factory, {"a": lambda: make_test_model(ROLE2)}, [lineup("2p", "a", "a")], collected=col,
                  wait_for_first_weights=lambda: False)
    lines = [r for r in caplog.records if "first weights" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    assert "['a']" in lines[0].getMessage() and "no match starts" in lines[0].getMessage()


def test_the_worker_process_plays_no_match_before_the_first_weights(restore_torch_threads):
    """In-process worker with the wait on: no env step, chunk or result (so nothing is journaled) until the learner's
    first payload; the first results then count for that payload's version."""
    weights, results, trajectories, stop = NumpyOnlyQueue(), NumpyOnlyQueue(), NumpyOnlyQueue(), threading.Event()
    counter = SharedCounter()
    thread = threading.Thread(target=rollout_worker_process, kwargs=dict(
        worker_id=0, env_fn=_solo_env, num_envs=2, chunk_length=3, agent_ids=["a"], agent_roles={"a": ["player"]},
        model_factories={"a": _solo_model}, trajectory_queues={"a": trajectories}, weight_queues={"a": weights},
        stop_event=stop, env_step_counter=counter, lineups=[Lineup("solo", [SeatAssignment("a")])] * 2,
        results_queue=results, seed=0, wait_for_first_weights=True,
    ))
    thread.start()
    try:
        time.sleep(0.5)
        idle = (thread.is_alive(), counter.value, results.qsize(), trajectories.qsize())
        weights.put(WeightPayload("a", 5, state_dict_to_numpy(_solo_model().state_dict())))
        deadline = time.monotonic() + 10
        while results.empty() and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=10)
    assert idle == (True, 0, 0, 0)
    assert not thread.is_alive()
    played = [r for _ in range(results.qsize()) for r in results.get_nowait()]
    assert played and {s.version_start for r in played for s in r.seats} == {5}


def test_a_worker_process_stopped_while_it_waits_exits_promptly(restore_torch_threads):
    stop = threading.Event()
    thread = threading.Thread(target=rollout_worker_process, kwargs=dict(
        worker_id=0, env_fn=_solo_env, num_envs=1, chunk_length=3, agent_ids=["a"], agent_roles={"a": ["player"]},
        model_factories={"a": _solo_model}, trajectory_queues={"a": NumpyOnlyQueue()},
        weight_queues={"a": NumpyOnlyQueue()}, stop_event=stop, lineups=[Lineup("solo", [SeatAssignment("a")])],
        results_queue=NumpyOnlyQueue(), wait_for_first_weights=True,
    ))
    thread.start()
    time.sleep(0.2)
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def _recorded_worker_kwargs(monkeypatch) -> dict:
    import colosseum.worker.rollout_worker as rollout_worker

    recorded: dict = {}
    monkeypatch.setattr(rollout_worker, "rollout_worker_process", lambda **kwargs: recorded.update(kwargs))
    return recorded


def test_the_node_worker_waits_for_the_first_weights(monkeypatch):
    """``train``'s worker (the node's, SP5 T2.5 b) turns the wait on: its loop plays nothing until the node's
    ``WeightsReady`` (or the stop)."""
    from node_kit import run_worker_play

    cfg = make_test_config("turns")
    lineups = [Lineup("2p", [SeatAssignment("agent_0"), SeatAssignment("agent_0")])] * cfg.rollout.envs_per_worker
    recorded, io = run_worker_play(monkeypatch, cfg, lineups)
    wait = recorded["wait_for_first_weights"]
    assert callable(wait) and wait() is False                       # no WeightsReady yet: keep waiting
    io.weights_ready = True
    assert wait() is True


def test_the_distributed_worker_waits_for_the_first_weights(monkeypatch):
    from colosseum.distributed import _dist_worker_main
    from colosseum.hub.hub import setup_run

    recorded = _recorded_worker_kwargs(monkeypatch)
    cfg = make_test_config("turns")
    setup = setup_run(cfg, validate=False)
    lineups = [Lineup("2p", [SeatAssignment("agent_0"), SeatAssignment("agent_0")])] * cfg.rollout.envs_per_worker
    _dist_worker_main(worker_id=0, config=cfg, agent_ids=["agent_0"], agent_roles=setup.agent_roles,
                      agent_configs=setup.agent_configs, role_specs=setup.role_specs,
                      weight_store_address="localhost:1", learner_addresses={"agent_0": "localhost:2"},
                      stop_event=None, total_timesteps=100, lineups=lineups)
    assert recorded["wait_for_first_weights"] is True
