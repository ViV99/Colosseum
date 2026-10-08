"""RolloutLoop lineups, commands, checkpoints, versions and parking without loss (T3.4)."""

from __future__ import annotations

import logging
import random

import pytest

from colosseum.sp2.core.types import SeatAssignment, WeightPayload, WorkerCommand, state_dict_to_numpy
from colosseum.sp2.worker.rollout_loop import RolloutLoop
from game_harness import Collected, GameFactory, kinds, lineup, make_loop, run_steps, slot_steps
from game_helpers import Tick, TickGame, make_test_model

ROLE2 = TickGame([Tick(acting={0})], 2).spec.roles["player"]


def _episode(length):
    return [Tick(acting={0, 1}, rewards={0: 1.0, 1: 1.0}) for _ in range(length)] + [
        Tick(over=True, rewards={0: 1.0, 1: -1.0})]


def _random_lineup(rng):
    seats = [SeatAssignment(rng.choice("ab"), collect=rng.random() < 0.7) for _ in range(2)]
    return lineup("2p", *seats)


def test_lineup_changes_park_buffers_without_losing_or_duplicating_acts():
    rng = random.Random(0)
    models = {"a": make_test_model(ROLE2), "b": make_test_model(ROLE2)}
    loop, col = make_loop(GameFactory((_episode(3), 2), (_episode(5), 2)),
                          {"a": lambda: models["a"], "b": lambda: models["b"]},
                          [lineup("2p", "a", "a"), lineup("2p", "b", "b")], chunk_length=4)
    max_parked = 0
    for i in range(200):
        if i % 3 == 0:
            col.commands.append(WorkerCommand(lineups=[_random_lineup(rng), _random_lineup(rng)]))
        loop.step()
        max_parked = max(max_parked, loop.stats["parked_buffers"])
        assert loop.stats["parked_buffers"] <= 2 * 2 * 2           # agents x envs x seats
    assert max_parked > 0, "the scenario must park buffers"
    stats = loop.stats
    for aid in "ab":
        sent = sum(c.num_acts for c in col.chunks if c.agent_id == aid)
        assert sent + stats[f"buffered_transitions/{aid}"] == stats[f"recorded_transitions/{aid}"]
    seen, resumed = set(), 0
    for chunk in col.chunks:
        slots, letters = slot_steps(chunk), kinds(chunk)
        if len({(s.env, s.seat) for s in slots}) > 1:
            resumed += 1
        for i, (slot, letter) in enumerate(zip(slots, letters)):
            if letter in "AT":
                key = (slot.env, slot.ep, slot.t, slot.seat)
                assert key not in seen, f"ACT {key} sent twice"
                seen.add(key)
            if i + 1 == len(slots):
                continue
            nxt = slots[i + 1]
            if letter == "A":       # an open ACT continues in the same (env, episode, seat)
                assert (nxt.env, nxt.ep, nxt.seat, nxt.t) == (slot.env, slot.ep, slot.seat, slot.t + 1)
            elif letters[i + 1] in "AT":   # after an episode end the next ACT starts an episode
                assert nxt.t == 0
    assert resumed > 0, "no chunk continues a parked buffer in another (env, seat)"


def test_command_checkpoint_is_loaded_at_once_and_seated_from_the_next_episode():
    created = []

    def factory():
        created.append(make_test_model(ROLE2))
        return created[-1]

    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": factory}, [lineup("2p", "a", "a")])
    ckpt = state_dict_to_numpy(make_test_model(ROLE2).state_dict())
    col.commands.append(WorkerCommand(
        lineups=[lineup("2p", "a", SeatAssignment("a", "ckpt_v9", collect=False))],
        new_checkpoints={"a": {"ckpt_v9": ckpt}},
    ))
    loop.step()
    assert len(created) == 2 and loop.get("a", "ckpt_v9") is created[1]     # loaded at once
    assert loop.get("a", "latest") is created[0]
    loop.step()                                              # episode 0 ends -> lineup applied
    run_steps(loop, 2)                                       # episode 1
    first, second = col.results
    assert [s.network_id for s in first.seats] == ["latest", "latest"]
    assert [s.network_id for s in second.seats] == ["latest", "ckpt_v9"]
    assert loop.stats["recorded_transitions"] == 2 * 2 + 2   # seat 1 stopped collecting in episode 1


def test_a_missing_checkpoint_is_replaced_by_latest_which_collects(caplog):
    model = make_test_model(ROLE2)
    with caplog.at_level(logging.WARNING):
        loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model},
                              [lineup("2p", "a", SeatAssignment("a", "ckpt_v3", collect=False))])
    run_steps(loop, 2)
    assert col.results[0].seats[1].network_id == "latest"
    assert loop.stats["recorded_transitions"] == 4          # both seats collect
    assert any("ckpt_v3" in r.getMessage() for r in caplog.records)


def test_chunk_version_is_the_version_at_its_first_slot_also_after_parking():
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((_episode(3), 2)), {"a": lambda: model, "b": lambda: model},
                          [lineup("2p", "a", "b")], chunk_length=8)
    col.commands.append(WorkerCommand(lineups=[lineup("2p", "a", SeatAssignment("b", collect=False))]))
    run_steps(loop, 4)                                       # episode 0 ends: b's 3 ACTs are parked
    assert loop.stats["parked_buffers"] == 1 and loop.stats["buffered_transitions/b"] == 3
    col.weights["b"] = [WeightPayload.from_model("b", 5, model)]
    col.commands.append(WorkerCommand(lineups=[lineup("2p", "a", "b")]))
    loop.step()                                              # loads b v5; the lineup waits for the episode end
    run_steps(loop, 3)                                       # episode 1 ends: seat 1 resumes the parked buffer
    assert loop.stats["parked_buffers"] == 0
    run_steps(loop, 8)
    b_chunks = [c for c in col.chunks if c.agent_id == "b"]
    assert b_chunks and b_chunks[0].policy_version == 0       # parked under v0, resumed under v5
    assert [s.ep for s in slot_steps(b_chunks[0])][:5] == [0, 0, 0, 2, 2]   # episode 1 was not collected


def test_results_are_reported_per_episode_and_stats_have_the_contract_keys():
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model, "b": lambda: model},
                          [lineup("2p", "a", "b"), lineup("2p", "b", "a")])
    run_steps(loop, 4)
    assert [r.match_id for r in col.results] == ["w0_e0_ep0", "w0_e1_ep0", "w0_e0_ep1", "w0_e1_ep1"]
    assert [s.agent_id for s in col.results[1].seats] == ["b", "a"]
    assert col.results[0].outcome_kind == "wdl" and col.results[0].teams[0].rank == 1.0
    assert set(loop.stats) == {
        "chunks_sent", "env_steps", "episodes", "parked_buffers", "dropped_reward_episodes",
        "recorded_transitions", "recorded_transitions/a", "recorded_transitions/b",
        "buffered_transitions/a", "buffered_transitions/b",
    }
    assert loop.stats["env_steps"] == 8 and loop.stats["episodes"] == 4
    assert col.env_steps == [2] * 4


@pytest.mark.parametrize("bad", ["unknown agent", "wrong layout"])
def test_bad_initial_lineups_fail_fast(bad):
    model = make_test_model(ROLE2)
    lu = lineup("2p", "zzz", "a") if bad == "unknown agent" else lineup("3p", "a", "a", "a")
    with pytest.raises(ValueError):
        make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model}, [lu])


def test_a_lineup_count_other_than_num_envs_fails_before_any_env_is_built():
    model = make_test_model(ROLE2)
    games = GameFactory((_episode(2), 2))
    with pytest.raises(ValueError, match="2 envs"):
        RolloutLoop(worker_id=0, env_fn=games, num_envs=2, chunk_length=4, agent_ids=["a"],
                    agent_roles={"a": ["player"]}, model_factories={"a": lambda: model},
                    io=Collected().io(), lineups=[lineup("2p", "a", "a")])
    assert games.created == []


def test_a_command_with_more_lineups_than_envs_fails():
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model}, [lineup("2p", "a", "a")])
    col.commands.append(WorkerCommand(lineups=[lineup("2p", "a", "a"), lineup("2p", "a", "a")]))
    with pytest.raises(ValueError, match="1 envs"):
        loop.step()


class _ClosingTickGame(TickGame):
    """TickGame that records ``close()`` calls in a shared list."""

    def __init__(self, closed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._closed = closed

    def close(self) -> None:
        self._closed.append(self.tag)


@pytest.mark.parametrize("bad", ["unknown agent", "missing agent_roles entry"])
def test_a_failing_constructor_closes_the_vector_env(bad):
    model = make_test_model(ROLE2)
    closed, built = [], []

    def env_fn():
        built.append(_ClosingTickGame(closed, _episode(2), 2, tag=len(built)))
        return built[-1]

    lu = lineup("2p", "zzz", "a") if bad == "unknown agent" else lineup("2p", "a", "a")
    roles = {"a": ["player"]} if bad == "unknown agent" else {}
    with pytest.raises((ValueError, KeyError)):
        RolloutLoop(worker_id=0, env_fn=env_fn, num_envs=2, chunk_length=4, agent_ids=["a"],
                    agent_roles=roles, model_factories={"a": lambda: model},
                    io=Collected().io(), lineups=[lu, lu])
    assert len(built) == 2 and sorted(closed) == [0, 1]
