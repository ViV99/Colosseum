"""Scripted kickstart teacher on the worker (DAgger, spec block 6): labels in the ACT slots of the
collecting seats, one teacher instance per (student, env, seat), resets at episode starts, infos,
legality, the teacher flag, and the learner still reproduces the worker's log-probs (T4.5)."""
from __future__ import annotations

from collections import defaultdict

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.core.errors import PlayerError
from colosseum.core.types import SLOT_ACT, SeatAssignment, TrajectoryChunk, WeightPayload
from colosseum.envs.game import RoleSpec
from colosseum.players.registry import BotSpec
from game_harness import Collected, GameFactory, learner_eval, lineup, make_loop, run_steps, run_until_chunks
from game_helpers import SCRIPT_OBS_SPACE, LabelBot, Tick, make_test_model

ROLE = RoleSpec(SCRIPT_OBS_SPACE, gymnasium.spaces.Discrete(3))
TEACHER = {"a": BotSpec("game_helpers.LabelBot", {"action": 2})}
THREE_MOVES = [Tick(acting={0, 1}), Tick(acting={0, 1}), Tick(acting={0, 1}), Tick(over=True)]
SOLO = [Tick(acting={0}), Tick(acting={0}), Tick(acting={0}), Tick(over=True)]


@pytest.fixture(autouse=True)
def clear_label_bot():
    LabelBot.calls.clear()
    yield
    LabelBot.calls.clear()


def factory():
    torch.manual_seed(0)
    return make_test_model(ROLE)


def calls(kind: str) -> list[tuple]:
    return [c for c in LabelBot.calls if c[0] == kind]


def test_act_slots_of_collecting_seats_carry_the_teachers_action():
    loop, col = make_loop(GameFactory((THREE_MOVES, 2)), {"a": factory}, [lineup("2p", "a", "a")],
                          chunk_length=8, teachers=TEACHER)
    for chunk in run_until_chunks(loop, col, 4):
        chunk = TrajectoryChunk.from_payload(chunk.to_payload())
        is_act = chunk.kind == SLOT_ACT
        assert torch.equal(chunk.has_teacher, is_act)
        assert (chunk.teacher_action[is_act] == 2).all() and (chunk.teacher_action[~is_act] == 0).all()
        assert chunk.teacher_action.dtype == chunk.actions.dtype
    loop.close()


def test_only_collecting_seats_ask_their_teacher():
    loop, _col = make_loop(GameFactory((THREE_MOVES, 2)), {"a": factory},
                           [lineup("2p", "a", SeatAssignment("a", collect=False))], chunk_length=8, teachers=TEACHER)
    run_steps(loop, 12)
    acts = calls("act")
    assert acts and {int(obs[3]) for _kind, _id, obs, _info in acts} == {0}     # obs[3] is the seat
    loop.close()


def test_one_teacher_per_student_env_and_seat_reset_at_every_episode_start():
    loop, _col = make_loop(GameFactory((THREE_MOVES, 2)), {"a": factory}, [lineup("2p", "a", "a")] * 2,
                           chunk_length=8, teachers=TEACHER)
    run_steps(loop, 9)                             # 3 episodes of 3 env steps in each env
    assert len(calls("init")) == 4                 # 2 envs x 2 seats
    episodes_by_bot: dict[int, set] = defaultdict(set)
    for _kind, bot, obs, _info in calls("act"):
        episodes_by_bot[bot].add((int(obs[0]), int(obs[1])))    # (env tag, episode)
    resets = calls("reset")
    for bot, episodes in episodes_by_bot.items():
        assert len(episodes) == 3 and len({e for e, _ in episodes}) == 1
        assert len([r for r in resets if r[1] == bot]) == 3     # once per episode it collected in
    assert {(r[2], r[4]) for r in resets} == {("player", "2p")} and {r[3] for r in resets} == {0, 1}
    loop.close()


def test_teacher_rng_is_reproducible_with_the_seed():
    def draws(seed: int) -> list[float]:
        LabelBot.calls.clear()
        loop, _col = make_loop(GameFactory((SOLO, 1)), {"a": factory}, [lineup("solo", "a")], chunk_length=8,
                               teachers=TEACHER, seed=seed)
        run_steps(loop, 8)
        loop.close()
        return [r[5] for r in calls("reset")]

    assert draws(3) == draws(3) and draws(3) != draws(4)


def test_the_teacher_sees_the_seats_infos():
    loop, _col = make_loop(GameFactory((SOLO, 1), infos=True), {"a": factory}, [lineup("solo", "a")],
                           chunk_length=8, teachers=TEACHER)
    run_steps(loop, 4)                              # 3 decisions of episode 0, the first of episode 1
    assert [info for _kind, _id, _obs, info in calls("act")] == [
        {"k": 0, "t": 0, "seat": 0}, {"k": 0, "t": 1, "seat": 0}, {"k": 0, "t": 2, "seat": 0},
        {"k": 1, "t": 0, "seat": 0}]
    loop.close()


@pytest.mark.parametrize("bot, message", [
    (BotSpec("game_helpers.LabelBot", {"action": 2}), "kickstart teacher of agent 'a'"),
    (BotSpec("game_helpers.LabelBot", {"action": 1, "fail": "raise"}), "act raised RuntimeError: teacher bot failure"),
])
def test_an_illegal_action_or_a_failure_of_the_teacher_is_a_player_error(bot, message):
    masked = GameFactory((SOLO, 1), mask_fn=lambda k, t, seat: np.array([True, True, False]))
    loop, _col = make_loop(masked, {"a": factory}, [lineup("solo", "a")], chunk_length=8, teachers={"a": bot})
    with pytest.raises(PlayerError, match="worker 0, env 0, seat 0, episode step 0, layout solo") as info:
        run_steps(loop, 1)
    assert message in str(info.value)
    loop.close()


def test_the_teacher_flag_stops_the_queries():
    col = Collected()
    loop, col = make_loop(GameFactory((SOLO, 1)), {"a": factory}, [lineup("solo", "a")], chunk_length=4,
                          teachers=TEACHER, collected=col)
    run_steps(loop, 6)
    asked = len(calls("act"))
    assert asked == 6
    col.weights["a"] = [WeightPayload.from_model("a", 1, factory(), teacher_active=False)]
    run_steps(loop, 1)                              # this step still asks; the sync after it reads the flag
    run_steps(loop, 8)
    assert len(calls("act")) == asked + 1
    last = TrajectoryChunk.from_payload(col.chunks[-1].to_payload())
    assert last.policy_version == 1 and not last.has_teacher.any()
    loop.close()


def test_the_learner_reproduces_the_worker_with_labels_in_the_chunks():
    loop, col = make_loop(GameFactory((THREE_MOVES, 2)), {"a": factory}, [lineup("2p", "a", "a")],
                          chunk_length=8, teachers=TEACHER)
    chunks = run_until_chunks(loop, col, 3)
    view = learner_eval(factory(), chunks, ROLE)
    assert torch.allclose(view.log_probs[view.is_act], view.worker_log_probs[view.is_act], atol=1e-5)
    loop.close()
