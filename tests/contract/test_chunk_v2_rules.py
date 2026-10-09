"""Chunk v2 slot rules on the real RolloutLoop + MatchRunner (spec block 4, T3.4).

TickGame observations are ``[env, episode, step, seat, is_final]``; ``slot_steps``
decodes them, ``kinds`` spells a chunk's slots (A = ACT, T = terminal ACT, B = BOOT,
R = BOOT with reset_after, P = PAD).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from colosseum.core.types import SeatAssignment, WorkerCommand
from colosseum.networks.state import tree_leaves
from game_harness import GameFactory, kinds, lineup, make_loop, run_steps, seat_chunks, slot_steps
from game_helpers import Tick, TickGame, make_test_model

ROLE1 = TickGame([Tick(acting={0})], 1).spec.roles["player"]
ROLE2 = TickGame([Tick(acting={0})], 2).spec.roles["player"]


def _solo(script, *, chunk_length=4, core="none", **game_kwargs):
    model = make_test_model(ROLE1, core=core)
    loop, col = make_loop(GameFactory((script, 1), **game_kwargs), {"a": lambda: model},
                          [lineup("solo", "a")], chunk_length=chunk_length)
    return loop, col


def _acts(n, *, reward=lambda t: 0.0):
    """Ticks 0..n-1 make seat 0 act; the step after the ACT at step t rewards it ``reward(t)``."""
    return [Tick(acting={0})] + [Tick(acting={0}, rewards={0: reward(t)}) for t in range(n - 1)]


def test_mid_episode_boundary_writes_a_boot_and_continues_in_the_same_buffer():
    script = _acts(10, reward=lambda t: t + 1.0) + [Tick(over=True, rewards={0: 10.0})]
    loop, col = _solo(script)
    run_steps(loop, 7)
    assert [kinds(c) for c in col.chunks] == ["AAAB", "AAAB"]
    first, second = col.chunks
    assert [s.t for s in slot_steps(first)] == [0, 1, 2, 3]      # the BOOT is step 3's observation...
    assert [s.t for s in slot_steps(second)] == [3, 4, 5, 6]     # ...and the next chunk's first ACT
    assert first.reward.tolist() == [1.0, 2.0, 3.0, 0.0]
    assert first.reset_after.tolist() == [False] * 4 and first.terminal.tolist() == [False] * 4
    assert float(first.behavior_logp[3]) == 0.0 and bool(first.behavior_logp[:3].lt(0).all())


def test_episode_end_by_rules_marks_the_open_act_terminal_and_pads():
    script = _acts(3) + [Tick(over=True, rewards={0: 5.0})]
    loop, col = _solo(script)
    run_steps(loop, 3)                        # the episode ends: terminal ACT at slot 2, no chunk yet
    assert col.chunks == []
    run_steps(loop, 1)                        # next episode's first ACT: PAD + seal, ACT into slot 0
    (chunk,) = col.chunks
    assert kinds(chunk) == "AATP"
    assert chunk.reward.tolist() == [0.0, 0.0, 5.0, 0.0]
    assert slot_steps(chunk)[3] == slot_steps(chunk)[2]   # the PAD copies the terminal ACT's obs
    assert chunk.reset_after.tolist() == [False, False, True, True]
    run_steps(loop, 3)
    assert kinds(col.chunks[1]) == "AATP" and slot_steps(col.chunks[1])[0].ep == 1


def test_chunks_span_episodes_and_end_with_a_boot_mid_episode():
    loop, col = _solo(_acts(2) + [Tick(over=True)])
    run_steps(loop, 4)
    (chunk,) = col.chunks
    assert kinds(chunk) == "ATAB"
    assert [(s.ep, s.t) for s in slot_steps(chunk)] == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_truncation_with_the_open_act_at_s_minus_2_writes_the_final_boot_and_seals():
    script = _acts(3, reward=lambda t: 1.0) + [Tick(over=True, truncated=True, rewards={0: 2.0})]
    loop, col = _solo(script)
    run_steps(loop, 3)
    (chunk,) = col.chunks                                     # sealed at the episode end
    assert kinds(chunk) == "AAAR"
    assert (slot_steps(chunk)[3].t, slot_steps(chunk)[3].final) == (3, 1)   # final_obs
    assert chunk.reward.tolist() == [1.0, 1.0, 2.0, 0.0]
    assert chunk.terminal.tolist() == [False] * 4             # truncation is not termination
    assert chunk.reset_after.tolist() == [False, False, False, True]


def test_truncation_with_the_open_act_at_s_minus_3_boots_then_pads_at_the_next_act():
    loop, col = _solo(_acts(2) + [Tick(over=True, truncated=True)])
    run_steps(loop, 2)
    assert col.chunks == []
    run_steps(loop, 1)                                        # next episode's first ACT: PAD + seal
    (chunk,) = col.chunks
    assert kinds(chunk) == "AARP"
    assert slot_steps(chunk)[2] == slot_steps(chunk)[3]
    assert (slot_steps(chunk)[2].t, slot_steps(chunk)[2].final) == (2, 1)
    run_steps(loop, 3)
    assert kinds(col.chunks[1]) == "AARP" and slot_steps(col.chunks[1])[0].ep == 1


def test_a_seat_eliminated_in_the_truncation_step_is_terminal_and_gets_no_boot():
    script = [
        Tick(acting={0, 1}),
        Tick(acting={0, 1}, rewards={0: 0.5, 1: 0.5}),
        Tick(over=True, truncated=True, rewards={0: 1.0, 1: -1.0}, terminated={1}),
    ]
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((script, 2), global_state=True), {"a": lambda: model},
                          [lineup("2p", "a", "a")])
    run_steps(loop, 2)                                        # the truncation step eliminates seat 1
    assert col.chunks == []
    run_steps(loop, 2)                                        # episode 1: seat 0 pads, seat 1 boots
    (seat0,) = seat_chunks(col, 0)
    assert kinds(seat0) == "AARP"                             # live seat: BOOT(final_obs) with reset
    assert (slot_steps(seat0)[2].t, slot_steps(seat0)[2].final) == (2, 1)
    assert seat0.reward.tolist() == [0.5, 1.0, 0.0, 0.0]
    assert seat0.global_state[:, 1].tolist() == [0.0, 1.0, 2.0, 2.0]
    (seat1,) = seat_chunks(col, 1)
    assert kinds(seat1) == "ATAB"                             # eliminated seat: terminal ACT, no BOOT
    assert [(s.ep, s.t, s.final) for s in slot_steps(seat1)] == [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    assert seat1.reward.tolist() == [0.5, -1.0, 0.5, 0.0]               # slot 2: episode 1, step 1
    assert seat1.terminal.tolist() == [False, True, False, False]


def test_elimination_at_s_minus_2_is_terminal_with_its_step_reward_then_a_pad():
    script = [
        Tick(acting={0, 1}),
        Tick(acting={0, 1}, rewards={1: 0.5}),
        Tick(acting={0, 1}),
        Tick(acting={0}, rewards={0: 1.0, 1: -1.0}, terminated={1}),
        Tick(acting={0}),
        Tick(over=True, rewards={0: 1.0}),
    ]
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((script, 2)), {"a": lambda: model}, [lineup("2p", "a", "a")])
    run_steps(loop, 5)
    assert seat_chunks(col, 1) == []                          # terminal ACT at slot 2, slot 3 free
    run_steps(loop, 1)                                        # seat 1 acts in episode 1 -> PAD + seal
    (chunk,) = seat_chunks(col, 1)
    assert kinds(chunk) == "AATP"
    assert chunk.reward.tolist() == [0.5, 0.0, -1.0, 0.0]     # the elimination reward lands before the mark
    assert chunk.terminal.tolist() == [False, False, True, False]


def test_dead_teammate_keeps_its_open_act_until_the_final_reward():
    script = [
        Tick(acting={0, 1}),
        Tick(acting={0}, rewards={1: 0.25}),                  # seat 1 stops acting but stays live
        Tick(acting={0}, rewards={0: 1.0, 1: 0.25}),
        Tick(over=True, rewards={0: 3.0, 1: 3.0}),
    ]
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((script, 2)), {"a": lambda: model}, [lineup("2p", "a", "a")],
                          chunk_length=2)
    run_steps(loop, 4)                                        # episode 0, then seat 1 acts in episode 1
    (chunk,) = seat_chunks(col, 1)
    assert kinds(chunk) == "TP"
    assert chunk.reward.tolist() == [3.5, 0.0]                # every reward up to the episode end


def test_rewards_before_the_first_act_are_carried_into_it():
    script = [
        Tick(acting={0}),
        Tick(acting={1}, rewards={1: 0.5}),                   # seat 1's reward arrives with its first turn
        Tick(acting={0}, rewards={0: 1.0}),
        Tick(over=True, rewards={0: 1.0, 1: -1.0}),
    ]
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((script, 2)), {"a": lambda: model}, [lineup("2p", "a", "a")],
                          chunk_length=2)
    run_steps(loop, 5)
    assert [kinds(c) for c in seat_chunks(col, 0)] == ["AB", "TP"]
    assert [c.reward.tolist() for c in seat_chunks(col, 0)] == [[1.0, 0.0], [1.0, 0.0]]
    (seat1,) = seat_chunks(col, 1)
    assert kinds(seat1) == "TP"
    assert seat1.reward.tolist() == [-0.5, 0.0]               # 0.5 pending + (-1.0) final
    assert loop.stats["dropped_reward_episodes"] == 0


def test_a_seat_that_never_acts_drops_its_reward_and_is_counted(caplog):
    script = [
        Tick(acting={0}),
        Tick(acting={0}, rewards={1: 2.0}),
        Tick(over=True, rewards={0: 1.0, 1: 1.0}),
    ]
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((script, 2)), {"a": lambda: model}, [lineup("2p", "a", "a")])
    with caplog.at_level(logging.WARNING, logger="colosseum.worker.rollout_loop"):
        run_steps(loop, 6)                                    # 3 episodes
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1                                 # once per worker, further drops are only counted
    assert warnings[0].startswith("worker 0, env 0, seat 1, episode step 2, layout 2p: agent 'a' loses reward 3.0")
    assert "docs/ENV_GUIDE.md" in warnings[0] and "dropped_reward_episodes" in warnings[0]
    stats = loop.stats
    assert stats["dropped_reward_episodes"] == 3
    assert stats["recorded_transitions"] == 6
    assert sum(c.num_acts for c in col.chunks) + stats["buffered_transitions/a"] == 6
    assert all(s.seat == 0 for c in col.chunks for s in slot_steps(c))


def test_global_state_is_recorded_for_acts_and_the_final_boot():
    loop, col = _solo(_acts(2) + [Tick(over=True, truncated=True)], global_state=True)
    run_steps(loop, 3)
    (chunk,) = col.chunks
    assert kinds(chunk) == "AARP"
    assert chunk.global_state[:, 1].tolist() == [0.0, 1.0, 2.0, 2.0]   # [k, t]; the PAD copies the BOOT


@pytest.mark.parametrize("core", ["lstm", "gru"])
def test_mid_episode_boundary_with_a_parked_buffer_continues_in_the_seat_buffer(core):
    long_ep = [Tick(acting={0, 1})] * 10 + [Tick(over=True)]
    short_ep = [Tick(acting={0, 1})] * 2 + [Tick(over=True)]
    model = make_test_model(ROLE2, core=core)
    loop, col = make_loop(GameFactory((long_ep, 2), (short_ep, 2)), {"a": lambda: model},
                          [lineup("2p", "a", "a"), lineup("2p", "a", "a")])
    col.commands.append(WorkerCommand(lineups=[None, lineup("2p", "a", SeatAssignment("a", collect=False))]))
    run_steps(loop, 2)                        # env 1's episode ends: its seat 1 buffer ("AT") is parked
    assert loop.stats["parked_buffers"] == 1
    run_steps(loop, 5)                        # env 0's seats cross a chunk boundary after 3 ACTs
    assert loop.stats["parked_buffers"] == 1  # ...and the parked buffer was not taken
    first, second = seat_chunks(col, 0, env=0)[:2]
    assert kinds(first) == "AAAB" and kinds(second) == "AAAB"
    assert [(s.env, s.t) for s in slot_steps(second)] == [(0, 3), (0, 4), (0, 5), (0, 6)]
    # the continuation starts from the state after the first chunk's 3 ACTs, not from zeros
    state = model.initial_state(1)
    with torch.no_grad():
        for s in range(3):
            state = model.step(first.obs[s:s + 1], state).state
    for got, want in zip(tree_leaves(second.initial_state), tree_leaves(state)):
        assert torch.allclose(got, want, atol=1e-6)
    assert any(float(x.abs().sum()) > 0 for x in tree_leaves(second.initial_state))


def test_reward_accounting_matches_the_env_except_documented_drops():
    script = [
        Tick(acting={0, 1, 2}),
        Tick(acting={0, 1}, rewards={0: 0.5, 2: -1.0, 3: 0.75}, terminated={2}),
        Tick(acting={0}, rewards={1: 0.25, 3: 0.75}),           # seat 1 is a dead teammate from here on
        Tick(acting={0}, rewards={0: 1.0, 1: 0.5}),
        Tick(over=True, truncated=True, rewards={0: 2.0, 1: 2.0}),
    ]
    episodes, length = 5, 4
    model = make_test_model(TickGame(script, 4).spec.roles["player"])
    loop, col = make_loop(GameFactory((script, 4)), {"a": lambda: model}, [lineup("4p", "a", "a", "a", "a")],
                          chunk_length=3)
    run_steps(loop, episodes * length)                         # stops exactly at an episode end
    per_episode = sum(sum(t.rewards.values()) for t in script)
    dropped = 0.75 + 0.75                                      # seat 3 never acts
    sent = sum(float(c.reward.sum()) for c in col.chunks)
    assert sent + loop.buffered_reward("a") == pytest.approx(episodes * (per_episode - dropped))
    assert loop.stats["dropped_reward_episodes"] == episodes
    acts = loop.stats["recorded_transitions/a"]
    assert acts == episodes * (3 + 2 + 1 + 1)
    assert sum(c.num_acts for c in col.chunks) + loop.stats["buffered_transitions/a"] == acts
    assert all(np.all(c.reward[c.kind != 0].numpy() == 0.0) for c in col.chunks)
