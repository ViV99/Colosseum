"""RolloutBuffer act/boot/pad slots and BufferPool parking (T3.2)."""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.types import SLOT_ACT, SLOT_BOOT, SLOT_PAD
from colosseum.sp2.envs.spaces import Units
from colosseum.sp2.worker.buffers import BufferPool, BufferSpec, RolloutBuffer

OBS_SPACE = gymnasium.spaces.Dict({
    "grid": gymnasium.spaces.Box(0, 255, (2, 2), np.uint8),
    "vec": gymnasium.spaces.Box(-1.0, 1.0, (3,), np.float32),
})
GS_SPACE = gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32)


def _spec(*, units: bool = False, global_state: bool = False) -> BufferSpec:
    action = Units(3, gymnasium.spaces.Discrete(4)) if units else gymnasium.spaces.Discrete(3)
    return BufferSpec(
        obs=ObsSpec.from_space(OBS_SPACE),
        action=ActionSpec.from_space(action),
        global_state=ObsSpec.from_space(GS_SPACE) if global_state else None,
    )


def _obs(v: int) -> dict:
    return {"grid": np.full((2, 2), v, np.uint8), "vec": np.full(3, v / 10, np.float32)}


def _act(buf: RolloutBuffer, v: int, *, reward: float = 0.0, mask=None, gs=None, unit_lp=None) -> None:
    action = np.full(3, v % 4, np.int64) if unit_lp is not None else np.int64(v % 3)
    buf.write_act(_obs(v), gs, mask, action, -float(v), unit_lp, reward)


def test_act_never_takes_the_last_slot():
    buf = RolloutBuffer(3, _spec())
    _act(buf, 1)
    _act(buf, 2)
    assert buf.free_slots == 1 and buf.has_open
    with pytest.raises(RuntimeError, match="last slot"):
        _act(buf, 3)


def test_boot_requires_an_open_act_and_pad_requires_an_episode_end():
    buf = RolloutBuffer(4, _spec())
    with pytest.raises(RuntimeError):
        buf.write_boot(_obs(0), None, reset_after=False)
    with pytest.raises(RuntimeError):
        buf.write_pad()                       # empty
    _act(buf, 1)
    with pytest.raises(RuntimeError, match="BOOT"):
        buf.write_pad()                       # after an open ACT
    buf.mark_terminal()
    assert not buf.has_open and buf.ends_episode
    with pytest.raises(RuntimeError):
        buf.write_boot(_obs(2), None, reset_after=True)   # terminal ACT is not open
    buf.write_pad()


def test_rewards_go_to_the_open_act_and_terminal_sets_reset_after():
    buf = RolloutBuffer(4, _spec())
    _act(buf, 1, reward=0.5)
    buf.add_reward(1.0)
    buf.add_reward(-0.25)
    buf.mark_terminal()
    with pytest.raises(RuntimeError):
        buf.add_reward(1.0)
    buf.write_pad()
    _act(buf, 2, reward=2.0)
    buf.write_boot(_obs(3), None, reset_after=False)
    chunk = buf.build_chunk("a")
    assert chunk.kind.tolist() == [SLOT_ACT, SLOT_PAD, SLOT_ACT, SLOT_BOOT]
    assert chunk.reward.tolist() == [1.25, 0.0, 2.0, 0.0]
    assert chunk.terminal.tolist() == [True, False, False, False]
    assert chunk.reset_after.tolist() == [True, True, False, False]
    assert chunk.behavior_logp.tolist() == [-1.0, 0.0, -2.0, 0.0]
    assert buf.reward_sum == pytest.approx(3.25)


def test_boot_and_pad_contents():
    spec = _spec(units=True, global_state=True)
    buf = RolloutBuffer(5, spec)
    mask = {"unit": np.array([True, False, True]), "action": np.ones((3, 4), bool)}
    mask["action"][0, 1:] = False
    gs = np.array([0.1, 0.2], np.float32)
    buf.begin(None, 3)
    _act(buf, 1, mask=mask, gs=gs, unit_lp=np.array([-0.1, 0.0, -0.3], np.float32))
    buf.mark_terminal()
    buf.write_pad()                                       # copies slot 0's obs and global_state
    _act(buf, 2, mask=mask, gs=gs * 2, unit_lp=np.array([-0.2, 0.0, -0.4], np.float32))
    buf.write_boot(_obs(9), gs * 3, reset_after=True)     # truncation boot
    buf.write_pad()                                       # copies the boot's obs
    chunk = buf.build_chunk("a")
    assert chunk.kind.tolist() == [SLOT_ACT, SLOT_PAD, SLOT_ACT, SLOT_BOOT, SLOT_PAD]
    assert chunk.reset_after.tolist() == [True, True, False, True, True]
    assert chunk.obs["grid"].dtype == torch.uint8
    assert chunk.obs["grid"][:, 0, 0].tolist() == [1, 1, 2, 9, 9]
    assert torch.allclose(chunk.global_state[:, 0], torch.tensor([0.1, 0.1, 0.2, 0.3, 0.3]))
    # boot/pad masks are ActionSpec.boot_mask() (units: unit=False), actions are zeros
    for s in (1, 3, 4):
        assert not chunk.action_masks["unit"][s].any()
        assert chunk.action_masks["action"][s].all()
        assert int(chunk.actions[s].abs().sum()) == 0
        assert float(chunk.behavior_unit_logp[s].abs().sum()) == 0.0
    assert chunk.action_masks["unit"][0].tolist() == [True, False, True]
    assert chunk.behavior_unit_logp[2].tolist() == pytest.approx([-0.2, 0.0, -0.4])
    assert chunk.policy_version == 3 and chunk.num_acts == 2


def test_boot_and_pad_overwrite_stale_values_of_a_reused_buffer():
    buf = RolloutBuffer(2, _spec())
    mask = np.array([True, False, False])
    _act(buf, 2, mask=mask)
    buf.write_boot(_obs(4), None, reset_after=False)
    buf.build_chunk("a")
    buf.reset()
    _act(buf, 5, mask=np.array([False, True, False]), reward=1.0)
    buf.mark_terminal()
    buf.write_pad()
    chunk = buf.build_chunk("a")
    assert chunk.action_masks[1].tolist() == [True, True, True]
    assert int(chunk.actions[1]) == 0 and float(chunk.reward[1]) == 0.0


def test_unit_log_probs_are_required_only_for_multi_decider_actions():
    single = RolloutBuffer(3, _spec())
    _act(single, 1)                                      # K == 1: no unit log-probs
    multi = RolloutBuffer(3, _spec(units=True))
    with pytest.raises(ValueError, match="unit_log_probs"):
        multi.write_act(_obs(1), None, None, np.zeros(3, np.int64), -1.0, None, 0.0)


def test_global_state_is_required_when_the_role_declares_it():
    buf = RolloutBuffer(3, _spec(global_state=True))
    with pytest.raises(ValueError, match="global_state"):
        _act(buf, 1)


def test_begin_clones_state_and_only_on_an_empty_buffer():
    buf = RolloutBuffer(3, _spec())
    batched = torch.arange(6.0).reshape(3, 2)
    row = batched[1:2]
    buf.begin({"h": row}, 4)
    batched.zero_()
    assert buf.initial_state["h"].tolist() == [[2.0, 3.0]]
    assert buf.initial_state["h"].untyped_storage().nbytes() == 2 * 4
    _act(buf, 1)
    with pytest.raises(RuntimeError):
        buf.begin(None, 5)


def test_build_chunk_needs_a_full_buffer_and_reset_empties_it():
    buf = RolloutBuffer(2, _spec())
    _act(buf, 1)
    with pytest.raises(RuntimeError, match="partial"):
        buf.build_chunk("a")
    buf.write_boot(_obs(2), None, reset_after=False)
    assert buf.is_full and not buf.ends_episode
    buf.reset()
    assert buf.slots_used == 0 and buf.num_acts == 0 and buf.ends_episode


def test_ends_episode_skips_pads():
    buf = RolloutBuffer(5, _spec())
    _act(buf, 1)
    assert not buf.ends_episode
    buf.write_boot(_obs(2), None, reset_after=True)
    assert buf.ends_episode
    buf.write_pad()
    assert buf.ends_episode


def test_pool_prefers_parked_then_free_buffers_per_agent():
    pool = BufferPool(4, {"a": _spec(), "b": _spec(units=True)})
    a1 = pool.acquire("a")
    _act(a1, 1)
    with pytest.raises(RuntimeError, match="middle of an episode"):
        pool.park("a", a1)
    a1.mark_terminal()
    pool.park("a", a1)
    assert pool.acquire("a") is a1                   # a parked buffer comes back first
    fresh = pool.acquire("a")                        # nothing parked or free: a new buffer
    assert fresh is not a1
    pool.park("a", fresh)                            # an empty buffer becomes a free one
    pool.park("a", a1)
    assert pool.parked_count() == 1 and pool.parked_acts("a") == 1 and pool.parked_acts("b") == 0
    assert pool.acquire("b").spec.action.has_units  # never another agent's buffer
    assert pool.acquire("a") is a1                   # parked before free
    assert pool.acquire("a") is fresh
    assert pool.parked_count() == 0


def test_pool_refuses_full_buffers_and_unknown_agents():
    pool = BufferPool(2, {"a": _spec()})
    buf = pool.acquire("a")
    _act(buf, 1)
    buf.write_boot(_obs(2), None, reset_after=True)
    with pytest.raises(RuntimeError, match="full"):
        pool.park("a", buf)
    with pytest.raises(KeyError):
        pool.acquire("zzz")


def test_parked_reward_sums_unsent_rewards():
    pool = BufferPool(4, {"a": _spec()})
    buf = pool.acquire("a")
    _act(buf, 1, reward=1.5)
    buf.add_reward(0.5)
    buf.mark_terminal()
    pool.park("a", buf)
    assert pool.parked_reward("a") == pytest.approx(2.0)
    assert pool.parked_reward("b") == 0.0
