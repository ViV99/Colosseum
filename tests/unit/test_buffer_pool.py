"""RolloutBuffer and BufferPool: agent-owned buffers with parking (T2.6)."""
import numpy as np
import pytest
import torch

from colosseum.worker.slots import BufferPool, RolloutBuffer, SlotTrack


def _buf(T=3, mask_size=0):
    return RolloutBuffer(chunk_length=T, obs_shape=(2,), action_shape=(), action_dtype=np.int64,
                         mask_size=mask_size)


def test_open_add_reward_done_and_build_chunk():
    b = _buf(T=3, mask_size=2)
    b.begin_chunk({"h": torch.ones(1, 4)}, policy_version=5)
    b.open(np.array([1, 2]), 1, -0.5, 0.25, np.array([True, False]), reward=0.5)
    b.add_reward(1.0)
    b.open(np.array([3, 4]), 0, -0.1, 0.5, None)
    b.mark_done()
    b.open(np.array([5, 6]), 1, -0.2, 0.75, None, reward=2.0)
    assert b.is_full
    c = b.build_chunk("a", bootstrap_value=0.9)
    assert c.agent_id == "a" and c.behavior_policy_version == 5
    assert c.rewards.tolist() == [1.5, 0.0, 2.0]
    assert c.dones.tolist() == [False, True, False] and c.dones.dtype == torch.bool
    assert c.actions.tolist() == [1, 0, 1]
    assert float(c.bootstrap_value) == pytest.approx(0.9)
    assert c.action_masks.tolist() == [[True, False], [True, True], [True, True]]
    assert torch.equal(c.initial_state["h"], torch.ones(1, 4))
    b.reset()
    assert b.steps == 0 and b.initial_state is None


def test_chunk_has_no_masks_when_none_were_given():
    b = _buf(T=1, mask_size=3)
    b.begin_chunk(None, 0)
    b.open(np.zeros(2), 0, 0.0, 0.0, None)
    assert b.build_chunk("a", 0.0).action_masks is None


def test_build_chunk_requires_full_buffer_and_begin_requires_empty():
    b = _buf(T=2)
    b.begin_chunk(None, 0)
    b.open(np.zeros(2), 0, 0.0, 0.0, None)
    with pytest.raises(RuntimeError):
        b.build_chunk("a", 0.0)
    with pytest.raises(RuntimeError):
        b.begin_chunk(None, 1)


def test_pool_prefers_parked_buffers_of_the_same_agent():
    pool = BufferPool(chunk_length=3, obs_shape=(2,), action_shape=(), action_dtype=np.int64)
    a1 = pool.acquire("a")
    a1.begin_chunk(None, 0)
    a1.open(np.zeros(2), 0, 0.0, 0.0, None)
    a1.mark_done()
    pool.park("a", a1)
    assert pool.parked_count() == 1 and pool.parked_transitions("a") == 1
    assert pool.acquire("b") is not a1
    assert pool.acquire("a") is a1
    assert pool.parked_count() == 0


def test_pool_recycles_empty_buffers_and_rejects_open_episodes():
    pool = BufferPool(chunk_length=3, obs_shape=(2,), action_shape=(), action_dtype=np.int64)
    empty = pool.acquire("a")
    pool.park("a", empty)
    assert pool.parked_count() == 0
    assert pool.acquire("b") is empty
    open_buf = pool.acquire("a")
    open_buf.begin_chunk(None, 0)
    open_buf.open(np.zeros(2), 0, 0.0, 0.0, None)
    with pytest.raises(RuntimeError, match="not done"):
        pool.park("a", open_buf)


def test_slot_track_defaults():
    t = SlotTrack(buffer=None)
    assert (t.has_open, t.pending_reward, t.state) == (False, 0.0, None)


def test_add_reward_and_mark_done_require_a_transition():
    b = _buf(T=2)
    with pytest.raises(RuntimeError, match="empty buffer"):
        b.add_reward(1.0)
    with pytest.raises(RuntimeError, match="empty buffer"):
        b.mark_done()
    assert b.steps == 0
    b.begin_chunk(None, 0)
    b.open(np.zeros(2), 0, 0.0, 0.0, None)
    b.reset()
    with pytest.raises(RuntimeError, match="empty buffer"):  # also after a reset
        b.mark_done()


def test_built_chunk_is_unaffected_by_reusing_the_buffer():
    b = _buf(T=2, mask_size=2)
    b.begin_chunk({"h": torch.ones(1, 3)}, 1)
    b.open(np.array([1, 1]), 1, -0.1, 0.1, np.array([True, False]), reward=1.0)
    b.open(np.array([2, 2]), 0, -0.2, 0.2, None, reward=2.0)
    b.mark_done()
    first = b.build_chunk("a", 0.0)
    snapshot = {k: v.clone() for k, v in vars(first).items() if isinstance(v, torch.Tensor)}
    b.reset()
    b.begin_chunk({"h": torch.full((1, 3), 7.0)}, 2)
    b.open(np.array([9, 9]), 2, -0.9, 0.9, np.array([False, True]), reward=9.0)
    b.open(np.array([8, 8]), 2, -0.8, 0.8, np.array([False, True]), reward=8.0)
    second = b.build_chunk("a", 0.5)
    for name, value in snapshot.items():
        assert torch.equal(getattr(first, name), value), name
    assert torch.equal(first.initial_state["h"], torch.ones(1, 3))
    assert first.behavior_policy_version == 1 and second.behavior_policy_version == 2
    assert second.observations.tolist() == [[9.0, 9.0], [8.0, 8.0]]


def test_pool_rejects_parking_a_full_buffer():
    pool = BufferPool(chunk_length=2, obs_shape=(2,), action_shape=(), action_dtype=np.int64)
    buf = pool.acquire("a")
    buf.begin_chunk(None, 0)
    buf.open(np.zeros(2), 0, 0.0, 0.0, None)
    buf.open(np.zeros(2), 0, 0.0, 0.0, None)
    buf.mark_done()
    with pytest.raises(RuntimeError, match="full buffer"):
        pool.park("a", buf)
    assert pool.parked_count() == 0
