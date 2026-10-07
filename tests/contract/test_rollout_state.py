"""RolloutLoop keeps an opaque per-slot State and records it at chunk start."""

from __future__ import annotations

import pytest
import torch

from colosseum.networks.state import tree_leaves
from harness import make_loop, run_steps, simple_factory
from helpers import CORE_KINDS


@pytest.mark.parametrize("core", CORE_KINDS)
def test_chunks_carry_initial_state_with_batch_one(core):
    torch.manual_seed(0)
    loop, rec = make_loop(model_factories={"agent_0": lambda: simple_factory(core)},
                          num_envs=2, chunk_length=4)
    run_steps(loop, 8)
    loop.close()
    assert len(rec.chunks) == 8
    for c in rec.chunks:
        if core == "none":
            assert c.initial_state is None
        else:
            leaves = list(c.initial_state.values())
            assert all(leaf.shape[0] == 1 for leaf in leaves)
    if core != "none":
        # The first chunk of each slot starts at an episode start (zero state);
        # the second one starts mid-episode (non-zero state).
        first, second = rec.chunks[:4], rec.chunks[4:]
        for c in first:
            leaves = tree_leaves(c.initial_state)
            assert leaves
            assert all(int(torch.count_nonzero(leaf)) == 0 for leaf in leaves), sorted(c.initial_state)
        assert any(any(int(torch.count_nonzero(leaf)) > 0 for leaf in tree_leaves(c.initial_state))
                   for c in second)


def test_slot_state_is_reset_at_episode_end():
    torch.manual_seed(0)
    loop, _ = make_loop(model_factories={"agent_0": lambda: simple_factory("gru")},
                        num_envs=1, chunk_length=4)
    run_steps(loop, 4)
    assert float(loop._slot_states[(0, 0)]["h"].abs().sum()) > 0
    run_steps(loop, 1)  # 5th step ends the episode (episode_length=5)
    assert float(loop._slot_states[(0, 0)]["h"].abs().sum()) == 0.0
    loop.close()


def test_chunk_initial_state_owns_compact_storage():
    """A chunk's state leaves are not views into the group's batched state (no storage bloat)."""
    torch.manual_seed(0)
    loop, rec = make_loop(model_factories={"agent_0": lambda: simple_factory("lstm")},
                          num_envs=2, chunk_length=4)
    run_steps(loop, 8)
    loop.close()
    mid_episode = rec.chunks[4:]
    assert mid_episode
    for c in mid_episode:
        for leaf in c.initial_state.values():
            assert leaf.untyped_storage().nbytes() == leaf.numel() * leaf.element_size()
