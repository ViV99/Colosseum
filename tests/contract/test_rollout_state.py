"""RolloutLoop keeps an opaque per-slot State and records it at chunk start."""

from __future__ import annotations

import pytest
import torch

from colosseum.core.types import state_dict_to_numpy
from colosseum.networks.distributions import CategoricalDist
from colosseum.networks.model import PolicyModel, StepOutput
from colosseum.networks.state import tree_leaves
from harness import NUM_ACTIONS, make_loop, run_steps, simple_factory
from helpers import CORE_KINDS


@pytest.mark.parametrize("core", CORE_KINDS)
def test_chunks_carry_initial_state_with_batch_one(core):
    torch.manual_seed(0)
    loop, rec = make_loop(model_factories={"agent_0": lambda: simple_factory(core)},
                          num_envs=2, chunk_length=4)
    run_steps(loop, 9)  # the 2nd chunk of each slot is full after step 8, sealed on step 9 (T3.1)
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
    assert float(loop._tracks[0][0].state["h"].abs().sum()) > 0
    run_steps(loop, 1)  # 5th step ends the episode (episode_length=5)
    assert float(loop._tracks[0][0].state["h"].abs().sum()) == 0.0
    loop.close()


def test_chunk_initial_state_owns_compact_storage():
    """A chunk's state leaves are not views into the group's batched state (no storage bloat)."""
    torch.manual_seed(0)
    loop, rec = make_loop(model_factories={"agent_0": lambda: simple_factory("lstm")},
                          num_envs=2, chunk_length=4)
    run_steps(loop, 9)  # the 2nd chunk of each slot is full after step 8, sealed on step 9 (T3.1)
    loop.close()
    mid_episode = rec.chunks[4:]
    assert mid_episode
    for c in mid_episode:
        for leaf in c.initial_state.values():
            assert leaf.untyped_storage().nbytes() == leaf.numel() * leaf.element_size()


class LearnedInitStateModel(PolicyModel):
    """Stateful model whose initial state is a buffer (``h0``) that checkpoints carry."""

    def __init__(self, h0: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("h0", torch.full((1, 2), float(h0)))

    def initial_state(self, batch_size: int, device="cpu"):
        return {"h": self.h0.expand(batch_size, -1).clone().to(device)}

    def step(self, obs, state, action_mask=None) -> StepOutput:
        batch = obs.shape[0]
        dist = CategoricalDist(torch.zeros(batch, NUM_ACTIONS), mask=action_mask)
        return StepOutput(dist=dist, value=state["h"].sum(-1), state={"h": state["h"] + 1.0})


def test_episode_start_state_comes_from_the_seated_model():
    """A frozen checkpoint seat starts every episode from the CHECKPOINT's initial state."""
    ckpt = state_dict_to_numpy(LearnedInitStateModel(h0=5.0).state_dict())
    loop, _ = make_loop(
        model_factories={"agent_0": LearnedInitStateModel}, num_envs=1, chunk_length=4,
        slot_network_map=[["latest", "ckpt_v1"]], collect_mask=[[True, False]],
        checkpoint_state_dicts_by_agent={"agent_0": {"ckpt_v1": ckpt}},
    )

    def h(p):
        return loop._tracks[0][p].state["h"].tolist()

    assert (h(0), h(1)) == ([[0.0, 0.0]], [[5.0, 5.0]])
    run_steps(loop, 2)
    assert (h(0), h(1)) == ([[2.0, 2.0]], [[7.0, 7.0]])
    run_steps(loop, 3)  # 5th step ends the episode (episode_length=5)
    assert (h(0), h(1)) == ([[0.0, 0.0]], [[5.0, 5.0]])
    loop.close()
