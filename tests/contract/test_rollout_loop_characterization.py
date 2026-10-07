"""Characterization tests for RolloutLoop (driven in-process, no worker processes)."""

from __future__ import annotations

import numpy as np
import torch

from colosseum.core.types import WorkerCommand
from harness import (
    NUM_ACTIONS,
    make_loop,
    player_index,
    run_steps,
    simple_factory,
    step_index,
    weights_payload,
)

EPISODE = 5


def test_chunk_count_shapes_and_stats():
    torch.manual_seed(0)
    loop, rec = make_loop(num_envs=2, chunk_length=4)
    run_steps(loop, 20)  # 20 transitions per slot, 4 slots -> 5 chunks per slot
    loop.close()

    assert len(rec.chunks) == 20
    assert loop.stats["chunks_sent"] == 20
    assert loop.stats["env_steps"] == 40
    for c in rec.chunks:
        assert c.agent_id == "agent_0"
        assert c.observations.shape == (4, 4)
        assert c.actions.shape == (4,) and c.actions.dtype == torch.int64
        assert c.action_log_probs.shape == (4,)
        assert c.values.shape == (4,)
        assert c.rewards.shape == (4,)
        assert c.dones.shape == (4,)
        assert c.bootstrap_value.shape == ()
        assert torch.isfinite(c.action_log_probs).all() and (c.action_log_probs <= 0).all()
        assert torch.isfinite(c.values).all()
        assert c.behavior_policy_version == 0
        assert c.action_masks is None


def test_transitions_are_consecutive_and_rewards_dones_align():
    torch.manual_seed(0)
    loop, rec = make_loop(num_envs=2, chunk_length=4)
    run_steps(loop, 20)
    loop.close()

    for c in rec.chunks:
        ts = step_index(c, EPISODE)
        (p,) = player_index(c)  # every chunk belongs to exactly one slot
        for i in range(len(ts) - 1):
            assert ts[i + 1] == (ts[i] + 1) % EPISODE
        for i, t in enumerate(ts):
            assert bool(c.dones[i]) == (t == EPISODE - 1)
            expected_reward = 1.0 if int(c.actions[i]) == (t + p) % NUM_ACTIONS else 0.0
            assert float(c.rewards[i]) == expected_reward
        if bool(c.dones[-1]):
            assert float(c.bootstrap_value) == 0.0


def test_multi_agent_routing_by_slot():
    torch.manual_seed(0)
    loop, rec = make_loop(
        agent_ids=["a", "b"], num_envs=2, chunk_length=4,
        slot_agent_map=[["a", "b"], ["a", "b"]],
    )
    run_steps(loop, 8)
    loop.close()

    by_agent = {"a": 0, "b": 0}
    for c in rec.chunks:
        (p,) = player_index(c)
        assert c.agent_id == ("a" if p == 0 else "b")
        by_agent[c.agent_id] += 1
    assert by_agent == {"a": 4, "b": 4}


def test_non_collecting_checkpoint_slot_produces_no_chunks():
    torch.manual_seed(0)
    ckpt = {k: v.clone() for k, v in simple_factory().state_dict().items()}
    loop, rec = make_loop(
        num_envs=2, chunk_length=4,
        collect_mask=[[True, False], [True, False]],
        slot_network_map=[["latest", "ckpt_v1"], ["latest", "ckpt_v1"]],
        checkpoint_state_dicts_by_agent={"agent_0": {"ckpt_v1": ckpt}},
    )
    run_steps(loop, 8)
    loop.close()

    assert len(rec.chunks) == 4
    assert all(player_index(c) == {0} for c in rec.chunks)


def test_episode_results_are_reported_per_env():
    torch.manual_seed(0)
    loop, rec = make_loop(num_envs=2, chunk_length=4)
    run_steps(loop, 20)
    loop.close()

    assert len(rec.results) == 8  # 2 envs x 4 episodes of 5 steps
    for r in rec.results:
        assert r.episode_length == EPISODE
        assert set(r.player_outcomes) == {"agent_0:latest"}  # both seats share one key
        assert set(r.total_rewards) == {"agent_0:latest"}
    assert loop.stats["episodes"] == 8


def test_initial_and_periodic_weight_sync_set_policy_version():
    torch.manual_seed(0)
    src = simple_factory()
    loop, rec = make_loop(
        num_envs=1, chunk_length=4, weight_sync_interval=0.0,
        initial_weights={"agent_0": weights_payload("agent_0", src, 7)},
    )
    run_steps(loop, 4)
    assert [c.behavior_policy_version for c in rec.chunks] == [7, 7]

    rec.pending_weights["agent_0"] = weights_payload("agent_0", src, 9)
    run_steps(loop, 1)  # the sync at the end of this step picks up version 9
    run_steps(loop, 3)
    loop.close()
    assert rec.chunks[-1].behavior_policy_version == 9
    latest = loop._networks["agent_0"]["latest"]
    for k, v in src.state_dict().items():
        assert torch.equal(latest.state_dict()[k], v)


def test_command_reassignment_applies_at_episode_boundary():
    torch.manual_seed(0)
    loop, rec = make_loop(num_envs=2, chunk_length=4)
    run_steps(loop, 10)  # two full episodes in each env
    rec.pending_commands.append(WorkerCommand(
        slot_agent_map=[["agent_0", "agent_0"], ["agent_0", "agent_0"]],
        slot_network_map=[["latest", "latest"], ["latest", "latest"]],
        collect_mask=[[True, False], [True, False]],
        new_checkpoints={},
    ))
    sent_before_boundary = None
    for i in range(25):
        loop.step()
        if i == 4:  # the 3rd episode ends on this step; the command applies here
            sent_before_boundary = len(rec.chunks)
    loop.close()

    after = rec.chunks[sent_before_boundary:]
    assert after, "player 0 must keep producing chunks"
    assert all(player_index(c) == {0} for c in after)


def test_run_respects_max_env_steps_and_stop():
    torch.manual_seed(0)
    loop, _ = make_loop(num_envs=2, chunk_length=4)
    loop.run(should_stop=lambda: True)
    assert loop.stats["env_steps"] == 0
    loop.run(should_stop=lambda: False, max_env_steps=12)
    assert loop.stats["env_steps"] == 12
    loop.close()


def test_same_seed_gives_identical_chunks():
    def collect():
        torch.manual_seed(0)
        loop, rec = make_loop(num_envs=2, chunk_length=4, seed=7)
        run_steps(loop, 12)
        loop.close()
        return rec.chunks

    a, b = collect(), collect()
    assert len(a) == len(b) > 0
    for x, y in zip(a, b):
        assert torch.equal(x.actions, y.actions)
        assert torch.equal(x.observations, y.observations)
        assert np.allclose(x.action_log_probs.numpy(), y.action_log_probs.numpy())
