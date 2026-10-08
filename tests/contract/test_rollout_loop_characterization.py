"""Characterization tests for RolloutLoop (driven in-process, no worker processes)."""

from __future__ import annotations

import numpy as np
import torch

from colosseum.core.types import WorkerCommand, state_dict_to_numpy
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
    run_steps(loop, 9)  # the 2nd chunk of each slot is full after step 8, sealed on step 9 (T3.1)
    loop.close()

    by_agent = {"a": 0, "b": 0}
    for c in rec.chunks:
        (p,) = player_index(c)
        assert c.agent_id == ("a" if p == 0 else "b")
        by_agent[c.agent_id] += 1
    assert by_agent == {"a": 4, "b": 4}


def test_non_collecting_checkpoint_slot_produces_no_chunks():
    torch.manual_seed(0)
    ckpt = state_dict_to_numpy(simple_factory().state_dict())
    loop, rec = make_loop(
        num_envs=2, chunk_length=4,
        collect_mask=[[True, False], [True, False]],
        slot_network_map=[["latest", "ckpt_v1"], ["latest", "ckpt_v1"]],
        checkpoint_state_dicts_by_agent={"agent_0": {"ckpt_v1": ckpt}},
    )
    run_steps(loop, 9)  # the 2nd chunk of each slot is full after step 8, sealed on step 9 (T3.1)
    loop.close()

    assert len(rec.chunks) == 4
    assert all(player_index(c) == {0} for c in rec.chunks)
    # The checkpoint network really occupies seat 1 (results key it separately).
    assert len(rec.results) == 2
    assert all(set(r.player_outcomes) == {"agent_0:latest", "agent_0:ckpt_v1"} for r in rec.results)


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
    # Version 9 is offered before step 5. Step 5 acts under version 7: it seals the
    # chunk of steps 1..4 (T3.1: sealed when the slot acts again) and opens the
    # chunk of steps 5..8 under version 7; the sync at the END of step 5 loads v9.
    rec.weights["agent_0"] = [weights_payload("agent_0", src, 9)]
    run_steps(loop, 1)
    assert [c.behavior_policy_version for c in rec.chunks] == [7, 7]
    run_steps(loop, 4)  # step 9 seals the chunk of steps 5..8
    # behavior_policy_version is the version at a chunk's FIRST transition (T2.6):
    # these chunks started on step 5, before the sync at its end loaded version 9.
    assert [c.behavior_policy_version for c in rec.chunks[2:]] == [7, 7]
    run_steps(loop, 4)  # chunks of steps 9..12 (sealed on step 13), entirely under version 9
    loop.close()
    assert [c.behavior_policy_version for c in rec.chunks[4:]] == [9, 9]
    latest = loop._models["agent_0"]["latest"]
    for k, v in src.state_dict().items():
        assert torch.equal(latest.state_dict()[k], v)


def test_command_reassignment_applies_at_episode_boundary():
    # Both envs run in lockstep: episodes end on steps 5, 10, 15, 20, ...
    torch.manual_seed(0)
    loop, rec = make_loop(
        agent_ids=["a", "b"], num_envs=2, chunk_length=4,
        slot_agent_map=[["a", "a"], ["a", "a"]],
    )
    run_steps(loop, 10)  # two full episodes per env; every buffer now holds 2/4 transitions
    chunks_warm, results_warm = len(rec.chunks), len(rec.results)
    assert (chunks_warm, results_warm) == (8, 4)

    # Env 0 seat 1 switches to agent "b"; env 1 seat 1 stops collecting.
    rec.commands.append(WorkerCommand(
        slot_agent_map=[["a", "b"], ["a", "a"]],
        slot_network_map=[["latest", "latest"], ["latest", "latest"]],
        collect_mask=[[True, True], [True, False]],
        new_checkpoints={},
    ))
    run_steps(loop, 5)  # steps 11..15: command polled on step 11, 3rd episode ends on step 15
    chunks_boundary, results_boundary = len(rec.chunks), len(rec.results)
    run_steps(loop, 20)  # steps 16..35
    loop.close()

    # Until the boundary the OLD assignment holds: step 12 seals one agent-"a" chunk per slot.
    before = rec.chunks[chunks_warm:chunks_boundary]
    assert sorted((c.agent_id, *player_index(c)) for c in before) == [("a", 0), ("a", 0), ("a", 1), ("a", 1)]
    boundary_results = rec.results[results_warm:results_boundary]
    assert len(boundary_results) == 2
    assert all(set(r.player_outcomes) == {"a:latest"} for r in boundary_results)
    # Env 0's next episode is played under the new assignment.
    assert set(rec.results[results_boundary].player_outcomes) == {"a:latest", "b:latest"}

    after = rec.chunks[chunks_boundary:]
    a_after = [c for c in after if c.agent_id == "a"]
    b_after = [c for c in after if c.agent_id == "b"]
    assert len(a_after) + len(b_after) == len(after)
    # Seat 0 keeps collecting in both envs (its buffers are untouched): 2 x 5 chunks.
    assert len(a_after) == 10 and all(player_index(c) == {0} for c in a_after)
    # Agent "b" starts from an empty buffer at the episode start (no stale agent-"a"
    # transitions), and env 1 seat 1 no longer produces chunks at all.
    assert all(player_index(c) == {1} for c in b_after)
    assert [step_index(c, EPISODE) for c in b_after] == [
        [0, 1, 2, 3], [4, 0, 1, 2], [3, 4, 0, 1], [2, 3, 4, 0], [1, 2, 3, 4],
    ]


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
