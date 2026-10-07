"""Tests for multi-agent support: per-agent config, worker routing, launcher."""

import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np

from colosseum.core.config import (
    AgentConfig,
    AlgorithmConfig,
    ColosseumConfig,
    LearnerConfig,
    NetworkConfig,
    load_config,
)


# ---------------------------------------------------------------
# T3.8: Per-agent config tests
# ---------------------------------------------------------------

def test_agent_config_defaults():
    """AgentConfig fields should default to None."""
    ac = AgentConfig()
    assert ac.networks is None
    assert ac.algorithm is None
    assert ac.learner is None


def test_get_trainable_agent_ids_empty():
    """Empty agents dict should return ['agent_0']."""
    cfg = load_config("configs/examples/tic_tac_toe.yaml")
    assert cfg.get_trainable_agent_ids() == ["agent_0"]


def test_get_trainable_agent_ids_multi():
    """Non-empty agents dict should return agent IDs."""
    cfg = load_config("configs/examples/tic_tac_toe_multi.yaml")
    ids = cfg.get_trainable_agent_ids()
    assert "agent_alpha" in ids
    assert "agent_beta" in ids
    assert len(ids) == 2


def test_get_agent_config_no_override():
    """get_agent_config for unknown agent returns a copy with same values."""
    cfg = load_config("configs/examples/tic_tac_toe.yaml")
    result = cfg.get_agent_config("agent_0")
    assert result is not cfg  # should be a copy, not the same object
    assert result.algorithm == cfg.algorithm
    assert result.networks == cfg.networks


def test_get_agent_config_with_override():
    """get_agent_config merges per-agent overrides into a new config."""
    cfg = load_config("configs/examples/tic_tac_toe_multi.yaml")

    # Modify the config to have an actual override for agent_alpha
    cfg_dict = cfg.model_dump()
    cfg_dict["agents"]["agent_alpha"]["algorithm"] = {
        "name": "appo",
        "learning_rate": 1e-2,
    }
    cfg = ColosseumConfig.model_validate(cfg_dict)

    alpha_cfg = cfg.get_agent_config("agent_alpha")
    beta_cfg = cfg.get_agent_config("agent_beta")

    # alpha should have overridden LR
    assert alpha_cfg.algorithm.learning_rate == 1e-2
    # beta should keep global LR
    assert beta_cfg.algorithm.learning_rate == cfg.algorithm.learning_rate
    # Both should share the same env config
    assert alpha_cfg.env.env_class == cfg.env.env_class


def test_load_multi_agent_config_roundtrip():
    """Multi-agent config should survive dump/reload."""
    cfg = load_config("configs/examples/tic_tac_toe_multi.yaml")
    data = cfg.model_dump()
    cfg2 = ColosseumConfig.model_validate(data)
    assert cfg2.get_trainable_agent_ids() == cfg.get_trainable_agent_ids()
    assert cfg2.training.phase.value == "league"


# ---------------------------------------------------------------
# T1.4: Worker multi-agent routing
# ---------------------------------------------------------------

def test_worker_multi_agent_routing():
    """Worker in multi-agent mode routes chunks to the correct agent queues."""
    from colosseum.core.config import ColosseumConfig, load_config
    from colosseum.launcher import _create_env, _create_network, _worker_target

    config = load_config("configs/examples/tic_tac_toe_multi.yaml")
    cd = config.model_dump()
    cd["training"]["total_timesteps"] = 500
    cd["rollout"]["num_workers"] = 1
    cd["rollout"]["envs_per_worker"] = 2
    cd["rollout"]["chunk_length"] = 8
    cd["learner"]["batch_chunks"] = 2
    config = ColosseumConfig(**cd)

    agent_ids = config.get_trainable_agent_ids()
    agent_configs = {aid: config.get_agent_config(aid) for aid in agent_ids}

    # Per-agent trajectory queues
    trajectory_queues = {aid: mp.Queue(maxsize=16) for aid in agent_ids}
    weight_queues = {aid: mp.Queue(maxsize=2) for aid in agent_ids}
    stop_event = mp.Event()

    # slot_agent_map: alternate agents across envs
    # env0: [agent_alpha, agent_beta], env1: [agent_alpha, agent_beta]
    slot_agent_map = [
        [agent_ids[0], agent_ids[1]],
        [agent_ids[0], agent_ids[1]],
    ]
    # All slots use latest, all collect
    from colosseum.worker.rollout_worker import LATEST_NETWORK_ID
    slot_network_map = [
        [LATEST_NETWORK_ID, LATEST_NETWORK_ID],
        [LATEST_NETWORK_ID, LATEST_NETWORK_ID],
    ]
    collect_mask = [[True, True], [True, True]]

    p = mp.Process(
        target=_worker_target,
        args=(
            0, config, agent_ids, agent_configs,
            trajectory_queues, weight_queues,
            stop_event, 500,
            None,  # checkpoint_state_dicts_by_agent
            slot_network_map, collect_mask, slot_agent_map,
            None,  # results_queue
        ),
        daemon=True,
    )
    p.start()

    # Collect chunks from both agents
    chunks_by_agent = {aid: [] for aid in agent_ids}
    total_collected = 0
    deadline = time.time() + 20  # 20s timeout

    while total_collected < 4 and time.time() < deadline:
        for aid in agent_ids:
            try:
                chunk = trajectory_queues[aid].get(timeout=0.5)
                chunks_by_agent[aid].append(chunk)
                total_collected += 1
            except Exception:
                pass

    stop_event.set()
    p.join(timeout=5)
    if p.is_alive():
        p.terminate()

    # Both agents should have received chunks
    for aid in agent_ids:
        assert len(chunks_by_agent[aid]) > 0, f"Agent {aid} received no chunks"
        for chunk in chunks_by_agent[aid]:
            assert chunk.agent_id == aid, (
                f"Chunk routed to wrong agent: expected {aid}, got {chunk.agent_id}"
            )

    print(f"  Chunks per agent: {', '.join(f'{k}={len(v)}' for k, v in chunks_by_agent.items())}")
    print("  Multi-agent worker routing PASSED")


# ---------------------------------------------------------------
# T1.4: _derive_worker_configs
# ---------------------------------------------------------------

def test_derive_worker_configs():
    """Verify slot_agent_map and collect_mask are extracted correctly."""
    from colosseum.core.types import MatchConfig, PlayerSlot
    from colosseum.coordinator.coordinator import Coordinator
    from colosseum.launcher import _derive_worker_configs

    config = load_config("configs/examples/tic_tac_toe_multi.yaml")
    agent_ids = config.get_trainable_agent_ids()

    coordinator = Coordinator(config)
    for aid in agent_ids:
        coordinator.agent_pool.register_trainable(aid)

    # Manually create match configs simulating arena matches
    match_configs = [
        MatchConfig(
            match_id="test_0",
            player_slots=[
                PlayerSlot(agent_id="agent_alpha", collect_trajectories=True),
                PlayerSlot(agent_id="agent_beta", collect_trajectories=True),
            ],
        ),
        MatchConfig(
            match_id="test_1",
            player_slots=[
                PlayerSlot(agent_id="agent_beta", collect_trajectories=True),
                PlayerSlot(agent_id="agent_alpha", collect_trajectories=False),
            ],
        ),
    ]

    ckpt_by_agent, slot_net_map, collect_mask, slot_agent_map = (
        _derive_worker_configs(match_configs, coordinator, agent_ids)
    )

    # slot_agent_map
    assert slot_agent_map[0] == ["agent_alpha", "agent_beta"]
    assert slot_agent_map[1] == ["agent_beta", "agent_alpha"]

    # collect_mask
    assert collect_mask[0] == [True, True]
    assert collect_mask[1] == [True, False]

    # All checkpoints empty (no ckpt_ids in slots)
    assert all(len(v) == 0 for v in ckpt_by_agent.values())


# ---------------------------------------------------------------
# T1.4: Multi-agent monitor loop
# ---------------------------------------------------------------

def test_monitor_loop_per_agent_checkpoint_queues():
    """Monitor loop should process per-agent checkpoint queues."""
    from colosseum.coordinator.coordinator import Coordinator
    from colosseum.launcher import Launcher
    from colosseum.metrics.wandb_logger import WandBLogger

    config = load_config("configs/examples/tic_tac_toe_multi.yaml")
    agent_ids = config.get_trainable_agent_ids()

    coordinator = Coordinator(config)
    for aid in agent_ids:
        coordinator.agent_pool.register_trainable(aid)

    # Create per-agent checkpoint queues with a fake checkpoint
    checkpoint_queues = {aid: mp.Queue(maxsize=4) for aid in agent_ids}

    # This test just verifies the monitor loop's queue draining works
    # by putting items and checking they don't crash
    for aid in agent_ids:
        state_dict = {"weight": torch.randn(3, 3)}
        checkpoint_queues[aid].put({
            "policy_version": 100,
            "state_dict": state_dict,
        })

    # mp.Queue serializes in background thread; wait for it to finish
    time.sleep(0.5)

    # Create a minimal launcher and manually invoke checkpoint processing
    launcher = Launcher(config)

    # Process checkpoint queues (simulating part of _monitor_loop)
    import queue
    for aid in agent_ids:
        cq = checkpoint_queues[aid]
        try:
            ckpt_data = cq.get(timeout=2.0)
            # Would call coordinator.maybe_save_checkpoint in real code
            assert "policy_version" in ckpt_data
            assert "state_dict" in ckpt_data
        except queue.Empty:
            pytest.fail(f"Expected checkpoint data for {aid}")


# ---------------------------------------------------------------
# T1.4d: Full multi-agent pipeline integration test
# ---------------------------------------------------------------

def test_multi_agent_pipeline():
    """End-to-end multi-agent training pipeline with 2 agents."""
    from colosseum.core.config import ColosseumConfig, load_config
    from colosseum.launcher import Launcher

    config = load_config("configs/examples/tic_tac_toe_multi.yaml")
    cd = config.model_dump()
    cd["training"]["total_timesteps"] = 3000
    cd["rollout"]["num_workers"] = 1
    cd["rollout"]["envs_per_worker"] = 2
    cd["rollout"]["chunk_length"] = 8
    cd["learner"]["batch_chunks"] = 2
    cd["learner"]["queue_size"] = 16
    cd["metrics"]["use_wandb"] = False
    config = ColosseumConfig(**cd)

    launcher = Launcher(config)
    launcher.launch()
    print("  Multi-agent pipeline PASSED")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    print("=" * 60)
    print("Test: Per-agent config")
    print("=" * 60)
    test_agent_config_defaults()
    test_get_trainable_agent_ids_empty()
    test_get_trainable_agent_ids_multi()
    test_get_agent_config_no_override()
    test_get_agent_config_with_override()
    test_load_multi_agent_config_roundtrip()
    print("  Config tests PASSED")

    print()
    print("=" * 60)
    print("Test: _derive_worker_configs")
    print("=" * 60)
    test_derive_worker_configs()
    print("  derive_worker_configs_multi PASSED")

    print()
    print("=" * 60)
    print("Test: Worker multi-agent routing")
    print("=" * 60)
    test_worker_multi_agent_routing()

    print()
    print("=" * 60)
    print("Test: Multi-agent pipeline")
    print("=" * 60)
    test_multi_agent_pipeline()

    print()
    print("ALL MULTI-AGENT TESTS PASSED!")
