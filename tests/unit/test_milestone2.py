"""Milestone 2 integration test: self-play + checkpoint management.

Tests:
1. CheckpointManager save/load/FIFO
2. SelfPlayMatchmaker generates correct match configs
3. Full pipeline with checkpoint saving
"""
import os
import tempfile

import torch

from helpers import example_config


def test_checkpoint_manager():
    """Test CheckpointManager: save, load, list, FIFO eviction."""
    from colosseum.coordinator.checkpoint_manager import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(base_dir=tmpdir, pool_size=3, save_optimizer=True)

        # Save 5 checkpoints (pool_size=3, so oldest 2 should be evicted)
        for v in [10, 20, 30, 40, 50]:
            state = {"weight": torch.randn(4, 4)}
            opt_state = {"step": v}
            mgr.save("agent_0", v, state, opt_state, metrics={"reward": float(v)})

        # Should have exactly 3 checkpoints (30, 40, 50)
        ckpts = mgr.list_checkpoints("agent_0")
        assert len(ckpts) == 3, f"Expected 3 checkpoints, got {len(ckpts)}"
        versions = [c.policy_version for c in ckpts]
        assert versions == [30, 40, 50], f"Expected versions [30,40,50], got {versions}"

        # Load latest
        latest = mgr.get_latest("agent_0")
        assert latest is not None
        assert latest.policy_version == 50

        # Load specific
        sd = mgr.load("agent_0", "ckpt_v40")
        assert "weight" in sd

        # Load optimizer
        opt = mgr.load_optimizer("agent_0", "ckpt_v40")
        assert opt is not None
        assert opt["step"] == 40

        # Check evicted checkpoints are gone from disk
        assert not os.path.exists(os.path.join(tmpdir, "agent_0", "ckpt_v10"))
        assert not os.path.exists(os.path.join(tmpdir, "agent_0", "ckpt_v20"))

    print("PASS: test_checkpoint_manager")


def test_matchmaker():
    """Test SimpleSelfPlayMatchmaker and SelfPlayMatchmaker."""
    from colosseum.coordinator.checkpoint_manager import CheckpointManager
    from colosseum.coordinator.matchmaker import SelfPlayMatchmaker, SimpleSelfPlayMatchmaker

    # SimpleSelfPlayMatchmaker: all slots collect, no checkpoints
    simple = SimpleSelfPlayMatchmaker()
    configs = simple.generate_matches("agent_0", num_envs=4, num_players=2)
    assert len(configs) == 4
    for mc in configs:
        assert len(mc.player_slots) == 2
        for slot in mc.player_slots:
            assert slot.agent_id == "agent_0"
            assert slot.checkpoint_id is None
            assert slot.collect_trajectories is True

    # SelfPlayMatchmaker: slot 0 always latest+collecting
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(base_dir=tmpdir, pool_size=5)
        # Save some checkpoints
        for v in [10, 20, 30]:
            ckpt_mgr.save("agent_0", v, {"w": torch.randn(2, 2)})

        sp = SelfPlayMatchmaker(ckpt_mgr, latest_prob=0.0)  # always use checkpoint
        configs = sp.generate_matches("agent_0", num_envs=10, num_players=2)
        assert len(configs) == 10

        has_checkpoint_slot = False
        for mc in configs:
            # Slot 0: always latest, always collects
            assert mc.player_slots[0].checkpoint_id is None
            assert mc.player_slots[0].collect_trajectories is True
            # Slot 1: should be checkpoint (latest_prob=0.0)
            if mc.player_slots[1].checkpoint_id is not None:
                has_checkpoint_slot = True
                assert mc.player_slots[1].collect_trajectories is False

        assert has_checkpoint_slot, "Expected at least one checkpoint opponent"

    print("PASS: test_matchmaker")


def test_coordinator():
    """Test Coordinator integrates agent pool, checkpoints, matchmaking."""
    from colosseum.coordinator.coordinator import Coordinator
    from colosseum.core.config import ColosseumConfig, load_config

    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config(example_config("tic_tac_toe.yaml"))
        cd = config.model_dump()
        cd["checkpoint"]["dir"] = tmpdir
        cd["self_play"]["checkpoint_interval"] = 5
        cd["self_play"]["pool_size"] = 3
        config = ColosseumConfig(**cd)

        coord = Coordinator(config)
        coord.agent_pool.register_trainable("agent_0")

        # Initially no checkpoints → SimpleSelfPlayMatchmaker
        configs = coord.generate_match_configs("agent_0", num_envs=4)
        assert len(configs) == 4
        for mc in configs:
            assert all(s.collect_trajectories for s in mc.player_slots)

        # Save a checkpoint at version 5
        ckpt_id = coord.maybe_save_checkpoint(
            "agent_0", 5, {"w": torch.randn(2, 2)},
        )
        assert ckpt_id == "ckpt_v5"

        # Now should use SelfPlayMatchmaker
        configs = coord.generate_match_configs("agent_0", num_envs=4)
        assert len(configs) == 4
        # With default latest_prob=0.5, some slots may use checkpoints

    print("PASS: test_coordinator")


def test_derive_worker_configs():
    """Test _derive_worker_configs extracts collect_mask, checkpoint and slot_agent_map."""
    from colosseum.coordinator.coordinator import Coordinator
    from colosseum.core.config import ColosseumConfig, load_config
    from colosseum.core.types import MatchConfig, PlayerSlot
    from colosseum.launcher import _derive_worker_configs

    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config(example_config("tic_tac_toe.yaml"))
        cd = config.model_dump()
        cd["checkpoint"]["dir"] = tmpdir
        config = ColosseumConfig(**cd)

        coord = Coordinator(config)

        # Save a checkpoint
        state = {"w": torch.randn(2, 2)}
        coord.checkpoint_manager.save("agent_0", 10, state)

        # Create match configs with a checkpoint slot
        match_configs = [
            MatchConfig(
                match_id="test_1",
                env_config={},
                player_slots=[
                    PlayerSlot(agent_id="agent_0", checkpoint_id=None, collect_trajectories=True),
                    PlayerSlot(agent_id="agent_0", checkpoint_id="ckpt_v10", collect_trajectories=False),
                ],
            ),
            MatchConfig(
                match_id="test_2",
                env_config={},
                player_slots=[
                    PlayerSlot(agent_id="agent_0", checkpoint_id=None, collect_trajectories=True),
                    PlayerSlot(agent_id="agent_0", checkpoint_id=None, collect_trajectories=True),
                ],
            ),
        ]

        ckpt_by_agent, slot_map, collect_mask, slot_agent_map = _derive_worker_configs(
            match_configs, coord, ["agent_0"],
        )

        # Should have loaded checkpoint weights
        assert "agent_0" in ckpt_by_agent
        assert "ckpt_v10" in ckpt_by_agent["agent_0"]
        assert "w" in ckpt_by_agent["agent_0"]["ckpt_v10"]

        # slot_network_map should map slots to the right networks
        assert slot_map[0][0] == "latest"
        assert slot_map[0][1] == "ckpt_v10"
        assert slot_map[1][0] == "latest"
        assert slot_map[1][1] == "latest"

        # collect_mask should reflect the slots
        assert collect_mask == [[True, False], [True, True]]

        # slot_agent_map should be all agent_0
        assert slot_agent_map == [["agent_0", "agent_0"], ["agent_0", "agent_0"]]

    print("PASS: test_derive_worker_configs")
