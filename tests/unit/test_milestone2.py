"""Milestone 2 integration test: self-play + checkpoint management.

Tests:
1. CheckpointManager save/load/FIFO
2. _derive_worker_configs builds worker slot maps from match configs
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
