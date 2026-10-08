"""Regression tests for fixes from the critical review (КРИТИЧЕСКИЙ_ОБЗОР.md).

Covers:
  C4  — PFSP sees real cross-agent win rates from reported match results.
  C9  — action_masks survive chunk serialization (gRPC path).
  C14 — match outcomes prefer the env's authoritative rank/outcome signal.
"""
import tempfile
from pathlib import Path

import torch

from dataflow_helpers import two_seat_result
from helpers import make_test_run_dir

# ---------------------------------------------------------------------------
# C14: outcome derivation
# ---------------------------------------------------------------------------

def test_outcomes_from_rewards():
    from colosseum.core.outcomes import player_outcomes

    assert player_outcomes([0.0, 5.0]) == [0.0, 1.0]
    assert player_outcomes([3.0, 3.0]) == [0.5, 0.5]
    assert player_outcomes([1.0, -1.0, -1.0]) == [1.0, 0.0, 0.0]


def test_outcomes_prefer_env_rank_over_reward():
    from colosseum.core.outcomes import player_outcomes

    # Player 1 accumulated more shaped reward, but the env says player 0 ranked best.
    out = player_outcomes([1.0, 9.0], {0: {"rank": 1}, 1: {"rank": 2}}, num_players=2)
    assert out[0] > out[1]
    assert out == [1.0, 0.0]


def test_outcomes_prefer_env_explicit_outcome():
    from colosseum.core.outcomes import player_outcomes

    out = player_outcomes([9.0, 0.0], {0: {"outcome": 0.0}, 1: {"outcome": 1.0}}, num_players=2)
    assert out == [0.0, 1.0]


def test_outcomes_fall_back_when_signal_incomplete():
    from colosseum.core.outcomes import player_outcomes

    # Only one player has a rank → not authoritative → fall back to reward.
    out = player_outcomes([5.0, 0.0], {0: {"rank": 1}, 1: {}}, num_players=2)
    assert out == [1.0, 0.0]


# ---------------------------------------------------------------------------
# C4: PFSP using real win rates
# ---------------------------------------------------------------------------

def _make_cfg(phase="league", latest_prob=0.5, envs_per_worker=8):
    from colosseum.core.config import (
        ColosseumConfig,
        EnvConfig,
        NetworkConfig,
        RolloutConfig,
        SelfPlayConfig,
        TrainingConfig,
    )

    return ColosseumConfig(
        env=EnvConfig(env_class="examples.tic_tac_toe.env.TicTacToeEnv", num_players=2),
        networks=NetworkConfig(
            encoder_class="examples.tic_tac_toe.networks.TicTacToeEncoder",
            policy_class="examples.tic_tac_toe.networks.TicTacToePolicy",
            value_class="examples.tic_tac_toe.networks.TicTacToeValue",
        ),
        rollout=RolloutConfig(envs_per_worker=envs_per_worker),
        training=TrainingConfig(phase=phase),
        self_play=SelfPlayConfig(latest_prob=latest_prob),
    )


def _make_coordinator(tmpdir, phase="league"):
    from colosseum.coordinator.coordinator import Coordinator

    return Coordinator(_make_cfg(phase=phase), checkpoint_dir=tmpdir)


def test_pfsp_uses_win_rates():
    """PFSP should prioritize the harder opponent (lower win rate)."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _make_coordinator(tmp)
        for aid in ("alpha", "beta", "gamma"):
            coord.agent_pool.register_trainable(aid)

        # alpha crushes beta but loses to gamma.
        for _ in range(20):
            coord.report_match_result(two_seat_result("alpha", 1.0, "beta", 0.0, match_id="m1"))
            coord.report_match_result(two_seat_result("alpha", 0.0, "gamma", 1.0, match_id="m2"))

        import collections

        from colosseum.coordinator.matchmaker import PFSPMatchmaker

        assert isinstance(coord._matchmaker, PFSPMatchmaker)
        picks = collections.Counter(
            coord._matchmaker.select_opponents("alpha", ["beta", "gamma"], k=400))

        # gamma (win rate 0 → priority 1) should be picked far more than beta
        # (win rate 1 → priority ~0).
        assert picks["gamma"] > picks["beta"] * 3, picks


# ---------------------------------------------------------------------------
# C9: action_masks survive serialization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# C1: runtime matchmaking refresh closes the loop
# ---------------------------------------------------------------------------

def test_refresh_pushes_new_checkpoint_to_worker():
    """Main process should push freshly-saved checkpoints to workers (delta only)."""
    import multiprocessing as mp

    from colosseum.coordinator.coordinator import Coordinator
    from colosseum.core.registry import build_model
    from colosseum.core.types import WorkerCommand
    from colosseum.launcher import Launcher

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(phase="self_play", latest_prob=0.0, envs_per_worker=4)
        coord = Coordinator(cfg, checkpoint_dir=tmp)
        coord.agent_pool.register_trainable("agent_0")

        # Save a checkpoint; the matchmaker lists checkpoints on every match.
        sd = {k: v.detach().cpu().numpy() for k, v in build_model(cfg).state_dict().items()}
        coord.checkpoint_manager.save("agent_0", 50, sd)

        launcher = Launcher(cfg, make_test_run_dir(cfg, Path(tmp)))
        cq = mp.Queue(maxsize=4)
        broadcast = [{"agent_0": set()}]

        launcher._refresh_worker_matches(coord, ["agent_0"], [cq], broadcast)
        cmd = cq.get(timeout=5)
        assert isinstance(cmd, WorkerCommand)
        assert "ckpt_v50" in cmd.new_checkpoints.get("agent_0", {}), \
            "freshly-saved checkpoint was not shipped to the worker"
        # The slot maps should reference the checkpoint as an opponent.
        assert any("ckpt_v50" in row for row in cmd.slot_network_map)

        # A second refresh must NOT resend the same checkpoint (delta only).
        launcher._refresh_worker_matches(coord, ["agent_0"], [cq], broadcast)
        cmd2 = cq.get(timeout=5)
        assert "ckpt_v50" not in cmd2.new_checkpoints.get("agent_0", {})


def test_worker_applies_command_loads_checkpoint_and_stages_maps():
    """Worker side: a WorkerCommand loads new checkpoints and stages slot maps.

    Drives the real RolloutLoop (T2.6 folded ``_apply_command`` into
    ``RolloutLoop._poll_command``): the checkpoint enters the model pool at once,
    the slot maps only at the env's next episode boundary.
    """
    from colosseum.core.types import WorkerCommand, state_dict_to_numpy
    from colosseum.worker.rollout_loop import LATEST_NETWORK_ID
    from dataflow_helpers import EnvFactory, GridStepEnv, make_loop
    from helpers import make_simple_model

    def factory():
        return make_simple_model(obs_dim=4, num_actions=3)

    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(3,)), factory,
                          agent_ids=("agent_0",), num_envs=1)
    ckpt_sd = state_dict_to_numpy(factory().state_dict())
    col.commands.append(WorkerCommand(
        slot_agent_map=[["agent_0", "agent_0"]],
        slot_network_map=[["latest", "ckpt_v50"]],
        collect_mask=[[True, False]],
        new_checkpoints={"agent_0": {"ckpt_v50": ckpt_sd}},
    ))
    loop.step()

    assert "ckpt_v50" in loop._models["agent_0"], "checkpoint not loaded into pool"
    # Staged, not applied mid-episode.
    assert loop._slot_network_map == [[LATEST_NETWORK_ID, LATEST_NETWORK_ID]]
    assert loop._collect_mask == [[True, True]]
    loop.step()
    loop.step()  # episode end -> assignment applied
    assert loop._slot_network_map == [[LATEST_NETWORK_ID, "ckpt_v50"]]
    assert loop._collect_mask == [[True, False]]
    loop.close()


# ---------------------------------------------------------------------------
# C9: action_masks survive serialization
# ---------------------------------------------------------------------------

def test_chunk_serialization_preserves_action_masks():
    from colosseum.core.types import TrajectoryChunk
    from colosseum.transport.serialization import deserialize_chunk, serialize_chunk

    T, A = 6, 4
    masks = torch.zeros(T, A, dtype=torch.bool)
    masks[:, 0] = True
    masks[:, 2] = True
    chunk = TrajectoryChunk(
        agent_id="agent_0",
        observations=torch.randn(T, 8),
        actions=torch.randint(0, A, (T,)),
        action_log_probs=torch.randn(T),
        rewards=torch.randn(T),
        dones=torch.zeros(T),
        values=torch.randn(T),
        bootstrap_value=torch.tensor(0.0),
        behavior_policy_version=3,
        action_masks=masks,
    )

    data, compressed = serialize_chunk(chunk)
    restored = deserialize_chunk("agent_0", 3, data, compressed)

    assert restored.action_masks is not None, "action_masks dropped during serialization"
    assert torch.equal(restored.action_masks, masks)
