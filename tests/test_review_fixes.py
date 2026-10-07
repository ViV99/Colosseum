"""Regression tests for fixes from the critical review (КРИТИЧЕСКИЙ_ОБЗОР.md).

Covers:
  C4  — coordinator aggregates composite "agent:net" outcome keys to the base
        agent_id so ELO / win-rate / PFSP actually see cross-agent results.
  C5  — recurrent BPTT resets the hidden state at episode boundaries within a
        chunk (post-done timesteps must not depend on pre-done inputs).
  C9  — action_masks survive chunk serialization (gRPC path).
  C14 — match outcomes prefer the env's authoritative rank/outcome signal.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from colosseum.networks.actor_critic import ActorCriticNetwork
from tests.helpers import SimpleEncoder, SimplePolicy, SimpleValue


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
# C4: coordinator key aggregation + PFSP using real win rates
# ---------------------------------------------------------------------------

def _make_cfg(tmpdir, phase="league", latest_prob=0.5, envs_per_worker=8):
    from colosseum.core.config import (
        CheckpointConfig, ColosseumConfig, EnvConfig, NetworkConfig, RolloutConfig,
        SelfPlayConfig, TrainingConfig,
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
        checkpoint=CheckpointConfig(dir=tmpdir),
    )


def _make_coordinator(tmpdir, phase="league"):
    from colosseum.coordinator.coordinator import Coordinator

    return Coordinator(_make_cfg(tmpdir, phase=phase))


def test_coordinator_aggregates_composite_keys():
    from colosseum.core.types import MatchResult

    with tempfile.TemporaryDirectory() as tmp:
        coord = _make_coordinator(tmp)
        coord.agent_pool.register_trainable("agent_alpha")
        coord.agent_pool.register_trainable("agent_beta")

        # Worker reports outcomes keyed by "agent:network_id".
        for _ in range(5):
            coord.report_match_result(MatchResult(
                match_id="m",
                player_outcomes={"agent_alpha:latest": 1.0, "agent_beta:latest": 0.0},
                total_rewards={"agent_alpha:latest": 1.0, "agent_beta:latest": -1.0},
            ))

        # Win rate must be queryable by BASE agent_id (what PFSP uses), not 0.5 default.
        wr = coord.win_rates.get_win_rate("agent_alpha", "agent_beta")
        assert wr == 1.0, f"expected 1.0, got {wr}"
        assert coord.elo.get("agent_alpha") > coord.elo.get("agent_beta")


def test_coordinator_skips_same_agent_pairs():
    from colosseum.core.types import MatchResult

    with tempfile.TemporaryDirectory() as tmp:
        coord = _make_coordinator(tmp)
        coord.agent_pool.register_trainable("agent_alpha")

        # Solo self-play: latest vs a historical checkpoint of the SAME agent.
        coord.report_match_result(MatchResult(
            match_id="m",
            player_outcomes={"agent_alpha:latest": 1.0, "agent_alpha:ckpt_v1": 0.0},
        ))
        # No cross-agent signal should be recorded.
        assert coord.win_rates.get_win_rate("agent_alpha", "agent_alpha") == 0.5


def test_pfsp_uses_win_rates():
    """PFSP should prioritize the harder opponent (lower win rate)."""
    from colosseum.core.types import MatchResult

    with tempfile.TemporaryDirectory() as tmp:
        coord = _make_coordinator(tmp)
        for aid in ("alpha", "beta", "gamma"):
            coord.agent_pool.register_trainable(aid)

        # alpha crushes beta but loses to gamma.
        for _ in range(20):
            coord.report_match_result(MatchResult(
                match_id="m1",
                player_outcomes={"alpha:latest": 1.0, "beta:latest": 0.0},
            ))
            coord.report_match_result(MatchResult(
                match_id="m2",
                player_outcomes={"alpha:latest": 0.0, "gamma:latest": 1.0},
            ))

        coord.setup_matchmaker("alpha")
        mm = coord._matchmaker
        candidates = [a for a in coord.agent_pool.list_trainable() if a.agent_id != "alpha"]

        import collections
        picks = collections.Counter()
        for _ in range(400):
            picks[mm._select_opponent("alpha", candidates).agent_id] += 1

        # gamma (win rate 0 → priority 1) should be picked far more than beta
        # (win rate 1 → priority ~0).
        assert picks["gamma"] > picks["beta"] * 3, picks


# ---------------------------------------------------------------------------
# C5: recurrent hidden-state reset at episode boundaries
# ---------------------------------------------------------------------------

def _make_recurrent_net(obs_dim=4, hidden=8, num_actions=3):
    enc = SimpleEncoder(obs_dim=obs_dim, hidden_dim=hidden)
    pol = SimplePolicy(hidden_dim=hidden, num_actions=num_actions)
    val = SimpleValue(hidden_dim=hidden)
    rnn = nn.LSTM(hidden, hidden, 1, batch_first=False)
    net = ActorCriticNetwork(enc, pol, val, recurrent=rnn)
    net.eval()
    return net


def test_recurrent_reset_isolates_post_done_steps():
    """With a done at t=1, outputs at t>=2 must NOT depend on inputs at t<=1."""
    torch.manual_seed(0)
    net = _make_recurrent_net()
    T, B = 4, 1

    obs_a = torch.randn(T, B, 4)
    obs_b = obs_a.clone()
    obs_b[0] = torch.randn(B, 4)  # differ before the boundary
    obs_b[1] = torch.randn(B, 4)  # differ at the boundary
    actions = torch.zeros(T, B, dtype=torch.long)
    dones = torch.zeros(T, B)
    dones[1, 0] = 1.0  # episode ends at t=1

    h0 = net.initial_hidden(B)

    _, va, _ = net.evaluate_actions_recurrent(obs_a, actions, h0, dones_seq=dones)
    _, vb, _ = net.evaluate_actions_recurrent(obs_b, actions, h0, dones_seq=dones)

    # After the reset, t>=2 depend only on obs[2:], which are identical.
    assert torch.allclose(va[2:], vb[2:], atol=1e-6), "hidden state leaked across episode boundary"
    # t<=1 differ (different inputs).
    assert not torch.allclose(va[:2], vb[:2], atol=1e-6)


def test_recurrent_without_reset_carries_state():
    """Sanity: without dones, state carries over → post-boundary outputs differ."""
    torch.manual_seed(0)
    net = _make_recurrent_net()
    T, B = 4, 1

    obs_a = torch.randn(T, B, 4)
    obs_b = obs_a.clone()
    obs_b[0] = torch.randn(B, 4)
    obs_b[1] = torch.randn(B, 4)
    actions = torch.zeros(T, B, dtype=torch.long)
    h0 = net.initial_hidden(B)

    _, va, _ = net.evaluate_actions_recurrent(obs_a, actions, h0, dones_seq=None)
    _, vb, _ = net.evaluate_actions_recurrent(obs_b, actions, h0, dones_seq=None)
    assert not torch.allclose(va[2:], vb[2:], atol=1e-6)


# ---------------------------------------------------------------------------
# C9: action_masks survive serialization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# C1: runtime matchmaking refresh closes the loop
# ---------------------------------------------------------------------------

def test_refresh_pushes_new_checkpoint_to_worker():
    """Main process should push freshly-saved checkpoints to workers (delta only)."""
    import multiprocessing as mp

    from colosseum.core.registry import build_network
    from colosseum.core.types import WorkerCommand
    from colosseum.coordinator.coordinator import Coordinator
    from colosseum.launcher import Launcher

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(tmp, phase="self_play", latest_prob=0.0, envs_per_worker=4)
        coord = Coordinator(cfg)
        coord.agent_pool.register_trainable("agent_0")

        # Save a checkpoint, then refresh the matchmaker so it uses it.
        sd = {k: v.cpu() for k, v in build_network(cfg).state_dict().items()}
        coord.checkpoint_manager.save("agent_0", 50, sd)
        coord.setup_matchmaker("agent_0")

        launcher = Launcher(cfg)
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
    """Worker side: a WorkerCommand loads new checkpoints and stages slot maps."""
    from colosseum.core.types import WorkerCommand
    from colosseum.worker.rollout_worker import LATEST_NETWORK_ID, _apply_command
    from tests.helpers import make_simple_network

    def factory():
        return make_simple_network(obs_dim=4, num_actions=3)

    networks_by_agent = {"agent_0": {LATEST_NETWORK_ID: factory()}}
    factories = {"agent_0": factory}
    ckpt_sd = factory().state_dict()

    cmd = WorkerCommand(
        slot_agent_map=[["agent_0", "agent_0"]],
        slot_network_map=[["latest", "ckpt_v50"]],
        collect_mask=[[True, False]],
        new_checkpoints={"agent_0": {"ckpt_v50": ckpt_sd}},
    )
    pending = {"slot_agent_map": None, "slot_network_map": None, "collect_mask": None}

    _apply_command(cmd, networks_by_agent, factories, pending)

    assert "ckpt_v50" in networks_by_agent["agent_0"], "checkpoint not loaded into pool"
    assert pending["slot_agent_map"] == [["agent_0", "agent_0"]]
    assert pending["collect_mask"] == [[True, False]]


# ---------------------------------------------------------------------------
# C11: resume_from is honored
# ---------------------------------------------------------------------------

def test_resume_from_path(tmp_path):
    from colosseum.core.registry import build_network
    from colosseum.coordinator.coordinator import Coordinator
    from colosseum.launcher import _resolve_resume_state

    cfg = _make_cfg(str(tmp_path), phase="self_play")
    weights_path = str(tmp_path / "bc_weights.pt")
    sd = build_network(cfg).state_dict()
    torch.save(sd, weights_path)

    cfg.training.resume_from = weights_path
    coord = Coordinator(cfg)
    state = _resolve_resume_state(cfg, "agent_0", coord)
    assert state is not None
    assert set(state["state_dict"].keys()) == set(sd.keys())


def test_resume_from_none_returns_none(tmp_path):
    from colosseum.coordinator.coordinator import Coordinator
    from colosseum.launcher import _resolve_resume_state

    cfg = _make_cfg(str(tmp_path), phase="self_play")
    coord = Coordinator(cfg)
    assert _resolve_resume_state(cfg, "agent_0", coord) is None


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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
