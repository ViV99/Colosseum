"""Tests for RNN/LSTM support (T1.2)."""



import numpy as np
import torch
import torch.nn as nn

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.types import TrajectoryChunk
from colosseum.networks.actor_critic import ActorCriticNetwork
from helpers import SimpleEncoder, SimplePolicy, SimpleValue

OBS_DIM = 8
HIDDEN_SIZE = 16
NUM_ACTIONS = 4
NUM_LAYERS = 1


def _make_recurrent_net(rnn_type="lstm"):
    encoder = SimpleEncoder(OBS_DIM, HIDDEN_SIZE)
    policy = SimplePolicy(HIDDEN_SIZE, NUM_ACTIONS)
    value = SimpleValue(HIDDEN_SIZE)
    if rnn_type == "lstm":
        recurrent = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=False)
    else:
        recurrent = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=False)
    return ActorCriticNetwork(encoder, policy, value, recurrent=recurrent)


def _make_feedforward_net():
    return ActorCriticNetwork(
        SimpleEncoder(OBS_DIM, HIDDEN_SIZE),
        SimplePolicy(HIDDEN_SIZE, NUM_ACTIONS),
        SimpleValue(HIDDEN_SIZE),
    )


# ---------------------------------------------------------------
# ActorCriticNetwork recurrent tests
# ---------------------------------------------------------------


def test_is_recurrent_property():
    """is_recurrent should return True for LSTM/GRU, False for feedforward."""
    assert _make_recurrent_net("lstm").is_recurrent is True
    assert _make_recurrent_net("gru").is_recurrent is True
    assert _make_feedforward_net().is_recurrent is False


def test_initial_hidden_shape():
    """initial_hidden should return (h, c) with correct shapes."""
    net = _make_recurrent_net("lstm")
    batch_size = 5
    h, c = net.initial_hidden(batch_size)
    assert h.shape == (NUM_LAYERS, batch_size, HIDDEN_SIZE)
    assert c.shape == (NUM_LAYERS, batch_size, HIDDEN_SIZE)
    assert (h == 0).all()
    assert (c == 0).all()


def test_recurrent_act_returns_4tuple_with_hidden():
    """act() on a recurrent net should return new_hidden as 4th element."""
    net = _make_recurrent_net("lstm")
    net.eval()
    obs = torch.randn(3, OBS_DIM)
    h0, c0 = net.initial_hidden(3)

    actions, log_probs, values, new_hidden = net.act(obs, hidden=(h0, c0))
    assert actions.shape == (3,)
    assert log_probs.shape == (3,)
    assert values.shape == (3,)
    assert new_hidden is not None
    assert new_hidden[0].shape == (NUM_LAYERS, 3, HIDDEN_SIZE)
    assert new_hidden[1].shape == (NUM_LAYERS, 3, HIDDEN_SIZE)


def test_recurrent_act_hidden_changes_with_input():
    """Hidden state should change after processing different inputs."""
    net = _make_recurrent_net("lstm")
    net.eval()
    h0, c0 = net.initial_hidden(1)

    obs1 = torch.randn(1, OBS_DIM)
    _, _, _, hidden1 = net.act(obs1, hidden=(h0, c0))

    obs2 = torch.randn(1, OBS_DIM)
    _, _, _, hidden2 = net.act(obs2, hidden=(h0, c0))

    # Different inputs should produce different hidden states
    assert not torch.allclose(hidden1[0], hidden2[0])


def test_feedforward_act_returns_none_hidden():
    """act() on a feedforward net should return None for hidden."""
    net = _make_feedforward_net()
    net.eval()
    obs = torch.randn(3, OBS_DIM)

    actions, log_probs, values, new_hidden = net.act(obs)
    assert new_hidden is None
    assert actions.shape == (3,)


def test_gru_act():
    """GRU variant should also return valid hidden state."""
    net = _make_recurrent_net("gru")
    net.eval()
    obs = torch.randn(2, OBS_DIM)
    h0, c0 = net.initial_hidden(2)

    actions, log_probs, values, new_hidden = net.act(obs, hidden=(h0, c0))
    assert new_hidden is not None
    assert new_hidden[0].shape == (NUM_LAYERS, 2, HIDDEN_SIZE)
    # GRU c should be zeros
    assert (new_hidden[1] == 0).all()


# ---------------------------------------------------------------
# evaluate_actions_recurrent tests
# ---------------------------------------------------------------


def test_evaluate_actions_recurrent_shapes():
    """evaluate_actions_recurrent should return [T, B] tensors."""
    net = _make_recurrent_net("lstm")
    T, B = 8, 4
    obs_seq = torch.randn(T, B, OBS_DIM)
    actions_seq = torch.randint(0, NUM_ACTIONS, (T, B))
    h0, c0 = net.initial_hidden(B)

    log_probs, values, entropy = net.evaluate_actions_recurrent(
        obs_seq, actions_seq, (h0, c0),
    )
    assert log_probs.shape == (T, B)
    assert values.shape == (T, B)
    assert entropy.shape == (T, B)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(entropy).all()


def test_evaluate_actions_recurrent_with_masks():
    """Recurrent evaluation should respect action masks."""
    net = _make_recurrent_net("lstm")
    T, B = 4, 2
    obs_seq = torch.randn(T, B, OBS_DIM)
    # Only actions 0 and 1 are valid
    actions_seq = torch.randint(0, 2, (T, B))
    masks = torch.zeros(T, B, NUM_ACTIONS, dtype=torch.bool)
    masks[:, :, 0] = True
    masks[:, :, 1] = True

    h0, c0 = net.initial_hidden(B)
    log_probs, values, entropy = net.evaluate_actions_recurrent(
        obs_seq, actions_seq, (h0, c0), action_mask_seq=masks,
    )
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()


def test_evaluate_actions_recurrent_gradients():
    """Recurrent evaluation should produce gradients for all params."""
    net = _make_recurrent_net("lstm")
    T, B = 4, 3
    obs_seq = torch.randn(T, B, OBS_DIM)
    actions_seq = torch.randint(0, NUM_ACTIONS, (T, B))
    h0, c0 = net.initial_hidden(B)

    log_probs, values, entropy = net.evaluate_actions_recurrent(
        obs_seq, actions_seq, (h0, c0),
    )
    loss = -log_probs.mean() + values.mean() - 0.01 * entropy.mean()
    loss.backward()

    # Check gradients exist for all parameter groups
    has_encoder_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in net.encoder.parameters()
    )
    has_rnn_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in net.recurrent.parameters()
    )
    has_policy_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in net.policy.parameters()
    )
    has_value_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in net.value.parameters()
    )
    assert has_encoder_grad, "Encoder should have gradients"
    assert has_rnn_grad, "RNN should have gradients"
    assert has_policy_grad, "Policy should have gradients"
    assert has_value_grad, "Value should have gradients"


# ---------------------------------------------------------------
# Stateless evaluate_actions backward compat
# ---------------------------------------------------------------


def test_feedforward_evaluate_actions_unchanged():
    """evaluate_actions on feedforward net should work exactly as before."""
    net = _make_feedforward_net()
    obs = torch.randn(10, OBS_DIM)
    actions = torch.randint(0, NUM_ACTIONS, (10,))

    log_probs, values, entropy = net.evaluate_actions(obs, actions)
    assert log_probs.shape == (10,)
    assert values.shape == (10,)
    assert entropy.shape == (10,)


# ---------------------------------------------------------------
# APPO recurrent training
# ---------------------------------------------------------------


def _make_recurrent_chunks(net, num_chunks=4, chunk_len=8):
    """Generate trajectory chunks with LSTM hidden states."""
    chunks = []
    for _ in range(num_chunks):
        h, c = net.initial_hidden(1)
        # Squeeze to [num_layers, hidden_size] for per-chunk storage
        h_init = h.squeeze(1)
        c_init = c.squeeze(1)

        chunk = TrajectoryChunk(
            agent_id="test",
            observations=torch.randn(chunk_len, OBS_DIM),
            actions=torch.randint(0, NUM_ACTIONS, (chunk_len,)),
            action_log_probs=torch.randn(chunk_len),
            rewards=torch.randn(chunk_len),
            dones=torch.zeros(chunk_len),
            values=torch.randn(chunk_len),
            bootstrap_value=torch.tensor(0.0),
            behavior_policy_version=0,
            lstm_hidden=(h_init, c_init),
        )
        chunks.append(chunk)
    return chunks


def test_appo_recurrent_train_step():
    """APPO should train with recurrent chunks end-to-end."""
    net = _make_recurrent_net("lstm")
    config = AlgorithmConfig(name="appo", num_epochs=1, minibatch_chunks=0)
    appo = APPO(net, config, device="cpu")

    chunks = _make_recurrent_chunks(net, num_chunks=4, chunk_len=8)
    metrics = appo.train_step(chunks)

    assert "total_loss" in metrics
    assert np.isfinite(metrics["total_loss"])
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics


def test_appo_recurrent_multiple_steps():
    """Multiple recurrent training steps should converge (loss should change)."""
    net = _make_recurrent_net("lstm")
    config = AlgorithmConfig(
        name="appo", num_epochs=1, minibatch_chunks=0,
        lr_schedule="constant", learning_rate=1e-3,
    )
    appo = APPO(net, config, device="cpu")

    losses = []
    for _ in range(3):
        chunks = _make_recurrent_chunks(net, num_chunks=4, chunk_len=8)
        metrics = appo.train_step(chunks)
        losses.append(metrics["total_loss"])

    # Losses should all be finite
    assert all(np.isfinite(loss) for loss in losses)


def test_appo_feedforward_backward_compat():
    """APPO with feedforward net and no lstm_hidden should use stateless path."""
    net = _make_feedforward_net()
    config = AlgorithmConfig(name="appo", num_epochs=1, minibatch_chunks=0)
    appo = APPO(net, config, device="cpu")

    # Chunks WITHOUT lstm_hidden (feedforward mode)
    chunks = []
    for _ in range(4):
        chunk = TrajectoryChunk(
            agent_id="test",
            observations=torch.randn(8, OBS_DIM),
            actions=torch.randint(0, NUM_ACTIONS, (8,)),
            action_log_probs=torch.randn(8),
            rewards=torch.randn(8),
            dones=torch.zeros(8),
            values=torch.randn(8),
            bootstrap_value=torch.tensor(0.0),
            behavior_policy_version=0,
        )
        chunks.append(chunk)

    metrics = appo.train_step(chunks)
    assert "total_loss" in metrics
    assert np.isfinite(metrics["total_loss"])


# ---------------------------------------------------------------
# TrajectoryChunk lstm_hidden support
# ---------------------------------------------------------------


def test_trajectory_chunk_lstm_hidden_to_device():
    """TrajectoryChunk with lstm_hidden should preserve it through to_device."""
    h = torch.randn(NUM_LAYERS, HIDDEN_SIZE)
    c = torch.randn(NUM_LAYERS, HIDDEN_SIZE)

    chunk = TrajectoryChunk(
        agent_id="test",
        observations=torch.randn(8, OBS_DIM),
        actions=torch.randint(0, NUM_ACTIONS, (8,)),
        action_log_probs=torch.randn(8),
        rewards=torch.randn(8),
        dones=torch.zeros(8),
        values=torch.randn(8),
        bootstrap_value=torch.tensor(0.0),
        behavior_policy_version=0,
        lstm_hidden=(h, c),
    )

    chunk2 = chunk.to_device("cpu")
    assert chunk2.lstm_hidden is not None
    assert torch.allclose(chunk2.lstm_hidden[0], h)
    assert torch.allclose(chunk2.lstm_hidden[1], c)


def test_trajectory_chunk_no_lstm_hidden():
    """TrajectoryChunk without lstm_hidden should keep it as None."""
    chunk = TrajectoryChunk(
        agent_id="test",
        observations=torch.randn(8, OBS_DIM),
        actions=torch.randint(0, NUM_ACTIONS, (8,)),
        action_log_probs=torch.randn(8),
        rewards=torch.randn(8),
        dones=torch.zeros(8),
        values=torch.randn(8),
        bootstrap_value=torch.tensor(0.0),
        behavior_policy_version=0,
    )
    assert chunk.lstm_hidden is None
    chunk2 = chunk.to_device("cpu")
    assert chunk2.lstm_hidden is None


# ---------------------------------------------------------------
# Registry: build_network with recurrent
# ---------------------------------------------------------------


def test_build_network_with_lstm():
    """build_network should create LSTM trunk when recurrent_type='lstm'."""
    from colosseum.core.config import ColosseumConfig, EnvConfig, NetworkConfig
    from colosseum.core.registry import build_network

    net_cfg = NetworkConfig(
        encoder_class="helpers.SimpleEncoder",
        policy_class="helpers.SimplePolicy",
        value_class="helpers.SimpleValue",
        recurrent_type="lstm",
        recurrent_hidden_size=HIDDEN_SIZE,
        recurrent_num_layers=NUM_LAYERS,
    )
    env_cfg = EnvConfig(env_class="examples.tic_tac_toe.env.TicTacToeEnv")
    config = ColosseumConfig(env=env_cfg, networks=net_cfg)

    net = build_network(config)
    assert net.is_recurrent
    assert isinstance(net.recurrent, nn.LSTM)
    assert net.recurrent_hidden_size == HIDDEN_SIZE
    assert net.recurrent_num_layers == NUM_LAYERS


def test_build_network_with_gru():
    """build_network should create GRU trunk when recurrent_type='gru'."""
    from colosseum.core.config import ColosseumConfig, EnvConfig, NetworkConfig
    from colosseum.core.registry import build_network

    net_cfg = NetworkConfig(
        encoder_class="helpers.SimpleEncoder",
        policy_class="helpers.SimplePolicy",
        value_class="helpers.SimpleValue",
        recurrent_type="gru",
        recurrent_hidden_size=HIDDEN_SIZE,
        recurrent_num_layers=1,
    )
    env_cfg = EnvConfig(env_class="examples.tic_tac_toe.env.TicTacToeEnv")
    config = ColosseumConfig(env=env_cfg, networks=net_cfg)

    net = build_network(config)
    assert net.is_recurrent
    assert isinstance(net.recurrent, nn.GRU)


def test_build_network_feedforward_default():
    """build_network with no recurrent_type should create feedforward net."""
    from colosseum.core.config import ColosseumConfig, EnvConfig, NetworkConfig
    from colosseum.core.registry import build_network

    net_cfg = NetworkConfig(
        encoder_class="helpers.SimpleEncoder",
        policy_class="helpers.SimplePolicy",
        value_class="helpers.SimpleValue",
    )
    env_cfg = EnvConfig(env_class="examples.tic_tac_toe.env.TicTacToeEnv")
    config = ColosseumConfig(env=env_cfg, networks=net_cfg)

    net = build_network(config)
    assert not net.is_recurrent
    assert net.recurrent is None


# ---------------------------------------------------------------
# RolloutBuffer LSTM init storage
# ---------------------------------------------------------------


def test_rollout_buffer_lstm_init():
    """RolloutBuffer should store and clear LSTM init hidden state."""
    from colosseum.worker.rollout_worker import RolloutBuffer

    buf = RolloutBuffer(chunk_length=4, obs_shape=(OBS_DIM,))
    assert buf._lstm_h_init is None
    assert buf._lstm_c_init is None

    h = torch.randn(NUM_LAYERS, 1, HIDDEN_SIZE)
    c = torch.randn(NUM_LAYERS, 1, HIDDEN_SIZE)
    buf.set_lstm_init(h, c)

    assert buf._lstm_h_init is not None
    assert torch.allclose(buf._lstm_h_init, h)
    assert torch.allclose(buf._lstm_c_init, c)

    buf.reset()
    assert buf._lstm_h_init is None
    assert buf._lstm_c_init is None


def test_build_chunk_includes_lstm_hidden():
    """_build_chunk should include lstm_hidden when buffer has it."""
    from colosseum.worker.rollout_worker import RolloutBuffer, _build_chunk

    buf = RolloutBuffer(chunk_length=2, obs_shape=(OBS_DIM,))
    h = torch.randn(NUM_LAYERS, 1, HIDDEN_SIZE)
    c = torch.randn(NUM_LAYERS, 1, HIDDEN_SIZE)
    buf.set_lstm_init(h, c)

    # Fill buffer
    for _ in range(2):
        buf.append(
            obs=np.zeros(OBS_DIM, dtype=np.float32),
            action=0,
            log_prob=0.0,
            reward=0.0,
            done=False,
            value=0.0,
        )

    chunk = _build_chunk(buf, "test_agent", 0.0, 0)
    assert chunk.lstm_hidden is not None
    assert torch.allclose(chunk.lstm_hidden[0], h)
    assert torch.allclose(chunk.lstm_hidden[1], c)


def test_build_chunk_no_lstm_hidden():
    """_build_chunk should leave lstm_hidden as None when buffer has none."""
    from colosseum.worker.rollout_worker import RolloutBuffer, _build_chunk

    buf = RolloutBuffer(chunk_length=2, obs_shape=(OBS_DIM,))
    for _ in range(2):
        buf.append(
            obs=np.zeros(OBS_DIM, dtype=np.float32),
            action=0,
            log_prob=0.0,
            reward=0.0,
            done=False,
            value=0.0,
        )

    chunk = _build_chunk(buf, "test_agent", 0.0, 0)
    assert chunk.lstm_hidden is None
