"""Tests for RNN/LSTM support (T1.2)."""



import torch
import torch.nn as nn

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
# Registry: build_network with recurrent
# ---------------------------------------------------------------


def _legacy_cfg(core=None):
    from colosseum.core.config import ColosseumConfig, EnvConfig, NetworkConfig

    net_cfg = NetworkConfig(
        encoder_class="helpers.SimpleEncoder",
        core=core,
        policy_class="helpers.SimplePolicy",
        value_class="helpers.SimpleValue",
    )
    return ColosseumConfig(env=EnvConfig(env_class="examples.tic_tac_toe.env.TicTacToeEnv"), networks=net_cfg)


def test_build_network_with_lstm():
    """Transitional build_network maps an LSTMCore config to an nn.LSTM trunk."""
    from colosseum.core.registry import build_network

    net = build_network(_legacy_cfg({"class": "colosseum.networks.cores.LSTMCore",
                                     "kwargs": {"hidden_size": 24, "num_layers": NUM_LAYERS}}))
    assert net.is_recurrent
    assert isinstance(net.recurrent, nn.LSTM)
    assert net.recurrent_hidden_size == 24
    assert net.recurrent_num_layers == NUM_LAYERS
    assert net.policy.fc.in_features == 24  # in_dim passed to the head


def test_build_network_with_gru():
    """Transitional build_network maps a GRUCore config to an nn.GRU trunk."""
    from colosseum.core.registry import build_network

    net = build_network(_legacy_cfg({"class": "colosseum.networks.cores.GRUCore",
                                     "kwargs": {"hidden_size": 24}}))
    assert net.is_recurrent
    assert isinstance(net.recurrent, nn.GRU)


def test_build_network_feedforward_default():
    """Transitional build_network without a core builds a feedforward net."""
    from colosseum.core.registry import build_network

    net = build_network(_legacy_cfg())
    assert not net.is_recurrent
    assert net.recurrent is None
