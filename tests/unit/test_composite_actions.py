"""Tests for composite action space support (Dict, Tuple, MultiDiscrete)."""



import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.core.action_spec import ActionSpec
from colosseum.networks.distributions import (
    CategoricalDist,
    CompositeDist,
    DiagGaussianDist,
)

# ===================================================================
# ActionSpec unit tests
# ===================================================================


def test_action_spec_discrete():
    spec = ActionSpec.from_space(gymnasium.spaces.Discrete(5))
    assert not spec.is_composite
    assert spec.flat_size == 1
    assert spec.action_shape == ()
    assert spec.numpy_dtype == np.int64
    assert spec.flat_mask_size == 5
    assert spec.space_type == "discrete"


def test_action_spec_box():
    spec = ActionSpec.from_space(gymnasium.spaces.Box(low=-1, high=1, shape=(3,)))
    assert not spec.is_composite
    assert spec.flat_size == 3
    assert spec.action_shape == (3,)
    assert spec.numpy_dtype == np.float32
    assert spec.flat_mask_size == 0
    assert spec.space_type == "box"


def test_action_spec_dict():
    space = gymnasium.spaces.Dict({
        "direction": gymnasium.spaces.Discrete(4),
        "speed": gymnasium.spaces.Box(low=0, high=1, shape=(2,)),
    })
    spec = ActionSpec.from_space(space)
    assert spec.is_composite
    assert spec.space_type == "dict"
    # gymnasium sorts plain-dict keys: "direction" → offset=0, size=1; "speed" → offset=1, size=2
    assert spec.flat_size == 3
    assert spec.action_shape == (3,)
    assert spec.numpy_dtype == np.float32
    # mask: direction has 4 categories, speed has 0
    assert spec.flat_mask_size == 4

    # Check components
    assert len(spec.components) == 2
    d = spec.components[0]
    assert d.name == "direction" and d.offset == 0 and d.size == 1
    assert d.is_discrete and d.num_categories == 4
    assert d.mask_offset == 0 and d.mask_size == 4
    s = spec.components[1]
    assert s.name == "speed" and s.offset == 1 and s.size == 2
    assert not s.is_discrete


def test_action_spec_multi_discrete():
    spec = ActionSpec.from_space(gymnasium.spaces.MultiDiscrete([4, 6]))
    assert spec.is_composite
    assert spec.space_type == "multi_discrete"
    assert spec.flat_size == 2
    assert spec.action_shape == (2,)
    assert spec.numpy_dtype == np.float32
    assert spec.flat_mask_size == 10  # 4 + 6


def test_action_spec_tuple():
    space = gymnasium.spaces.Tuple((
        gymnasium.spaces.Discrete(3),
        gymnasium.spaces.Box(low=0, high=1, shape=(2,)),
    ))
    spec = ActionSpec.from_space(space)
    assert spec.is_composite
    assert spec.space_type == "tuple"
    assert spec.flat_size == 3  # 1 + 2
    assert spec.flat_mask_size == 3  # only Discrete(3)


def test_action_spec_nested_raises():
    space = gymnasium.spaces.Dict({
        "outer": gymnasium.spaces.Dict({"inner": gymnasium.spaces.Discrete(2)}),
    })
    try:
        ActionSpec.from_space(space)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Nested" in str(e)


def test_action_spec_large_discrete_raises():
    try:
        ActionSpec.from_space(gymnasium.spaces.Discrete(2**24 + 1))
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "float32" in str(e)


# ===================================================================
# ActionSpec encode/decode round-trip
# ===================================================================


def test_encode_decode_dict():
    space = gymnasium.spaces.Dict({
        "type": gymnasium.spaces.Discrete(3),
        "pos": gymnasium.spaces.Box(low=-1, high=1, shape=(2,)),
    })
    spec = ActionSpec.from_space(space)
    # gymnasium sorts plain-dict keys: "pos" before "type" → flat = [pos[0], pos[1], type]
    structured = {"type": 2, "pos": np.array([0.5, -0.3], dtype=np.float32)}
    flat = spec.encode(structured)
    assert flat.shape == (3,)
    assert flat.dtype == np.float32
    assert flat[0] == 0.5    # pos[0]
    assert flat[1] == -0.3   # pos[1]
    assert flat[2] == 2.0    # type

    decoded = spec.decode(flat)
    assert isinstance(decoded, dict)
    assert decoded["type"] == 2
    np.testing.assert_allclose(decoded["pos"], [0.5, -0.3], atol=1e-6)


def test_encode_decode_multi_discrete():
    spec = ActionSpec.from_space(gymnasium.spaces.MultiDiscrete([4, 6]))
    structured = np.array([3, 5])
    flat = spec.encode(structured)
    assert flat.shape == (2,)
    decoded = spec.decode(flat)
    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype == np.int64
    np.testing.assert_array_equal(decoded, [3, 5])


def test_encode_decode_tuple():
    space = gymnasium.spaces.Tuple((
        gymnasium.spaces.Discrete(3),
        gymnasium.spaces.Box(low=0, high=1, shape=(1,)),
    ))
    spec = ActionSpec.from_space(space)
    structured = (1, np.array([0.7], dtype=np.float32))
    flat = spec.encode(structured)
    decoded = spec.decode(flat)
    assert isinstance(decoded, tuple)
    assert decoded[0] == 1
    assert abs(decoded[1] - 0.7) < 1e-5


def test_decode_passthrough_discrete():
    spec = ActionSpec.from_space(gymnasium.spaces.Discrete(5))
    val = np.int64(3)
    assert spec.decode(val) is val  # pass-through


def test_decode_passthrough_box():
    spec = ActionSpec.from_space(gymnasium.spaces.Box(low=0, high=1, shape=(2,)))
    val = np.array([0.5, 0.3], dtype=np.float32)
    assert spec.decode(val) is val  # pass-through


# ===================================================================
# ActionSpec flatten_mask
# ===================================================================


def test_flatten_mask_flat_array():
    spec = ActionSpec.from_space(gymnasium.spaces.Dict({
        "a": gymnasium.spaces.Discrete(3),
        "b": gymnasium.spaces.Discrete(4),
    }))
    flat = np.array([True, False, True, True, True, False, True], dtype=bool)
    result = spec.flatten_mask(flat)
    np.testing.assert_array_equal(result, flat)


def test_flatten_mask_dict():
    spec = ActionSpec.from_space(gymnasium.spaces.Dict({
        "a": gymnasium.spaces.Discrete(3),
        "b": gymnasium.spaces.Discrete(4),
    }))
    mask_dict = {
        "a": np.array([True, False, True]),
        "b": np.array([True, True, False, True]),
    }
    result = spec.flatten_mask(mask_dict)
    assert result.shape == (7,)
    np.testing.assert_array_equal(result[:3], [True, False, True])
    np.testing.assert_array_equal(result[3:], [True, True, False, True])


def test_flatten_mask_partial_dict():
    """Missing keys should default to all-True."""
    spec = ActionSpec.from_space(gymnasium.spaces.Dict({
        "a": gymnasium.spaces.Discrete(3),
        "b": gymnasium.spaces.Discrete(4),
    }))
    mask_dict = {"a": np.array([True, False, True])}
    result = spec.flatten_mask(mask_dict)
    np.testing.assert_array_equal(result[:3], [True, False, True])
    np.testing.assert_array_equal(result[3:], [True, True, True, True])  # b defaults


# ===================================================================
# CompositeDist unit tests
# ===================================================================


def test_composite_dist_sample_shape():
    B = 5
    dists = {
        "type": CategoricalDist(torch.randn(B, 3)),
        "pos": DiagGaussianDist(torch.randn(B, 2), torch.zeros(B, 2)),
    }
    cd = CompositeDist(dists)
    assert cd.action_dim == 3  # 1 (type) + 2 (pos)
    s = cd.sample()
    assert s.shape == (B, 3)
    assert s.dtype == torch.float32
    # Insertion order: "type" at [0] (integer-valued), "pos" at [1:3]
    assert (s[:, 0] == s[:, 0].long().float()).all()


def test_composite_dist_log_prob():
    B = 4
    logits = torch.tensor([[1.0, 0.0, -1.0]] * B)
    mean = torch.tensor([[0.5, -0.5]] * B)
    log_std = torch.zeros(B, 2)

    # Insertion order: "type" at [0], "pos" at [1:3]
    cd = CompositeDist({
        "type": CategoricalDist(logits),
        "pos": DiagGaussianDist(mean, log_std),
    })
    flat = cd.sample()
    lp = cd.log_prob(flat)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()

    cat_lp = CategoricalDist(logits).log_prob(flat[:, 0].long())
    gauss_lp = DiagGaussianDist(mean, log_std).log_prob(flat[:, 1:3])
    torch.testing.assert_close(lp, cat_lp + gauss_lp)


def test_composite_dist_entropy():
    B = 3
    cd = CompositeDist({
        "a": CategoricalDist(torch.randn(B, 4)),
        "b": DiagGaussianDist(torch.randn(B, 2), torch.zeros(B, 2)),
    })
    ent = cd.entropy()
    assert ent.shape == (B,)
    assert (ent > 0).all()


def test_composite_dist_mode():
    B = 3
    # Insertion order: "type" at [0], "pos" at [1:3]
    cd = CompositeDist({
        "type": CategoricalDist(torch.tensor([[10.0, 0.0, 0.0]] * B)),
        "pos": DiagGaussianDist(torch.tensor([[0.5, -0.3]] * B), torch.zeros(B, 2)),
    })
    m = cd.mode()
    assert m.shape == (B, 3)
    assert (m[:, 0] == 0.0).all()
    torch.testing.assert_close(m[:, 1], torch.tensor([0.5] * B))
    torch.testing.assert_close(m[:, 2], torch.tensor([-0.3] * B))


def test_composite_dist_apply_mask():
    B = 2
    cd = CompositeDist({
        "action": CategoricalDist(torch.randn(B, 4)),
        "speed": DiagGaussianDist(torch.randn(B, 1), torch.zeros(B, 1)),
    })
    # Mask: 4 categories for "action", 0 for "speed" → flat_mask_size=4
    assert cd._flat_mask_size == 4
    mask = torch.tensor([[True, True, False, False]] * B)
    masked_cd = cd.apply_mask(mask)
    s = masked_cd.sample()
    # Sampled discrete action should be 0 or 1 (only first two valid)
    assert (s[:, 0].long() < 2).all()


def test_composite_dist_kl_divergence():
    B = 3
    cd1 = CompositeDist({
        "a": CategoricalDist(torch.randn(B, 3)),
        "b": DiagGaussianDist(torch.randn(B, 2), torch.zeros(B, 2)),
    })
    cd2 = CompositeDist({
        "a": CategoricalDist(torch.randn(B, 3)),
        "b": DiagGaussianDist(torch.randn(B, 2), torch.zeros(B, 2)),
    })
    kl = cd1.kl_divergence(cd2)
    assert kl.shape == (B,)
    assert torch.isfinite(kl).all()
    assert (kl >= 0).all()


def test_composite_dist_kl_key_mismatch_raises():
    B = 2
    cd1 = CompositeDist({"a": CategoricalDist(torch.randn(B, 3))})
    cd2 = CompositeDist({"b": CategoricalDist(torch.randn(B, 3))})
    try:
        cd1.kl_divergence(cd2)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


def test_composite_dist_gradient_flow():
    """Gradients should flow through all components via entropy."""
    B = 4
    type_logits = torch.randn(B, 3, requires_grad=True)
    pos_mean = torch.randn(B, 2, requires_grad=True)
    log_std = torch.zeros(B, 2, requires_grad=True)

    cd = CompositeDist({
        "type": CategoricalDist(type_logits),
        "pos": DiagGaussianDist(pos_mean, log_std),
    })

    # Use fixed actions (not own sample) so log_prob gradients flow through mean
    fixed_actions = torch.tensor([[1.0, 0.1, -0.2]] * B)  # type=1 at [0], pos at [1:3]
    lp = cd.log_prob(fixed_actions)
    loss = -lp.mean() - 0.01 * cd.entropy().mean()
    loss.backward()

    assert type_logits.grad is not None and type_logits.grad.abs().sum() > 0
    assert pos_mean.grad is not None and pos_mean.grad.abs().sum() > 0


def test_action_dim_property():
    assert CategoricalDist(torch.randn(2, 5)).action_dim == 1
    assert DiagGaussianDist(torch.randn(2, 3), torch.zeros(2, 3)).action_dim == 3


# ===================================================================
# Integration: VectorEnv with composite actions
# ===================================================================


class DictActionEnv:
    """Minimal env with Dict action space for testing."""

    def __init__(self):
        self._step = 0

    @property
    def num_players(self):
        return 2

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Dict({
            "direction": gymnasium.spaces.Discrete(4),
            "speed": gymnasium.spaces.Box(low=0, high=1, shape=(1,)),
        })

    def reset(self, seed=None):
        self._step = 0
        obs = {i: np.zeros(4, dtype=np.float32) for i in range(2)}
        info = {i: {} for i in range(2)}
        return obs, info

    def step(self, actions):
        self._step += 1
        # Validate action structure
        for p in range(2):
            a = actions[p]
            assert isinstance(a, dict), f"Expected dict, got {type(a)}"
            assert "direction" in a and "speed" in a
            assert isinstance(a["direction"], (int, np.integer))
            assert 0 <= a["direction"] < 4

        obs = {i: np.random.rand(4).astype(np.float32) for i in range(2)}
        rewards = {i: float(np.random.rand()) for i in range(2)}
        done = self._step >= 5
        terminated = {i: done for i in range(2)}
        truncated = {i: False for i in range(2)}
        info = {i: {} for i in range(2)}
        return obs, rewards, terminated, truncated, info

    def close(self):
        pass


def test_vector_env_composite_decode():
    """VectorEnv should decode flat actions to dicts for composite action spaces."""
    from colosseum.envs.vec_env import VectorEnv

    vec_env = VectorEnv(DictActionEnv, 2)
    assert vec_env.action_spec.is_composite
    assert vec_env.action_spec.flat_size == 2  # 1 (direction) + 1 (speed)

    obs, _ = vec_env.reset_all()

    # Create flat actions: [num_envs, num_players, flat_size]
    actions = np.zeros((2, 2, 2), dtype=np.float32)
    actions[0, 0] = [2.0, 0.5]   # direction=2, speed=0.5
    actions[0, 1] = [1.0, 0.3]
    actions[1, 0] = [0.0, 0.8]
    actions[1, 1] = [3.0, 0.1]

    # This should not raise — VectorEnv decodes flat to dict internally
    next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)
    assert next_obs.shape == (2, 2, 4)
    vec_env.close()


# ===================================================================
# Integration: APPO with composite actions
# ===================================================================


def test_appo_composite_train_step():
    """APPO should train on chunks with composite actions (flat [T, flat_size] tensors)."""
    from colosseum.algorithms.appo import APPO
    from colosseum.core.config import AlgorithmConfig
    from colosseum.core.types import TrajectoryChunk
    from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
    from colosseum.networks.composed import ComposedModel
    from colosseum.networks.cores import NoCore

    FLAT_SIZE = 3  # direction(1) + speed(2)
    OBS_DIM = 4
    LATENT = 8

    class Enc(BaseEncoder):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(OBS_DIM, LATENT)

        @property
        def latent_dim(self):
            return LATENT

        def forward(self, x):
            return torch.relu(self.fc(x))

    class Pol(BasePolicy):
        def __init__(self):
            super().__init__()
            self.dir_head = nn.Linear(LATENT, 4)
            self.spd_mean = nn.Linear(LATENT, 2)
            self.spd_logstd = nn.Parameter(torch.zeros(2))

        def forward(self, latent):
            return CompositeDist({
                "direction": CategoricalDist(self.dir_head(latent)),
                "speed": DiagGaussianDist(
                    self.spd_mean(latent),
                    self.spd_logstd.expand(latent.shape[0], -1),
                ),
            })

    class Val(BaseValue):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(LATENT, 1)

        def forward(self, latent):
            return self.fc(latent).squeeze(-1)

    net = ComposedModel(Enc(), NoCore(LATENT), Pol(), Val())
    config = AlgorithmConfig(name="appo", num_epochs=1, minibatch_chunks=0)
    appo = APPO(net, config, device="cpu")

    # Create chunks with composite actions: [T, flat_size]
    # Insertion order: "direction" (Discrete(4)) at [0], "speed" (Box(2)) at [1:3]
    chunks = []
    for _ in range(4):
        T = 8
        actions = torch.zeros(T, FLAT_SIZE)
        actions[:, 0] = torch.randint(0, 4, (T,)).float()  # direction: valid 0-3
        actions[:, 1:] = torch.randn(T, 2)                  # speed: continuous
        chunk = TrajectoryChunk(
            agent_id="test",
            observations=torch.randn(T, OBS_DIM),
            actions=actions,
            action_log_probs=torch.randn(T),
            rewards=torch.randn(T),
            dones=torch.zeros(T),
            values=torch.randn(T),
            bootstrap_value=torch.tensor(0.0),
            behavior_policy_version=0,
        )
        chunks.append(chunk)

    metrics = appo.train_step(chunks)
    assert "total_loss" in metrics
    assert np.isfinite(metrics["total_loss"])
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])
