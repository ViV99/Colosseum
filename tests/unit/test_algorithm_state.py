"""Algorithm state round trip and deep-copy semantics (T4.6)."""
import copy

import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.algorithms.base import deep_cpu_copy
from colosseum.bc.kickstart import KickstartLoss
from colosseum.core.config import AlgorithmConfig
from colosseum.networks.model import PolicyModel
from helpers import MaskedToyEnv, make_simple_model, rollout_chunks

STATE_KEYS = {"optimizer", "progress", "scaler", "kickstart", "policy_version", "consumed_samples"}


def tiny_model(core: str = "none", seed: int | None = None) -> PolicyModel:
    """Model matching MaskedToyEnv (4-dim observation, Discrete(4))."""
    return make_simple_model(obs_dim=4, num_actions=4, core=core, seed=seed)


def _tensors(tree):
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _tensors(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _tensors(value)


def _assert_tree_equal(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_tree_equal(x, y)
    else:
        assert a == b


def test_deep_cpu_copy_detaches_and_copies():
    t = torch.ones(3, requires_grad=True)
    tree = {"a": [t, (t * 2,)], "b": 5, "c": "x"}
    out = deep_cpu_copy(tree)
    assert out["a"][0].data_ptr() != t.data_ptr()
    assert not out["a"][0].requires_grad
    assert isinstance(out["a"][1], tuple)
    assert out["b"] == 5 and out["c"] == "x"


def test_state_dict_keys_and_values():
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    algo = APPO(model, AlgorithmConfig(), device="cpu")
    algo.set_progress(0.25)
    algo.train_step(chunks)
    state = algo.state_dict()
    assert set(state) == STATE_KEYS
    assert state["policy_version"] == 1
    assert state["consumed_samples"] == 32 == algo.consumed_samples
    assert state["progress"] == pytest.approx(0.25)
    assert state["scaler"] is None          # no AMP on CPU
    assert state["kickstart"] is None
    assert all(t.device.type == "cpu" for t in _tensors(state))


def test_state_dict_is_a_deep_copy():
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    algo = APPO(model, AlgorithmConfig(), device="cpu")
    algo.train_step(chunks)
    state = algo.state_dict()
    frozen = copy.deepcopy(state)
    live_ptrs = {
        v.data_ptr()
        for per_param in algo._optimizer.state.values()
        for v in per_param.values()
        if isinstance(v, torch.Tensor)
    }
    assert not any(t.data_ptr() in live_ptrs for t in _tensors(state))
    algo.train_step(chunks)
    algo.train_step(chunks)
    _assert_tree_equal(state, frozen)


def test_resume_reproduces_the_next_update_exactly():
    cfg = AlgorithmConfig(num_epochs=2, minibatch_chunks=2, learning_rate=1e-3, lr_schedule="linear")
    model = tiny_model(core="lstm")
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    teacher = copy.deepcopy(model)

    original = APPO(model, cfg, device="cpu",
                    kickstart=KickstartLoss(copy.deepcopy(teacher), initial_lambda=1.0, decay_steps=10))
    torch.manual_seed(1)
    original.set_progress(0.1)
    original.train_step(chunks)
    torch.manual_seed(2)
    original.set_progress(0.2)
    original.train_step(chunks)

    model_state = {k: v.detach().clone() for k, v in original.model.state_dict().items()}
    algo_state = original.state_dict()
    algo_state_frozen = copy.deepcopy(algo_state)

    resumed = APPO(tiny_model(core="lstm", seed=99), cfg, device="cpu",
                   kickstart=KickstartLoss(copy.deepcopy(teacher), initial_lambda=1.0, decay_steps=10))
    resumed.model.load_state_dict(model_state)
    resumed.load_state_dict(algo_state)
    assert resumed.policy_version == original.policy_version == 2
    assert resumed.consumed_samples == original.consumed_samples
    assert resumed._optimizer.param_groups[0]["lr"] == original._optimizer.param_groups[0]["lr"]

    torch.manual_seed(3)
    m_original = original.train_step(chunks)
    torch.manual_seed(3)
    m_resumed = resumed.train_step(chunks)

    for key, value in original.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key
    assert m_original == m_resumed

    # load_state_dict copied its input: training `resumed` did not touch algo_state.
    _assert_tree_equal(algo_state, algo_state_frozen)


def test_base_algorithm_state_methods_raise_not_implemented():
    from colosseum.algorithms.base import BaseAlgorithm

    class Minimal(BaseAlgorithm):
        def compute_loss(self, chunks):
            return {}

        def train_step(self, chunks):
            return {}

        @property
        def model(self):
            return None

        @property
        def policy_version(self):
            return 0

    with pytest.raises(NotImplementedError, match="Minimal"):
        Minimal().state_dict()
    with pytest.raises(NotImplementedError, match="Minimal"):
        Minimal().load_state_dict({})


@pytest.mark.gpu
def test_grad_scaler_state_round_trips_on_cuda():
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=2, chunk_length=8)
    cfg = AlgorithmConfig(use_amp=True, amp_dtype="float16")
    algo = APPO(model, cfg, device="cuda")
    algo.train_step(chunks)
    state = algo.state_dict()
    assert state["scaler"] is not None and "scale" in state["scaler"]
    other = APPO(tiny_model(seed=5), cfg, device="cuda")
    other.load_state_dict(state)
    assert other._scaler.get_scale() == algo._scaler.get_scale()


@pytest.mark.gpu
def test_state_dict_round_trip_with_model_on_cuda():
    """Saved tensors are CPU copies; loading puts the optimizer state back on the model's device."""
    model = tiny_model(core="lstm")
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    cfg = AlgorithmConfig(num_epochs=1, minibatch_chunks=0)
    algo = APPO(model, cfg, device="cuda")
    algo.set_progress(0.3)
    algo.train_step(chunks)
    state = algo.state_dict()
    assert set(state) == STATE_KEYS
    assert all(t.device.type == "cpu" for t in _tensors(state))

    resumed = APPO(tiny_model(core="lstm", seed=7), cfg, device="cuda")
    resumed.model.load_state_dict(algo.model.state_dict())
    resumed.load_state_dict(state)
    assert resumed.policy_version == 1
    assert resumed.consumed_samples == algo.consumed_samples
    assert resumed._optimizer.param_groups[0]["lr"] == algo._optimizer.param_groups[0]["lr"]
    for per_param in resumed._optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            assert per_param[name].device.type == "cuda"
    _assert_tree_equal(resumed.state_dict(), state)
    metrics = resumed.train_step(chunks)
    assert metrics["policy_version"] == 2.0
