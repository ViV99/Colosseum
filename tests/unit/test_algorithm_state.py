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


def test_load_state_dict_warns_about_mismatched_scaler_and_kickstart_state(caplog):
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=2, chunk_length=8)
    plain = APPO(model, AlgorithmConfig(), device="cpu")
    plain.train_step(chunks)
    plain_state = plain.state_dict()

    rich = APPO(tiny_model(seed=3), AlgorithmConfig(), device="cpu",
                kickstart=KickstartLoss(tiny_model(seed=4), initial_lambda=1.0, decay_steps=10))
    rich._scaler = torch.amp.GradScaler("cpu")
    rich.train_step(chunks)
    rich_state = rich.state_dict()

    caplog.set_level("WARNING", logger="colosseum.algorithms.appo")
    APPO(tiny_model(), AlgorithmConfig(), device="cpu").load_state_dict(rich_state)
    text = caplog.text
    assert "GradScaler state" in text and "ignored" in text
    assert "kickstart state" in text

    caplog.clear()
    fresh_rich = APPO(tiny_model(), AlgorithmConfig(), device="cpu",
                      kickstart=KickstartLoss(tiny_model(seed=4), initial_lambda=1.0, decay_steps=10))
    fresh_rich._scaler = torch.amp.GradScaler("cpu")
    fresh_rich.load_state_dict(plain_state)
    text = caplog.text
    assert "no GradScaler state" in text and "no kickstart state" in text

    caplog.clear()
    APPO(tiny_model(), AlgorithmConfig(), device="cpu").load_state_dict(plain_state)
    assert caplog.text == ""


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


def _assert_scaler_round_trip(algo: APPO, other: APPO) -> None:
    """Give ``algo``'s scaler a non-default state, save it, restore it into ``other``'s fresh scaler."""
    algo._scaler.update(new_scale=1234.0)            # distinctive scale (default init_scale is 65536)
    state = algo.state_dict()
    saved = state["scaler"]
    assert saved is not None and saved["scale"] == 1234.0
    assert saved["_growth_tracker"] > 0              # successful unskipped steps since the last growth
    fresh = other._scaler.state_dict()
    assert fresh["scale"] != saved["scale"] and fresh["_growth_tracker"] != saved["_growth_tracker"]
    other.load_state_dict(state)
    assert other._scaler.state_dict() == saved
    assert other._scaler.get_scale() == 1234.0


def test_grad_scaler_state_round_trips_with_a_cpu_scaler():
    """CPU coverage of the scaler save/restore path: APPO creates a GradScaler only for
    AMP on CUDA, so this test installs a CPU GradScaler (torch.amp supports it) by hand."""
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    cfg = AlgorithmConfig(num_epochs=1, minibatch_chunks=2)
    algo = APPO(model, cfg, device="cpu")
    algo._scaler = torch.amp.GradScaler("cpu")
    algo.train_step(chunks)
    algo.train_step(chunks)
    other = APPO(tiny_model(seed=5), cfg, device="cpu")
    other._scaler = torch.amp.GradScaler("cpu")
    _assert_scaler_round_trip(algo, other)


@pytest.mark.gpu
def test_grad_scaler_state_round_trips_on_cuda():
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    cfg = AlgorithmConfig(num_epochs=2, minibatch_chunks=2, use_amp=True, amp_dtype="float16")
    algo = APPO(model, cfg, device="cuda")
    for _ in range(3):
        algo.train_step(chunks)
    other = APPO(tiny_model(seed=5), cfg, device="cuda")
    _assert_scaler_round_trip(algo, other)


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
