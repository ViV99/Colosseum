"""NormalizeObs: explicit update(), pure forward, one update per train step (T4.2)."""
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.types import WeightPayload
from colosseum.networks.normalization import NormalizeObs
from helpers import MaskedToyEnv, make_simple_model, rollout_chunks


def _norm(model) -> NormalizeObs:
    found = [m for m in model.modules() if isinstance(m, NormalizeObs)]
    assert len(found) == 1
    return found[0]


def test_forward_never_updates_stats_even_in_train_mode():
    norm = NormalizeObs(shape=(3,))
    norm.train()
    before = {k: v.clone() for k, v in norm.state_dict().items()}
    norm(torch.randn(32, 3) * 5 + 2)
    for key, value in norm.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_update_matches_batch_statistics():
    norm = NormalizeObs(shape=(3,), clip=0.0)
    x = torch.randn(1000, 3) * torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([5.0, -1.0, 0.5])
    norm.update(x)
    assert norm.rms.count.item() == pytest.approx(1000.0, abs=1e-3)
    torch.testing.assert_close(norm.rms.mean, x.mean(0), atol=1e-3, rtol=0)
    torch.testing.assert_close(norm.rms.var, x.var(0, unbiased=False), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(norm(x).mean(0), torch.zeros(3), atol=1e-2, rtol=0)


def test_update_accepts_leading_dims_and_rejects_wrong_shape():
    norm = NormalizeObs(shape=(3,))
    norm.update(torch.randn(4, 5, 3))
    assert norm.rms.count.item() == pytest.approx(20.0, abs=1e-3)
    with pytest.raises(ValueError, match="NormalizeObs"):
        norm.update(torch.randn(10, 4))


def test_update_normalizers_without_normalizer_is_noop():
    model = make_simple_model(obs_dim=4, num_actions=4, normalize=False, seed=0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.update_normalizers(torch.randn(8, 4))
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key


@pytest.mark.parametrize("num_epochs,minibatch_chunks", [(1, 0), (3, 1), (2, 2)])
def test_train_step_counts_each_sample_once(num_epochs, minibatch_chunks):
    model = make_simple_model(obs_dim=4, num_actions=4, normalize=True, seed=0)
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    samples = sum(c.chunk_length for c in chunks)
    algo = APPO(model, AlgorithmConfig(num_epochs=num_epochs, minibatch_chunks=minibatch_chunks), device="cpu")
    norm = _norm(algo.model)
    start = norm.rms.count.item()
    algo.train_step(chunks)
    assert norm.rms.count.item() - start == pytest.approx(samples, abs=1e-3)
    algo.train_step(chunks)
    assert norm.rms.count.item() - start == pytest.approx(2 * samples, abs=1e-3)


def test_stats_reach_workers_through_weight_payload():
    model = make_simple_model(obs_dim=4, num_actions=4, normalize=True, seed=0)
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=2, chunk_length=8)
    algo = APPO(model, AlgorithmConfig(), device="cpu")
    algo.train_step(chunks)
    learner_norm = _norm(algo.model)
    assert learner_norm.rms.count.item() > 1.0

    payload = WeightPayload.from_model("a", algo.policy_version, algo.model)
    worker_model = make_simple_model(obs_dim=4, num_actions=4, normalize=True, seed=123)
    worker_model.load_state_dict(payload.to_torch_state_dict())
    worker_norm = _norm(worker_model)
    torch.testing.assert_close(worker_norm.rms.mean, learner_norm.rms.mean)
    torch.testing.assert_close(worker_norm.rms.var, learner_norm.rms.var)
    torch.testing.assert_close(worker_norm.rms.count, learner_norm.rms.count)
