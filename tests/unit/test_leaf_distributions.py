"""Leaf distributions: exact log-probs, masked entropy without NaN, KL, cat (SP2 T2.1)."""
import math

import pytest
import torch

from colosseum.sp2.networks.dist import CategoricalDist, DiagGaussianDist, MultiCategoricalDist

LOG2, LOG3, LOG4 = math.log(2), math.log(3), math.log(4)


def test_categorical_masked_log_prob_entropy_and_validity():
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    dist = CategoricalDist(torch.zeros(2, 4), mask)
    a = torch.tensor([1, 3])
    assert dist.batch_size == 2 and dist.num_deciders == 1
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-LOG2], [-LOG4]]))
    assert torch.allclose(dist.log_prob(a), torch.tensor([-LOG2, -LOG4]))
    assert torch.allclose(dist.unit_entropy(a), torch.tensor([[LOG2], [LOG4]]))
    assert dist.unit_valid(a).tolist() == [[True], [True]]


def test_categorical_mode_and_sample_respect_the_mask():
    dist = CategoricalDist(torch.tensor([[5.0, 1.0, 0.0]]), torch.tensor([[False, True, True]]))
    assert dist.mode().tolist() == [1]
    torch.manual_seed(0)
    samples = torch.stack([dist.sample() for _ in range(200)])
    assert samples.dtype == torch.int64 and set(samples.flatten().tolist()) == {1, 2}


def test_masked_entropy_gradient_is_finite_under_loss_scaling():
    logits = torch.randn(3, 5, requires_grad=True)
    mask = torch.tensor([[True, False, True, False, False]] * 3)
    loss = CategoricalDist(logits, mask).unit_entropy(None).sum() * 65536.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[:, [1, 3, 4]] == 0)


def test_all_illegal_row_stays_finite():
    dist = CategoricalDist(torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.bool))
    assert torch.allclose(dist.unit_log_prob(torch.tensor([2])), torch.tensor([[-LOG4]]))


def test_categorical_kl_exact_and_masked():
    p = CategoricalDist(torch.log(torch.tensor([[0.25, 0.75]])))
    q = CategoricalDist(torch.zeros(1, 2))
    expected = 0.25 * math.log(0.5) + 0.75 * math.log(1.5)
    assert p.unit_kl(q, None).item() == pytest.approx(expected, abs=1e-6)
    assert p.unit_kl(p, None).item() == pytest.approx(0.0, abs=1e-7)
    # A one-sided mask: the KL is taken on the legal intersection, renormalized -> 0 here.
    masked = CategoricalDist(torch.tensor([[3.0, 0.0, 0.0]]), torch.tensor([[False, True, True]]))
    assert masked.unit_kl(CategoricalDist(torch.zeros(1, 3)), None).item() == pytest.approx(0.0, abs=1e-7)
    with pytest.raises(TypeError):
        p.unit_kl(DiagGaussianDist(torch.zeros(1, 2), torch.zeros(2)), None)


def test_categorical_apply_mask_combines_and_cat_fills_missing_masks():
    dist = CategoricalDist(torch.zeros(1, 3), torch.tensor([[True, True, False]]))
    both = dist.apply_mask(torch.tensor([[False, True, True]]))
    assert both.mask.tolist() == [[False, True, False]] and both.mode().tolist() == [1]
    assert dist.apply_mask(None) is dist
    joined = CategoricalDist.cat([dist, CategoricalDist(torch.zeros(2, 3))])
    assert joined.batch_size == 3 and joined.mask.tolist() == [[True, True, False]] + [[True] * 3] * 2


def test_shape_errors():
    with pytest.raises(ValueError, match=r"logits must be \[B, n\]"):
        CategoricalDist(torch.zeros(4))
    with pytest.raises(ValueError, match="mask shape"):
        CategoricalDist(torch.zeros(2, 3), torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match=r"\[B, 5\] for nvec \(2, 3\)"):
        MultiCategoricalDist(torch.zeros(2, 4), [2, 3])


def test_multi_categorical_is_one_decider():
    mask = torch.tensor([[True, True, True, False, True]])           # nvec (2, 3): second part [T, F, T]
    dist = MultiCategoricalDist(torch.zeros(1, 5), [2, 3], mask)
    a = torch.tensor([[1, 2]])
    assert dist.num_deciders == 1
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-2 * LOG2]]))
    assert torch.allclose(dist.unit_entropy(a), torch.tensor([[2 * LOG2]]))
    sample = dist.sample()
    assert sample.shape == (1, 2) and sample.dtype == torch.int64 and sample[0, 1].item() in (0, 2)
    assert dist.mode().shape == (1, 2)
    assert dist.unit_kl(dist, a).item() == pytest.approx(0.0, abs=1e-7)
    assert MultiCategoricalDist.cat([dist, dist]).batch_size == 2


def test_gaussian_exact_values():
    dist = DiagGaussianDist(torch.zeros(3, 2), torch.zeros(2))            # log_std broadcast from [d]
    a = torch.zeros(3, 2)
    assert torch.allclose(dist.unit_log_prob(a), torch.full((3, 1), -math.log(2 * math.pi)))
    assert torch.allclose(dist.unit_entropy(a), torch.full((3, 1), 1.0 + math.log(2 * math.pi)))
    other = DiagGaussianDist(torch.ones(3, 2), torch.zeros(3, 2))
    assert torch.allclose(dist.unit_kl(other, a), torch.full((3, 1), 1.0))   # 0.5 per dim
    assert dist.mode().tolist() == [[0.0, 0.0]] * 3 and dist.sample().shape == (3, 2)
    assert dist.apply_mask(None) is dist
    with pytest.raises(ValueError, match="takes no mask"):
        dist.apply_mask(torch.ones(3, 2, dtype=torch.bool))
    joined = DiagGaussianDist.cat([dist, other])
    assert joined.batch_size == 6 and joined.log_std.shape == (6, 2)


def test_gaussian_log_prob_matches_torch():
    mean, log_std = torch.randn(4, 3), torch.randn(4, 3) * 0.3
    a = torch.randn(4, 3)
    ref = torch.distributions.Normal(mean, log_std.exp()).log_prob(a).sum(-1)
    assert torch.allclose(DiagGaussianDist(mean, log_std).log_prob(a), ref, atol=1e-5)
