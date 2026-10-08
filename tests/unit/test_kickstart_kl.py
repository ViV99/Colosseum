"""Masked KL and kickstart (T4.3)."""
import copy
import math

import pytest
import torch
from pydantic import ValidationError

from colosseum.algorithms.appo import APPO
from colosseum.bc.kickstart import KickstartLoss
from colosseum.core.config import AlgorithmConfig, TrainingConfig
from colosseum.networks.distributions import CategoricalDist, CompositeDist, DiagGaussianDist
from colosseum.networks.model import PolicyModel, StepOutput, UnrollOutput
from helpers import MaskedToyEnv, make_simple_model, rollout_chunks

P = [0.5, 0.5]
Q = [0.9, 0.1]
KL_PQ = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)   # 0.5108
KL_QP = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)   # 0.3681


def _masked_toy_model(core: str = "none") -> PolicyModel:
    """Model matching MaskedToyEnv (4-dim observation, Discrete(4))."""
    return make_simple_model(obs_dim=4, num_actions=4, core=core)


class FixedTeacher(PolicyModel):
    """Stateless teacher with the same action probabilities for every observation."""

    def __init__(self, probs):
        super().__init__()
        self.register_buffer("logits", torch.log(torch.tensor(probs)))

    def step(self, obs, state, action_mask=None):
        dist = CategoricalDist(self.logits.expand(obs.shape[0], -1))
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        return StepOutput(dist, torch.zeros(obs.shape[0]), state)

    def unroll(self, obs, state0, dones, action_mask=None):
        T, B = obs.shape[:2]
        flat_mask = None if action_mask is None else action_mask.reshape(T * B, -1)
        out = self.step(obs.reshape(T * B, *obs.shape[2:]), state0, flat_mask)
        return UnrollOutput(out.dist, out.value)


def _student(probs, n):
    return CategoricalDist(torch.log(torch.tensor(probs)).expand(n, -1))


def test_categorical_kl_known_values():
    p, q = _student(P, 1), _student(Q, 1)
    assert p.kl_divergence(q).item() == pytest.approx(KL_PQ, abs=1e-6)
    assert q.kl_divergence(p).item() == pytest.approx(KL_QP, abs=1e-6)


@pytest.mark.parametrize("direction,expected", [("forward", KL_PQ), ("reverse", KL_QP)])
def test_kickstart_direction(direction, expected):
    T, B = 3, 2
    ks = KickstartLoss(FixedTeacher(P), initial_lambda=1.0, decay_steps=10, direction=direction)
    obs = torch.zeros(T, B, 1)
    dones = torch.zeros(T, B, dtype=torch.bool)
    loss = ks.compute(_student(Q, T * B), obs, dones, None, None)
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_kickstart_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        KickstartLoss(FixedTeacher(P), direction="sideways")
    assert TrainingConfig().kickstart_kl == "forward"
    assert TrainingConfig(kickstart_kl="reverse").kickstart_kl == "reverse"
    with pytest.raises(ValidationError):
        TrainingConfig(kickstart_kl="sideways")


MASK = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)


@pytest.mark.parametrize("case", ["none", "teacher_only", "student_only", "both", "disjoint"])
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_masked_kl_is_finite_with_finite_grad(case, direction):
    torch.manual_seed(0)
    student_logits = torch.randn(3, 4, requires_grad=True)
    teacher_logits = torch.randn(3, 4)
    s_mask = {"none": None, "teacher_only": None, "student_only": MASK, "both": MASK, "disjoint": ~MASK}[case]
    t_mask = {"none": None, "teacher_only": MASK, "student_only": None, "both": MASK, "disjoint": MASK}[case]
    student = CategoricalDist(student_logits, s_mask)
    teacher = CategoricalDist(teacher_logits, t_mask)
    kl = teacher.kl_divergence(student) if direction == "forward" else student.kl_divergence(teacher)
    assert kl.shape == (3,)
    assert torch.isfinite(kl).all()
    assert (kl >= -1e-6).all()
    kl.sum().backward()
    assert torch.isfinite(student_logits.grad).all()
    if case == "disjoint":
        assert torch.equal(kl, torch.zeros(3))


def test_masked_kl_renormalizes_over_legal_actions():
    p = torch.softmax(torch.tensor([1.0, 2.0, 0.0]), 0)
    q = torch.softmax(torch.tensor([0.0, 0.0, 5.0]), 0)
    mask = torch.tensor([[True, True, False]])
    kl = CategoricalDist(torch.log(p)[None], mask).kl_divergence(CategoricalDist(torch.log(q)[None]))
    pp, qq = p[:2] / p[:2].sum(), q[:2] / q[:2].sum()
    assert kl.item() == pytest.approx((pp * (pp / qq).log()).sum().item(), abs=1e-6)


def test_masked_kl_of_identical_distributions_is_zero():
    logits = torch.randn(3, 4)
    kl = CategoricalDist(logits, MASK).kl_divergence(CategoricalDist(logits.clone(), MASK))
    assert torch.equal(kl, torch.zeros(3))


def test_apply_mask_combines_masks():
    dist = CategoricalDist(torch.zeros(1, 4), torch.tensor([[True, True, True, False]]))
    both = dist.apply_mask(torch.tensor([[False, True, True, True]]))
    assert both.mask.tolist() == [[False, True, True, False]]


def test_composite_kl_with_masks_is_finite():
    def make(logits, mask):
        return CompositeDist({
            "a": CategoricalDist(logits, mask),
            "b": DiagGaussianDist(torch.zeros(3, 2), torch.zeros(3, 2)),
        })
    kl = make(torch.randn(3, 4), MASK).kl_divergence(make(torch.randn(3, 4), None))
    assert torch.isfinite(kl).all()


def test_categorical_cat_concatenates_masks_and_kl_stays_finite():
    torch.manual_seed(0)
    masked = CategoricalDist(torch.randn(3, 4), MASK)
    unmasked = CategoricalDist(torch.randn(2, 4))
    cat = CategoricalDist.cat([masked, unmasked])
    assert cat.mask.tolist() == MASK.tolist() + [[True] * 4] * 2
    assert CategoricalDist.cat([unmasked, unmasked]).mask is None

    other = CategoricalDist(torch.randn(5, 4))
    for kl in (cat.kl_divergence(other), other.kl_divergence(cat)):
        assert kl.shape == (5,)
        assert torch.isfinite(kl).all()
    # The masked rows keep their masked KL after concatenation.
    torch.testing.assert_close(cat.kl_divergence(other)[:3], masked.kl_divergence(CategoricalDist(other.logits[:3])))


def test_composite_cat_concatenates_component_masks_and_kl_stays_finite():
    torch.manual_seed(0)

    def make(n, mask):
        return CompositeDist({
            "a": CategoricalDist(torch.randn(n, 4), mask),
            "b": DiagGaussianDist(torch.zeros(n, 2), torch.zeros(n, 2)),
        })

    cat = CompositeDist.cat([make(3, MASK), make(3, ~MASK)])
    assert cat._dists["a"].mask.tolist() == MASK.tolist() + (~MASK).tolist()
    kl = cat.kl_divergence(make(6, None))
    assert kl.shape == (6,)
    assert torch.isfinite(kl).all()


def test_lambda_decays_linearly_and_round_trips():
    ks = KickstartLoss(FixedTeacher(P), initial_lambda=2.0, decay_steps=4)
    assert ks.current_lambda == pytest.approx(2.0)
    ks.step()
    ks.step()
    assert ks.current_lambda == pytest.approx(1.0)
    state = ks.state_dict()
    assert state == {"step": 2}
    other = KickstartLoss(FixedTeacher(P), initial_lambda=2.0, decay_steps=4)
    other.load_state_dict(state)
    assert other.current_lambda == pytest.approx(1.0)
    for _ in range(10):
        other.step()
    assert other.current_lambda == 0.0
    loss = other.compute(_student(Q, 2), torch.zeros(1, 2, 1), torch.zeros(1, 2, dtype=torch.bool), None)
    assert loss.item() == 0.0


def test_kickstart_teacher_is_frozen():
    teacher = _masked_toy_model()
    KickstartLoss(teacher)
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not teacher.training


def test_kickstart_lstm_masked_teacher_equal_to_student_gives_zero_kl():
    model = _masked_toy_model(core="lstm")
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    # Chunks of 8 over 5-step episodes: dones inside chunks and chunks starting mid-episode.
    assert any(bool(torch.as_tensor(c.dones).any()) for c in chunks)
    assert all(c.initial_state is not None for c in chunks)
    teacher = copy.deepcopy(model)
    algo = APPO(
        model, AlgorithmConfig(num_epochs=1, minibatch_chunks=0, learning_rate=1e-2),
        device="cpu", kickstart=KickstartLoss(teacher, initial_lambda=1.0, decay_steps=100),
    )
    first = algo.train_step(chunks)
    # Same weights, same initial states, same masks and resets -> KL is exactly 0.
    assert first["kickstart_loss"] < 1e-6
    second = algo.train_step(chunks)
    assert second["kickstart_loss"] > 1e-8
    assert all(math.isfinite(v) for v in second.values())


def test_appo_rejects_teacher_with_different_state_layout():
    with pytest.raises(ValueError, match="state layout"):
        APPO(_masked_toy_model(core="lstm"), AlgorithmConfig(), device="cpu",
             kickstart=KickstartLoss(_masked_toy_model(core="none")))


@pytest.mark.gpu
def test_kickstart_forward_kl_on_cuda_with_masks_and_lstm():
    model = _masked_toy_model(core="lstm")
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    assert all(c.action_masks is not None for c in chunks)
    teacher = _masked_toy_model(core="lstm")
    ks = KickstartLoss(teacher, initial_lambda=1.0, decay_steps=100, direction="forward")
    algo = APPO(model, AlgorithmConfig(num_epochs=1, minibatch_chunks=0), device="cuda", kickstart=ks)
    assert next(teacher.parameters()).device.type == "cuda"
    assert next(algo.model.parameters()).device.type == "cuda"
    metrics = algo.train_step(chunks)
    assert metrics["kickstart_loss"] > 0
    assert all(math.isfinite(v) for v in metrics.values())
