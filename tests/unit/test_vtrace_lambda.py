"""V-trace(lambda): c_t = lambda * min(c_bar, rho_t) (T4.1)."""
import pytest
import torch
from pydantic import ValidationError

import colosseum.algorithms.appo as appo_module
from colosseum.algorithms.appo import APPO
from colosseum.algorithms.vtrace import compute_vtrace
from colosseum.core.config import AlgorithmConfig
from helpers import REPO_ROOT, MaskedToyEnv, make_simple_model, rollout_chunks


def reference_vtrace(blp, tlp, r, v, boot, d, gamma, rho_bar, c_bar, lam):
    """Slow per-element V-trace (Espeholt et al. 2018, eq. 1) with c_i = lam * min(c_bar, rho_i)."""
    T, B = r.shape
    rho = torch.exp(tlp - blp)
    crho = rho.clamp(max=rho_bar)
    c = lam * rho.clamp(max=c_bar)
    vs = torch.zeros(T, B, dtype=torch.float64)
    adv = torch.zeros(T, B, dtype=torch.float64)
    for b in range(B):
        def value(t, b=b):
            return boot[b] if t == T else v[t, b]

        for s in range(T):
            acc = 0.0
            for t in range(s, T):
                prod, cut = 1.0, False
                for i in range(s, t):
                    if d[i, b]:
                        cut = True
                        break
                    prod *= gamma * c[i, b]
                if cut:
                    break
                not_done = 0.0 if d[t, b] else 1.0
                acc += prod * crho[t, b] * (r[t, b] + gamma * not_done * value(t + 1) - v[t, b])
            vs[s, b] = v[s, b] + acc
        for s in range(T):
            not_done = 0.0 if d[s, b] else 1.0
            v_next = boot[b] if s == T - 1 else vs[s + 1, b]
            adv[s, b] = crho[s, b] * (r[s, b] + gamma * not_done * v_next - v[s, b])
    return vs, adv


def gae(r, v, boot, d, gamma, lam):
    T, B = r.shape
    out = torch.zeros(T, B, dtype=r.dtype)
    last = torch.zeros(B, dtype=r.dtype)
    for t in reversed(range(T)):
        not_done = 1.0 - d[t].to(r.dtype)
        v_next = boot if t == T - 1 else v[t + 1]
        delta = r[t] + gamma * not_done * v_next - v[t]
        last = delta + gamma * lam * not_done * last
        out[t] = last
    return out


@pytest.mark.parametrize("lam", [0.5, 0.95, 1.0])
@pytest.mark.parametrize("rho_bar,c_bar", [(1.0, 1.0), (2.0, 0.5)])
def test_vtrace_lambda_matches_reference(lam, rho_bar, c_bar):
    g = torch.Generator().manual_seed(0)
    T, B = 12, 4
    blp = torch.randn(T, B, generator=g, dtype=torch.float64) * 0.5 - 1.0
    tlp = blp + torch.randn(T, B, generator=g, dtype=torch.float64) * 0.7
    r = torch.randn(T, B, generator=g, dtype=torch.float64)
    v = torch.randn(T, B, generator=g, dtype=torch.float64)
    boot = torch.randn(B, generator=g, dtype=torch.float64)
    d = torch.rand(T, B, generator=g, dtype=torch.float64) < 0.2
    assert d.any()
    vs, adv = compute_vtrace(blp, tlp, r, v, boot, d, gamma=0.97, rho_bar=rho_bar, c_bar=c_bar, lam=lam)
    ref_vs, ref_adv = reference_vtrace(blp, tlp, r, v, boot, d, 0.97, rho_bar, c_bar, lam)
    torch.testing.assert_close(vs, ref_vs, atol=1e-10, rtol=0)
    torch.testing.assert_close(adv, ref_adv, atol=1e-10, rtol=0)


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
def test_on_policy_vtrace_lambda_equals_gae(lam):
    g = torch.Generator().manual_seed(1)
    T, B = 10, 3
    lp = torch.zeros(T, B, dtype=torch.float64)
    r = torch.randn(T, B, generator=g, dtype=torch.float64)
    v = torch.randn(T, B, generator=g, dtype=torch.float64)
    boot = torch.randn(B, generator=g, dtype=torch.float64)
    d = torch.rand(T, B, generator=g, dtype=torch.float64) < 0.2
    vs, _ = compute_vtrace(lp, lp, r, v, boot, d, gamma=0.97, lam=lam)
    torch.testing.assert_close(vs - v, gae(r, v, boot, d, 0.97, lam), atol=1e-10, rtol=0)


def test_default_lambda_is_plain_vtrace():
    g = torch.Generator().manual_seed(2)
    T, B = 6, 2
    blp = torch.randn(T, B, generator=g)
    tlp = torch.randn(T, B, generator=g)
    r, v = torch.randn(T, B, generator=g), torch.randn(T, B, generator=g)
    boot = torch.randn(B, generator=g)
    d = torch.zeros(T, B)
    a = compute_vtrace(blp, tlp, r, v, boot, d)
    b = compute_vtrace(blp, tlp, r, v, boot, d, lam=1.0)
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


def test_config_has_vtrace_lambda_and_no_gae_lambda():
    assert "gae_lambda" not in AlgorithmConfig.model_fields
    assert AlgorithmConfig().vtrace_lambda == 1.0
    assert AlgorithmConfig(vtrace_lambda=0.9).vtrace_lambda == 0.9
    with pytest.raises(ValidationError):
        AlgorithmConfig(vtrace_lambda=1.5)
    with pytest.raises(ValidationError):
        AlgorithmConfig(vtrace_lambda=-0.1)


def test_no_gae_lambda_left_in_example_configs():
    offenders = [p.name for p in (REPO_ROOT / "configs").rglob("*.yaml") if "gae_lambda" in p.read_text()]
    assert offenders == []


def test_appo_passes_vtrace_lambda(monkeypatch):
    seen = []
    real = appo_module.compute_vtrace

    def spy(*args, **kwargs):
        seen.append(kwargs.get("lam"))
        return real(*args, **kwargs)

    monkeypatch.setattr(appo_module, "compute_vtrace", spy)
    model = make_simple_model(obs_dim=4, num_actions=4, seed=0)
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=2, chunk_length=8)
    algo = APPO(model, AlgorithmConfig(vtrace_lambda=0.7), device="cpu")
    algo.train_step(chunks)
    assert seen and all(lam == 0.7 for lam in seen)
