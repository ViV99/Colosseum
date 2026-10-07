"""Compare compute_vtrace against a slow reference implementation (Espeholt et al. 2018 eq. 1)."""
import torch
from colosseum.algorithms.vtrace import compute_vtrace
torch.manual_seed(0)

def ref_vtrace(blp, tlp, r, v, boot, d, gamma, rho_bar, c_bar):
    T, B = r.shape
    rho = torch.exp(tlp - blp)
    crho = rho.clamp(max=rho_bar); c = rho.clamp(max=c_bar)
    vs = torch.zeros(T, B, dtype=torch.float64); adv = torch.zeros(T, B, dtype=torch.float64)
    for b in range(B):
        def V(t):  # value of state t, within episode chain
            return boot[b] if t == T else v[t, b]
        for s in range(T):
            acc = 0.0
            for t in range(s, T):
                # product of discounts & c_i for i in [s, t-1], break if done in between
                prod = 1.0; cut = False
                for i in range(s, t):
                    if d[i, b]:
                        cut = True; break
                    prod *= gamma * c[i, b]
                if cut: break
                nd = 0.0 if d[t, b] else 1.0
                delta = crho[t, b] * (r[t, b] + gamma * nd * V(t + 1) - v[t, b])
                acc += prod * delta
            vs[s, b] = v[s, b] + acc
        for s in range(T):
            nd = 0.0 if d[s, b] else 1.0
            vnext = boot[b] if s == T - 1 else vs[s + 1, b]
            adv[s, b] = crho[s, b] * (r[s, b] + gamma * nd * vnext - v[s, b])
    return vs, adv

T, B = 12, 5
for trial in range(5):
    blp = torch.randn(T, B, dtype=torch.float64) * 0.5 - 1
    tlp = blp + torch.randn(T, B, dtype=torch.float64) * 0.7
    r = torch.randn(T, B, dtype=torch.float64)
    v = torch.randn(T, B, dtype=torch.float64)
    boot = torch.randn(B, dtype=torch.float64)
    d = (torch.rand(T, B) < 0.2).double()
    for rb, cb in [(1.0, 1.0), (2.0, 0.5)]:
        vs, adv = compute_vtrace(blp, tlp, r, v, boot, d, 0.97, rb, cb)
        rvs, radv = ref_vtrace(blp, tlp, r, v, boot, d.bool(), 0.97, rb, cb)
        print(f"trial {trial} rho_bar={rb} c_bar={cb}: max|vs-ref|={ (vs-rvs).abs().max():.2e}  max|adv-ref|={(adv-radv).abs().max():.2e}")

# Truncation bias: constant reward 1 per step, infinite-horizon value = 1/(1-g)=100.
# Episode truncated by time limit at t=T-1 -> 'done'=1 -> target collapses to r.
T, B = 8, 1
r = torch.ones(T, B); v = torch.full((T, B), 100.0); boot = torch.tensor([100.0])
lp = torch.zeros(T, B)
d = torch.zeros(T, B); d[-1] = 1
vs, adv = compute_vtrace(lp, lp, r, v, boot, d, 0.99)
print("truncation: vs at last step =", vs[-1].item(), "(true value of non-terminal state ~100); adv =", adv[-1].item())
