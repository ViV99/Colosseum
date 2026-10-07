import numpy as np
from scipy.stats import norm
rng=np.random.default_rng(0)
def elo_to_p(d): return 1/(1+10**(-d/400))
def p_to_elo(p): return 400*np.log10(p/(1-p))
print("== Games needed to detect dElo (H0: p=0.5, one-sided alpha=0.05, power 0.8; binary outcomes, no draws), normal approx; and with draw ratio 0.3 (variance 0.25*(1-d))")
za,zb=norm.ppf(0.95),norm.ppf(0.8)
for dE in [5,10,20,30,50,100,200]:
    p=elo_to_p(dE)
    for d in (0.0,0.3,0.6):
        # score variance per game under H0: 0.25*(1-d) ; under H1 approx same
        s=np.sqrt(0.25*(1-d))
        n=((za+zb)*s/(p-0.5))**2
        print(f"dElo={dE:4d} p={p:.4f} draw={d:.1f} n={n:9.0f}", end=" | ")
    print()
print("\n== 95% CI half-width in Elo (binary, p=0.5) for N games, Wilson")
def wilson(k,n,z=1.96):
    ph=k/n; den=1+z*z/n; c=(ph+z*z/(2*n))/den; h=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return c-h,c+h
for n in [50,100,200,400,1000,2000,5000,10000,50000]:
    lo,hi=wilson(n//2,n); print(n, round(p_to_elo(lo),1), round(p_to_elo(hi),1), "halfwidth Elo", round((p_to_elo(hi)-p_to_elo(lo))/2,1))
print("\n== SPRT expected games, normalized Elo, alpha=beta=0.05: T=1046535/(en1-en0)^2 (worst case); draw=0 so normalized=logistic")
for (e0,e1) in [(0,2),(0,5),(0,10),(0,20),(0,50),(-1.75,0.25)]:
    print((e0,e1), round(1046535/(e1-e0)**2))
print("\n== SPRT trinomial/binomial LLR bounds")
for a,b in [(0.05,0.05),(0.01,0.05),(0.05,0.2),(0.01,0.01)]:
    print(a,b, "lower",round(np.log(b/(1-a)),4),"upper",round(np.log((1-b)/a),4))
