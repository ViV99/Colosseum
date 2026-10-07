import numpy as np
from scipy.stats import binom
from scipy.optimize import minimize
rng=np.random.default_rng(1)
def p_of(d): return 1/(1+10**(-d/400))
def elo_of(p): return 400*np.log10(p/(1-p))
print("== Gating pass probabilities: AGZ 400 games, pass if wins>=220 (55%); KataGo 200 games, wins>=100")
for dE in [-20,-10,0,10,20,35,50,70,100]:
    p=p_of(dE)
    print(f"true dElo={dE:4d} p={p:.3f}  AGZ-pass={binom.sf(219,400,p):.3f}  KataGo-pass={binom.sf(99,200,p):.3f}")
print("55% win rate = dElo", round(elo_of(0.55),1), "; 60% =",round(elo_of(0.60),1))

print("\n== Binary SPRT sim (no draws), alpha=beta=0.05")
def sprt(elo0,elo1,true_d,a=0.05,b=0.05,maxn=2_000_000):
    p0,p1,p=p_of(elo0),p_of(elo1),p_of(true_d)
    lw=np.log(p1/p0); ll=np.log((1-p1)/(1-p0))
    lo,hi=np.log(b/(1-a)),np.log((1-b)/a)
    llr=0.0;n=0
    while lo<llr<hi and n<maxn:
        # draw in blocks
        blk=rng.random(64)<p
        for w in blk:
            llr+= lw if w else ll; n+=1
            if not (lo<llr<hi): break
    return llr>=hi, n
for (e0,e1,td) in [(0,10,0),(0,10,5),(0,10,10),(0,20,0),(0,20,10),(0,20,20),(0,50,0),(0,50,25),(0,50,50)]:
    res=[sprt(e0,e1,td) for _ in range(300)]
    acc=np.mean([r[0] for r in res]); N=np.mean([r[1] for r in res])
    print(f"H0={e0} H1={e1} true={td}: P(accept H1)={acc:.3f} mean games={N:.0f}")

print("\n== Winner's curse: K identical-strength checkpoints (true winrate vs benchmark = 0.5), each evaluated with n games; expected max observed winrate; and K ckpts with true Elo ~ N(0,sd)")
def wc(K,n,sd_elo,reps=2000):
    bias=[];correct=[]
    for _ in range(reps):
        d=rng.normal(0,sd_elo,K); p=p_of(d)
        obs=rng.binomial(n,p)/n
        i=np.argmax(obs); bias.append(elo_of(np.clip(obs[i],1e-3,1-1e-3))-d[i]); correct.append(i==np.argmax(d))
    return np.mean(bias),np.mean(correct)
for K in (5,20,100):
    for n in (100,400,1600):
        for sd in (0,20,50):
            b,c=wc(K,n,sd); print(f"K={K:3d} n={n:5d} sdElo={sd:3d}: mean overestimate of selected = {b:6.1f} Elo; P(pick true best)={c:.2f}")
