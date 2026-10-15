import numpy as np
from math import sqrt
from statistics import NormalDist
class norm:
    ppf=staticmethod(NormalDist().inv_cdf)
# sample sizes
def n_one_sample(p1,p0=0.5,alpha=0.05,power=0.8):
    za=norm.ppf(1-alpha/2); zb=norm.ppf(power)
    return ((za*sqrt(p0*(1-p0))+zb*sqrt(p1*(1-p1)))/(p1-p0))**2
for p in [0.55,0.52,0.51,0.60,0.65]:
    print("p",p,"games to detect vs 0.5 (80% power, two-sided 5%):",round(n_one_sample(p)))
# CI half-width to resolve +-: n for halfwidth h at p=.5
for h in [0.05,0.02,0.01]:
    print("halfwidth",h,"n=",round(1.96**2*0.25/h**2))
# draws reduce info: with draw rate d, variance of score per game ~ (0.25 - d/4)... (score 0/.5/1)
for d in [0,0.3,0.6]:
    var=0.25*(1-d)
    print("draw rate",d,"per-game score var",var)
rng=np.random.default_rng(0)
# winner's curse: K snapshots, true win prob vs anchor ~ N(0.5,0.02) clipped, n games each
for K,n in [(20,200),(50,200),(50,1000),(200,200)]:
    sd=0.02
    bias=[];regret=[];re_eval=[]
    for _ in range(2000):
        p=np.clip(rng.normal(0.5,sd,K),0,1)
        w=rng.binomial(n,p)/n
        i=np.argmax(w)
        bias.append(w[i]-p[i]); regret.append(p.max()-p[i])
    print(f"K={K} n={n} true sd {sd}: mean selection bias of max est {np.mean(bias):.4f}, mean regret(true best - chosen) {np.mean(regret):.4f}")
# two-stage: top-3 re-evaluated with fresh 1000 games
K,n=50,200;sd=0.02
reg1=[];reg2=[]
for _ in range(2000):
    p=np.clip(rng.normal(0.5,sd,K),0,1)
    w=rng.binomial(n,p)/n
    reg1.append(p.max()-p[np.argmax(w)])
    top=np.argsort(-w)[:5]
    w2=rng.binomial(1000,p[top])/1000
    reg2.append(p.max()-p[top[np.argmax(w2)]])
print("one-stage regret",np.mean(reg1),"two-stage(top5 +1000 fresh games each)",np.mean(reg2), "extra games",5*1000, "vs stage1",K*n)
# successive halving budget compare: uniform 50*200=10000 + 5000; SH with 15000 total
def sh(p,budget):
    K=len(p);arms=list(range(K));rounds=int(np.ceil(np.log2(K)));wins=np.zeros(K);cnt=np.zeros(K)
    per=budget/rounds
    for r in range(rounds):
        m=int(per//len(arms))
        if m<1: break
        wins[arms]+=rng.binomial(m,p[arms]);cnt[arms]+=m
        est=wins[arms]/cnt[arms]
        keep=max(1,int(np.ceil(len(arms)/2)))
        arms=[arms[j] for j in np.argsort(-est)[:keep]]
    return arms[0]
reg3=[];reg4=[]
for _ in range(2000):
    p=np.clip(rng.normal(0.5,sd,K),0,1)
    reg3.append(p.max()-p[sh(p,15000)])
    w=rng.binomial(300,p)/300
    reg4.append(p.max()-p[np.argmax(w)])
print("SH 15000 games regret",np.mean(reg3),"uniform 300 each (15000)",np.mean(reg4))
