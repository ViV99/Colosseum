import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(2)
def p_of(d): return 1/(1+10**(-d/400))
print("== Online Elo noise: 2 equal players, N games, shuffled; std of final rating diff")
for K in (4,8,16,32,64):
    fin=[]
    for _ in range(400):
        ra=rb=1000.0
        for g in range(2000):
            s=1.0 if rng.random()<0.5 else 0.0
            e=p_of(ra-rb); ra+=K*(s-e); rb-=K*(s-e)
        fin.append(ra-rb)
    print(f"K={K}: std of (ra-rb) after 2000 games = {np.std(fin):.1f} Elo (so single-player std ~ half)")
print("BT-MLE std for N games, p=.5: ", {N: round(400/np.log(10)*np.sqrt(4/N),1) for N in (200,1000,2000,10000)})

print("\n== RPS + Nash averaging with clones (logit scale in Elo units: A_ij = 400*log10(p/(1-p)))")
def elo_matrix(P):
    P=np.clip(P,1e-3,1-1e-3); return 400*np.log10(P/(1-P))
# Agents: R,P,S cyclic with 75% wins; W weak agent: loses to all 90%.
names=["R","P","S","W"]
P=np.full((4,4),0.5)
beats={(0,2):0.75,(1,0):0.75,(2,1):0.75}  # R beats S, P beats R, S beats P
for (i,j),p in beats.items(): P[i,j]=p;P[j,i]=1-p
for i in range(3): P[i,3]=0.9;P[3,i]=0.1
A=elo_matrix(P)
def maxent_nash(A):
    n=len(A); x0=np.ones(n)/n
    cons=[{'type':'eq','fun':lambda p: p.sum()-1},{'type':'ineq','fun':lambda p: -(A@p)/100.0}]
    r=minimize(lambda p: np.sum(p*np.log(p+1e-12)), x0, bounds=[(0,1)]*n, constraints=cons, method='SLSQP',options={'maxiter':500,'ftol':1e-12})
    return r.x
def report(A,names):
    elo=A.mean(axis=1); p=maxent_nash(A); nash=A@p
    print("names",names); print(" Elo(div) ",np.round(elo,1)); print(" Nash p   ",np.round(p,3)); print(" Nash avg ",np.round(nash,1))
report(A,names)
# clone W 10 times -> Elo of cyclic agents inflates relative
idx=[0,1,2]+[3]*11; A2=A[np.ix_(idx,idx)]; 
report(A2,[names[i] for i in idx])
# weak agents inflate Elo of RPS? Compare elo of R before/after
print(" Elo(R) before %.1f after adding 10 clones of W %.1f" % (A.mean(1)[0], A2.mean(1)[0]))
# add a strong clone of R (R2) -> check
