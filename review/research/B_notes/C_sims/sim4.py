import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(3)
def p_of(d): return 1/(1+10**(-d/400))
C=100; delta=10.0; true=np.arange(C)*delta
G=40  # games per new checkpoint
def run(scheme,K=16,window=None,seed=0):
    rng=np.random.default_rng(seed)
    r=np.zeros(C); games=[]
    for i in range(1,C):
        r[i]=r[i-1] if scheme in("online","frozen") else 0   # inherit from parent
        lo=0 if window is None else max(0,i-window)
        for g in range(G):
            j=rng.integers(lo,i)
            s=1.0 if rng.random()<p_of(true[i]-true[j]) else 0.0
            games.append((i,j,s))
            e=p_of(r[i]-r[j])
            r[i]+=K*(s-e)
            if scheme=="online": r[j]-=K*(s-e)
    if scheme=="bt":
        W=np.zeros((C,C)); 
        for i,j,s in games: W[i,j]+=s; W[j,i]+=1-s
        def nll(x):
            x=np.concatenate([[0],x]); d=(x[:,None]-x[None,:])*np.log(10)/400
            lp=-np.logaddexp(0,-d); l=W*lp; return -l.sum()
        def grad(x):
            xf=np.concatenate([[0],x]); d=(xf[:,None]-xf[None,:])*np.log(10)/400
            p=1/(1+np.exp(-d)); N=W+W.T
            g=-(W-N*p).sum(1)*np.log(10)/400   # d nll/d x_i
            return g[1:]
        res=minimize(nll,np.zeros(C-1),jac=grad,method="L-BFGS-B"); r=np.concatenate([[0],res.x])
    return r
for name,kw in [("online K=16",dict(scheme="online",K=16)),("online K=32",dict(scheme="online",K=32)),("frozen-opp K=16 (only new ckpt updates)",dict(scheme="frozen",K=16)),("frozen-opp K=32",dict(scheme="frozen",K=32)),("BT-MLE batch",dict(scheme="bt"))]:
    for w in (None,10):
        errs=[];tot=[]
        for sd in range(5):
            r=run(window=w,seed=sd,**kw)
            r=r-r[0]; errs.append(np.sqrt(np.mean((r-true)**2))); tot.append(r[-1])
        print(f"{name:42s} window={str(w):5s}: RMSE vs truth={np.mean(errs):6.1f} Elo; est range of last ckpt={np.mean(tot):7.1f} (true {true[-1]:.0f})")
