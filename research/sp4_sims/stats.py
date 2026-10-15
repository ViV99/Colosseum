import math, random
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
# games for half-width
for hw in (0.10,0.05,0.03,0.02):
    n=math.ceil(1.96**2*0.25/hw**2); print("halfwidth",hw,"n",n)
# n so that observed 55% has lower Wilson>0.5
for p in (0.55,0.52,0.60,0.65):
    n=10
    while wilson(round(p*n),n)[0]<=0.5: n+=1
    print("obs",p,"n for Wilson lower>0.5 ~",n)
# winner's curse: K identical candidates, n games each, max observed wr
random.seed(1)
def sim(K,n,trials=2000,spread=0.0):
    m=0; tm=0; picked_true=0
    for _ in range(trials):
        tr=[0.5+random.gauss(0,spread) for _ in range(K)]
        ob=[sum(random.random()<p for _ in range(n))/n for p in tr]
        i=max(range(K),key=lambda j:ob[j]); m+=ob[i]; picked_true+=tr[i]; tm+=max(tr)
    return m/trials,picked_true/trials,tm/trials
for K,n in ((10,100),(20,100),(20,400),(20,1000),(50,200)):
    print("identical K",K,"n",n,"mean max observed %.3f"%sim(K,n,500)[0])
for K,n,s in ((20,100,0.03),(20,400,0.03),(20,1000,0.03)):
    a,b,c=sim(K,n,500,s); print("spread sd",s,"K",K,"n",n,"obs max %.3f picked true %.3f best true %.3f"%(a,b,c))
# Elo: win prob vs rating diff
for d in (10,25,50,100,200): print("elo diff",d,"expected wr %.3f"%(1/(1+10**(-d/400))))
# trueskill 8.3 ~ 80%: beta=25/6; 
