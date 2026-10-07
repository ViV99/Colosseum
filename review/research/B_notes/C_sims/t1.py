from openskill.models import PlackettLuce, BradleyTerryFull, BradleyTerryPart, ThurstoneMostellerFull
import inspect
m = PlackettLuce()
print("defaults", m.mu, m.sigma, m.beta, m.tau, m.kappa)
a,b,c,d = [m.rating(name=n) for n in "abcd"]
print(a, a.ordinal() if hasattr(a,'ordinal') else None)
# 1v1
[[a1],[b1]] = m.rate([[a],[b]])
print("1v1 win:", a1.mu, a1.sigma, b1.mu, b1.sigma)
# FFA ranks (lower=better), tie for 2nd
r = m.rate([[a],[b],[c],[d]], ranks=[1,2,2,4])
print([ (x[0].mu, x[0].sigma) for x in r])
# scores
r = m.rate([[a],[b],[c]], scores=[10,5,5])
print([ (x[0].mu, round(x[0].sigma,3)) for x in r])
print("predict_win", m.predict_win([[a],[b]]), m.predict_draw([[a],[b]]))
print("predict_rank", m.predict_rank([[a],[b],[c]]))
print("ordinal", a1.ordinal(), a1.ordinal(z=2.0), a1.ordinal(alpha=1, target=0))
import openskill; print(openskill.__version__ if hasattr(openskill,'__version__') else '')
# draws after many games for fresh
for M in (PlackettLuce, BradleyTerryFull, BradleyTerryPart, ThurstoneMostellerFull):
    mm=M(); x,y=mm.rating(),mm.rating()
    [[x1],[y1]] = mm.rate([[x],[y]])
    print(M.__name__, round(x1.mu,3), round(x1.sigma,3), round(y1.mu,3))
print(inspect.signature(PlackettLuce.rate))
