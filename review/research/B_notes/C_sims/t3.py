import numpy as np, choix, evalica, inspect
from evalica import Winner
# choix
data=[(0,1),(0,1),(1,0),(1,2),(2,1),(1,2),(0,2),(0,2),(2,0)]  # (winner, loser)
p = choix.ilsr_pairwise(3, data, alpha=0.01); print("ilsr", p)
p2 = choix.opt_pairwise(3, data, alpha=0.01); print("opt", p2)
print("prob 0 beats 1", choix.probabilities([0,1], p)[0])
mean,cov = choix.ep_pairwise(3, data, alpha=0.01); print("ep", mean, np.sqrt(np.diag(cov)))
# evalica
xs=['a','a','b','b','c','a']; ys=['b','b','c','c','a','c']
ws=[Winner.X,Winner.X,Winner.Y,Winner.X,Winner.X,Winner.Draw]
r = evalica.bradley_terry(xs,ys,ws); print(type(r), r.scores)
r = evalica.elo(xs,ys,ws); print(r.scores)
print([n for n in dir(evalica) if not n.startswith('_')])
print(inspect.signature(evalica.bradley_terry)); print(inspect.signature(evalica.elo)); print(inspect.signature(evalica.pairwise_frame))
r = evalica.bradley_terry(xs,ys,ws); print(r.scores.index.tolist() if hasattr(r.scores,'index') else None)
