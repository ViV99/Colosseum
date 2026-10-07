import numpy as np
from open_spiel.python.egt import alpharank
# RPS payoff (symmetric, zero-sum), plus a weak 4th
import inspect
print(inspect.signature(alpharank.compute))
print(inspect.signature(alpharank.sweep_pi_vs_alpha))
M = np.array([[0,-1,1],[1,0,-1],[-1,1,0]],dtype=float)
rhos, rho_m, pi, nprof, nstrat = alpharank.compute([M], alpha=1e2)
print("RPS pi", pi)
M4 = np.zeros((4,4)); M4[:3,:3]=M; M4[:3,3]=1; M4[3,:3]=-1
rhos, rho_m, pi, nprof, nstrat = alpharank.compute([M4], alpha=1e2, m=50)
print("RPS+weak pi", np.round(pi,4))
