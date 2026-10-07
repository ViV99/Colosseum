from common import *
from colosseum.envs.vec_env import VectorEnv
class E(StepEnv):
    def reset(self, seed=None):
        o, _ = super().reset(seed); return o, {p: {} for p in range(2)}   # reset info has no mask
    def step(self, a):
        o, r, te, tr, i = super().step(a)
        for p in range(2): i[p]["action_mask"] = np.array([self.t < self.L, True, True, True]); i[p]["outcome"] = 0.123
        return o, r, te, tr, i
v = VectorEnv(lambda: E(L=2), 1); v.reset_all()
for s in range(2):
    obs, r, te, tr, infos = v.step(np.zeros((1, 2), np.int64))
print("done:", te, "obs after reset (ep,t):", obs[0, 0, :2], "info used as PRE-step info for next episode:", {k: (v_ if k != 'terminal_info' else '...') for k, v_ in infos[0][0].items() if k != 'terminal_observation'})
