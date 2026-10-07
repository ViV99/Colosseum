from common import *
from colosseum.envs.subproc_vec_env import SubprocessVectorEnv
class Boom(StepEnv):
    def step(self, a):
        if self.t == 2: raise ValueError("env bug at t=2")
        return super().step(a)
def mk(): return Boom(L=10)
if __name__ == "__main__":
    import multiprocessing as mp
    v = SubprocessVectorEnv(mk, 2, num_workers=1); v.reset_all()
    try:
        for _ in range(4): v.step(np.zeros((2, 2), np.int64))
    except Exception as e:
        print("parent saw:", type(e).__name__, repr(e)[:120])
    v.close()
