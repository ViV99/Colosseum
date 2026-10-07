from common import *
class NanEnv(StepEnv):
    def _obs(self):
        o = super()._obs()
        if self.t == 2: o[0][1] = np.nan
        return o
try:
    run_worker(lambda: NanEnv(L=5), steps=10, chunk_length=4)
    print("no crash")
except Exception as e:
    print("worker died on NaN obs:", type(e).__name__, str(e)[:150])
# all-False action mask (e.g. stale terminal mask / player with no legal move)
class NoLegal(StepEnv):
    def _info(self):
        d = super()._info()
        for p in range(2): d[p]["action_mask"] = np.array([self.t != 2] * 4)
        return d
try:
    tq, _ = run_worker(lambda: NoLegal(L=5), steps=10, chunk_length=4)
    lp = torch.cat([c.action_log_probs for c in tq["a"].items])
    print("all-false mask: no crash; logp finite:", bool(torch.isfinite(lp).all()), lp.tolist()[:6])
except Exception as e:
    print("worker died on all-False mask:", type(e).__name__, str(e)[:150])
