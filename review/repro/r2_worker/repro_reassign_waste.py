"""Measure fraction of collected transitions discarded by per-episode re-assignment (pending maps
are applied at episode end; partial buffers are reset when collect flag/agent changes)."""
from common import *
import random; torch.set_num_threads(1)
from colosseum.core.types import WorkerCommand
import colosseum.worker.rollout_worker as rw
class CmdQ:
    """Delivers a fresh random self-play assignment every `every` worker iterations (like a refresh)."""
    def __init__(self, n_envs, every): self.n, self.every, self.i = n_envs, every, 0
    def get_nowait(self):
        self.i += 1
        if self.i % self.every: raise queue.Empty
        col = [[True, random.random() < 0.5] for _ in range(self.n)]
        return WorkerCommand(slot_agent_map=[["a","a"]]*self.n,
                             slot_network_map=[["latest", "latest" if c[1] else "ckpt"] for c in col],
                             collect_mask=col, new_checkpoints={})
discarded = [0]
orig = rw.RolloutBuffer.reset
def counting(self):
    if 0 < self._cursor < self.chunk_length: discarded[0] += self._cursor
    return orig(self)
rw.RolloutBuffer.reset = counting
random.seed(0)
for every in (1000000, 200, 20):
    discarded[0] = 0
    tq, _ = run_worker(lambda: StepEnv(L=9), steps=8*1500, num_envs=8, chunk_length=128, results=False, command_queue=CmdQ(8, every))
    sent = sum(c.chunk_length for c in tq["a"].items)
    print(f"refresh every {every} iters: shipped={sent} transitions, discarded on re-assignment={discarded[0]} ({100*discarded[0]/max(1,sent+discarded[0]):.1f}%)")
