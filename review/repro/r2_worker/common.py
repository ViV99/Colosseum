"""Shared helpers for R2 worker repros."""
import queue, threading, sys
sys.path.insert(0, "/home/viv/dev/repos/Colosseum")
sys.path.insert(0, "/home/viv/dev/repos/Colosseum/tests")
import numpy as np, torch, torch.nn as nn, gymnasium
from colosseum.envs.base_env import BaseEnv
from colosseum.worker.rollout_worker import rollout_worker_process
from colosseum.networks.actor_critic import ActorCriticNetwork
from helpers import SimpleEncoder, SimplePolicy, SimpleValue

OBS_DIM = 4

class StepEnv(BaseEnv):
    """Deterministic 2-player env. obs[p] = [ep_id, t, p, 0]; reward[p] = 100*ep + t + 0.1*p
    (t = index of the step that PRODUCED the reward, i.e. obs t was acted on).
    Episode length L; ends with terminated (or truncated if trunc=True).
    turn_based=True: info[p]['active'] = (t % 2 == p); final reward of -1/+1 given at
    last step to BOTH players (loser inactive)."""
    def __init__(self, L=5, trunc=False, turn_based=False, mask=False):
        self.L, self.trunc, self.turn_based, self.mask = L, trunc, turn_based, mask
        self.ep = -1; self.t = 0
    @property
    def num_players(self): return 2
    @property
    def observation_space(self): return gymnasium.spaces.Box(-1e9, 1e9, (OBS_DIM,), np.float32)
    @property
    def action_space(self): return gymnasium.spaces.Discrete(4)
    def _obs(self):
        return {p: np.array([self.ep, self.t, p, 0], np.float32) for p in range(2)}
    def _info(self):
        d = {}
        for p in range(2):
            i = {}
            if self.turn_based: i["active"] = (self.t % 2 == p)
            if self.mask: i["action_mask"] = np.array([True, True, True, self.t < self.L - 1])
            d[p] = i
        return d
    def reset(self, seed=None):
        self.ep += 1; self.t = 0
        return self._obs(), self._info()
    def step(self, actions):
        t = self.t
        rew = {p: 100 * self.ep + t + 0.1 * p for p in range(2)}
        self.t += 1
        done = self.t >= self.L
        if done and self.turn_based:
            mover = t % 2
            rew = {mover: 1.0, 1 - mover: -1.0}
        term = {p: done and not self.trunc for p in range(2)}
        tr = {p: done and self.trunc for p in range(2)}
        info = self._info()
        if done:
            for p in range(2):
                info[p]["rank"] = 1 if (not self.turn_based or p == t % 2) else 2
        return self._obs(), rew, term, tr, info

def make_net(hidden=16, recurrent=None):
    enc = SimpleEncoder(OBS_DIM, hidden)
    rec = None
    lat = hidden
    if recurrent == "lstm":
        rec = nn.LSTM(hidden, 8, 1); lat = 8
    return ActorCriticNetwork(enc, SimplePolicy(lat, 4), SimpleValue(lat), rec)

class ListQ:
    def __init__(self): self.items = []
    def put(self, x, timeout=None): self.items.append(x)
    def put_nowait(self, x): self.items.append(x)
    def get_nowait(self): raise queue.Empty
    def cancel_join_thread(self): pass

def run_worker(env_fn, steps, num_envs=1, agent_ids=("a",), net_fn=make_net, chunk_length=4,
               results=True, **kw):
    tq = {a: ListQ() for a in agent_ids}
    wq = {a: ListQ() for a in agent_ids}
    rq = ListQ() if results else None
    torch.manual_seed(0)
    nets = {a: (lambda: net_fn()) for a in agent_ids}
    rollout_worker_process(0, env_fn, num_envs, chunk_length, list(agent_ids), nets, tq, wq,
                           threading.Event(), total_timesteps=steps, results_queue=rq, seed=0, **kw)
    return tq, rq
