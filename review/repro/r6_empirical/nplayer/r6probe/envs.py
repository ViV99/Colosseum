"""Tiny probe envs for num_players != 2 (outside the repo)."""
import gymnasium, numpy as np, torch.nn as nn
from colosseum.envs.base_env import BaseEnv
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import CategoricalDist


class NPick(BaseEnv):
    """N players each pick 0..4 for 5 steps; reward +1 to the highest unique pick each step (FFA)."""
    def __init__(self, n=3):
        self.n = n; self.t = 0
    @property
    def num_players(self): return self.n
    @property
    def observation_space(self): return gymnasium.spaces.Box(0, 1, (4,), np.float32)
    @property
    def action_space(self): return gymnasium.spaces.Discrete(5)
    def _obs(self): return {i: np.array([self.t / 5, i / self.n, 1, 0], np.float32) for i in range(self.n)}
    def reset(self, seed=None):
        self.t = 0; return self._obs(), {i: {} for i in range(self.n)}
    def step(self, actions):
        self.t += 1
        picks = [int(actions[i]) for i in range(self.n)]
        r = {i: 0.0 for i in range(self.n)}
        uniq = [p for p in picks if picks.count(p) == 1]
        if self.n == 1:
            r[0] = picks[0] / 4.0
        elif uniq:
            r[picks.index(max(uniq))] = 1.0
        d = self.t >= 5
        return self._obs(), r, {i: d for i in range(self.n)}, {i: False for i in range(self.n)}, {i: {} for i in range(self.n)}


class Solo(NPick):
    def __init__(self): super().__init__(1)


class FFA3(NPick):
    def __init__(self): super().__init__(3)


class Enc(BaseEncoder):
    def __init__(self, **kw):
        super().__init__(); self.net = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
    @property
    def latent_dim(self): return 32
    def forward(self, x): return self.net(x)


class Pol(BasePolicy):
    def __init__(self, **kw):
        super().__init__(); self.net = nn.Linear(32, 5)
    def forward(self, z): return CategoricalDist(logits=self.net(z))


class Val(BaseValue):
    def __init__(self, **kw):
        super().__init__(); self.net = nn.Linear(32, 1)
    def forward(self, z): return self.net(z).squeeze(-1)


from examples.tic_tac_toe.env import TicTacToeEnv as _TTT


class TTTActive(_TTT):
    """TicTacToe + info['active'] (whose turn) + info['action_mask'] (legal cells)."""
    def _aug(self, infos):
        legal = self._board == 0
        if not legal.any():
            legal = np.ones(9, bool)
        for i in range(2):
            infos[i]["active"] = (i == self._current_player)
            infos[i]["action_mask"] = legal.copy()
        return infos
    def reset(self, seed=None):
        o, i = super().reset(seed); return o, self._aug(i)
    def step(self, a):
        o, r, t, tr, i = super().step(a); return o, r, t, tr, self._aug(i)
