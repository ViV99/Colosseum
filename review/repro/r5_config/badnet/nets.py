import torch.nn as nn
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import CategoricalDist
class Enc(BaseEncoder):
    def __init__(self, **kw):
        super().__init__(); self.net = nn.Sequential(nn.Flatten(), nn.Linear(27, 64))
    @property
    def latent_dim(self): return 64
    def forward(self, o): return self.net(o)
class Pol(BasePolicy):
    def __init__(self, **kw):
        super().__init__(); self.fc = nn.Linear(32, 9)   # WRONG: latent is 64
    def forward(self, z): return CategoricalDist(logits=self.fc(z))
class Pol10(BasePolicy):
    def __init__(self, **kw):
        super().__init__(); self.fc = nn.Linear(64, 10)   # WRONG: 10 actions, env has 9
    def forward(self, z): return CategoricalDist(logits=self.fc(z))
class Val(BaseValue):
    def __init__(self, **kw):
        super().__init__(); self.fc = nn.Linear(64, 1)
    def forward(self, z): return self.fc(z).squeeze(-1)
class ValNoSqueeze(BaseValue):
    def __init__(self, **kw):
        super().__init__(); self.fc = nn.Linear(64, 1)
    def forward(self, z): return self.fc(z)
