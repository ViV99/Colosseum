import sys, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
sys.path.insert(0, "/home/viv/dev/repos/Colosseum/tests")
from helpers import SimpleEncoder, SimplePolicy, SimpleValue, make_simple_network
from colosseum.networks.actor_critic import ActorCriticNetwork
from colosseum.networks.normalization import NormalizeObs
from colosseum.networks.base import BaseEncoder
from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.types import TrajectoryChunk
from colosseum.bc.kickstart import KickstartLoss
torch.manual_seed(0)
OBS, A, T = 8, 4, 16

def chunk(T=T):
    return TrajectoryChunk("a", torch.randn(T, OBS), torch.randint(0, A, (T,)), torch.full((T,), -1.386),
                           torch.randn(T), torch.zeros(T), torch.randn(T), torch.tensor(0.0), 0)

# 1) mixed-dtype mse_loss (what happens under AMP fp16 when value head returns half)
v = torch.randn(10, dtype=torch.bfloat16, requires_grad=True)
try:
    F.mse_loss(v, torch.randn(10)).backward(); print("mse_loss(bf16 pred, fp32 target).backward(): OK, grad dtype", v.grad.dtype)
except Exception as e:
    print("mse_loss(bf16, fp32) FAILED:", type(e).__name__, str(e)[:150])

# 2) NormalizeObs: how many samples counted per train_step
class NEnc(BaseEncoder):
    def __init__(s):
        super().__init__(); s.norm = NormalizeObs((OBS,)); s.fc = nn.Linear(OBS, 16)
    @property
    def latent_dim(s): return 16
    def forward(s, o): return torch.relu(s.fc(s.norm(o)))
def nnet(): return ActorCriticNetwork(NEnc(), SimplePolicy(16, A), SimpleValue(16))
for epochs, mb, ks in [(1, 0, False), (4, 1, False), (1, 0, True)]:
    n = nnet()
    kick = KickstartLoss(nnet()) if ks else None
    appo = APPO(n, AlgorithmConfig(num_epochs=epochs, minibatch_chunks=mb), kickstart=kick)
    appo.train_step([chunk() for _ in range(4)])
    print(f"NormalizeObs: epochs={epochs} minibatch_chunks={mb} kickstart={ks}: unique samples=64, rms.count={n.encoder.norm.rms.count.item():.0f}; network.training={n.training}")

# 3) LR schedule + kickstart lambda on resume (mimics learner.py resume path)
n = make_simple_network(OBS, 16, A)
appo = APPO(n, AlgorithmConfig(lr_schedule="linear", learning_rate=1e-3), kickstart=KickstartLoss(make_simple_network(OBS, 16, A), decay_steps=10))
appo.setup_lr_schedule(10)
for _ in range(5): appo.train_step([chunk()])
print("before 'checkpoint': lr =", appo._optimizer.param_groups[0]["lr"], "kickstart lambda =", appo._kickstart.current_lambda)
opt_sd = appo.optimizer_state_dict; sd = n.state_dict()
n2 = make_simple_network(OBS, 16, A); n2.load_state_dict(sd)
appo2 = APPO(n2, AlgorithmConfig(lr_schedule="linear", learning_rate=1e-3), kickstart=KickstartLoss(make_simple_network(OBS, 16, A), decay_steps=10))
appo2._optimizer.load_state_dict(opt_sd); appo2._policy_version = 5
appo2.setup_lr_schedule(10)
print("after resume (learner.py path): lr =", appo2._optimizer.param_groups[0]["lr"], "kickstart lambda =", appo2._kickstart.current_lambda)

# 4) Policy lag: with behavior log-probs from an old policy, fraction of samples clipped on FIRST minibatch
n = make_simple_network(OBS, 16, A)
c = chunk(256)
with torch.no_grad():
    lp, _, _ = n.evaluate_actions(c.observations, c.actions)
c.action_log_probs = lp + 0.0
appo = APPO(n, AlgorithmConfig())
r = appo.compute_loss([c]); print("on-policy: clip_fraction", r["clip_fraction"].item(), "approx_kl", r["approx_kl"].item())

# 5) evaluate_actions_recurrent: per-step loop vs cuDNN-style single pass timing (CPU)
import time
rnn = nn.LSTM(16, 64, 1)
netr = ActorCriticNetwork(SimpleEncoder(OBS, 16), SimplePolicy(64, A), SimpleValue(64), recurrent=rnn)
Tt, B = 256, 32
obs = torch.randn(Tt, B, OBS); act = torch.randint(0, A, (Tt, B)); h = netr.initial_hidden(B)
dn = torch.zeros(Tt, B)
for name, d in [("dones_seq=None (single pass)", None), ("dones_seq given (per-step loop)", dn)]:
    t0 = time.perf_counter()
    for _ in range(3):
        lp, v, e = netr.evaluate_actions_recurrent(obs, act, h, dones_seq=d); (lp.sum() + v.sum()).backward()
    print(f"recurrent fwd+bwd T={Tt} B={B} {name}: {(time.perf_counter()-t0)/3*1000:.1f} ms")

# 6) Offline BC ignores action_masks in data
from colosseum.bc.offline_bc import OfflineBCTrainer
import tempfile, os
d = tempfile.mkdtemp(dir=".")
masks = torch.zeros(64, A, dtype=torch.bool); masks[:, 0] = True
torch.save({"observations": torch.randn(64, OBS), "actions": torch.zeros(64, dtype=torch.long), "action_masks": masks}, os.path.join(d, "x.pt"))
bc = OfflineBCTrainer(make_simple_network(OBS, 16, A)); bc.load_data(d)
print("BC trainer stored masks?", hasattr(bc, "_action_masks"), "; add_data signature:", OfflineBCTrainer.add_data.__code__.co_varnames[:3])
# 7) BC with Box actions and default action_type='discrete'
from colosseum.networks.distributions import DiagGaussianDist
from colosseum.networks.base import BasePolicy
class GP(BasePolicy):
    def __init__(s):
        super().__init__(); s.fc = nn.Linear(16, 2); s.log_std = nn.Parameter(torch.zeros(2))
    def forward(s, l): return DiagGaussianDist(s.fc(l), s.log_std.expand(l.shape[0], 2))
gnet = ActorCriticNetwork(SimpleEncoder(OBS, 16), GP(), SimpleValue(16))
bc = OfflineBCTrainer(gnet)  # default action_type="discrete"
bc.add_data(torch.randn(32, OBS), torch.rand(32, 2) * 0.9)
try:
    print("BC Box actions with default action_type: loss", bc.train(num_epochs=1)["bc_loss"], "(continuous actions in [0,0.9) were .long()'d to 0)")
except Exception as e:
    print("BC Box default action_type FAILED:", type(e).__name__, str(e)[:120])
