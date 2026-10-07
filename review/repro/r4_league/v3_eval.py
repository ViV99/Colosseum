"""Eval statistics, recurrent eval/kickstart bypass, heterogeneous architectures."""
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/viv/dev/repos/Colosseum")
torch.set_num_threads(1)
from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.registry import build_network, import_class
from colosseum.eval import _wilson_ci, evaluate_agents

BASE = "/home/viv/dev/repos/Colosseum/configs/examples/tic_tac_toe.yaml"
cfg = load_config(BASE)

print("=== A. Draw handling in EvalResult (reversed entry) ===")
from colosseum.eval import EvalMatrix
wins_a, wins_b, draws = 10, 10, 80
lo, hi = _wilson_ci(wins_a, 100)
print(f"A: wr={wins_a/100:.2f} CI=[{lo:.3f},{hi:.3f}]  -> reversed B entry: wr={wins_b/100:.2f} "
      f"CI=[{1-hi:.3f},{1-lo:.3f}]  (B's point estimate lies OUTSIDE its CI)")

print("\n=== B. Real eval run, tic-tac-toe, two random-init agents (many draws? first-move advantage) ===")
random.seed(0); torch.manual_seed(0)
env_fn = lambda: import_class(cfg.env.env_class)(**cfg.env.kwargs)
nf = lambda: build_network(cfg)
sds = {n: {"state_dict": nf().state_dict()} for n in ("A", "B")}
m = evaluate_agents(sds, env_fn, nf, num_matches=400, num_envs=8)
for k in [("A", "B"), ("B", "A")]:
    r = m.get(*k)
    print(k, f"wins={r.wins_a}/{r.wins_b} draws={r.draws} wr_a={r.win_rate_a:.3f} CI=[{r.ci_lower_a:.3f},{r.ci_upper_a:.3f}]")

print("\n=== C. Seat (first-mover) advantage in tic-tac-toe with identical agents ===")
# same network in both seats, count seat-0 wins
from colosseum.envs.vec_env import VectorEnv
from colosseum.core.outcomes import player_outcomes
net = nf(); net.eval()
venv = VectorEnv(env_fn, 8)
obs, infos = venv.reset_all()
from colosseum.worker.rollout_worker import _extract_action_masks
ep = np.zeros((8, 2)); res = []
while len(res) < 2000:
    masks = _extract_action_masks(infos, 8, 2, action_spec=venv.action_spec)
    o = torch.tensor(obs.reshape(16, *obs.shape[2:]), dtype=torch.float32)
    mk = None if masks is None else torch.tensor(masks, dtype=torch.bool)
    with torch.no_grad():
        a, _, _, _ = net.act(o, action_mask=mk)
    obs, r, term, trunc, infos = venv.step(a.numpy().reshape(8, 2))
    ep += r
    for e in range(8):
        if term[e] or trunc[e]:
            res.append(player_outcomes(ep[e]))
            ep[e] = 0
res = np.array(res)
print("identical agent self-match: seat0 score=", res[:, 0].mean().round(3), "seat1 score=", res[:, 1].mean().round(3),
      "masks provided:", masks is not None)

print("\n=== D. Recurrent policies in eval/kickstart: LSTM is bypassed when hidden=None ===")
cd = cfg.model_dump()
cd["networks"]["recurrent_type"] = "lstm"
cd["networks"]["recurrent_hidden_size"] = 64
rcfg = ColosseumConfig(**cd)
rnet = build_network(rcfg); rnet.eval()
x = torch.randn(4, 3, 3, 3)
with torch.no_grad():
    d_none, _, h_none = rnet.forward(x, None)
    d_h, _, h_new = rnet.forward(x, rnet.initial_hidden(4))
print("hidden=None returns new_hidden:", h_none, "| max |logit diff| (no-LSTM vs LSTM path):",
      (d_none.logits - d_h.logits).abs().max().item())
print("eval.py calls net.act(obs, action_mask=..., deterministic=...) without hidden -> LSTM never used in eval")

print("\n=== E. Heterogeneous architectures in CLI eval: single network_factory from global cfg ===")
cd = cfg.model_dump(); cd["networks"]["recurrent_type"] = "gru"; cd["networks"]["recurrent_hidden_size"] = 64
other_sd = build_network(ColosseumConfig(**cd)).state_dict()
try:
    build_network(cfg).load_state_dict(other_sd)
    print("loaded OK")
except RuntimeError as e:
    print("CLI eval would fail for an agent with a different architecture:", str(e).splitlines()[0][:120])
