"""Build a chunk exactly the way the worker does and feed it to APPO.train_step.
Also check kickstart / BC / eval-style act() on recurrent nets."""
import sys, traceback, torch, torch.nn as nn, numpy as np
sys.path.insert(0, "/home/viv/dev/repos/Colosseum/tests")
from helpers import SimpleEncoder, SimplePolicy, SimpleValue
from colosseum.networks.actor_critic import ActorCriticNetwork
from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.worker.rollout_worker import RolloutBuffer, _build_chunk, _run_inference_group
from colosseum.bc.kickstart import KickstartLoss
from colosseum.bc.offline_bc import OfflineBCTrainer
torch.manual_seed(0)
OBS, LAT, HID, A, T = 8, 16, 32, 4, 6

def net(rnn="lstm", hid=HID):
    r = nn.LSTM(LAT, hid, 1) if rnn == "lstm" else nn.GRU(LAT, hid, 1)
    return ActorCriticNetwork(SimpleEncoder(OBS, LAT), SimplePolicy(hid, A), SimpleValue(hid), recurrent=r)

for rnn in ["lstm", "gru"]:
    n = net(rnn)
    # --- emulate worker: hidden_states[(env,p)] from initial_hidden(1), updated by _run_inference_group
    hidden_states = {(0, 0): n.initial_hidden(1)}
    buf = RolloutBuffer(chunk_length=T, obs_shape=(OBS,), action_shape=(), action_dtype=np.int64)
    out_a = np.zeros(1, np.int64); out_lp = np.zeros(1, np.float32); out_v = np.zeros(1, np.float32)
    for t in range(T):
        if buf.steps == 0:
            buf.set_lstm_init(*hidden_states[(0, 0)])
        obs = np.random.randn(1, OBS).astype(np.float32)
        _run_inference_group(n, [(0, 0, 0)], obs, None, hidden_states, out_a, out_lp, out_v)
        buf.append(obs[0], out_a[0], out_lp[0], 0.1, False, out_v[0])
    chunk = _build_chunk(buf, "a", 0.0, 0)
    print(f"[{rnn}] worker-built chunk lstm_hidden[0].shape = {tuple(chunk.lstm_hidden[0].shape)}  (types.py doc says [num_layers, hidden])")
    appo = APPO(n, AlgorithmConfig())
    try:
        m = appo.train_step([chunk, chunk])
        print(f"[{rnn}] APPO.train_step on worker-built chunks: OK loss={m['total_loss']:.3f}")
    except Exception as e:
        print(f"[{rnn}] APPO.train_step on worker-built chunks FAILED: {type(e).__name__}: {str(e)[:200]}")

# --- consistency: does learner recompute the same log-probs as the worker for an on-policy chunk?
n = net("lstm")
hs = {(0, 0): n.initial_hidden(1)}
buf = RolloutBuffer(chunk_length=T, obs_shape=(OBS,), action_shape=(), action_dtype=np.int64)
dones = [0, 0, 1, 0, 0, 0]
for t in range(T):
    if buf.steps == 0: buf.set_lstm_init(*hs[(0, 0)])
    obs = np.random.randn(1, OBS).astype(np.float32)
    _run_inference_group(n, [(0, 0, 0)], obs, None, hs, out_a, out_lp, out_v)
    buf.append(obs[0], out_a[0], out_lp[0], 0.0, dones[t], out_v[0])
    if dones[t]: hs[(0, 0)] = n.initial_hidden(1)  # worker resets after done
chunk = _build_chunk(buf, "a", 0.0, 0)
h0 = chunk.lstm_hidden[0].reshape(1, 1, HID); c0 = chunk.lstm_hidden[1].reshape(1, 1, HID)
with torch.no_grad():
    lp, v, _ = n.evaluate_actions_recurrent(chunk.observations[:, None], chunk.actions[:, None], (h0, c0), dones_seq=chunk.dones[:, None])
print("on-policy recompute max|lp_learner - lp_worker| =", (lp[:, 0] - chunk.action_log_probs).abs().max().item(),
      " max|v diff| =", (v[:, 0] - chunk.values).abs().max().item())

# --- kickstart on recurrent net
n = net("lstm"); teacher = net("lstm")
ks = KickstartLoss(teacher)
try:
    ks.compute(n, torch.randn(5, OBS)); print("kickstart on recurrent(latent!=hidden): OK")
except Exception as e:
    print("kickstart on recurrent(latent!=hidden) FAILED:", type(e).__name__, str(e)[:120])
n_eq = net("lstm", hid=LAT); t_eq = net("lstm", hid=LAT)
out = KickstartLoss(t_eq).compute(n_eq, torch.randn(5, OBS))
print("kickstart on recurrent(latent==hidden): silently runs, RNN bypassed, loss =", out.item())

# --- BC on recurrent net
bc = OfflineBCTrainer(net("lstm"))
bc.add_data(torch.randn(32, OBS), torch.randint(0, A, (32,)))
try:
    bc.train(num_epochs=1); print("BC on recurrent: OK")
except Exception as e:
    print("BC on recurrent(latent!=hidden) FAILED:", type(e).__name__, str(e)[:120])

# --- act() without hidden on a recurrent net (what eval.py does)
n = net("lstm")
try:
    n.act(torch.randn(3, OBS)); print("act() without hidden on recurrent net: OK")
except Exception as e:
    print("act() without hidden on recurrent net (eval.py path) FAILED:", type(e).__name__, str(e)[:120])
