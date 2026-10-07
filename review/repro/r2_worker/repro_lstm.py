from common import *
from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
fn = lambda: make_net(recurrent="lstm")
# 1) shape of lstm_hidden produced by the worker and whether APPO can consume it
tq, _ = run_worker(lambda: StepEnv(L=5), steps=16, chunk_length=4, net_fn=fn, results=False)
cs = tq["a"].items
print("worker lstm_hidden h shape:", tuple(cs[0].lstm_hidden[0].shape), " (TrajectoryChunk doc + APPO expect [num_layers, hidden])")
torch.manual_seed(0); net = fn()
appo = APPO(net, AlgorithmConfig(name="appo", num_epochs=1, minibatch_chunks=0), device="cpu")
try:
    m = appo.train_step(cs[:4]); print("APPO train_step OK", m.get("total_loss"))
except Exception as e:
    print("APPO train_step FAILED:", type(e).__name__, str(e)[:200])

# 2) behaviour vs learner log-prob consistency (same weights, no update) -- plain env and turn-based env
def consistency(env_fn, label):
    torch.manual_seed(0)
    tq, _ = run_worker(env_fn, steps=40, chunk_length=4, net_fn=fn, results=False)
    torch.manual_seed(0); net = fn()   # identical init to worker's 'latest' (same seed order)
    diffs = []
    for c in tq["a"].items:
        h = c.lstm_hidden[0].reshape(1, 1, -1); cc = c.lstm_hidden[1].reshape(1, 1, -1)
        with torch.no_grad():
            lp, v, _ = net.evaluate_actions_recurrent(c.observations.unsqueeze(1), c.actions.unsqueeze(1), (h, cc), dones_seq=c.dones.unsqueeze(1))
        diffs.append((lp.squeeze(1) - c.action_log_probs).abs().max().item())
    print(f"{label}: max |learner_logp - behaviour_logp| over chunks = {max(diffs):.2e}")
consistency(lambda: StepEnv(L=5), "simultaneous env")
consistency(lambda: StepEnv(L=5, turn_based=True), "turn-based env (info['active'])")
