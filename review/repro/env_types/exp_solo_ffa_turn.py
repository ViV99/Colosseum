"""Experiments: solo, FFA with elimination, turn-based. Prints evidence."""
from __future__ import annotations

import sys
import traceback
from collections import defaultdict

import numpy as np
import torch

from harness import make_net, run_worker
from toy_envs import (AlternatingEnv, CatPolicy, FFAElimEnv, MLPEncoder, SoloEnv,
                      ValueHead)

torch.manual_seed(0)


def section(name):
    print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)


def per_seat_stats(chunks, n_seats, id_slice):
    """Attribute each chunk to a seat via the one-hot id in obs and sum rewards/dones."""
    rew = defaultdict(float)
    dones = defaultdict(int)
    steps = defaultdict(int)
    for c in chunks:
        seat = int(c.observations[0, id_slice].argmax())
        rew[seat] += float(c.rewards.sum())
        dones[seat] += int(c.dones.sum())
        steps[seat] += c.chunk_length
    return dict(rew), dict(dones), dict(steps)


# ---------------------------------------------------------------------------
section("SOLO: 1-player env through worker -> APPO -> coordinator -> eval")
from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.coordinator.coordinator import Coordinator

fac = lambda: make_net(MLPEncoder(4), CatPolicy(32, 4), ValueHead(32))
chunks, results = run_worker(SoloEnv, fac, steps=400, num_envs=2, chunk_length=10)
print("chunks:", len(chunks["a"]), "results:", len(results))
print("first result:", results[0])
algo = APPO(fac(), AlgorithmConfig())
m = algo.train_step(chunks["a"][:8])
print("APPO metrics finite:", all(np.isfinite(v) for v in m.values()), {k: round(v, 4) for k, v in m.items()})

from colosseum.core.config import ColosseumConfig, EnvConfig, NetworkConfig
cfg = ColosseumConfig(env=EnvConfig(env_class="toy_envs.SoloEnv", num_players=1),
                      networks=NetworkConfig(encoder_class="toy_envs.MLPEncoder",
                                             policy_class="toy_envs.CatPolicy",
                                             value_class="toy_envs.ValueHead"))
coord = Coordinator(cfg)
coord.agent_pool.register_trainable("a")
for r in results:
    coord.report_match_result(r)
print("coordinator ratings after solo results:", coord.get_ratings_summary())
print("mean total reward in results (never logged anywhere):",
      np.mean([list(r.total_rewards.values())[0] for r in results]))

from colosseum.eval import evaluate_agents
net = fac()
mat = evaluate_agents({"a": {"state_dict": net.state_dict()}}, SoloEnv, fac, num_matches=20, num_envs=2)
print("eval results for solo agent:", mat.results, "\nsummary:\n" + mat.summary())

# ---------------------------------------------------------------------------
section("FFA-3 elimination, mode=per_player_term (eliminated player terminated=True)")
fac3 = lambda: make_net(MLPEncoder(5), CatPolicy(32, 3), ValueHead(32))
chunks, results = run_worker(lambda: FFAElimEnv("per_player_term"), fac3, steps=120,
                             num_envs=1, chunk_length=6)
print("episode lengths reported:", sorted({r.episode_length for r in results}),
      "(true game length is 6)")
print("outcomes of first result:", results[0].player_outcomes)

section("FFA-3 elimination, mode=active (documented convention)")
chunks, results = run_worker(lambda: FFAElimEnv("active"), fac3, steps=600,
                             num_envs=1, chunk_length=6)
n_eps = len(results)
rew, dones, steps = per_seat_stats(chunks["a"], 3, slice(0, 3))
print(f"episodes={n_eps}")
print("per-episode env reward per seat: seat0=-2 (elim -1 + rank3 -1), seat1=-1 (elim -1 + rank2 0), seat2=+1")
for s in range(3):
    eps_in_chunks = steps.get(s, 0)
    print(f" seat {s}: recorded steps={steps.get(s,0)}, sum reward in chunks={rew.get(s,0):.1f}, "
          f"dones in chunks={dones.get(s,0)}")
print("result outcomes (rank-based):", results[0].player_outcomes)
# show one chunk of seat 0
for c in chunks["a"]:
    if int(c.observations[0, 0:3].argmax()) == 0:
        print(" seat0 chunk rewards:", c.rewards.tolist(), "dones:", c.dones.tolist(),
              "t-feature:", [round(float(x), 2) for x in c.observations[:, 4]])
        break

# ---------------------------------------------------------------------------
section("TURN-BASED (alternating), info['active'] convention")
fac2 = lambda: make_net(MLPEncoder(4), CatPolicy(32, 3), ValueHead(32))
chunks, results = run_worker(lambda: AlternatingEnv(use_active=True), fac2, steps=400,
                             num_envs=1, chunk_length=4)
rew, dones, steps = per_seat_stats(chunks["a"], 2, slice(0, 2))
print(f"episodes={len(results)}; env gives seat0 +1 and seat1 -1 per episode")
for s in range(2):
    print(f" seat {s}: recorded steps={steps.get(s,0)}, sum reward={rew.get(s,0):.1f}, dones={dones.get(s,0)}")
for c in chunks["a"]:
    if int(c.observations[0, 0:2].argmax()) == 0:
        print(" seat0 chunk rewards:", c.rewards.tolist(), "dones:", c.dones.tolist(),
              "t:", [round(float(x), 2) for x in c.observations[:, 3]])
        break

section("TURN-BASED without 'active' (every slot recorded, incl. ignored moves)")
chunks, results = run_worker(lambda: AlternatingEnv(use_active=False), fac2, steps=400,
                             num_envs=1, chunk_length=4)
rew, dones, steps = per_seat_stats(chunks["a"], 2, slice(0, 2))
for s in range(2):
    print(f" seat {s}: recorded steps={steps.get(s,0)}, sum reward={rew.get(s,0):.1f}, dones={dones.get(s,0)}")
frac_noop = np.mean([float(c.observations[t, 2] == 0) for c in chunks["a"] for t in range(c.chunk_length)])
print(f" fraction of recorded transitions where the slot's action is ignored by env: {frac_noop:.2f}")

section("TURN-BASED with all-False action mask for the inactive player")
try:
    run_worker(lambda: AlternatingEnv(use_active=True, mask_inactive_all_false=True), fac2,
               steps=20, num_envs=1, chunk_length=4)
    print("no error")
except Exception as e:
    tb = traceback.format_exc().strip().splitlines()
    print("CRASH:", type(e).__name__, str(e)[:200])
    print("\n".join(tb[-8:]))
