"""Experiments: Dict obs, per-unit actions, ordering, ratio scaling, recurrent eval."""
from __future__ import annotations

import traceback

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from harness import make_net, run_worker
from toy_envs import (CatPolicy, DictObsEnv, MLPEncoder, MultiUnitCompositePolicy,
                      SoloEnv, UnitsEnv, ValueHead)

torch.manual_seed(0)


def section(name):
    print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)


def show_exc(e):
    tb = traceback.format_exc().strip().splitlines()
    print("ERROR:", type(e).__name__, str(e)[:300])
    print("\n".join(tb[-6:]))


# ---------------------------------------------------------------------------
section("DICT OBSERVATIONS: VectorEnv / validate_config")
from colosseum.envs.vec_env import VectorEnv
try:
    ve = VectorEnv(DictObsEnv, 2)
    obs, _ = ve.reset_all()
    print("VectorEnv obs dtype/shape:", obs.dtype, obs.shape)
    fac = lambda: make_net(MLPEncoder(4), CatPolicy(32, 4), ValueHead(32))
    run_worker(DictObsEnv, fac, steps=8, num_envs=2, chunk_length=4)
except Exception as e:
    show_exc(e)

from colosseum.core.config import ColosseumConfig, EnvConfig, NetworkConfig
from colosseum.core.registry import validate_config
try:
    validate_config(ColosseumConfig(
        env=EnvConfig(env_class="toy_envs.DictObsEnv", num_players=2),
        networks=NetworkConfig(encoder_class="toy_envs.MLPEncoder",
                               policy_class="toy_envs.CatPolicy",
                               value_class="toy_envs.ValueHead")))
    print("validate_config OK")
except Exception as e:
    show_exc(e)

# ---------------------------------------------------------------------------
section("PER-UNIT ACTIONS: MultiDiscrete([A]*K) + CompositeDist + per-unit mask")
K, A = 8, 4
facu = lambda: make_net(MLPEncoder(K + 1), MultiUnitCompositePolicy(32, K, A), ValueHead(32))
try:
    chunks, results = run_worker(lambda: UnitsEnv(K, A), facu, steps=64, num_envs=2, chunk_length=8)
    c = chunks["a"][0]
    print("chunk actions shape/dtype:", tuple(c.actions.shape), c.actions.dtype,
          "masks:", None if c.action_masks is None else tuple(c.action_masks.shape))
    from colosseum.algorithms.appo import APPO
    from colosseum.core.config import AlgorithmConfig
    algo = APPO(facu(), AlgorithmConfig())
    m = algo.train_step(chunks["a"])
    print("APPO metrics:", {k: round(v, 4) for k, v in m.items()})
except Exception as e:
    show_exc(e)

section("PER-UNIT: dead units with all-False mask row")
try:
    run_worker(lambda: UnitsEnv(K, A, dead_mask_all_false=True), facu, steps=16, num_envs=1, chunk_length=8)
    print("no error")
except Exception as e:
    show_exc(e)

section("PER-UNIT: GridNet-style 2-D MultiDiscrete nvec")
from colosseum.core.action_spec import ActionSpec
try:
    ActionSpec.from_space(gymnasium.spaces.MultiDiscrete(np.full((2, 4), 4)))
    print("ok")
except Exception as e:
    show_exc(e)

section("PER-UNIT: Box action of shape (K, D)")
spec = ActionSpec.from_space(gymnasium.spaces.Box(-1, 1, shape=(3, 2)))
print("Box(3,2) action_shape:", spec.action_shape, "composite:", spec.is_composite)

section("PER-UNIT: unit ordering for K>10 (MultiDiscrete -> sorted string keys)")
K2 = 12
spec = ActionSpec.from_space(gymnasium.spaces.MultiDiscrete([K2] * K2))
print("component names in flat order:", [c.name for c in spec.components])
# policy: head named str(i) deterministically picks action i (so we can see which unit gets what)
from colosseum.networks.distributions import CategoricalDist, CompositeDist
logits = torch.full((1, K2, K2), -10.0)
for i in range(K2):
    logits[0, i, i] = 10.0
dist = CompositeDist({str(i): CategoricalDist(logits[:, i]) for i in range(K2)})
flat = dist.mode()[0].numpy()
env_action = spec.decode(flat)
print("env receives (index=unit, value=head that produced it):", env_action.tolist())
print("units whose action came from a DIFFERENT head:",
      [u for u, h in enumerate(env_action.tolist()) if u != h])
# masks: env gives a flat mask where unit u may only take action u
mask = np.zeros((K2, K2), dtype=bool)
for u in range(K2):
    mask[u, u] = True
md = dist.apply_mask(torch.from_numpy(mask.reshape(1, -1)))
print("head '10' after applying env's per-unit mask -> allowed action:",
      int(md._dists["10"].logits.argmax()), "(it got the mask row of env unit",
      [c.name for c in spec.components].index("10"), ")")

# ---------------------------------------------------------------------------
section("PER-UNIT: joint log-prob ratio scaling with K (same per-unit policy change)")
from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
for Kk in (1, 8, 32, 128):
    torch.manual_seed(1)
    # behaviour policy: uniform; target: per-unit logits perturbed slightly (per-unit KL ~ small)
    B = 4096
    beh = torch.zeros(B, Kk, 4)
    tgt = beh + 0.15 * torch.randn(1, Kk, 4)  # same small change for every sample
    db = CompositeDist({f"{i:03d}": CategoricalDist(beh[:, i]) for i in range(Kk)})
    dt = CompositeDist({f"{i:03d}": CategoricalDist(tgt[:, i]) for i in range(Kk)})
    a = db.sample()
    log_ratio = dt.log_prob(a) - db.log_prob(a)
    ratio = log_ratio.exp()
    per_unit_kl = (dt.kl_divergence(db) / Kk).mean().item() if False else None
    clip_frac = ((ratio - 1).abs() > 0.2).float().mean().item()
    rho_clipped = (ratio > 1.0).float().mean().item()
    print(f" K={Kk:4d}: std(log_ratio)={log_ratio.std().item():.3f}  "
          f"PPO clip_fraction(eps=0.2)={clip_frac:.2f}  "
          f"frac rho truncated at 1={rho_clipped:.2f}  mean c_t={ratio.clamp(max=1).mean().item():.3f}")

# ---------------------------------------------------------------------------
section("RECURRENT (partial observability) in eval")
from colosseum.eval import evaluate_agents


def rec_fac():
    enc = MLPEncoder(4, hidden=32)
    return make_net(enc, CatPolicy(16, 4), ValueHead(16), recurrent=nn.LSTM(32, 16))


net = rec_fac()
try:
    from colosseum.envs.base_env import BaseEnv

    class Solo2(SoloEnv):
        @property
        def num_players(self):
            return 2

        def _obs(self):
            o = super()._obs()[0]
            return {0: o, 1: o}

        def reset(self, seed=None):
            o, _ = super().reset(seed)
            return o, {0: {}, 1: {}}

        def step(self, actions):
            o, r, t, tr, i = super().step(actions)
            return o, {0: r[0], 1: 0.0}, {0: t[0], 1: t[0]}, {0: False, 1: False}, {0: {}, 1: {}}

    mat = evaluate_agents({"x": {"state_dict": net.state_dict()}, "y": {"state_dict": net.state_dict()}},
                          Solo2, rec_fac, num_matches=4, num_envs=2)
    print("eval ran:", mat.summary())
except Exception as e:
    show_exc(e)


def rec_fac_same():
    enc = MLPEncoder(4, hidden=16)
    return make_net(enc, CatPolicy(16, 4), ValueHead(16), recurrent=nn.LSTM(16, 16))


net = rec_fac_same()
with torch.no_grad():
    obs = torch.eye(4)[:2]
    h = net.initial_hidden(2)
    d_with, _, _ = net.forward(obs, h)
    d_without, _, _ = net.forward(obs, None)
print("latent_dim == hidden_size case: eval path (hidden=None) silently skips LSTM; "
      "max |logit diff| vs proper path =",
      float((d_with.logits - d_without.logits).abs().max()))
