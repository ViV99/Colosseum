# SP1 Plan — Part C: Algorithm (block 4) and Eval (block 7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** This part covers spec blocks 4 and 7:
- **Block 4 (algorithm):** V-trace(λ), explicit observation-normalizer updates, masked KL, a correct kickstart, a BC rewrite, natural `ActionSpec` component order, and resumable algorithm state plus diagnostic metrics.
- **Block 7 (eval):** an eval engine on `PolicyModel` with per-seat state and seat rotation, draw-aware statistics, solo mode, heterogeneous checkpoints and JSON output.

Every task assumes Parts A (`01-tooling-and-model.md`, T0.1–T1.7) and B (`02-dataflow-and-transitions.md`, T2.1–T3.4) are done exactly as the interface contract in `00-overview.md` says.

**Read `00-overview.md` first.** It holds the global constraints, the binding interface contract and the task order.

**Conventions used in every task:**
- Commands run from the repo root (`/home/viv/dev/repos/Colosseum`) with `.venv/bin/python`.
- The test layout from T0.2 is `tests/unit|contract|integration|learning`. `tests/helpers.py` is the shared toy module, imported as `from tests.helpers import ...`. Existing tests may have moved during T0.2, so tasks find them with `grep -rln` rather than by path.
- Ruff config comes from T0.3. Keep imports at the top of modules, sorted, with no unused names.
- "Full fast suite" means `.venv/bin/python -m pytest -m "not gpu and not slow" -q`. It must pass at the end of every task.
- Commits use a conventional prefix and have no attribution or co-author lines.

---

### Task T4.1: V-trace λ (`vtrace_lambda`), remove `gae_lambda`

Spec block 4 "λ в V-trace", finding R1-12. Today `gae_lambda` is dead config and the docs claim GAE. This task implements V-trace(λ): `c_t = λ·min(c̄, ρ_t)` (IMPALA, Remark 2), renames the knob to `algorithm.vtrace_lambda` (default 1.0 = current behaviour), and removes every mention of `gae_lambda`/GAE.

The task also creates the **algorithm/eval test kit** in `tests/helpers.py`, which every later task in this part reuses:
- tiny `ComposedModel`s;
- a masked solo env;
- `rollout_chunks`, which produces real chunks through `RolloutLoop`.

**Files:**
- Modify: `src/colosseum/algorithms/vtrace.py` (whole file)
- Modify: `src/colosseum/core/config.py` (`AlgorithmConfig`, the `gae_lambda` line)
- Modify: `src/colosseum/algorithms/appo.py` (the `compute_vtrace` call in `compute_loss`)
- Modify: `configs/examples/*.yaml` (every file with `gae_lambda`)
- Modify: `README.md` (algorithm table row), `CLAUDE.md` (two GAE mentions), `src/colosseum/core/types.py` (docstring mention of GAE)
- Modify: `tests/helpers.py` (append the test kit)
- Test: `tests/unit/test_vtrace_lambda.py` (create)

**Interfaces:**
- Consumes:
  - `colosseum.networks.composed.ComposedModel(encoder, core, policy_head, value_head)` (T1.4).
  - `colosseum.networks.cores.NoCore(input_dim)`, `LSTMCore(input_dim, hidden_size, num_layers=1)`, `GRUCore(...)`, `WindowAttentionCore(input_dim, d_model, window, num_heads, num_layers=1)`, each with `.output_dim` (T1.3).
  - `colosseum.worker.rollout_loop.RolloutLoop(*, worker_id, env_fn, num_envs, chunk_length, agent_ids, model_factories, io, ..., seed)` and `LoopIO(send_chunk, poll_weights, ...)` (T0.5/T1.6). With `slot_agent_map=None`, every seat plays `agent_ids[0]` and collects. `send_chunk` receives a `TrajectoryChunk`.
  - `TrajectoryChunk.to_payload()` / `TrajectoryChunk.from_payload(payload)` (T2.2).
  - `APPO(model, config, device="cpu", pin_memory=False, kickstart=None)` (T1.5). It reads V-trace through the module-global name `compute_vtrace` in `colosseum.algorithms.appo`, either stored as `self._compute_vtrace` at construction or called directly.
- Produces:
  - `compute_vtrace(behavior_log_probs, target_log_probs, rewards, values, bootstrap_value, dones, gamma=0.99, rho_bar=1.0, c_bar=1.0, lam=1.0) -> (vs [T,B], advantages [T,B])`. `dones` may be bool or float.
  - `AlgorithmConfig.vtrace_lambda: float = 1.0` (ge=0, le=1); `AlgorithmConfig.gae_lambda` no longer exists.
  - `tests/helpers.py`:
    - `TinyEncoder(obs_dim=4, latent_dim=16, normalize=False)`, `TinyPolicy(in_dim=16, num_actions=4)`, `TinyValue(in_dim=16)`;
    - `tiny_model(obs_dim=4, num_actions=4, core="none"|"lstm"|"gru"|"attention", latent_dim=16, hidden=24, normalize=False, seed=0) -> ComposedModel`;
    - `MaskedToyEnv(use_mask=True)`, with `.received`, `.expert_action()`, `EPISODE_LENGTH = 5`;
    - `rollout_chunks(model, env_fn, num_chunks, chunk_length=8, num_envs=2, seed=0) -> list[TrajectoryChunk]`.

- [ ] **Step 1: Append the test kit to `tests/helpers.py`.**

Add these imports at the top of `tests/helpers.py`. Skip any that are already there, and keep the existing `torch`, `torch.nn as nn`, `BaseEncoder/BasePolicy/BaseValue` and `CategoricalDist` imports.

```python
import copy

import gymnasium
import numpy as np

from colosseum.core.types import TrajectoryChunk
from colosseum.envs.base_env import BaseEnv
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import GRUCore, LSTMCore, NoCore, WindowAttentionCore
from colosseum.networks.normalization import NormalizeObs
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop
```

Then append this block at the end of the file:

```python
# ---------------------------------------------------------------------------
# Algorithm / eval test kit (SP1 blocks 4 and 7)
# ---------------------------------------------------------------------------


class TinyEncoder(BaseEncoder):
    """[B, obs_dim] -> optional NormalizeObs -> Linear -> tanh -> [B, latent_dim]."""

    def __init__(self, obs_dim: int = 4, latent_dim: int = 16, normalize: bool = False) -> None:
        super().__init__()
        self.norm = NormalizeObs(shape=(obs_dim,)) if normalize else None
        self.fc = nn.Linear(obs_dim, latent_dim)
        self._latent_dim = latent_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float().reshape(obs.shape[0], -1)
        if self.norm is not None:
            x = self.norm(x)
        return torch.tanh(self.fc(x))


class TinyPolicy(BasePolicy):
    def __init__(self, in_dim: int = 16, num_actions: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_actions)

    def forward(self, latent: torch.Tensor) -> CategoricalDist:
        return CategoricalDist(self.fc(latent))


class TinyValue(BaseValue):
    def __init__(self, in_dim: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.fc(latent).squeeze(-1)


def tiny_model(
    obs_dim: int = 4,
    num_actions: int = 4,
    core: str = "none",
    latent_dim: int = 16,
    hidden: int = 24,
    normalize: bool = False,
    seed: int = 0,
) -> ComposedModel:
    """Small ComposedModel. ``core`` is "none", "lstm", "gru" or "attention".

    ``hidden`` differs from ``latent_dim`` on purpose: a silently skipped core
    then fails with a shape error instead of passing.
    """
    torch.manual_seed(seed)
    encoder = TinyEncoder(obs_dim, latent_dim, normalize)
    if core == "none":
        core_module = NoCore(input_dim=latent_dim)
    elif core == "lstm":
        core_module = LSTMCore(input_dim=latent_dim, hidden_size=hidden)
    elif core == "gru":
        core_module = GRUCore(input_dim=latent_dim, hidden_size=hidden)
    elif core == "attention":
        core_module = WindowAttentionCore(input_dim=latent_dim, d_model=hidden, window=8, num_heads=2)
    else:
        raise ValueError(f"unknown core {core!r}")
    out_dim = core_module.output_dim
    return ComposedModel(encoder, core_module, TinyPolicy(out_dim, num_actions), TinyValue(out_dim))


class MaskedToyEnv(BaseEnv):
    """Solo env: random 4-vector observation, Discrete(4), two random legal actions.

    Each step exposes ``info["action_mask"]`` (if ``use_mask``) with exactly two
    legal actions. Episodes last ``EPISODE_LENGTH`` steps. The reward is 1 for the
    scripted expert's choice (the legal action with the largest observation
    entry), else 0. Every received action is appended to ``self.received``.
    """

    EPISODE_LENGTH = 5

    def __init__(self, use_mask: bool = True) -> None:
        self._use_mask = use_mask
        self._rng = np.random.default_rng(0)
        self._t = 0
        self._obs = np.zeros(4, dtype=np.float32)
        self._mask = np.ones(4, dtype=bool)
        self.received: list[int] = []

    @property
    def num_players(self) -> int:
        return 1

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(-1.0, 1.0, (4,), np.float32)

    @property
    def action_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(4)

    def expert_action(self) -> int:
        return int(np.argmax(np.where(self._mask, self._obs, -np.inf)))

    def _draw(self) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
        self._obs = self._rng.uniform(-1.0, 1.0, 4).astype(np.float32)
        self._mask = np.zeros(4, dtype=bool)
        self._mask[self._rng.choice(4, size=2, replace=False)] = True
        info = {"action_mask": self._mask.copy()} if self._use_mask else {}
        return {0: self._obs.copy()}, {0: info}

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        return self._draw()

    def step(self, actions):
        action = int(np.asarray(actions[0]).reshape(-1)[0])
        self.received.append(action)
        reward = 1.0 if action == self.expert_action() else 0.0
        self._t += 1
        done = self._t >= self.EPISODE_LENGTH
        obs, info = self._draw()
        return obs, {0: reward}, {0: done}, {0: False}, info


def rollout_chunks(
    model,
    env_fn,
    num_chunks: int,
    chunk_length: int = 8,
    num_envs: int = 2,
    seed: int = 0,
) -> list[TrajectoryChunk]:
    """Collect ``num_chunks`` real chunks for agent "a" with an in-process RolloutLoop.

    The loop acts with a deep copy of ``model`` (same weights). Chunks go through
    ``to_payload()``/``from_payload()`` exactly as they would across processes.
    """
    chunks: list[TrajectoryChunk] = []

    def send_chunk(chunk) -> None:
        payload = chunk if isinstance(chunk, dict) else chunk.to_payload()
        chunks.append(TrajectoryChunk.from_payload(payload))

    io = LoopIO(send_chunk=send_chunk, poll_weights=lambda agent_id: None)
    loop = RolloutLoop(
        worker_id=0, env_fn=env_fn, num_envs=num_envs, chunk_length=chunk_length,
        agent_ids=["a"], model_factories={"a": lambda: copy.deepcopy(model)},
        io=io, seed=seed,
    )
    try:
        for _ in range(100_000):
            if len(chunks) >= num_chunks:
                break
            loop.step()
    finally:
        loop.close()
    assert len(chunks) >= num_chunks, f"RolloutLoop produced only {len(chunks)} chunks"
    return chunks[:num_chunks]
```

- [ ] **Step 2: Write the failing test** `tests/unit/test_vtrace_lambda.py`:

```python
"""V-trace(lambda): c_t = lambda * min(c_bar, rho_t) (T4.1)."""
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

import colosseum.algorithms.appo as appo_module
from colosseum.algorithms.appo import APPO
from colosseum.algorithms.vtrace import compute_vtrace
from colosseum.core.config import AlgorithmConfig
from tests.helpers import MaskedToyEnv, rollout_chunks, tiny_model

REPO_ROOT = Path(__file__).resolve().parents[2]


def reference_vtrace(blp, tlp, r, v, boot, d, gamma, rho_bar, c_bar, lam):
    """Slow per-element V-trace (Espeholt et al. 2018, eq. 1) with c_i = lam * min(c_bar, rho_i)."""
    T, B = r.shape
    rho = torch.exp(tlp - blp)
    crho = rho.clamp(max=rho_bar)
    c = lam * rho.clamp(max=c_bar)
    vs = torch.zeros(T, B, dtype=torch.float64)
    adv = torch.zeros(T, B, dtype=torch.float64)
    for b in range(B):
        def value(t):
            return boot[b] if t == T else v[t, b]

        for s in range(T):
            acc = 0.0
            for t in range(s, T):
                prod, cut = 1.0, False
                for i in range(s, t):
                    if d[i, b]:
                        cut = True
                        break
                    prod *= gamma * c[i, b]
                if cut:
                    break
                not_done = 0.0 if d[t, b] else 1.0
                acc += prod * crho[t, b] * (r[t, b] + gamma * not_done * value(t + 1) - v[t, b])
            vs[s, b] = v[s, b] + acc
        for s in range(T):
            not_done = 0.0 if d[s, b] else 1.0
            v_next = boot[b] if s == T - 1 else vs[s + 1, b]
            adv[s, b] = crho[s, b] * (r[s, b] + gamma * not_done * v_next - v[s, b])
    return vs, adv


def gae(r, v, boot, d, gamma, lam):
    T, B = r.shape
    out = torch.zeros(T, B, dtype=r.dtype)
    last = torch.zeros(B, dtype=r.dtype)
    for t in reversed(range(T)):
        not_done = 1.0 - d[t].to(r.dtype)
        v_next = boot if t == T - 1 else v[t + 1]
        delta = r[t] + gamma * not_done * v_next - v[t]
        last = delta + gamma * lam * not_done * last
        out[t] = last
    return out


@pytest.mark.parametrize("lam", [0.5, 0.95, 1.0])
@pytest.mark.parametrize("rho_bar,c_bar", [(1.0, 1.0), (2.0, 0.5)])
def test_vtrace_lambda_matches_reference(lam, rho_bar, c_bar):
    g = torch.Generator().manual_seed(0)
    T, B = 12, 4
    blp = torch.randn(T, B, generator=g, dtype=torch.float64) * 0.5 - 1.0
    tlp = blp + torch.randn(T, B, generator=g, dtype=torch.float64) * 0.7
    r = torch.randn(T, B, generator=g, dtype=torch.float64)
    v = torch.randn(T, B, generator=g, dtype=torch.float64)
    boot = torch.randn(B, generator=g, dtype=torch.float64)
    d = torch.rand(T, B, generator=g, dtype=torch.float64) < 0.2
    assert d.any()
    vs, adv = compute_vtrace(blp, tlp, r, v, boot, d, gamma=0.97, rho_bar=rho_bar, c_bar=c_bar, lam=lam)
    ref_vs, ref_adv = reference_vtrace(blp, tlp, r, v, boot, d, 0.97, rho_bar, c_bar, lam)
    torch.testing.assert_close(vs, ref_vs, atol=1e-10, rtol=0)
    torch.testing.assert_close(adv, ref_adv, atol=1e-10, rtol=0)


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
def test_on_policy_vtrace_lambda_equals_gae(lam):
    g = torch.Generator().manual_seed(1)
    T, B = 10, 3
    lp = torch.zeros(T, B, dtype=torch.float64)
    r = torch.randn(T, B, generator=g, dtype=torch.float64)
    v = torch.randn(T, B, generator=g, dtype=torch.float64)
    boot = torch.randn(B, generator=g, dtype=torch.float64)
    d = torch.rand(T, B, generator=g, dtype=torch.float64) < 0.2
    vs, _ = compute_vtrace(lp, lp, r, v, boot, d, gamma=0.97, lam=lam)
    torch.testing.assert_close(vs - v, gae(r, v, boot, d, 0.97, lam), atol=1e-10, rtol=0)


def test_default_lambda_is_plain_vtrace():
    g = torch.Generator().manual_seed(2)
    T, B = 6, 2
    blp = torch.randn(T, B, generator=g)
    tlp = torch.randn(T, B, generator=g)
    r, v = torch.randn(T, B, generator=g), torch.randn(T, B, generator=g)
    boot = torch.randn(B, generator=g)
    d = torch.zeros(T, B)
    a = compute_vtrace(blp, tlp, r, v, boot, d)
    b = compute_vtrace(blp, tlp, r, v, boot, d, lam=1.0)
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


def test_config_has_vtrace_lambda_and_no_gae_lambda():
    assert "gae_lambda" not in AlgorithmConfig.model_fields
    assert AlgorithmConfig().vtrace_lambda == 1.0
    assert AlgorithmConfig(vtrace_lambda=0.9).vtrace_lambda == 0.9
    with pytest.raises(ValidationError):
        AlgorithmConfig(vtrace_lambda=1.5)
    with pytest.raises(ValidationError):
        AlgorithmConfig(vtrace_lambda=-0.1)


def test_no_gae_lambda_left_in_example_configs():
    offenders = [p.name for p in (REPO_ROOT / "configs").rglob("*.yaml") if "gae_lambda" in p.read_text()]
    assert offenders == []


def test_appo_passes_vtrace_lambda(monkeypatch):
    seen = []
    real = appo_module.compute_vtrace

    def spy(*args, **kwargs):
        seen.append(kwargs.get("lam"))
        return real(*args, **kwargs)

    monkeypatch.setattr(appo_module, "compute_vtrace", spy)
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=2, chunk_length=8)
    algo = APPO(model, AlgorithmConfig(vtrace_lambda=0.7), device="cpu")
    algo.train_step(chunks)
    assert seen and all(lam == 0.7 for lam in seen)
```

- [ ] **Step 3: Run it and watch it fail.**

Run: `.venv/bin/python -m pytest tests/unit/test_vtrace_lambda.py -v`

Expected failures:
- `TypeError: compute_vtrace() got an unexpected keyword argument 'lam'` in the V-trace tests;
- `AssertionError` in `test_config_has_vtrace_lambda_and_no_gae_lambda` and `test_no_gae_lambda_left_in_example_configs`;
- `assert seen and ...` failing in the APPO test.

If an import error mentions a helper (`TinyEncoder`, `rollout_chunks`, ...), Step 1 was not applied.

- [ ] **Step 4: Replace `src/colosseum/algorithms/vtrace.py` with:**

```python
"""V-trace off-policy correction for IMPALA-style training.

Implements V-trace from "IMPALA: Scalable Distributed Deep-RL with Importance
Weighted Actor-Learner Architectures" (Espeholt et al., 2018), including the
lambda variant of Remark 2: trace coefficients c_t = lambda * min(c_bar, rho_t).

All functions are pure (no classes, no state), operating on tensors.
"""

from __future__ import annotations

import torch


def compute_vtrace(
    behavior_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute V-trace(lambda) value targets and policy-gradient advantages.

    Args:
        behavior_log_probs: [T, B] log mu(a_t|x_t) recorded by the worker (behavior policy).
        target_log_probs:   [T, B] log pi(a_t|x_t) from the learner's current policy.
        rewards:            [T, B] rewards r_t.
        values:             [T, B] V(x_t) from the learner's current network.
        bootstrap_value:    [B]    V(x_T) for the state after the chunk (0 if terminal).
        dones:              [T, B] bool or float; 1 = transition t ends its episode.
        gamma:              discount factor.
        rho_bar:            truncation of the importance weights rho_t.
        c_bar:              truncation of the trace coefficients c_t.
        lam:                lambda in [0, 1]; c_t = lam * min(c_bar, rho_t).
                            lam = 1 is plain V-trace; on-policy, vs - V equals GAE(lambda).

    Returns:
        vs:         [T, B] V-trace targets.
        advantages: [T, B] rho_t * (r_t + gamma * vs_{t+1} - V(x_t)), with vs_T = bootstrap.

    Math:
        rho_t = min(rho_bar, pi/mu),  c_t = lam * min(c_bar, pi/mu)
        delta_t = rho_t * (r_t + gamma * (1 - done_t) * V(x_{t+1}) - V(x_t))
        vs_t - V(x_t) = delta_t + gamma * (1 - done_t) * c_t * (vs_{t+1} - V(x_{t+1}))
    """
    T, B = behavior_log_probs.shape

    log_rhos = torch.clamp(target_log_probs - behavior_log_probs, -20.0, 20.0)
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs = lam * torch.clamp(rhos, max=c_bar)

    not_done = 1.0 - dones.to(rewards.dtype)

    values_plus = torch.cat([values, bootstrap_value.unsqueeze(0)], dim=0)  # [T+1, B]
    deltas = clipped_rhos * (rewards + gamma * not_done * values_plus[1:] - values_plus[:-1])

    vs_minus_v = torch.zeros(T + 1, B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        vs_minus_v[t] = deltas[t] + gamma * not_done[t] * cs[t] * vs_minus_v[t + 1]

    vs = values_plus[:-1] + vs_minus_v[:-1]

    vs_plus = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
    advantages = clipped_rhos * (rewards + gamma * not_done * vs_plus - values_plus[:-1])
    return vs, advantages
```

- [ ] **Step 5: Config.** In `src/colosseum/core/config.py`, class `AlgorithmConfig`, replace the line

```python
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0, description="GAE lambda.")
```

with

```python
    vtrace_lambda: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="V-trace lambda: trace coefficients c_t = lambda * min(c_bar, rho_t). "
                    "1.0 = plain V-trace; ~0.9-0.95 trades bias for lower variance "
                    "(on-policy it equals GAE(lambda)). GAE itself is not used.",
    )
```

- [ ] **Step 6: APPO passes λ.** In `src/colosseum/algorithms/appo.py`, `compute_loss`, the V-trace call currently ends with `c_bar=cfg.vtrace_c_bar,`. Add one keyword argument so that the call reads:

```python
            vtrace_targets, vtrace_advantages = self._compute_vtrace(
                behavior_log_probs=batch["behavior_log_probs"],
                target_log_probs=target_log_probs.detach(),
                rewards=batch["rewards"],
                values=new_values.detach(),
                bootstrap_value=batch["bootstrap_values"],
                dones=batch["dones"],
                gamma=cfg.gamma,
                rho_bar=cfg.vtrace_rho_bar,
                c_bar=cfg.vtrace_c_bar,
                lam=cfg.vtrace_lambda,
            )
```

Two cases depend on what T1.5 wrote:
- If T1.5 calls the module function `compute_vtrace(...)` directly instead of `self._compute_vtrace(...)`, add `lam=cfg.vtrace_lambda` to that call.
- If T1.5's local names differ (`batch`, `target_log_probs`, `new_values`), keep T1.5's names. The only change is the added `lam=` argument.

- [ ] **Step 7: Remove `gae_lambda` and the GAE claims everywhere.**

Run `grep -rn "gae_lambda" --include="*.py" --include="*.yaml" --include="*.md" src tests configs examples README.md CLAUDE.md`. Today it lists `configs/examples/chase.yaml`, `space_miners.yaml`, `tic_tac_toe.yaml`, `tic_tac_toe_multi.yaml` and `README.md`. Files added by Parts A/B may also appear. Leave `review/` and `docs/superpowers/` alone: they are historical.

In every YAML hit, replace the line `  gae_lambda: 0.95` with `  vtrace_lambda: 1.0`, keeping the indentation.

In `README.md`, replace the table row

```
| `gae_lambda` | `0.95` | GAE lambda |
```

with

```
| `vtrace_lambda` | `1.0` | V-trace λ: trace coefficients c_t = λ·min(c̄, ρ_t); 1.0 = plain V-trace (GAE is not used) |
```

In `CLAUDE.md`:
- replace the bullet `  - GAE for advantage estimation` with `  - V-trace(λ) targets and advantages (`algorithm.vtrace_lambda`; GAE is not used)`;
- replace `(GAE/V-trace reset at done, new episode starts in same chunk)` with `(V-trace traces are cut at done, new episode starts in same chunk)`.

In `src/colosseum/core/types.py`, change the docstring fragment `used for GAE / V-trace` to `used for V-trace`. If T1.5/T3.x rewrote that docstring, run `grep -n "GAE" src/colosseum/core/types.py` and fix whatever is left.

Finally, check that `grep -rn "gae_lambda\|GAE" src tests configs examples README.md CLAUDE.md` prints nothing except unrelated words.

- [ ] **Step 8: Run the new tests.**

Run: `.venv/bin/python -m pytest tests/unit/test_vtrace_lambda.py -v`
Expected: all PASS (14 tests).

- [ ] **Step 9: Run the full fast suite.**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: PASS. The older V-trace tests (`grep -rln "compute_vtrace" tests/`) still pass unchanged, because `lam` defaults to 1.0.

- [ ] **Step 10: Commit.**

```bash
git add src/colosseum/algorithms/vtrace.py src/colosseum/algorithms/appo.py src/colosseum/core/config.py \
        src/colosseum/core/types.py configs/examples README.md CLAUDE.md tests/helpers.py tests/unit/test_vtrace_lambda.py
git commit -m "feat: V-trace(lambda) via algorithm.vtrace_lambda; remove dead gae_lambda (R1-12)"
```

---
### Task T4.2: `NormalizeObs` explicit `update()`; `update_normalizers` once per train step

Spec block 4 "Нормализация наблюдений", finding R1-08. Today `NormalizeObs.forward` updates its running statistics whenever the module is in `train()` mode. The learner model is always in train mode, so every loss forward counts the same data again: once per epoch, per minibatch and per kickstart forward. The statistics also move between the worker's forward and the learner's.

After this task:
- `forward` never changes the statistics;
- `NormalizeObs.update(obs)` is the only way to change them;
- APPO calls `model.update_normalizers(obs)` exactly once per `train_step`, before the epochs, with the batch's fresh observations.

The statistics are buffers, so they ship to workers with the weights.

**Files:**
- Modify: `src/colosseum/networks/normalization.py` (whole file)
- Modify: `src/colosseum/networks/model.py` (`PolicyModel.update_normalizers` body)
- Modify: `src/colosseum/networks/composed.py` (delete a `ComposedModel.update_normalizers` override if T1.4 added one)
- Modify: `src/colosseum/algorithms/appo.py` (`train_step`: one call before the epoch loop)
- Test: `tests/unit/test_obs_normalization.py` (create)

**Interfaces:**
- Consumes:
  - `PolicyModel.update_normalizers(self, obs: Tensor) -> None` declared by T1.2 (default no-op), `ComposedModel` (T1.4), `APPO.train_step` (T1.5/T2.5).
  - `WeightPayload.from_model(agent_id, policy_version, model)` and `WeightPayload.to_torch_state_dict()` (T2.2).
  - Test kit from T4.1: `tiny_model(..., normalize=True)`, `MaskedToyEnv`, `rollout_chunks`.
- Produces:
  - `NormalizeObs.update(obs: Tensor) -> None`. `obs` has shape `[..., *shape]` (any leading dims) and raises `ValueError` if the trailing dims differ from `shape`. `NormalizeObs.shape: tuple[int, ...]`. `forward` is pure.
  - `PolicyModel.update_normalizers(obs)`: the default calls `update(obs)` on every `NormalizeObs` submodule and is a no-op when there are none. `ComposedModel` inherits it. This is a contract refinement; see Contract notes.
  - `APPO.train_step` calls `self._model.update_normalizers(cat of all chunk observations)` exactly once, before the epochs.

- [ ] **Step 1: Write the failing test** `tests/unit/test_obs_normalization.py`:

```python
"""NormalizeObs: explicit update(), pure forward, one update per train step (T4.2)."""
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.types import WeightPayload
from colosseum.networks.normalization import NormalizeObs
from tests.helpers import MaskedToyEnv, rollout_chunks, tiny_model


def _norm(model) -> NormalizeObs:
    found = [m for m in model.modules() if isinstance(m, NormalizeObs)]
    assert len(found) == 1
    return found[0]


def test_forward_never_updates_stats_even_in_train_mode():
    norm = NormalizeObs(shape=(3,))
    norm.train()
    before = {k: v.clone() for k, v in norm.state_dict().items()}
    norm(torch.randn(32, 3) * 5 + 2)
    for key, value in norm.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_update_matches_batch_statistics():
    norm = NormalizeObs(shape=(3,), clip=0.0)
    x = torch.randn(1000, 3) * torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([5.0, -1.0, 0.5])
    norm.update(x)
    assert norm.rms.count.item() == pytest.approx(1000.0, abs=1e-3)
    torch.testing.assert_close(norm.rms.mean, x.mean(0), atol=1e-3, rtol=0)
    torch.testing.assert_close(norm.rms.var, x.var(0, unbiased=False), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(norm(x).mean(0), torch.zeros(3), atol=1e-2, rtol=0)


def test_update_accepts_leading_dims_and_rejects_wrong_shape():
    norm = NormalizeObs(shape=(3,))
    norm.update(torch.randn(4, 5, 3))
    assert norm.rms.count.item() == pytest.approx(20.0, abs=1e-3)
    with pytest.raises(ValueError, match="NormalizeObs"):
        norm.update(torch.randn(10, 4))


def test_update_normalizers_without_normalizer_is_noop():
    model = tiny_model(normalize=False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.update_normalizers(torch.randn(8, 4))
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key


@pytest.mark.parametrize("num_epochs,minibatch_chunks", [(1, 0), (3, 1), (2, 2)])
def test_train_step_counts_each_sample_once(num_epochs, minibatch_chunks):
    model = tiny_model(normalize=True)
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    samples = sum(c.chunk_length for c in chunks)
    algo = APPO(model, AlgorithmConfig(num_epochs=num_epochs, minibatch_chunks=minibatch_chunks), device="cpu")
    norm = _norm(algo.model)
    start = norm.rms.count.item()
    algo.train_step(chunks)
    assert norm.rms.count.item() - start == pytest.approx(samples, abs=1e-3)
    algo.train_step(chunks)
    assert norm.rms.count.item() - start == pytest.approx(2 * samples, abs=1e-3)


def test_stats_reach_workers_through_weight_payload():
    model = tiny_model(normalize=True)
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=2, chunk_length=8)
    algo = APPO(model, AlgorithmConfig(), device="cpu")
    algo.train_step(chunks)
    learner_norm = _norm(algo.model)
    assert learner_norm.rms.count.item() > 1.0

    payload = WeightPayload.from_model("a", algo.policy_version, algo.model)
    worker_model = tiny_model(normalize=True, seed=123)
    worker_model.load_state_dict(payload.to_torch_state_dict())
    worker_norm = _norm(worker_model)
    torch.testing.assert_close(worker_norm.rms.mean, learner_norm.rms.mean)
    torch.testing.assert_close(worker_norm.rms.var, learner_norm.rms.var)
    torch.testing.assert_close(worker_norm.rms.count, learner_norm.rms.count)
```

- [ ] **Step 2: Run it and watch it fail.**

Run: `.venv/bin/python -m pytest tests/unit/test_obs_normalization.py -v`

Expected failures:
- `test_forward_never_updates_stats_even_in_train_mode`: the buffers changed;
- `AttributeError: 'NormalizeObs' object has no attribute 'update'`;
- `test_train_step_counts_each_sample_once`: the count is 0 or a multiple of the sample count, because the T1.2 default is a no-op or forward updates per minibatch.

- [ ] **Step 3: Replace `src/colosseum/networks/normalization.py` with:**

```python
"""Observation normalization (running mean/std).

``NormalizeObs`` keeps its running statistics in registered *buffers*, so they
are part of the model ``state_dict`` and ride along with the normal weight sync
(learner -> workers) and checkpoints.

The statistics change only through :meth:`NormalizeObs.update`; ``forward`` is
a pure function of the current statistics, in train and eval mode alike. The
algorithm calls ``PolicyModel.update_normalizers(obs)`` exactly once per train
step with the batch's fresh observations, before any loss forward. So every
sample is counted once, whatever the number of epochs and minibatches, and the
learner's forward uses the same statistics for every minibatch of a step.

Usage: put it inside your encoder::

    class MyEncoder(BaseEncoder):
        def __init__(self, obs_dim=...):
            super().__init__()
            self.norm = NormalizeObs(shape=(obs_dim,))
            self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU())

        def forward(self, obs):
            return self.net(self.norm(obs))

The default ``PolicyModel.update_normalizers`` feeds the raw observations to
every ``NormalizeObs`` submodule. If a normalizer sees something else (a slice
or a transform of the observation), override ``update_normalizers``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """Welford-style running mean/variance over a feature shape, in buffers."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update stats from a batch ``x`` of shape ``[N, *shape]``."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)


class NormalizeObs(nn.Module):
    """Normalize observations by running mean/std; statistics change only in ``update``.

    Args:
        shape: per-observation feature shape (e.g. ``(obs_dim,)`` or ``(C, H, W)``).
        clip: clip normalized values to ``[-clip, clip]`` (0 disables).
        epsilon: numerical floor for the variance.
    """

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.shape: tuple[int, ...] = tuple(int(s) for s in shape)
        self.rms = RunningMeanStd(self.shape)
        self._clip = clip
        self._eps = epsilon

    @torch.no_grad()
    def update(self, obs: torch.Tensor) -> None:
        """Add a batch of observations ``[..., *shape]`` (any leading dims) to the statistics."""
        n = len(self.shape)
        if obs.dim() < n or tuple(obs.shape[obs.dim() - n:]) != self.shape:
            raise ValueError(
                f"NormalizeObs(shape={self.shape}) cannot update from observations of shape "
                f"{tuple(obs.shape)}: the trailing dims must equal {self.shape}. If this "
                f"normalizer sees a transformed observation, override "
                f"PolicyModel.update_normalizers for your model."
            )
        flat = obs.reshape(-1, *self.shape).to(self.rms.mean.dtype)
        if flat.shape[0] == 0:
            return
        self.rms.update(flat)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normed = (obs - self.rms.mean) / torch.sqrt(self.rms.var + self._eps)
        if self._clip > 0:
            normed = torch.clamp(normed, -self._clip, self._clip)
        return normed
```

- [ ] **Step 4: `PolicyModel.update_normalizers` default.**

In `src/colosseum/networks/model.py`, add `from colosseum.networks.normalization import NormalizeObs` to the imports. Then replace the body of `PolicyModel.update_normalizers` (T1.2 made it a no-op) so the method reads:

```python
    @torch.no_grad()
    def update_normalizers(self, obs: Tensor) -> None:
        """Update running observation statistics from fresh training data.

        The algorithm calls this exactly once per train step, before any loss
        forward, with all new observations of the step (``[N, *obs_shape]``).
        Default: ``update(obs)`` on every ``NormalizeObs`` submodule (a no-op when
        there are none). Override when a normalizer sees something other than the
        raw observation.
        """
        for module in self.modules():
            if isinstance(module, NormalizeObs):
                module.update(obs)
```

Keep the existing `torch` and `Tensor` imports of `model.py` as they are. `networks/normalization.py` imports only `torch`, so there is no import cycle.

In `src/colosseum/networks/composed.py`, run `grep -n "update_normalizers" src/colosseum/networks/composed.py`. If `ComposedModel` defines its own `update_normalizers`, delete that method so that `ComposedModel` inherits the default above.

- [ ] **Step 5: APPO calls it once per train step.**

In `src/colosseum/algorithms/appo.py`, `train_step`, insert this block right after `cfg = self._config`, before the `for _epoch in range(cfg.num_epochs):` loop:

```python
        # Refresh observation-normalization statistics once per train step, from
        # this step's fresh samples only (never once per epoch/minibatch forward).
        obs_all = torch.cat([torch.as_tensor(c.observations) for c in chunks], dim=0)
        self._model.update_normalizers(obs_all.to(self._device))
```

If T1.5 named the model attribute differently from `self._model`, use T1.5's name (it is the object returned by the `model` property).

- [ ] **Step 6: Run the new tests.**

Run: `.venv/bin/python -m pytest tests/unit/test_obs_normalization.py -v`
Expected: all PASS (8 tests).

- [ ] **Step 7: Check that nothing else relied on forward-time updates.**

Run: `grep -rn "NormalizeObs\|RunningMeanStd\|\.rms\." src tests examples`. Any test that expected `forward()` in train mode to change the statistics is testing removed behaviour. Rewrite it to call `update()` first, or delete it if `test_obs_normalization.py` already covers it.

- [ ] **Step 8: Run the full fast suite.**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: PASS.

- [ ] **Step 9: Commit.**

```bash
git add src/colosseum/networks/normalization.py src/colosseum/networks/model.py src/colosseum/networks/composed.py \
        src/colosseum/algorithms/appo.py tests/unit/test_obs_normalization.py
git commit -m "fix: NormalizeObs updates only via update(), once per train step (R1-08)"
```

---
### Task T4.3: Masked KL; kickstart with forward KL, masks, `unroll` and the student's own distribution

Spec block 4 "Маски" and "Kickstart"; findings R1-13, R1-14, R1-02 (kickstart part), research B (G1).

Today:
- `CategoricalDist.kl_divergence` gives NaN or inf as soon as a mask is involved;
- kickstart uses reverse KL(student‖teacher), whereas the kickstarting paper, AlphaStar and VPT all use KL(teacher‖student);
- kickstart ignores masks;
- kickstart bypasses the recurrent core;
- kickstart does a second student forward.

After this task:
- **Masked KL.** `CategoricalDist` remembers its mask. Its KL is computed over the legal actions only, meaning the intersection of both distributions' legal sets, with both sides renormalized on that set. A row with no common legal action gives 0. Nothing is ever NaN or inf.
- **KL direction.** `training.kickstart_kl: "forward" | "reverse"`, default `"forward"` = KL(teacher‖student).
- **Teacher.** The teacher is a frozen `PolicyModel`. It is unrolled over the same `[T, B]` chunks as the student, with the chunks' masks, dones and initial states. In SP1 the teacher is built from the student's config, so it shares the student's state layout. APPO validates that at construction.
- **Student distribution.** The student's distribution comes from APPO's main `unroll`; there is no extra student forward.
- **Lambda.** Lambda decays linearly in train steps. `KickstartLoss.state_dict()` stores the step (used by T4.6).

**Files:**
- Modify: `src/colosseum/networks/distributions.py` (`CategoricalDist`: `__init__`, new `mask` property, `apply_mask`, `kl_divergence`)
- Modify: `src/colosseum/bc/kickstart.py` (whole file)
- Modify: `src/colosseum/algorithms/appo.py` (`__init__` teacher check; `compute_loss` whole method; module helper)
- Modify: `src/colosseum/core/config.py` (`TrainingConfig.kickstart_kl`, `kickstart_teacher` description)
- Modify: `src/colosseum/launcher.py`, `src/colosseum/distributed.py` (pass `direction=` to `KickstartLoss`)
- Modify: `CLAUDE.md` (two KL-direction mentions)
- Modify: existing kickstart tests (found with grep, Step 9)
- Test: `tests/unit/test_kickstart_kl.py` (create)

**Interfaces:**
- Consumes:
  - `PolicyModel.unroll(obs [T,B,...], state0, dones [T,B] bool, action_mask [T,B,A] | None) -> UnrollOutput(dist over T*B time-major, value [T*B])` (T1.2). `ComposedModel` applies `action_mask` with `dist.apply_mask` (T1.4).
  - `StepOutput`, `UnrollOutput`, `PolicyModel` from `colosseum.networks.model`; `cat_batch`, `state_to`, `tree_leaves` from `colosseum.networks.state` (T1.1).
  - APPO internals after T1.5/T2.5:
    - attributes `self._model`, `self._config`, `self._device`, `self._use_amp`, `self._amp_dtype`, `self._zero_loss`, `self._kickstart`, `self._compute_vtrace`;
    - `self._prepare_batch(chunks)` returns `[T, B, ...]` tensors under the keys `observations`, `actions`, `behavior_log_probs`, `rewards`, `dones`, `bootstrap_values` ([B]) and optional `action_masks`;
    - `TrajectoryChunk.initial_state` has leaves `[1, ...]`.
  - Test kit from T4.1.
- Produces:
  - `CategoricalDist.mask -> Tensor | None` (bool, legal = True).
  - `CategoricalDist.kl_divergence(other)` restricted to `self.mask & other.mask`.
  - `KickstartLoss(teacher: PolicyModel, initial_lambda: float = 1.0, decay_steps: int = 50_000, direction: Literal["forward","reverse"] = "forward")` with:
    - `.teacher`, `.direction`, `.current_lambda`, `.step_count`;
    - `.step()`, `.to(device)`;
    - `.state_dict() -> {"step": int}`, `.load_state_dict(state)`;
    - `.compute(student_dist, observations, dones, state0, action_mask=None) -> Tensor` (scalar, already multiplied by lambda).
  - `TrainingConfig.kickstart_kl: Literal["forward", "reverse"] = "forward"`.
  - In APPO, `compute_loss` binds `out` (`UnrollOutput`), `state0`, `masks` and `dones`. T4.6 builds on this method.

- [ ] **Step 1: Write the failing test** `tests/unit/test_kickstart_kl.py`:

```python
"""Masked KL and kickstart (T4.3)."""
import copy
import math

import pytest
import torch
from pydantic import ValidationError

from colosseum.algorithms.appo import APPO
from colosseum.bc.kickstart import KickstartLoss
from colosseum.core.config import AlgorithmConfig, TrainingConfig
from colosseum.networks.distributions import CategoricalDist, CompositeDist, DiagGaussianDist
from colosseum.networks.model import PolicyModel, StepOutput, UnrollOutput
from tests.helpers import MaskedToyEnv, rollout_chunks, tiny_model

P = [0.5, 0.5]
Q = [0.9, 0.1]
KL_PQ = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)   # 0.5108
KL_QP = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)   # 0.3681


class FixedTeacher(PolicyModel):
    """Stateless teacher with the same action probabilities for every observation."""

    def __init__(self, probs):
        super().__init__()
        self.register_buffer("logits", torch.log(torch.tensor(probs)))

    def step(self, obs, state, action_mask=None):
        dist = CategoricalDist(self.logits.expand(obs.shape[0], -1))
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        return StepOutput(dist, torch.zeros(obs.shape[0]), state)

    def unroll(self, obs, state0, dones, action_mask=None):
        T, B = obs.shape[:2]
        flat_mask = None if action_mask is None else action_mask.reshape(T * B, -1)
        out = self.step(obs.reshape(T * B, *obs.shape[2:]), state0, flat_mask)
        return UnrollOutput(out.dist, out.value)


def _student(probs, n):
    return CategoricalDist(torch.log(torch.tensor(probs)).expand(n, -1))


def test_categorical_kl_known_values():
    p, q = _student(P, 1), _student(Q, 1)
    assert p.kl_divergence(q).item() == pytest.approx(KL_PQ, abs=1e-6)
    assert q.kl_divergence(p).item() == pytest.approx(KL_QP, abs=1e-6)


@pytest.mark.parametrize("direction,expected", [("forward", KL_PQ), ("reverse", KL_QP)])
def test_kickstart_direction(direction, expected):
    T, B = 3, 2
    ks = KickstartLoss(FixedTeacher(P), initial_lambda=1.0, decay_steps=10, direction=direction)
    obs = torch.zeros(T, B, 1)
    dones = torch.zeros(T, B, dtype=torch.bool)
    loss = ks.compute(_student(Q, T * B), obs, dones, None, None)
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_kickstart_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        KickstartLoss(FixedTeacher(P), direction="sideways")
    assert TrainingConfig().kickstart_kl == "forward"
    assert TrainingConfig(kickstart_kl="reverse").kickstart_kl == "reverse"
    with pytest.raises(ValidationError):
        TrainingConfig(kickstart_kl="sideways")


MASK = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)


@pytest.mark.parametrize("case", ["none", "teacher_only", "student_only", "both", "disjoint"])
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_masked_kl_is_finite_with_finite_grad(case, direction):
    torch.manual_seed(0)
    student_logits = torch.randn(3, 4, requires_grad=True)
    teacher_logits = torch.randn(3, 4)
    s_mask = {"none": None, "teacher_only": None, "student_only": MASK, "both": MASK, "disjoint": ~MASK}[case]
    t_mask = {"none": None, "teacher_only": MASK, "student_only": None, "both": MASK, "disjoint": MASK}[case]
    student = CategoricalDist(student_logits, s_mask)
    teacher = CategoricalDist(teacher_logits, t_mask)
    kl = teacher.kl_divergence(student) if direction == "forward" else student.kl_divergence(teacher)
    assert kl.shape == (3,)
    assert torch.isfinite(kl).all()
    assert (kl >= -1e-6).all()
    kl.sum().backward()
    assert torch.isfinite(student_logits.grad).all()
    if case == "disjoint":
        assert torch.equal(kl, torch.zeros(3))


def test_masked_kl_renormalizes_over_legal_actions():
    p = torch.softmax(torch.tensor([1.0, 2.0, 0.0]), 0)
    q = torch.softmax(torch.tensor([0.0, 0.0, 5.0]), 0)
    mask = torch.tensor([[True, True, False]])
    kl = CategoricalDist(torch.log(p)[None], mask).kl_divergence(CategoricalDist(torch.log(q)[None]))
    pp, qq = p[:2] / p[:2].sum(), q[:2] / q[:2].sum()
    assert kl.item() == pytest.approx((pp * (pp / qq).log()).sum().item(), abs=1e-6)


def test_masked_kl_of_identical_distributions_is_zero():
    logits = torch.randn(3, 4)
    kl = CategoricalDist(logits, MASK).kl_divergence(CategoricalDist(logits.clone(), MASK))
    assert torch.equal(kl, torch.zeros(3))


def test_apply_mask_combines_masks():
    dist = CategoricalDist(torch.zeros(1, 4), torch.tensor([[True, True, True, False]]))
    both = dist.apply_mask(torch.tensor([[False, True, True, True]]))
    assert both.mask.tolist() == [[False, True, True, False]]


def test_composite_kl_with_masks_is_finite():
    def make(logits, mask):
        return CompositeDist({
            "a": CategoricalDist(logits, mask),
            "b": DiagGaussianDist(torch.zeros(3, 2), torch.zeros(3, 2)),
        })
    kl = make(torch.randn(3, 4), MASK).kl_divergence(make(torch.randn(3, 4), None))
    assert torch.isfinite(kl).all()


def test_lambda_decays_linearly_and_round_trips():
    ks = KickstartLoss(FixedTeacher(P), initial_lambda=2.0, decay_steps=4)
    assert ks.current_lambda == pytest.approx(2.0)
    ks.step()
    ks.step()
    assert ks.current_lambda == pytest.approx(1.0)
    state = ks.state_dict()
    assert state == {"step": 2}
    other = KickstartLoss(FixedTeacher(P), initial_lambda=2.0, decay_steps=4)
    other.load_state_dict(state)
    assert other.current_lambda == pytest.approx(1.0)
    for _ in range(10):
        other.step()
    assert other.current_lambda == 0.0
    loss = other.compute(_student(Q, 2), torch.zeros(1, 2, 1), torch.zeros(1, 2, dtype=torch.bool), None)
    assert loss.item() == 0.0


def test_kickstart_teacher_is_frozen():
    teacher = tiny_model()
    KickstartLoss(teacher)
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not teacher.training


def test_kickstart_lstm_masked_teacher_equal_to_student_gives_zero_kl():
    model = tiny_model(core="lstm")
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    # Chunks of 8 over 5-step episodes: dones inside chunks and chunks starting mid-episode.
    assert any(bool(torch.as_tensor(c.dones).any()) for c in chunks)
    assert all(c.initial_state is not None for c in chunks)
    teacher = copy.deepcopy(model)
    algo = APPO(
        model, AlgorithmConfig(num_epochs=1, minibatch_chunks=0, learning_rate=1e-2),
        device="cpu", kickstart=KickstartLoss(teacher, initial_lambda=1.0, decay_steps=100),
    )
    first = algo.train_step(chunks)
    # Same weights, same initial states, same masks and resets -> KL is exactly 0.
    assert first["kickstart_loss"] < 1e-6
    second = algo.train_step(chunks)
    assert second["kickstart_loss"] > 1e-8
    assert all(math.isfinite(v) for v in second.values())


def test_appo_rejects_teacher_with_different_state_layout():
    with pytest.raises(ValueError, match="state layout"):
        APPO(tiny_model(core="lstm"), AlgorithmConfig(), device="cpu",
             kickstart=KickstartLoss(tiny_model(core="none")))
```

- [ ] **Step 2: Run it and watch it fail.**

Run: `.venv/bin/python -m pytest tests/unit/test_kickstart_kl.py -v`

Expected failures:
- `TypeError: KickstartLoss.__init__() got an unexpected keyword argument 'direction'`;
- NaN/inf assertions in `test_masked_kl_is_finite_with_finite_grad`;
- `AttributeError: 'CategoricalDist' object has no attribute 'mask'`;
- `TrainingConfig` has no field `kickstart_kl`.

- [ ] **Step 3: Masked `CategoricalDist`.**

In `src/colosseum/networks/distributions.py`, class `CategoricalDist`, make these changes and leave `logits`, `action_dim`, `sample`, `log_prob`, `entropy` and `mode` as they are.

Replace `__init__` with:

```python
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self._mask: Optional[torch.Tensor] = None
        if mask is not None:
            self._mask = mask.bool()
            logits = logits.masked_fill(~self._mask, float("-inf"))
        self._dist = torch.distributions.Categorical(logits=logits)
```

Add the property:

```python
    @property
    def mask(self) -> Optional[torch.Tensor]:
        """Bool legal-action mask (True = legal), or None when unmasked."""
        return self._mask
```

Replace `kl_divergence` and `apply_mask` with:

```python
    def kl_divergence(self, other: Distribution) -> torch.Tensor:
        """KL(self || other) over legal actions only.

        The legal set is the intersection of both distributions' masks; both sides are
        renormalized on it. Rows with no common legal action give 0. The result is
        finite even when only one side is masked; it is computed in float32.
        """
        if not isinstance(other, CategoricalDist):
            raise TypeError(f"Cannot compute KL between CategoricalDist and {type(other).__name__}")
        legal = self._mask
        if other.mask is not None:
            legal = other.mask if legal is None else (legal & other.mask)
        p_logits = self.logits.float()
        q_logits = other.logits.float()
        if legal is None:
            log_p = F.log_softmax(p_logits, dim=-1)
            log_q = F.log_softmax(q_logits, dim=-1)
            return (log_p.exp() * (log_p - log_q)).sum(dim=-1)
        neg = torch.finfo(p_logits.dtype).min
        log_p = F.log_softmax(p_logits.masked_fill(~legal, neg), dim=-1)
        log_q = F.log_softmax(q_logits.masked_fill(~legal, neg), dim=-1)
        terms = torch.where(legal, log_p.exp() * (log_p - log_q), torch.zeros_like(log_p))
        return terms.sum(dim=-1)

    def apply_mask(self, mask: torch.Tensor) -> CategoricalDist:
        """Return a new CategoricalDist with invalid actions masked out (masks combine)."""
        mask = mask.bool()
        if self._mask is not None:
            mask = mask & self._mask
        return CategoricalDist(logits=self.logits, mask=mask)
```

`F` (`torch.nn.functional`) is already imported in this module.

- [ ] **Step 4: Replace `src/colosseum/bc/kickstart.py` with:**

```python
"""Online behavioral cloning via kickstarting (Schmitt et al., 2018).

Adds ``lambda * KL`` between a frozen teacher policy and the student to the RL
loss. Lambda decays linearly from ``initial_lambda`` to 0 over ``decay_steps``
train steps.

Direction (``training.kickstart_kl``):
- ``"forward"`` (default): KL(teacher || student), i.e. the teacher-to-student
  cross-entropy minus a constant, as in Kickstarting, AlphaStar and VPT. It is
  mode-covering, so the student keeps the teacher's diversity.
- ``"reverse"``: KL(student || teacher), mode-seeking.

The teacher is a ``PolicyModel`` unrolled over the same ``[T, B]`` sequences as
the student, with the chunks' action masks, episode resets (``dones``) and initial
states. The student distribution is passed in from the algorithm's main forward,
so there is no second student pass. Both distributions are masked, and the masked
KL is computed over legal actions only.

In SP1 the teacher is built from the student's config, so both share one state
layout and the chunk's ``initial_state``, recorded by the student's behavior
policy, is used as the teacher's ``state0``. That is exact when teacher == student
(the usual BC -> RL start) and an approximation afterwards. A separate teacher
config with its own state is SP3.
"""

from __future__ import annotations

from typing import Literal

import torch

from colosseum.networks.distributions import Distribution
from colosseum.networks.model import PolicyModel
from colosseum.networks.state import State

KLDirection = Literal["forward", "reverse"]


class KickstartLoss:
    """Decaying KL penalty between a frozen teacher and the student policy."""

    def __init__(
        self,
        teacher: PolicyModel,
        initial_lambda: float = 1.0,
        decay_steps: int = 50_000,
        direction: KLDirection = "forward",
    ) -> None:
        if direction not in ("forward", "reverse"):
            raise ValueError(f"kickstart direction must be 'forward' or 'reverse', got {direction!r}")
        self._teacher = teacher
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)
        self._initial_lambda = float(initial_lambda)
        self._decay_steps = max(1, int(decay_steps))
        self._direction: KLDirection = direction
        self._current_step = 0

    @property
    def teacher(self) -> PolicyModel:
        return self._teacher

    @property
    def direction(self) -> KLDirection:
        return self._direction

    @property
    def step_count(self) -> int:
        return self._current_step

    @property
    def current_lambda(self) -> float:
        """Current (decayed) lambda."""
        progress = min(1.0, self._current_step / self._decay_steps)
        return self._initial_lambda * (1.0 - progress)

    def step(self) -> None:
        """Advance the decay by one train step."""
        self._current_step += 1

    def to(self, device: str | torch.device) -> KickstartLoss:
        self._teacher.to(device)
        return self

    def state_dict(self) -> dict[str, int]:
        return {"step": self._current_step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._current_step = int(state["step"])

    def compute(
        self,
        student_dist: Distribution,
        observations: torch.Tensor,
        dones: torch.Tensor,
        state0: State,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scaled kickstart loss ``lambda * mean(KL)``.

        Args:
            student_dist: the student's (masked) distribution over ``T*B`` time-major
                rows, from the algorithm's main ``unroll``.
            observations: ``[T, B, *obs_shape]``.
            dones: ``[T, B]`` bool; the state is reset after a done step.
            state0: initial state for the teacher's unroll (leaves ``[B, ...]``).
            action_mask: ``[T, B, mask_size]`` bool or None.
        """
        lam = self.current_lambda
        if lam <= 0:
            return torch.zeros((), device=observations.device)
        with torch.no_grad():
            teacher_dist = self._teacher.unroll(observations, state0, dones.bool(), action_mask).dist
        if self._direction == "forward":
            kl = teacher_dist.kl_divergence(student_dist)
        else:
            kl = student_dist.kl_divergence(teacher_dist)
        return lam * kl.mean()
```

- [ ] **Step 5: APPO: teacher check in `__init__`.**

In `src/colosseum/algorithms/appo.py`, add `cat_batch`, `state_to` and `tree_leaves` to the `colosseum.networks.state` import (create the import line if T1.5 has none). Then add this module-level helper above `class APPO`:

```python
def _check_teacher_state_layout(student: PolicyModel, teacher: PolicyModel) -> None:
    """SP1 teachers reuse the student's chunk initial states, so the layouts must match."""
    student_shapes = [tuple(t.shape) for t in tree_leaves(student.initial_state(1))]
    teacher_shapes = [tuple(t.shape) for t in tree_leaves(teacher.initial_state(1))]
    if student_shapes != teacher_shapes:
        raise ValueError(
            "kickstart teacher must share the student's state layout in SP1 "
            f"(student state leaves {student_shapes}, teacher {teacher_shapes}); "
            "build the teacher from the student's networks config"
        )
```

In `APPO.__init__`, right after the line that moves the model to the device (`self._model = model.to(device)` or T1.5's equivalent) and before the optimizer is created, add:

```python
        if kickstart is not None:
            _check_teacher_state_layout(self._model, kickstart.teacher)
            kickstart.to(device)
```

- [ ] **Step 6: APPO: `compute_loss` uses the main-forward distribution for kickstart.**

Replace the whole `compute_loss` method with the version below. Lines marked `# (T1.5)` are what T1.5 already wrote: batch preparation, initial state and unroll. If T1.5 wrote them differently (for example extra assertions or pinning inside `_prepare_batch`), keep T1.5's version of those lines. You must still bind the names `batch`, `T`, `B`, `flat_actions`, `masks`, `dones`, `state0`, `amp_ctx` and `out`, because the rest of the method and T4.6 use them.

```python
    def compute_loss(self, chunks: list[TrajectoryChunk]) -> dict[str, torch.Tensor]:
        """APPO loss for one minibatch of chunks.

        1. Stack chunks to [T, B, ...]; unroll the model from the chunks' initial
           states with the chunks' dones and action masks.
        2. V-trace(lambda) targets and advantages from the recomputed log-probs/values.
        3. PPO clipped surrogate on the V-trace advantages, value MSE, entropy bonus.
        4. Optional kickstart KL on the same (masked) student distribution.
        """
        cfg = self._config
        batch = self._prepare_batch(chunks)                                       # (T1.5)
        T, B = batch["rewards"].shape                                             # (T1.5)
        act_shape = batch["actions"].shape[2:]                                    # (T1.5)
        flat_actions = batch["actions"].reshape(T * B, *act_shape)                # (T1.5)
        masks = batch.get("action_masks")                                         # (T1.5) [T, B, A] | None
        dones = batch["dones"].bool()                                             # (T1.5) [T, B]
        state0 = state_to(cat_batch([c.initial_state for c in chunks]), self._device)  # (T1.5)

        amp_ctx = torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=self._use_amp)
        with amp_ctx:
            out = self._model.unroll(batch["observations"], state0, dones, masks)  # (T1.5)
            target_log_probs = out.dist.log_prob(flat_actions).float().reshape(T, B)
            entropy = out.dist.entropy().float().reshape(T, B)
            new_values = out.value.float().reshape(T, B)

        # V-trace targets and advantages
        with torch.no_grad():
            vtrace_targets, vtrace_advantages = self._compute_vtrace(
                behavior_log_probs=batch["behavior_log_probs"],
                target_log_probs=target_log_probs.detach(),
                rewards=batch["rewards"],
                values=new_values.detach(),
                bootstrap_value=batch["bootstrap_values"],
                dones=dones,
                gamma=cfg.gamma,
                rho_bar=cfg.vtrace_rho_bar,
                c_bar=cfg.vtrace_c_bar,
                lam=cfg.vtrace_lambda,
            )

        # PPO clipped surrogate loss
        log_ratio = torch.clamp(target_log_probs - batch["behavior_log_probs"], -20.0, 20.0)
        ratio = torch.exp(log_ratio)
        adv = vtrace_advantages.detach()
        if cfg.normalize_advantages and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(new_values, vtrace_targets.detach())
        entropy_loss = -entropy.mean()
        total_loss = policy_loss + cfg.value_loss_coeff * value_loss + cfg.entropy_coeff * entropy_loss

        # Kickstart: KL between the frozen teacher and THIS forward's student distribution.
        kickstart_loss = self._zero_loss
        if self._kickstart is not None and self._kickstart.current_lambda > 0:
            with amp_ctx:
                kickstart_loss = self._kickstart.compute(
                    student_dist=out.dist,
                    observations=batch["observations"],
                    dones=dones,
                    state0=state0,
                    action_mask=masks,
                )
            total_loss = total_loss + kickstart_loss

        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > cfg.eps_clip).float().mean()

        result = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": -entropy_loss,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }
        if self._kickstart is not None:
            result["kickstart_loss"] = kickstart_loss.detach()
            result["kickstart_lambda"] = torch.tensor(self._kickstart.current_lambda)
        return result
```

`F` (`torch.nn.functional`) and `TrajectoryChunk` are already imported in `appo.py`. Leave `train_step` as it is: it already calls `self._kickstart.step()` once per train step.

- [ ] **Step 7: Config.** In `src/colosseum/core/config.py`:

Add `Literal` to the `typing` import if it is not there.

In `TrainingConfig`, replace the `kickstart_teacher` description string with:

```python
        description="Path to a frozen teacher .pt state_dict (e.g. a BC model), built from this "
                    "agent's networks config. When set, a decaying KL term between teacher and "
                    "student is added to the RL loss (direction: kickstart_kl). None disables it.",
```

Then add this field after `kickstart_decay_steps`:

```python
    kickstart_kl: Literal["forward", "reverse"] = Field(
        default="forward",
        description="Kickstart KL direction: 'forward' = KL(teacher || student) (Kickstarting / "
                    "AlphaStar / VPT, mode-covering); 'reverse' = KL(student || teacher).",
    )
```

- [ ] **Step 8: Pass the direction where the teacher is built.**

Run `grep -n "KickstartLoss(" src/colosseum/launcher.py src/colosseum/distributed.py`. In both files the call currently passes `initial_lambda=` and `decay_steps=`. Add the direction from the same config object the call already reads its other kickstart settings from:
- in `launcher.py` that object is `config`, so add `direction=config.training.kickstart_kl,`;
- in `distributed.py` it is `acfg`, so add `direction=acfg.training.kickstart_kl,`.

The `launcher.py` call then reads:

```python
            kickstart = KickstartLoss(
                teacher,
                initial_lambda=config.training.kickstart_lambda,
                decay_steps=config.training.kickstart_decay_steps,
                direction=config.training.kickstart_kl,
            )
```

- [ ] **Step 9: Replace the old kickstart tests.**

Run `grep -rln "KickstartLoss" tests/ | grep -v test_kickstart_kl.py`. In each listed file, delete every test function that constructs `KickstartLoss` or calls its `.compute(`. Before this part they were `test_kickstart_lambda_decay`, `test_kickstart_loss_computation`, `test_kickstart_zero_lambda` and `test_appo_with_kickstart`. They test the old signature, and `tests/unit/test_kickstart_kl.py` covers the same behaviour.

Then remove imports that are now unused (`ruff check tests` reports them). If a file ends up with no tests, delete the file.

- [ ] **Step 10: Docs.** In `CLAUDE.md`:
- replace `` - Online BC (kickstarting): `loss = RL_loss + λ * KL(policy || BC_policy)`, λ decays over training `` with `` - Online BC (kickstarting): `loss = RL_loss + λ * KL(BC_teacher || policy)` (forward KL by default, `training.kickstart_kl`), λ decays over training ``;
- replace the table cell `KL(student \|\| teacher) with linear lambda decay` with `KL(teacher \|\| student) by default (configurable), masked, unrolled teacher, linear lambda decay`.

- [ ] **Step 11: Run the new tests.**

Run: `.venv/bin/python -m pytest tests/unit/test_kickstart_kl.py -v`
Expected: all PASS.

- [ ] **Step 12: Run the full fast suite.**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: PASS. If an older test asserts `KL(unmasked || masked) == inf` or NaN, it encodes the R1-13 bug: delete it.

- [ ] **Step 13: Commit.**

```bash
git add src/colosseum/networks/distributions.py src/colosseum/bc/kickstart.py src/colosseum/algorithms/appo.py \
        src/colosseum/core/config.py src/colosseum/launcher.py src/colosseum/distributed.py CLAUDE.md tests
git commit -m "fix: masked KL over legal actions; kickstart forward KL with unrolled masked teacher (R1-13, R1-14)"
```

---
### Task T4.4: BC rewrite: masks, −log_prob for every distribution, stateful sequence training, strict action types

Spec block 4 "BC"; findings R1-05, R1-02 (BC part), R6-12 (misleading "Loss=" print). Problems today:
- BC ignores `action_masks`;
- it casts Box actions to `long` under the default `--action-type discrete`;
- it uses MSE on the mean for continuous actions, so `log_std` is never trained;
- it bypasses the recurrent core.

**Design:**
- Data keys are `observations`, `actions`, and the optional `action_masks` and `dones`.
- The loss is −log π(a|s) for Categorical, DiagGaussian and Composite. `--action-type` and the MSE path are gone.
- Masks go into the model, which applies them with `dist.apply_mask`.
- An expert action with zero probability (illegal under its mask) raises `ValueError`.
- Float actions for a `CategoricalDist` raise `ValueError`, and so do non-integer values in the discrete columns of a composite action.
- **Stateless models** train on shuffled transitions.
- **Stateful models** (`model.is_stateful`) train through `model.unroll` over contiguous windows of `seq_len` transitions:
  - each window starts from `initial_state`, and the state resets after every `done`;
  - a random window-grid offset each epoch moves the points where a window starts mid-episode;
  - windows are not carried over, because `UnrollOutput` has no final state; see Contract notes.
- Without `dones`, the transitions of each `add_data` call (one `.pt` file) count as one episode, with a warning.
- The window length comes from `bc.seq_len` in the config (default 64) or CLI `--seq-len`.

**Files:**
- Modify: `src/colosseum/bc/offline_bc.py` (whole file)
- Modify: `src/colosseum/networks/distributions.py` (`CompositeDist`: add a `components` property)
- Modify: `src/colosseum/core/config.py` (new `BCConfig`, `ColosseumConfig.bc`)
- Modify: `src/colosseum/cli.py` (`bc` command)
- Modify: existing BC tests (found with grep, Step 8)
- Test: `tests/unit/test_bc_trainer.py` (create), `tests/learning/test_bc_learns.py` (create), `tests/integration/test_bc_cli.py` (create)

**Interfaces:**
- Consumes:
  - `PolicyModel.step/unroll/initial_state/is_stateful` (T1.2), `build_model(config)` (T1.4), `load_config(path, overrides=None)` (T6.1 signature; a single positional path works before T6.1 too).
  - `ComposedModel`, `NoCore`, `LSTMCore` (T1.3/T1.4).
  - Test kit from T4.1: `TinyEncoder`, `TinyValue`, `tiny_model`, `MaskedToyEnv`.
- Produces:
  - `OfflineBCTrainer(model: PolicyModel, lr: float = 1e-3, device="cpu", seq_len: int = 64)` with:
    - `.model`, `.num_samples`;
    - `.add_data(observations, actions, action_masks=None, dones=None)`;
    - `.load_data(path) -> int`;
    - `.train(num_epochs=10, batch_size=256, log_interval=1) -> dict[str, float]`, whose keys are `bc_loss` (final-epoch mean NLL), `bc_loss_first_epoch`, `num_epochs`, `num_samples` and, when the action space has a discrete part, `accuracy`.
  - `colosseum.bc.offline_bc._window_index(n, seq_len, offset=0) -> LongTensor [W, seq_len]` (-1 = padding).
  - `CompositeDist.components -> list[tuple[str, int, int, bool]]`, i.e. `(name, flat_offset, size, is_discrete)` in layout order. T4.5 keeps this property.
  - `BCConfig(seq_len: int = 64, ge=1)` and `ColosseumConfig.bc: BCConfig`.
  - CLI: `colosseum bc -c CFG -d DATA -o OUT [--epochs] [--batch-size] [--lr] [--seq-len]`. `--action-type` is removed.

- [ ] **Step 1: Write the failing unit tests** `tests/unit/test_bc_trainer.py`:

```python
"""Offline BC: -log_prob for every distribution, masks, stateful windows, strict types (T4.4)."""
import logging

import pytest
import torch
import torch.nn as nn

from colosseum.bc.offline_bc import OfflineBCTrainer, _window_index
from colosseum.networks.base import BasePolicy
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import NoCore
from colosseum.networks.distributions import CategoricalDist, CompositeDist, DiagGaussianDist
from tests.helpers import TinyEncoder, TinyValue, tiny_model


class GaussPolicy(BasePolicy):
    def __init__(self, in_dim: int = 16, act_dim: int = 2) -> None:
        super().__init__()
        self.mean = nn.Linear(in_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, latent):
        return DiagGaussianDist(self.mean(latent), self.log_std.expand(latent.shape[0], -1))


class DirSpeedPolicy(BasePolicy):
    """Composite: direction Discrete(3) (flat column 0) + speed Box(1) (column 1)."""

    def __init__(self, in_dim: int = 16) -> None:
        super().__init__()
        self.direction = nn.Linear(in_dim, 3)
        self.speed = nn.Linear(in_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, latent):
        return CompositeDist({
            "direction": CategoricalDist(self.direction(latent)),
            "speed": DiagGaussianDist(self.speed(latent), self.log_std.expand(latent.shape[0], -1)),
        })


def _composed(policy):
    torch.manual_seed(0)
    return ComposedModel(TinyEncoder(4, 16), NoCore(input_dim=16), policy, TinyValue(16))


def _memory_data(num_episodes: int, seed: int = 0):
    """Episodes of 4 steps: a cue in {0,1,2} is visible only at step 0; the expert repeats it."""
    g = torch.Generator().manual_seed(seed)
    obs, actions, dones = [], [], []
    for _ in range(num_episodes):
        cue = int(torch.randint(0, 3, (1,), generator=g))
        for t in range(4):
            o = torch.zeros(4)
            if t == 0:
                o[cue] = 1.0
            o[3] = t / 3.0
            obs.append(o)
            actions.append(cue)
            dones.append(t == 3)
    return torch.stack(obs), torch.tensor(actions), torch.tensor(dones)


def test_window_index_tiles_data_in_order_with_padding():
    w = _window_index(10, 4, offset=0)
    assert w.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, -1, -1]]
    w = _window_index(10, 4, offset=3)
    assert w.tolist() == [[0, 1, 2, -1], [3, 4, 5, 6], [7, 8, 9, -1]]
    flat = w[w >= 0]
    assert flat.tolist() == list(range(10))


def test_discrete_bc_learns_and_reports_metrics():
    torch.manual_seed(0)
    obs = torch.randn(256, 4)
    actions = obs.argmax(dim=1)
    trainer = OfflineBCTrainer(tiny_model(), lr=1e-2)
    trainer.add_data(obs, actions, dones=torch.zeros(256, dtype=torch.bool))
    metrics = trainer.train(num_epochs=30, batch_size=64)
    assert set(metrics) >= {"bc_loss", "bc_loss_first_epoch", "num_epochs", "num_samples", "accuracy"}
    assert metrics["num_samples"] == 256
    assert metrics["bc_loss"] < metrics["bc_loss_first_epoch"]
    assert metrics["accuracy"] > 0.9


def test_masks_are_applied():
    # Exactly one legal action per sample, equal to the expert action -> NLL is 0 from the start.
    n = 64
    actions = torch.randint(0, 4, (n,))
    masks = torch.nn.functional.one_hot(actions, 4).bool()
    trainer = OfflineBCTrainer(tiny_model())
    trainer.add_data(torch.randn(n, 4), actions, action_masks=masks, dones=torch.zeros(n, dtype=torch.bool))
    metrics = trainer.train(num_epochs=1, batch_size=16)
    assert metrics["bc_loss_first_epoch"] < 1e-6
    assert metrics["accuracy"] == 1.0


def test_illegal_expert_action_raises():
    actions = torch.zeros(8, dtype=torch.long)
    masks = torch.ones(8, 4, dtype=torch.bool)
    masks[3, 0] = False
    trainer = OfflineBCTrainer(tiny_model())
    trainer.add_data(torch.randn(8, 4), actions, action_masks=masks, dones=torch.zeros(8, dtype=torch.bool))
    with pytest.raises(ValueError, match="illegal"):
        trainer.train(num_epochs=1, batch_size=8)


def test_float_actions_with_discrete_space_raise():
    trainer = OfflineBCTrainer(tiny_model())
    trainer.add_data(torch.randn(8, 4), torch.rand(8), dones=torch.zeros(8, dtype=torch.bool))
    with pytest.raises(ValueError, match="discrete"):
        trainer.train(num_epochs=1)


def test_continuous_bc_uses_log_prob_and_trains_log_std():
    torch.manual_seed(0)
    obs = torch.randn(512, 4)
    actions = 0.5 * obs[:, :2]
    model = _composed(GaussPolicy(16, 2))
    trainer = OfflineBCTrainer(model, lr=1e-2)
    trainer.add_data(obs, actions, dones=torch.zeros(512, dtype=torch.bool))
    metrics = trainer.train(num_epochs=40, batch_size=128)
    assert "accuracy" not in metrics
    assert metrics["bc_loss"] < metrics["bc_loss_first_epoch"]
    # MSE on the mean would never move log_std; NLL shrinks it on near-deterministic data.
    log_stds = [p for name, p in model.named_parameters() if name.endswith("log_std")]
    assert log_stds and (log_stds[0] < -0.5).all()


def test_composite_bc_and_strict_discrete_columns():
    torch.manual_seed(0)
    obs = torch.randn(512, 4)
    direction = obs[:, :3].argmax(dim=1).float()
    speed = obs[:, 3:4]
    actions = torch.cat([direction[:, None], speed], dim=1)          # flat layout [direction, speed]
    masks = torch.ones(512, 3, dtype=torch.bool)
    trainer = OfflineBCTrainer(_composed(DirSpeedPolicy(16)), lr=1e-2)
    trainer.add_data(obs, actions, action_masks=masks, dones=torch.zeros(512, dtype=torch.bool))
    metrics = trainer.train(num_epochs=30, batch_size=128)
    assert metrics["bc_loss"] < metrics["bc_loss_first_epoch"]
    assert metrics["accuracy"] > 0.85

    bad = actions.clone()
    bad[0, 0] = 1.5
    trainer2 = OfflineBCTrainer(_composed(DirSpeedPolicy(16)))
    trainer2.add_data(obs, bad, action_masks=masks, dones=torch.zeros(512, dtype=torch.bool))
    with pytest.raises(ValueError, match="non-integer"):
        trainer2.train(num_epochs=1)


def test_stateful_bc_uses_unroll_with_resets():
    obs, actions, dones = _memory_data(400)
    lstm = OfflineBCTrainer(tiny_model(num_actions=3, core="lstm", hidden=32), lr=5e-3, seq_len=16)
    lstm.add_data(obs, actions, dones=dones)
    lstm_metrics = lstm.train(num_epochs=25, batch_size=128)

    flat = OfflineBCTrainer(tiny_model(num_actions=3), lr=5e-3)
    flat.add_data(obs, actions, dones=dones)
    flat_metrics = flat.train(num_epochs=25, batch_size=128)

    # The cue is visible only at step 0: memory is required for steps 1..3.
    assert lstm_metrics["accuracy"] >= 0.95
    assert flat_metrics["accuracy"] < 0.75


def test_missing_dones_warns_and_trains(caplog):
    trainer = OfflineBCTrainer(tiny_model(core="lstm"), seq_len=8)
    with caplog.at_level(logging.WARNING, logger="colosseum.bc.offline_bc"):
        trainer.add_data(torch.randn(20, 4), torch.randint(0, 4, (20,)))
    assert "dones" in caplog.text
    metrics = trainer.train(num_epochs=1, batch_size=16)
    assert torch.isfinite(torch.tensor(metrics["bc_loss"]))


def test_load_data_from_files(tmp_path):
    for i in range(2):
        torch.save({
            "observations": torch.randn(10, 4),
            "actions": torch.randint(0, 4, (10,)),
            "action_masks": torch.ones(10, 4, dtype=torch.bool),
            "dones": torch.zeros(10, dtype=torch.bool),
        }, tmp_path / f"part{i}.pt")
    trainer = OfflineBCTrainer(tiny_model())
    assert trainer.load_data(tmp_path) == 20
    assert trainer.num_samples == 20
    torch.save({"observations": torch.randn(3, 4)}, tmp_path / "broken.pt")
    with pytest.raises(ValueError, match="actions"):
        OfflineBCTrainer(tiny_model()).load_data(tmp_path / "broken.pt")


def test_mixed_presence_of_masks_is_rejected():
    trainer = OfflineBCTrainer(tiny_model())
    trainer.add_data(torch.randn(4, 4), torch.zeros(4, dtype=torch.long),
                     action_masks=torch.ones(4, 4, dtype=torch.bool), dones=torch.zeros(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="action_masks"):
        trainer.add_data(torch.randn(4, 4), torch.zeros(4, dtype=torch.long), dones=torch.zeros(4, dtype=torch.bool))
```

- [ ] **Step 2: Write the failing learning test** `tests/learning/test_bc_learns.py`:

```python
"""BC on a scripted expert reaches high accuracy and plays well (T4.4)."""
import numpy as np
import torch

from colosseum.bc.offline_bc import OfflineBCTrainer
from colosseum.networks.model import act
from tests.helpers import MaskedToyEnv, tiny_model


def _record_expert(num_steps: int, seed: int):
    env = MaskedToyEnv()
    obs, info = env.reset(seed=seed)
    data = {"observations": [], "actions": [], "action_masks": [], "dones": []}
    for _ in range(num_steps):
        action = env.expert_action()
        data["observations"].append(obs[0])
        data["actions"].append(action)
        data["action_masks"].append(info[0]["action_mask"])
        obs, reward, terminated, truncated, info = env.step({0: action})
        assert reward[0] == 1.0
        done = terminated[0] or truncated[0]
        data["dones"].append(done)
        if done:
            obs, info = env.reset()
    return {
        "observations": torch.as_tensor(np.stack(data["observations"])),
        "actions": torch.as_tensor(data["actions"], dtype=torch.long),
        "action_masks": torch.as_tensor(np.stack(data["action_masks"])),
        "dones": torch.as_tensor(data["dones"]),
    }


def test_bc_imitates_masked_scripted_expert():
    data = _record_expert(2000, seed=0)
    model = tiny_model()
    trainer = OfflineBCTrainer(model, lr=1e-2)
    trainer.add_data(**data)
    metrics = trainer.train(num_epochs=30, batch_size=256)
    assert metrics["accuracy"] >= 0.95

    env = MaskedToyEnv()
    obs, info = env.reset(seed=123)
    rewards = []
    model.eval()
    for _ in range(500):
        mask = torch.as_tensor(info[0]["action_mask"])[None]
        with torch.no_grad():
            out = act(model, torch.as_tensor(obs[0])[None], model.initial_state(1), mask, deterministic=True)
        obs, reward, terminated, truncated, info = env.step({0: int(out.actions[0])})
        rewards.append(reward[0])
        if terminated[0] or truncated[0]:
            obs, info = env.reset()
    assert np.mean(rewards) >= 0.9
```

- [ ] **Step 3: Write the failing CLI test** `tests/integration/test_bc_cli.py`:

```python
"""`colosseum bc` CLI (T4.4)."""
import torch
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.config import load_config
from colosseum.core.registry import build_model

CONFIG = """\
env:
  env_class: examples.tic_tac_toe.env.TicTacToeEnv
  num_players: 2
networks:
  encoder_class: examples.tic_tac_toe.networks.TicTacToeEncoder
  policy_class: examples.tic_tac_toe.networks.TicTacToePolicy
  value_class: examples.tic_tac_toe.networks.TicTacToeValue
"""


def _write_data(path, n=64):
    actions = torch.randint(0, 9, (n,))
    masks = torch.rand(n, 9) < 0.5
    masks[torch.arange(n), actions] = True
    torch.save({
        "observations": torch.rand(n, 3, 3, 3),
        "actions": actions,
        "action_masks": masks,
        "dones": torch.arange(n) % 5 == 4,
    }, path)


def test_bc_cli_trains_and_saves_loadable_weights(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(CONFIG)
    data = tmp_path / "data.pt"
    _write_data(data)
    out = tmp_path / "bc.pt"
    result = CliRunner().invoke(main, [
        "bc", "-c", str(cfg), "-d", str(data), "-o", str(out),
        "--epochs", "2", "--batch-size", "16", "--seq-len", "8",
    ])
    assert result.exit_code == 0, result.output
    assert "final-epoch NLL" in result.output
    model = build_model(load_config(cfg))
    model.load_state_dict(torch.load(out, weights_only=True))


def test_bc_cli_has_no_action_type_option(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(CONFIG)
    data = tmp_path / "data.pt"
    _write_data(data)
    result = CliRunner().invoke(main, [
        "bc", "-c", str(cfg), "-d", str(data), "-o", str(tmp_path / "x.pt"), "--action-type", "discrete",
    ])
    assert result.exit_code == 2
    assert "No such option" in result.output
```

- [ ] **Step 4: Run them and watch them fail.**

Run: `.venv/bin/python -m pytest tests/unit/test_bc_trainer.py tests/learning/test_bc_learns.py tests/integration/test_bc_cli.py -v`

Expected failures:
- `ImportError: cannot import name '_window_index'`;
- `TypeError` for `seq_len=` / `action_masks=`;
- the CLI tests fail on the unknown `--seq-len` option.

- [ ] **Step 5: `CompositeDist.components`.**

In `src/colosseum/networks/distributions.py`, class `CompositeDist`, add this property right after `action_dim`:

```python
    @property
    def components(self) -> list[tuple[str, int, int, bool]]:
        """``(name, flat_offset, size, is_discrete)`` per component, in flat-layout order."""
        return [(k, *self._layout[k]) for k in self._keys]
```

- [ ] **Step 6: Replace `src/colosseum/bc/offline_bc.py` with:**

```python
"""Offline behavioral cloning (BC).

Trains a :class:`~colosseum.networks.model.PolicyModel` by maximum likelihood on
recorded expert transitions: loss = -log pi(a | s) for every distribution type
(Categorical, DiagGaussian, Composite), with the recorded action masks applied.

Data format: one or more ``.pt`` files (``torch.save`` of a dict) with
- ``observations``: ``[N, *obs_shape]`` (any numeric dtype; cast to float32);
- ``actions``: ``[N]`` integer tensor for a Discrete space, ``[N, D]`` float for
  a Box, ``[N, flat_size]`` float in the ``ActionSpec`` flat layout for
  Dict/Tuple/MultiDiscrete spaces;
- ``action_masks`` (optional): ``[N, flat_mask_size]`` bool, True = legal;
- ``dones`` (optional): ``[N]`` bool, True when transition t ends its episode.

Stateless models train on shuffled transitions. Stateful models
(``model.is_stateful``) train with ``model.unroll`` over contiguous windows of
``seq_len`` transitions. Each window starts from ``model.initial_state`` and the
state is reset after every ``done``. A window that starts mid-episode loses the
context before it (``UnrollOutput`` carries no final state); a random window
offset per epoch moves those cut points. Without ``dones``, the transitions of
each file (``add_data`` call) are treated as one episode, with a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from colosseum.networks.distributions import CategoricalDist, CompositeDist, Distribution
from colosseum.networks.model import PolicyModel

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {"observations", "actions"}
_KNOWN_KEYS = _REQUIRED_KEYS | {"action_masks", "dones"}
_EVAL_BATCH = 4096


def _window_index(n: int, seq_len: int, offset: int = 0) -> torch.Tensor:
    """``[W, seq_len]`` dataset indices of consecutive windows; -1 marks padding.

    The windows tile ``range(n)`` in order: ``[0, offset)`` when ``offset > 0``,
    then ``[offset, offset + seq_len)``, and so on; the last window may be shorter.
    """
    if n <= 0:
        return torch.empty(0, seq_len, dtype=torch.long)
    starts = list(range(offset, n, seq_len)) if offset > 0 else list(range(0, n, seq_len))
    if offset > 0:
        starts = [0] + starts
    ends = starts[1:] + [n]
    index = torch.full((len(starts), seq_len), -1, dtype=torch.long)
    for w, (start, end) in enumerate(zip(starts, ends)):
        index[w, : end - start] = torch.arange(start, end)
    return index


def _discrete_columns(dist: CompositeDist) -> list[int]:
    return [offset for _name, offset, _size, discrete in dist.components if discrete]


def _hits(dist: Distribution, actions: torch.Tensor) -> torch.Tensor | None:
    """1.0 where the mode matches the expert on every discrete component; None if none."""
    if isinstance(dist, CategoricalDist):
        return (dist.mode() == actions).float()
    if isinstance(dist, CompositeDist):
        cols = _discrete_columns(dist)
        if not cols:
            return None
        return (dist.mode()[:, cols] == actions[:, cols]).all(dim=-1).float()
    return None


class OfflineBCTrainer:
    """Supervised behavioral cloning from offline data.

    Usage::

        trainer = OfflineBCTrainer(build_model(cfg), lr=1e-3, seq_len=64)
        trainer.load_data("expert_games/")      # or add_data(obs, actions, masks, dones)
        metrics = trainer.train(num_epochs=10, batch_size=256)
    """

    def __init__(
        self,
        model: PolicyModel,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
        seq_len: int = 64,
    ) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        self._device = torch.device(device)
        self._model = model.to(self._device)
        self._seq_len = int(seq_len)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._observations: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []
        self._masks: list[torch.Tensor | None] = []
        self._dones: list[torch.Tensor] = []

    @property
    def model(self) -> PolicyModel:
        return self._model

    @property
    def num_samples(self) -> int:
        return sum(len(o) for o in self._observations)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def add_data(
        self,
        observations: Any,
        actions: Any,
        action_masks: Any = None,
        dones: Any = None,
    ) -> None:
        """Add ``N`` transitions (see the module docstring for shapes and dtypes)."""
        obs = torch.as_tensor(observations)
        acts = torch.as_tensor(actions)
        n = obs.shape[0]
        if n == 0:
            raise ValueError("BC data is empty")
        if acts.shape[0] != n:
            raise ValueError(f"BC data: {n} observations but {acts.shape[0]} actions")
        masks = None
        if action_masks is not None:
            masks = torch.as_tensor(action_masks).bool()
            if masks.dim() != 2 or masks.shape[0] != n:
                raise ValueError(f"BC action_masks must have shape [N, mask_size], got {tuple(masks.shape)}")
        if self._masks and (masks is None) != (self._masks[0] is None):
            raise ValueError("BC data: either every batch/file has 'action_masks' or none does")
        if dones is None:
            logger.warning("BC data has no 'dones': treating these %d transitions as one episode", n)
            done_t = torch.zeros(n, dtype=torch.bool)
            done_t[-1] = True
        else:
            done_t = torch.as_tensor(dones).bool().reshape(-1)
            if done_t.shape[0] != n:
                raise ValueError(f"BC data: {n} observations but {done_t.shape[0]} dones")
        self._observations.append(obs)
        self._actions.append(acts)
        self._masks.append(masks)
        self._dones.append(done_t)

    def load_data(self, path: str | Path) -> int:
        """Load one ``.pt`` file or every ``*.pt`` in a directory; returns samples loaded."""
        path = Path(path)
        files = sorted(path.glob("*.pt")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"no .pt files in {path}")
        total = 0
        for f in files:
            data = torch.load(f, map_location="cpu", weights_only=True)
            if not isinstance(data, dict):
                raise ValueError(f"{f}: expected a dict with keys {sorted(_KNOWN_KEYS)}")
            missing = _REQUIRED_KEYS - set(data)
            if missing:
                raise ValueError(f"{f}: missing BC data keys {sorted(missing)} (need observations, actions)")
            unknown = set(data) - _KNOWN_KEYS
            if unknown:
                logger.warning("%s: ignoring unknown BC data keys %s", f, sorted(unknown))
            self.add_data(data["observations"], data["actions"], data.get("action_masks"), data.get("dones"))
            n = len(data["observations"])
            total += n
            logger.info("Loaded %d BC samples from %s", n, f)
        logger.info("BC dataset: %d samples", self.num_samples)
        return total

    def _dataset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        obs = torch.cat(self._observations).float()
        actions = torch.cat(self._actions)
        masks = None if self._masks[0] is None else torch.cat(self._masks)
        dones = torch.cat(self._dones)
        return obs, actions, masks, dones

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, num_epochs: int = 10, batch_size: int = 256, log_interval: int = 1) -> dict[str, float]:
        """Run BC; returns final-epoch NLL ``bc_loss`` and, for discrete parts, ``accuracy``."""
        if not self._observations:
            raise ValueError("No training data. Call add_data() or load_data() first.")
        if num_epochs < 1 or batch_size < 1:
            raise ValueError("num_epochs and batch_size must be >= 1")
        obs, actions, masks, dones = self._dataset()
        actions = self._prepare_actions(obs, actions, masks)
        stateful = self._model.is_stateful
        logger.info(
            "BC training: %d samples, %d epochs, batch_size=%d, %s",
            len(obs), num_epochs, batch_size,
            f"stateful (seq_len={self._seq_len})" if stateful else "stateless",
        )
        self._model.train()
        epoch_losses: list[float] = []
        for epoch in range(num_epochs):
            if stateful:
                loss = self._sequence_epoch(obs, actions, masks, dones, batch_size)
            else:
                loss = self._flat_epoch(obs, actions, masks, batch_size)
            epoch_losses.append(loss)
            if (epoch + 1) % log_interval == 0:
                logger.info("BC epoch %d/%d: nll=%.4f", epoch + 1, num_epochs, loss)
        metrics = {
            "bc_loss": epoch_losses[-1],
            "bc_loss_first_epoch": epoch_losses[0],
            "num_epochs": float(num_epochs),
            "num_samples": float(len(obs)),
        }
        accuracy = self._accuracy(obs, actions, masks, dones)
        if accuracy is not None:
            metrics["accuracy"] = accuracy
        return metrics

    def _prepare_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor | None,
    ) -> torch.Tensor:
        """Check the action dtype/shape against the policy's distribution; return training actions."""
        with torch.no_grad():
            m = None if masks is None else masks[:1].to(self._device)
            dist = self._model.step(
                obs[:1].to(self._device), self._model.initial_state(1, self._device), m,
            ).dist
        if isinstance(dist, CategoricalDist):
            if actions.is_floating_point():
                raise ValueError(
                    "BC actions are floating point but the policy's action space is discrete "
                    "(CategoricalDist); store discrete actions as an integer tensor of shape [N]"
                )
            if actions.dim() != 1:
                raise ValueError(f"discrete BC actions must have shape [N], got {tuple(actions.shape)}")
            return actions.long()
        actions = actions.float()
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        try:
            action_dim = dist.action_dim
        except NotImplementedError:
            action_dim = None
        if action_dim is not None and (actions.dim() != 2 or actions.shape[1] != action_dim):
            raise ValueError(
                f"BC actions must have shape [N, {action_dim}] for {type(dist).__name__}, "
                f"got {tuple(actions.shape)}"
            )
        if isinstance(dist, CompositeDist):
            cols = _discrete_columns(dist)
            if cols:
                sub = actions[:, cols]
                if not torch.equal(sub, sub.round()):
                    raise ValueError(
                        f"BC actions have non-integer values in the discrete components "
                        f"(flat columns {cols}) of a composite action space"
                    )
        return actions

    @staticmethod
    def _weighted_nll(
        dist: Distribution, actions: torch.Tensor, weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nll = -dist.log_prob(actions)
        bad = ~torch.isfinite(nll) & (weights > 0)
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} BC expert actions have zero probability under the policy: "
                f"they are illegal under their action_masks (fix the data)"
            )
        nll = torch.where(weights > 0, nll, torch.zeros_like(nll))
        return (nll * weights).sum(), weights.sum()

    def _flat_epoch(
        self, obs: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor | None, batch_size: int,
    ) -> float:
        total, count = 0.0, 0.0
        perm = torch.randperm(len(obs))
        for start in range(0, len(obs), batch_size):
            idx = perm[start:start + batch_size]
            o = obs[idx].to(self._device)
            m = None if masks is None else masks[idx].to(self._device)
            dist = self._model.step(o, self._model.initial_state(len(idx), self._device), m).dist
            weights = torch.ones(len(idx), device=self._device)
            loss_sum, weight = self._weighted_nll(dist, actions[idx].to(self._device), weights)
            self._optimizer.zero_grad()
            (loss_sum / weight).backward()
            self._optimizer.step()
            total += float(loss_sum.detach())
            count += float(weight)
        return total / max(count, 1.0)

    def _sequence_forward(
        self,
        index: torch.Tensor,
        obs: torch.Tensor,
        masks: torch.Tensor | None,
        dones: torch.Tensor,
    ) -> tuple[Distribution, torch.Tensor]:
        """Unroll windows ``index [L, b]``; returns (dist over L*b time-major rows, weights)."""
        valid = index >= 0
        safe = index.clamp(min=0)
        o = obs[safe].to(self._device)                                   # [L, b, *obs]
        d = (dones[safe] | ~valid).to(self._device)                      # [L, b]
        m = None if masks is None else masks[safe].to(self._device)      # [L, b, A]
        state0 = self._model.initial_state(index.shape[1], self._device)
        dist = self._model.unroll(o, state0, d, m).dist
        return dist, valid.reshape(-1).to(self._device).float()

    def _sequence_epoch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor | None,
        dones: torch.Tensor,
        batch_size: int,
    ) -> float:
        seq_len = self._seq_len
        offset = int(torch.randint(0, seq_len, (1,)).item())
        windows = _window_index(len(obs), seq_len, offset)
        per_batch = max(1, batch_size // seq_len)
        order = torch.randperm(len(windows))
        total, count = 0.0, 0.0
        for start in range(0, len(windows), per_batch):
            index = windows[order[start:start + per_batch]].t()          # [L, b]
            dist, weights = self._sequence_forward(index, obs, masks, dones)
            a = actions[index.clamp(min=0)].reshape(index.numel(), *actions.shape[1:]).to(self._device)
            loss_sum, weight = self._weighted_nll(dist, a, weights)
            self._optimizer.zero_grad()
            (loss_sum / weight).backward()
            self._optimizer.step()
            total += float(loss_sum.detach())
            count += float(weight)
        return total / max(count, 1.0)

    @torch.no_grad()
    def _accuracy(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor | None,
        dones: torch.Tensor,
    ) -> float | None:
        self._model.eval()
        try:
            correct, total = 0.0, 0.0
            if self._model.is_stateful:
                windows = _window_index(len(obs), self._seq_len, 0)
                per_batch = max(1, _EVAL_BATCH // self._seq_len)
                for start in range(0, len(windows), per_batch):
                    index = windows[start:start + per_batch].t()
                    dist, weights = self._sequence_forward(index, obs, masks, dones)
                    a = actions[index.clamp(min=0)].reshape(index.numel(), *actions.shape[1:]).to(self._device)
                    hits = _hits(dist, a)
                    if hits is None:
                        return None
                    correct += float((hits * weights).sum())
                    total += float(weights.sum())
            else:
                for start in range(0, len(obs), _EVAL_BATCH):
                    sl = slice(start, start + _EVAL_BATCH)
                    o = obs[sl].to(self._device)
                    m = None if masks is None else masks[sl].to(self._device)
                    dist = self._model.step(o, self._model.initial_state(len(o), self._device), m).dist
                    hits = _hits(dist, actions[sl].to(self._device))
                    if hits is None:
                        return None
                    correct += float(hits.sum())
                    total += float(len(o))
            return correct / max(total, 1.0)
        finally:
            self._model.train()
```

- [ ] **Step 7: Config and CLI.**

In `src/colosseum/core/config.py`, add this section model right before `class ColosseumConfig`:

```python
class BCConfig(BaseModel):
    """Offline behavioral cloning (``colosseum bc``)."""

    seq_len: int = Field(
        default=64, ge=1,
        description="Window length (transitions) for stateful models; ignored by stateless "
                    "ones. CLI --seq-len overrides it.",
    )
```

Then add the field `bc: BCConfig = Field(default_factory=BCConfig)` to `ColosseumConfig`, next to the other section fields such as `checkpoint` and `metrics`. If T6.1 already introduced a shared strict base class or `model_config = ConfigDict(extra="forbid")` for sections, give `BCConfig` the same base or `model_config`.

In `src/colosseum/cli.py`, replace the whole `bc` command (decorators and function) with the code below. If T6.2/T6.5 changed how CLI commands set up logging (for example `setup_process_logging`), keep that setup line at the top of the function body in place of the `logging.basicConfig` call shown here.

```python
@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--data", "-d", required=True, type=click.Path(exists=True),
              help="BC data: a .pt file or a directory of .pt files (keys: observations, actions, "
                   "optional action_masks, dones)")
@click.option("--output", "-o", required=True, type=click.Path(), help="Where to save the trained state_dict (.pt)")
@click.option("--epochs", default=10, type=int, show_default=True, help="Number of BC epochs")
@click.option("--batch-size", default=256, type=int, show_default=True, help="Transitions per gradient step")
@click.option("--lr", default=1e-3, type=float, show_default=True, help="Adam learning rate")
@click.option("--seq-len", default=None, type=int,
              help="Window length for stateful models (default: bc.seq_len from the config, 64)")
def bc(
    config: str,
    data: str,
    output: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seq_len: int | None,
) -> None:
    """Train a policy by offline behavioral cloning (loss = -log pi(a|s), masks applied)."""
    import logging

    import torch

    from colosseum.bc.offline_bc import OfflineBCTrainer
    from colosseum.core.config import load_config
    from colosseum.core.registry import build_model

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = load_config(config)
    model = build_model(cfg)
    device = cfg.learner.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = OfflineBCTrainer(
        model, lr=lr, device=device,
        seq_len=seq_len if seq_len is not None else cfg.bc.seq_len,
    )
    trainer.load_data(data)
    metrics = trainer.train(num_epochs=epochs, batch_size=batch_size)

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output)
    message = f"BC training complete: final-epoch NLL={metrics['bc_loss']:.4f}"
    if "accuracy" in metrics:
        message += f", accuracy={metrics['accuracy']:.3f}"
    click.echo(message)
    click.echo(f"Weights saved to {output}")
```

- [ ] **Step 8: Replace the old BC tests.**

Run `grep -rln "OfflineBCTrainer" tests/ | grep -v "test_bc_trainer.py\|test_bc_learns.py"`. In each listed file, delete the tests that call `OfflineBCTrainer(...)` with `action_type=` or with the old two-argument `add_data`. Before this part they were `test_offline_bc_from_tensors`, `test_offline_bc_from_file` and `test_offline_bc_loss_decreases`; `test_bc_trainer.py` covers the same behaviour.

Also run `grep -rn "action-type\|action_type" src tests README.md`. No hits may remain other than unrelated words.

Then run `ruff check tests` and remove any now-unused imports.

- [ ] **Step 9: Run the new tests.**

Run: `.venv/bin/python -m pytest tests/unit/test_bc_trainer.py tests/learning/test_bc_learns.py tests/integration/test_bc_cli.py -v`
Expected: all PASS. `test_stateful_bc_uses_unroll_with_resets` takes a few seconds on CPU.

- [ ] **Step 10: Run the full fast suite.**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: PASS.

- [ ] **Step 11: Commit.**

```bash
git add src/colosseum/bc/offline_bc.py src/colosseum/networks/distributions.py src/colosseum/core/config.py \
        src/colosseum/cli.py tests
git commit -m "feat: BC with masks, NLL for all distributions, stateful window training, strict action types (R1-05)"
```

---
### Task T4.5: `ActionSpec` natural component order; `CompositeDist` keeps insertion order

Spec block 4 "Порядок компонент действий"; findings R2-03, ET-07.

**The bug.** `ActionSpec` names Tuple/MultiDiscrete components `"0"`, `"1"`, ... and then sorts the names as strings. `CompositeDist` sorts its keys the same way. With more than 10 components the order becomes `0, 1, 10, 11, 2, ...`, so unit 2 receives the action of head `"10"`. Masks stay aligned with the wrong heads, so nothing crashes and the error is silent.

**After this task:**
- `ActionSpec` orders Tuple/MultiDiscrete components by index, and Dict components in `space.spaces` order. Gymnasium sorts plain-dict keys alphabetically; an `OrderedDict` keeps its order.
- `CompositeDist` uses the insertion order of its dict and no longer sorts.
- `ActionSpec.check_distribution(dist)` verifies that a policy's `CompositeDist` matches the action space. `validate_config` calls it.
- `VectorEnv`/`SubprocessVectorEnv` decoding and `flatten_mask` iterate `ActionSpec.components`, so they become correct without code changes. The tests prove it end to end through the real `RolloutLoop`.

**Files:**
- Modify: `src/colosseum/core/action_spec.py` (`from_space` Dict branch, `_from_named_spaces` loop, new `component_names`, `check_distribution`)
- Modify: `src/colosseum/networks/distributions.py` (class `CompositeDist`, whole class)
- Modify: `src/colosseum/core/registry.py` (`validate_config`: one check after the dummy `step`)
- Modify: `tests/helpers.py` (append the 12-unit env and policies)
- Modify: the existing `CompositeDist` tests that assume sorted keys (Step 8)
- Test: `tests/unit/test_action_order.py` (create), `tests/contract/test_action_order_rollout.py` (create)

**Interfaces:**
- Consumes:
  - `rollout_chunks`, `TinyEncoder`, `TinyValue` (T4.1 kit);
  - `ComposedModel`, `NoCore` (T1.3/T1.4);
  - `act(model, obs, state, action_mask=None, deterministic=False) -> ActOutput` (T1.2);
  - `validate_config(config)` (T1.4), which builds the env and model and runs a dummy `model.step`;
  - `ConfigError` (T1.4);
  - `APPO` (T1.5).
- Produces:
  - `ActionSpec.component_names -> tuple[str, ...]`.
  - `ActionSpec.check_distribution(dist) -> None`. It raises `ValueError` on a component name/size/kind/order mismatch, or on a composite/non-composite mismatch.
  - `CompositeDist(dists: Mapping[str, Distribution])` in insertion order, with `.keys`, `.components` (kept from T4.4) and `.flat_mask_size`.
  - `tests/helpers.py`: `TWELVE_NVEC`, `TwelveUnitEnv(use_mask=True)` (with `.received`), `TwelveHeadPolicy(in_dim=16, peaked=True)`, `MisorderedTwelveHeadPolicy`, `twelve_unit_model(peaked=True)`.

- [ ] **Step 1: Append the 12-unit fixtures to `tests/helpers.py`.**

Add `from colosseum.networks.distributions import CompositeDist` to the imports at the top. `CategoricalDist` is already imported from that module, so extend that line. Then append:

```python
TWELVE_NVEC = tuple(range(2, 14))   # unit i has i + 2 actions


class TwelveUnitEnv(BaseEnv):
    """Solo env with a 12-unit MultiDiscrete action (unit i has i + 2 actions).

    With ``use_mask`` the flat mask (natural unit order) allows only action i for
    unit i. Every received action vector is appended to ``self.received``.
    Episodes last 5 steps; rewards are 0.
    """

    def __init__(self, use_mask: bool = True) -> None:
        self._use_mask = use_mask
        self._t = 0
        self.received: list[np.ndarray] = []

    @property
    def num_players(self) -> int:
        return 1

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(0.0, 1.0, (4,), np.float32)

    @property
    def action_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.MultiDiscrete(np.array(TWELVE_NVEC))

    def _infos(self) -> dict[int, dict]:
        if not self._use_mask:
            return {0: {}}
        mask = np.concatenate([np.arange(n) == i for i, n in enumerate(TWELVE_NVEC)])
        return {0: {"action_mask": mask}}

    def reset(self, seed=None):
        self._t = 0
        return {0: np.zeros(4, np.float32)}, self._infos()

    def step(self, actions):
        self.received.append(np.asarray(actions[0], dtype=np.int64).copy())
        self._t += 1
        done = self._t >= 5
        return {0: np.zeros(4, np.float32)}, {0: 0.0}, {0: done}, {0: False}, self._infos()


class TwelveHeadPolicy(BasePolicy):
    """One Categorical head per unit, keyed "0".."11" in unit order.

    ``peaked=True``: head i puts (almost) all mass on action i.
    ``peaked=False``: uniform logits (use it with the env's mask).
    """

    def __init__(self, in_dim: int = 16, peaked: bool = True) -> None:
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(in_dim, n) for n in TWELVE_NVEC)
        with torch.no_grad():
            for i, head in enumerate(self.heads):
                head.weight.zero_()
                head.bias.zero_()
                if peaked:
                    head.bias[i] = 50.0

    def _named_dists(self, latent: torch.Tensor) -> list[tuple[str, CategoricalDist]]:
        return [(str(i), CategoricalDist(head(latent))) for i, head in enumerate(self.heads)]

    def forward(self, latent: torch.Tensor) -> CompositeDist:
        return CompositeDist(dict(self._named_dists(latent)))


class MisorderedTwelveHeadPolicy(TwelveHeadPolicy):
    """Same heads inserted in string-sorted key order ("0", "1", "10", "11", "2", ...)."""

    def forward(self, latent: torch.Tensor) -> CompositeDist:
        return CompositeDist(dict(sorted(self._named_dists(latent), key=lambda kv: kv[0])))


def twelve_unit_model(peaked: bool = True) -> ComposedModel:
    torch.manual_seed(0)
    return ComposedModel(TinyEncoder(4, 16), NoCore(input_dim=16), TwelveHeadPolicy(16, peaked), TinyValue(16))
```

- [ ] **Step 2: Write the failing unit test** `tests/unit/test_action_order.py`:

```python
"""ActionSpec / CompositeDist component order (T4.5)."""
import collections

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.core.action_spec import ActionSpec
from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.registry import validate_config
from colosseum.networks.distributions import CategoricalDist
from colosseum.networks.model import act
from tests.helpers import (
    TWELVE_NVEC,
    MisorderedTwelveHeadPolicy,
    TwelveHeadPolicy,
    TwelveUnitEnv,
    twelve_unit_model,
)

Box, Discrete = gymnasium.spaces.Box, gymnasium.spaces.Discrete


def test_multidiscrete_components_follow_index_order():
    spec = ActionSpec.from_space(gymnasium.spaces.MultiDiscrete(np.array(TWELVE_NVEC)))
    assert spec.component_names == tuple(str(i) for i in range(12))
    assert [c.offset for c in spec.components] == list(range(12))
    assert [c.num_categories for c in spec.components] == list(TWELVE_NVEC)
    np.testing.assert_array_equal(spec.decode(np.arange(12, dtype=np.float32)), np.arange(12))


def test_multidiscrete_mask_segments_follow_index_order():
    env = TwelveUnitEnv(use_mask=True)
    _, info = env.reset()
    spec = ActionSpec.from_space(env.action_space)
    flat = spec.flatten_mask(info[0]["action_mask"])
    for i, comp in enumerate(spec.components):
        segment = flat[comp.mask_offset:comp.mask_offset + comp.mask_size]
        assert segment.sum() == 1 and int(np.argmax(segment)) == i


def test_tuple_components_follow_index_order():
    spec = ActionSpec.from_space(gymnasium.spaces.Tuple([Discrete(i + 2) for i in range(11)]))
    assert spec.component_names == tuple(str(i) for i in range(11))
    assert spec.decode(np.arange(11, dtype=np.float32)) == tuple(range(11))


def test_dict_components_follow_space_order():
    ordered = gymnasium.spaces.Dict(collections.OrderedDict([
        ("speed", Box(0.0, 1.0, (1,))), ("direction", Discrete(4)),
    ]))
    spec = ActionSpec.from_space(ordered)
    assert spec.component_names == ("speed", "direction")
    assert [c.offset for c in spec.components] == [0, 1]
    plain = gymnasium.spaces.Dict({"speed": Box(0.0, 1.0, (1,)), "direction": Discrete(4)})
    assert ActionSpec.from_space(plain).component_names == tuple(plain.spaces)


def test_composite_dist_keeps_insertion_order():
    dist = TwelveHeadPolicy(16)(torch.zeros(3, 16))
    assert dist.keys == [str(i) for i in range(12)]
    assert [offset for _name, offset, _size, _disc in dist.components] == list(range(12))
    np.testing.assert_array_equal(dist.mode()[0].numpy(), np.arange(12))


def test_check_distribution_accepts_matching_and_rejects_mismatches():
    spec = ActionSpec.from_space(TwelveUnitEnv().action_space)
    spec.check_distribution(TwelveHeadPolicy(16)(torch.zeros(2, 16)))
    with pytest.raises(ValueError, match="action space"):
        spec.check_distribution(MisorderedTwelveHeadPolicy(16)(torch.zeros(2, 16)))
    with pytest.raises(ValueError, match="CompositeDist"):
        spec.check_distribution(CategoricalDist(torch.zeros(2, 3)))
    ActionSpec.from_space(Discrete(3)).check_distribution(CategoricalDist(torch.zeros(2, 3)))
    with pytest.raises(ValueError, match="CompositeDist"):
        ActionSpec.from_space(Discrete(3)).check_distribution(TwelveHeadPolicy(16)(torch.zeros(2, 16)))


def test_twelve_masked_heads_step_and_decode():
    model = twelve_unit_model(peaked=False)
    env = TwelveUnitEnv(use_mask=True)
    obs, info = env.reset()
    spec = ActionSpec.from_space(env.action_space)
    mask = torch.as_tensor(spec.flatten_mask(info[0]["action_mask"]))[None]
    out = act(model, torch.as_tensor(obs[0])[None], model.initial_state(1), mask)
    np.testing.assert_array_equal(spec.decode(out.actions[0].numpy()), np.arange(12))


def _twelve_config(policy_class: str) -> ColosseumConfig:
    # use_mask=False: a misordered policy under the natural-order mask could get an
    # all-illegal head and fail inside the dummy step before the layout check runs.
    return ColosseumConfig.model_validate({
        "env": {"env_class": "tests.helpers.TwelveUnitEnv", "num_players": 1, "kwargs": {"use_mask": False}},
        "networks": {
            "encoder_class": "tests.helpers.TinyEncoder",
            "policy_class": policy_class,
            "value_class": "tests.helpers.TinyValue",
        },
    })


def test_validate_config_rejects_misordered_composite_policy():
    validate_config(_twelve_config("tests.helpers.TwelveHeadPolicy"))
    with pytest.raises(ConfigError, match="action space"):
        validate_config(_twelve_config("tests.helpers.MisorderedTwelveHeadPolicy"))


def test_chase_example_policy_matches_its_action_space():
    from examples.composite_action import networks as chase_networks
    from examples.composite_action.env import ChaseEnv

    spec = ActionSpec.from_space(ChaseEnv().action_space)
    spec.check_distribution(chase_networks.ChasePolicy()(torch.zeros(2, chase_networks._LATENT)))


def test_space_miners_example_policy_matches_its_action_space():
    pytest.importorskip("Box2D")
    from examples.space_miners import networks as sm_networks
    from examples.space_miners.env import SpaceMinersEnv

    spec = ActionSpec.from_space(SpaceMinersEnv().action_space)
    spec.check_distribution(sm_networks.SpaceMinersPolicy()(torch.zeros(2, sm_networks._LATENT)))
```

If T1.4 changed the example head constructors to require `in_dim`, pass `in_dim=chase_networks._LATENT` and `in_dim=sm_networks._LATENT` respectively.

- [ ] **Step 3: Write the failing contract test** `tests/contract/test_action_order_rollout.py`:

```python
"""12-unit MultiDiscrete through the real RolloutLoop: unit i gets head i's action (T4.5)."""
import math

import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from tests.helpers import TwelveUnitEnv, rollout_chunks, twelve_unit_model


@pytest.mark.parametrize("use_mask", [False, True])
def test_twelve_units_receive_their_own_head_through_rollout_loop(use_mask):
    envs = []

    def env_fn():
        env = TwelveUnitEnv(use_mask=use_mask)
        envs.append(env)
        return env

    # Unmasked: head i is peaked on action i. Masked: uniform heads, the mask leaves only action i.
    model = twelve_unit_model(peaked=not use_mask)
    chunks = rollout_chunks(model, env_fn, num_chunks=2, chunk_length=4, num_envs=1)

    received = [a for env in envs for a in env.received]
    assert received
    for action in received:
        np.testing.assert_array_equal(action, np.arange(12))
    for chunk in chunks:
        expected = np.tile(np.arange(12, dtype=np.float32), (chunk.chunk_length, 1))
        np.testing.assert_array_equal(np.asarray(chunk.actions), expected)
        assert torch.isfinite(torch.as_tensor(chunk.action_log_probs)).all()

    metrics = APPO(model, AlgorithmConfig(), device="cpu").train_step(chunks)
    assert all(math.isfinite(v) for v in metrics.values())
```

- [ ] **Step 4: Run them and watch them fail.**

Run: `.venv/bin/python -m pytest tests/unit/test_action_order.py tests/contract/test_action_order_rollout.py -v`

Expected failures:
- `AttributeError: 'ActionSpec' object has no attribute 'component_names'`;
- `CompositeDist` sorted keys (`['0', '1', '10', '11', '2', ...]`);
- in the contract test, the env receives `[0, 1, 10, 11, 2, ...]` (unmasked) or misaligned actions (masked).

- [ ] **Step 5: `ActionSpec` natural order and `check_distribution`.**

In `src/colosseum/core/action_spec.py`, change the class docstring line `flat ``float32`` vectors of length ``flat_size``.` to add one sentence after it:

```
    Components are laid out in natural order: Tuple/MultiDiscrete by index
    (named ``"0"``, ``"1"``, ...), Dict in ``space.spaces`` order.
```

In `from_space`, replace the Dict branch

```python
        if isinstance(space, gymnasium.spaces.Dict):
            return cls._from_named_spaces(
                {k: space.spaces[k] for k in sorted(space.spaces.keys())},
                space_type="dict",
            )
```

with

```python
        if isinstance(space, gymnasium.spaces.Dict):
            # space.spaces order: gymnasium sorts plain-dict keys, OrderedDict keeps its order.
            return cls._from_named_spaces(dict(space.spaces), space_type="dict")
```

In `_from_named_spaces`, change the docstring to `"""Build a composite ActionSpec from sub-spaces, in the given (natural) order."""`, and replace the loop header

```python
        for name in sorted(named.keys()):
            sub = named[name]
```

with

```python
        for name, sub in named.items():
```

Add these two members to `ActionSpec` right after `_from_named_spaces`:

```python
    @property
    def component_names(self) -> tuple[str, ...]:
        """Component names in flat-layout order."""
        return tuple(c.name for c in self.components)

    def check_distribution(self, dist: Any) -> None:
        """Raise ``ValueError`` unless ``dist`` has this spec's flat action layout.

        Composite spaces need a ``CompositeDist`` whose components are, in order,
        ``component_names`` with matching sizes and discrete/continuous kinds.
        """
        from colosseum.networks.distributions import CompositeDist

        if not self.is_composite:
            if isinstance(dist, CompositeDist):
                raise ValueError(
                    f"action space is {self.space_type} but the policy returned a CompositeDist; "
                    f"return a single distribution"
                )
            return
        if not isinstance(dist, CompositeDist):
            raise ValueError(
                f"action space is {self.space_type} with components {list(self.component_names)}; "
                f"the policy must return a CompositeDist with these components in this order, "
                f"got {type(dist).__name__}"
            )
        expected = [(c.name, c.size, c.is_discrete) for c in self.components]
        got = [(name, size, discrete) for name, _offset, size, discrete in dist.components]
        if got != expected:
            raise ValueError(
                "CompositeDist components do not match the action space. Expected "
                f"(name, size, is_discrete) in this order: {expected}; got {got}. "
                "Tuple/MultiDiscrete components are named '0', '1', ... in index order; "
                "Dict components follow action_space.spaces order."
            )
```

`Any` is already imported in this module.

- [ ] **Step 6: `CompositeDist` in insertion order.**

In `src/colosseum/networks/distributions.py`:
- add `from collections.abc import Mapping` to the imports;
- replace the whole `class CompositeDist` with the class below.

It keeps the T4.4 `components` property and all method bodies; only the ordering and the docstring change.

```python
class CompositeDist(Distribution):
    """Multi-head distribution for composite action spaces (Dict / Tuple / MultiDiscrete).

    Wraps an ordered mapping of sub-distributions. All public methods operate on
    *flat* ``float32`` tensors of shape ``[B, flat_size]``: a discrete component
    takes one column (the category index as a float), a continuous one takes
    ``action_dim`` columns.

    The component order is the insertion order of ``dists`` and defines the flat
    action and mask layout. It must equal the action space's component order
    (``ActionSpec.component_names``): Tuple/MultiDiscrete components are named
    ``"0"``, ``"1"``, ... in index order; Dict components follow
    ``action_space.spaces`` order (gymnasium sorts plain-dict keys, an
    ``OrderedDict`` keeps its order). ``ActionSpec.check_distribution`` verifies it.
    """

    def __init__(self, dists: Mapping[str, Distribution]) -> None:
        if not dists:
            raise ValueError("CompositeDist requires at least one sub-distribution")

        self._keys: list[str] = list(dists.keys())
        self._dists: dict[str, Distribution] = {k: dists[k] for k in self._keys}

        # Action layout: key -> (offset, size, is_discrete)
        offset = 0
        self._layout: dict[str, tuple[int, int, bool]] = {}
        for k in self._keys:
            d = self._dists[k]
            sz = d.action_dim
            self._layout[k] = (offset, sz, isinstance(d, CategoricalDist))
            offset += sz
        self._flat_size: int = offset

        # Mask layout (only discrete components have masks)
        mask_offset = 0
        self._mask_layout: dict[str, tuple[int, int]] = {}
        for k in self._keys:
            d = self._dists[k]
            ms = d.logits.shape[-1] if isinstance(d, CategoricalDist) else 0
            self._mask_layout[k] = (mask_offset, ms)
            mask_offset += ms
        self._flat_mask_size: int = mask_offset

    # ---- Layout -----------------------------------------------------------

    @property
    def keys(self) -> list[str]:
        """Component names in flat-layout order."""
        return list(self._keys)

    @property
    def components(self) -> list[tuple[str, int, int, bool]]:
        """``(name, flat_offset, size, is_discrete)`` per component, in flat-layout order."""
        return [(k, *self._layout[k]) for k in self._keys]

    @property
    def flat_mask_size(self) -> int:
        return self._flat_mask_size

    # ---- Distribution interface ------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._flat_size

    def sample(self) -> torch.Tensor:
        """Sample from all sub-distributions and return ``[B, flat_size]``."""
        parts: list[torch.Tensor] = []
        for k in self._keys:
            s = self._dists[k].sample()
            _, _, is_disc = self._layout[k]
            if is_disc:
                s = s.float().unsqueeze(-1)  # [B] -> [B, 1]
            elif s.dim() == 1:
                s = s.unsqueeze(-1)
            parts.append(s)
        return torch.cat(parts, dim=-1)

    def log_prob(self, flat_actions: torch.Tensor) -> torch.Tensor:
        """Log-probability of a flat action tensor ``[B, flat_size]``."""
        total: Optional[torch.Tensor] = None
        for k in self._keys:
            off, sz, is_disc = self._layout[k]
            sub = flat_actions[:, off].long() if is_disc else flat_actions[:, off:off + sz]
            lp = self._dists[k].log_prob(sub)
            total = lp if total is None else total + lp
        return total

    def entropy(self) -> torch.Tensor:
        total: Optional[torch.Tensor] = None
        for k in self._keys:
            e = self._dists[k].entropy()
            total = e if total is None else total + e
        return total

    def mode(self) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for k in self._keys:
            m = self._dists[k].mode()
            _, _, is_disc = self._layout[k]
            if is_disc:
                m = m.float().unsqueeze(-1)
            elif m.dim() == 1:
                m = m.unsqueeze(-1)
            parts.append(m)
        return torch.cat(parts, dim=-1)

    def apply_mask(self, flat_mask: torch.Tensor) -> CompositeDist:
        """Apply a flat mask ``[B, flat_mask_size]`` to the discrete sub-distributions."""
        new_dists: dict[str, Distribution] = {}
        for k in self._keys:
            d = self._dists[k]
            m_off, m_sz = self._mask_layout[k]
            if m_sz > 0 and flat_mask is not None:
                new_dists[k] = d.apply_mask(flat_mask[:, m_off:m_off + m_sz])
            else:
                new_dists[k] = d
        return CompositeDist(new_dists)

    def kl_divergence(self, other: Distribution) -> torch.Tensor:
        if not isinstance(other, CompositeDist):
            raise TypeError(f"Cannot compute KL between CompositeDist and {type(other).__name__}")
        if self._keys != other._keys:
            raise ValueError(f"Key mismatch: {self._keys} vs {other._keys}")
        total: Optional[torch.Tensor] = None
        for k in self._keys:
            kl = self._dists[k].kl_divergence(other._dists[k])
            total = kl if total is None else total + kl
        return total
```

- [ ] **Step 7: `validate_config` checks the layout.**

In `src/colosseum/core/registry.py`, add `from colosseum.core.action_spec import ActionSpec` to the imports if it is not there. `validate_config` (T1.4) builds an env instance and a model and runs a dummy `model.step(...)`. Right after that step call, add the block below, using the names T1.4 chose (here `env` is the env instance and `out` is the `StepOutput`):

```python
    try:
        ActionSpec.from_space(env.action_space).check_distribution(out.dist)
    except ValueError as exc:
        raise ConfigError(f"networks: policy distribution does not match the action space: {exc}") from exc
```

If `validate_config` runs the dummy step with an action mask, use the dist of that same call.

- [ ] **Step 8: Update the tests that assumed sorted keys.**

Run `grep -rn "Sorted\|sorted" tests/ | grep -i "composite\|key"`. Today the file is `test_composite_actions.py`, possibly moved by T0.2. Three tests build `CompositeDist({"type": ..., "pos": ...})` and expect `"pos"` first. Replace them with these versions (insertion order: `"type"` at column 0, `"pos"` at columns 1–2):

```python
def test_composite_dist_sample_shape():
    B = 5
    dists = {
        "type": CategoricalDist(torch.randn(B, 3)),
        "pos": DiagGaussianDist(torch.randn(B, 2), torch.zeros(B, 2)),
    }
    cd = CompositeDist(dists)
    assert cd.action_dim == 3  # 1 (type) + 2 (pos)
    s = cd.sample()
    assert s.shape == (B, 3)
    assert s.dtype == torch.float32
    # Insertion order: "type" at [0] (integer-valued), "pos" at [1:3]
    assert (s[:, 0] == s[:, 0].long().float()).all()


def test_composite_dist_log_prob():
    B = 4
    logits = torch.tensor([[1.0, 0.0, -1.0]] * B)
    mean = torch.tensor([[0.5, -0.5]] * B)
    log_std = torch.zeros(B, 2)

    # Insertion order: "type" at [0], "pos" at [1:3]
    cd = CompositeDist({
        "type": CategoricalDist(logits),
        "pos": DiagGaussianDist(mean, log_std),
    })
    flat = cd.sample()
    lp = cd.log_prob(flat)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()

    cat_lp = CategoricalDist(logits).log_prob(flat[:, 0].long())
    gauss_lp = DiagGaussianDist(mean, log_std).log_prob(flat[:, 1:3])
    torch.testing.assert_close(lp, cat_lp + gauss_lp)


def test_composite_dist_mode():
    B = 3
    # Insertion order: "type" at [0], "pos" at [1:3]
    cd = CompositeDist({
        "type": CategoricalDist(torch.tensor([[10.0, 0.0, 0.0]] * B)),
        "pos": DiagGaussianDist(torch.tensor([[0.5, -0.3]] * B), torch.zeros(B, 2)),
    })
    m = cd.mode()
    assert m.shape == (B, 3)
    assert (m[:, 0] == 0.0).all()
    torch.testing.assert_close(m[:, 1], torch.tensor([0.5] * B))
    torch.testing.assert_close(m[:, 2], torch.tensor([-0.3] * B))
```

Then check the remaining composite tests for order assumptions. Tests that use `"a"`/`"b"` or `"action"`/`"speed"` in alphabetical insertion order are unaffected. So are the `ActionSpec` Dict tests built from plain dicts, because gymnasium sorts those keys.

Finally, run `grep -rn "CompositeDist(" src examples`. Each policy must insert its keys in the action space's order:
- `examples/composite_action/networks.py`: `direction`, `speed`;
- `examples/space_miners/networks.py`: `accel`, `push_0`, `push_1`, `push_2`.

Both already match, and the Step 2 tests guard them. Any other policy found by the grep must be reordered to match `ActionSpec.from_space(<its env>.action_space).component_names`.

- [ ] **Step 9: Run the new tests.**

Run: `.venv/bin/python -m pytest tests/unit/test_action_order.py tests/contract/test_action_order_rollout.py -v`
Expected: all PASS. The space-miners test is skipped without Box2D.

- [ ] **Step 10: Run the full fast suite.**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: PASS.

- [ ] **Step 11: Commit.**

```bash
git add src/colosseum/core/action_spec.py src/colosseum/networks/distributions.py src/colosseum/core/registry.py tests
git commit -m "fix: natural ActionSpec component order, CompositeDist insertion order, layout check (R2-03, ET-07)"
```

---
### Task T4.6: `BaseAlgorithm.state_dict()/load_state_dict()`; extra APPO metrics

Spec block 4 "Состояние обучения и resume" and "Метрики APPO"; findings R1-07, R3-19, R3-20, R1-10 (metrics part).

**The problem today.** Resume restores only the model and, through a private attribute, the optimizer. The LR schedule, GradScaler and kickstart step restart from zero.

**After this task:**
- `algorithm.state_dict()` returns a deep CPU copy of the full training state:
  - the optimizer state;
  - `progress` (the LR is a pure function of progress since T2.5, so progress replaces a scheduler state);
  - the GradScaler state;
  - the kickstart step;
  - `policy_version`;
  - `consumed_samples`.
- `load_state_dict()` restores it, and the next update is bit-identical to the one an uninterrupted run would make.
- APPO also reports `explained_variance`, `grad_norm` (before clipping), `rho_mean`, `rho_clip_frac` and `lr`. The old `learning_rate` key is renamed to `lr`. `policy_lag_*` comes from the learner (T2.6) and is not duplicated here.

T5.3 writes `state_dict()` to `trainer_state.pt`, and resume calls `load_state_dict()`.

**Files:**
- Modify: `src/colosseum/algorithms/base.py` (add `deep_cpu_copy`, `state_dict`, `load_state_dict`)
- Modify: `src/colosseum/algorithms/appo.py` (`__init__` counters, `_explained_variance`, `compute_loss` whole method, `train_step` whole method, `state_dict`, `load_state_dict`, `consumed_samples`)
- Modify: tests asserting the `learning_rate` metric key (Step 8)
- Test: `tests/unit/test_algorithm_state.py` (create), `tests/unit/test_appo_metrics.py` (create)

**Interfaces:**
- Consumes:
  - `APPO.set_progress(progress)` (T2.5), which sets the optimizer LR from `progress` with no LR scheduler object;
  - `APPO.compute_loss` as written in T4.3 (binds `batch`, `dones`, `masks`, `state0`, `out`, `amp_ctx`);
  - `KickstartLoss.state_dict()/load_state_dict()` (T4.3);
  - `update_normalizers` call (T4.2);
  - test kit (T4.1).
- Produces:
  - `colosseum.algorithms.base.deep_cpu_copy(obj) -> obj`. Tensors become `.detach().cpu().clone()`, numpy arrays are copied, containers are rebuilt recursively, and anything else is `copy.deepcopy`'d.
  - `BaseAlgorithm.state_dict() -> dict[str, Any]` and `BaseAlgorithm.load_state_dict(state) -> None`. The base raises `NotImplementedError`.
  - `APPO.state_dict()` returns exactly the keys `optimizer`, `progress` (float), `scaler` (dict | None), `kickstart` (`{"step": int}` | None), `policy_version` (int) and `consumed_samples` (int), all deep CPU copies.
  - `APPO.load_state_dict(state)` deep-copies its input first, so the caller's dict is never aliased.
  - `APPO.consumed_samples -> int`: the sum of `chunk_length` over all chunks passed to `train_step`.
  - `APPO.train_step` metrics: `total_loss`, `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`, `explained_variance`, `grad_norm`, `rho_mean`, `rho_clip_frac`, `policy_version`, `lr`, plus `kickstart_loss`/`kickstart_lambda` with kickstart.

- [ ] **Step 1: Write the failing state test** `tests/unit/test_algorithm_state.py`:

```python
"""Algorithm state round trip and deep-copy semantics (T4.6)."""
import copy

import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.algorithms.base import deep_cpu_copy
from colosseum.bc.kickstart import KickstartLoss
from colosseum.core.config import AlgorithmConfig
from tests.helpers import MaskedToyEnv, rollout_chunks, tiny_model

STATE_KEYS = {"optimizer", "progress", "scaler", "kickstart", "policy_version", "consumed_samples"}


def _tensors(tree):
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _tensors(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _tensors(value)


def _assert_tree_equal(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_tree_equal(x, y)
    else:
        assert a == b


def test_deep_cpu_copy_detaches_and_copies():
    t = torch.ones(3, requires_grad=True)
    tree = {"a": [t, (t * 2,)], "b": 5, "c": "x"}
    out = deep_cpu_copy(tree)
    assert out["a"][0].data_ptr() != t.data_ptr()
    assert not out["a"][0].requires_grad
    assert isinstance(out["a"][1], tuple)
    assert out["b"] == 5 and out["c"] == "x"


def test_state_dict_keys_and_values():
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    algo = APPO(model, AlgorithmConfig(), device="cpu")
    algo.set_progress(0.25)
    algo.train_step(chunks)
    state = algo.state_dict()
    assert set(state) == STATE_KEYS
    assert state["policy_version"] == 1
    assert state["consumed_samples"] == 32 == algo.consumed_samples
    assert state["progress"] == pytest.approx(0.25)
    assert state["scaler"] is None          # no AMP on CPU
    assert state["kickstart"] is None
    assert all(t.device.type == "cpu" for t in _tensors(state))


def test_state_dict_is_a_deep_copy():
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    algo = APPO(model, AlgorithmConfig(), device="cpu")
    algo.train_step(chunks)
    state = algo.state_dict()
    frozen = copy.deepcopy(state)
    live_ptrs = {
        v.data_ptr()
        for per_param in algo._optimizer.state.values()
        for v in per_param.values()
        if isinstance(v, torch.Tensor)
    }
    assert not any(t.data_ptr() in live_ptrs for t in _tensors(state))
    algo.train_step(chunks)
    algo.train_step(chunks)
    _assert_tree_equal(state, frozen)


def test_resume_reproduces_the_next_update_exactly():
    cfg = AlgorithmConfig(num_epochs=2, minibatch_chunks=2, learning_rate=1e-3, lr_schedule="linear")
    model = tiny_model(core="lstm")
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    teacher = copy.deepcopy(model)

    original = APPO(model, cfg, device="cpu",
                    kickstart=KickstartLoss(copy.deepcopy(teacher), initial_lambda=1.0, decay_steps=10))
    torch.manual_seed(1)
    original.set_progress(0.1)
    original.train_step(chunks)
    torch.manual_seed(2)
    original.set_progress(0.2)
    original.train_step(chunks)

    model_state = {k: v.detach().clone() for k, v in original.model.state_dict().items()}
    algo_state = original.state_dict()
    algo_state_frozen = copy.deepcopy(algo_state)

    resumed = APPO(tiny_model(core="lstm", seed=99), cfg, device="cpu",
                   kickstart=KickstartLoss(copy.deepcopy(teacher), initial_lambda=1.0, decay_steps=10))
    resumed.model.load_state_dict(model_state)
    resumed.load_state_dict(algo_state)
    assert resumed.policy_version == original.policy_version == 2
    assert resumed.consumed_samples == original.consumed_samples
    assert resumed._optimizer.param_groups[0]["lr"] == original._optimizer.param_groups[0]["lr"]

    torch.manual_seed(3)
    m_original = original.train_step(chunks)
    torch.manual_seed(3)
    m_resumed = resumed.train_step(chunks)

    for key, value in original.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key
    assert m_original == m_resumed

    # load_state_dict copied its input: training `resumed` did not touch algo_state.
    _assert_tree_equal(algo_state, algo_state_frozen)


@pytest.mark.gpu
def test_grad_scaler_state_round_trips_on_cuda():
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=2, chunk_length=8)
    cfg = AlgorithmConfig(use_amp=True, amp_dtype="float16")
    algo = APPO(model, cfg, device="cuda")
    algo.train_step(chunks)
    state = algo.state_dict()
    assert state["scaler"] is not None and "scale" in state["scaler"]
    other = APPO(tiny_model(seed=5), cfg, device="cuda")
    other.load_state_dict(state)
    assert other._scaler.get_scale() == algo._scaler.get_scale()
```

- [ ] **Step 2: Write the failing metrics test** `tests/unit/test_appo_metrics.py`:

```python
"""APPO diagnostic metrics (T4.6)."""
import math

import pytest
import torch

from colosseum.algorithms.appo import APPO, _explained_variance
from colosseum.core.config import AlgorithmConfig
from tests.helpers import MaskedToyEnv, rollout_chunks, tiny_model

NEW_KEYS = ("explained_variance", "grad_norm", "rho_mean", "rho_clip_frac", "lr")


def test_explained_variance_helper():
    target = torch.randn(100)
    assert _explained_variance(target.clone(), target).item() == pytest.approx(1.0)
    assert _explained_variance(torch.full((100,), float(target.mean())), target).item() == pytest.approx(0.0, abs=1e-5)
    assert _explained_variance(torch.randn(10), torch.ones(10)).item() == 0.0   # constant target


def test_appo_reports_diagnostic_metrics_on_policy():
    cfg = AlgorithmConfig(num_epochs=1, minibatch_chunks=0, vtrace_rho_bar=1.5,
                          lr_schedule="constant", learning_rate=3e-4)
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    metrics = APPO(model, cfg, device="cpu").train_step(chunks)
    for key in NEW_KEYS:
        assert key in metrics and math.isfinite(metrics[key]), key
    assert "learning_rate" not in metrics
    assert metrics["lr"] == pytest.approx(3e-4)
    # First update on fresh chunks from identical weights: pi == mu.
    assert metrics["rho_mean"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["rho_clip_frac"] == 0.0
    assert metrics["grad_norm"] > 0.0
    assert metrics["explained_variance"] <= 1.0


def test_grad_norm_is_measured_before_clipping():
    cfg = AlgorithmConfig(num_epochs=1, minibatch_chunks=0, max_grad_norm=1e-6)
    model = tiny_model()
    chunks = rollout_chunks(model, MaskedToyEnv, num_chunks=4, chunk_length=8)
    metrics = APPO(model, cfg, device="cpu").train_step(chunks)
    assert metrics["grad_norm"] > 1e-3
```

- [ ] **Step 3: Run them and watch them fail.**

Run: `.venv/bin/python -m pytest tests/unit/test_algorithm_state.py tests/unit/test_appo_metrics.py -v`

Expected failures:
- `ImportError: cannot import name 'deep_cpu_copy'`;
- `ImportError: cannot import name '_explained_variance'`.

- [ ] **Step 4: `BaseAlgorithm`.**

In `src/colosseum/algorithms/base.py`, add `import copy` and `import numpy as np` to the imports (keep `torch`, `Any` and `Optional`). Add this function above `class BaseAlgorithm`:

```python
def deep_cpu_copy(obj: Any) -> Any:
    """Recursively copy ``obj`` so it shares no storage with live training state.

    Tensors -> ``.detach().cpu().clone()``; numpy arrays -> ``.copy()``; dicts,
    lists and tuples are rebuilt; anything else is ``copy.deepcopy``'d.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: deep_cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_cpu_copy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(deep_cpu_copy(v) for v in obj)
    return copy.deepcopy(obj)
```

Add these two methods to `BaseAlgorithm`, after `policy_version`:

```python
    def state_dict(self) -> dict[str, Any]:
        """Full training state except model weights, as a deep CPU copy.

        APPO keys: optimizer, progress, scaler, kickstart, policy_version,
        consumed_samples. Checkpoints store it as ``trainer_state.pt`` (T5.3).
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement state_dict()")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore what :meth:`state_dict` returned (model weights are loaded separately)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement load_state_dict()")
```

Keep the `optimizer_state_dict` property for now: the learner's checkpoint path uses it until T5.3 switches it to `state_dict()`.

- [ ] **Step 5: APPO counters and the explained-variance helper.**

In `src/colosseum/algorithms/appo.py`:

Add `from typing import Any` and `from colosseum.algorithms.base import BaseAlgorithm, deep_cpu_copy` (extend the existing base import).

Add this module-level helper above `class APPO`:

```python
def _explained_variance(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Var(target - predicted) / Var(target); 0 when the target is (near) constant."""
    var_target = target.float().var(unbiased=False)
    if var_target < 1e-8:
        return torch.zeros((), device=target.device)
    return 1.0 - (target.float() - predicted.float()).var(unbiased=False) / var_target
```

In `APPO.__init__`, after `self._policy_version = 0`, add:

```python
        self._consumed_samples = 0
        self._progress = 0.0
```

In `set_progress` (T2.5), make sure the first statement stores the clamped value it applies, `self._progress = float(progress)`, or the clamped variable T2.5 uses. That way `state_dict()` saves exactly the progress the LR was computed from. Do not change how T2.5 computes the LR.

Add the property next to `policy_version`:

```python
    @property
    def consumed_samples(self) -> int:
        """Total transitions passed to train_step (sum of chunk lengths)."""
        return self._consumed_samples
```

- [ ] **Step 6: `compute_loss` with metrics.**

Replace the whole `compute_loss` method with this version. It is the T4.3 method plus the `# (T4.6)` lines; keep any T1.5-specific front lines you kept in T4.3.

```python
    def compute_loss(self, chunks: list[TrajectoryChunk]) -> dict[str, torch.Tensor]:
        """APPO loss for one minibatch of chunks.

        1. Stack chunks to [T, B, ...]; unroll the model from the chunks' initial
           states with the chunks' dones and action masks.
        2. V-trace(lambda) targets and advantages from the recomputed log-probs/values.
        3. PPO clipped surrogate on the V-trace advantages, value MSE, entropy bonus.
        4. Optional kickstart KL on the same (masked) student distribution.
        """
        cfg = self._config
        batch = self._prepare_batch(chunks)
        T, B = batch["rewards"].shape
        act_shape = batch["actions"].shape[2:]
        flat_actions = batch["actions"].reshape(T * B, *act_shape)
        masks = batch.get("action_masks")
        dones = batch["dones"].bool()
        state0 = state_to(cat_batch([c.initial_state for c in chunks]), self._device)

        amp_ctx = torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=self._use_amp)
        with amp_ctx:
            out = self._model.unroll(batch["observations"], state0, dones, masks)
            target_log_probs = out.dist.log_prob(flat_actions).float().reshape(T, B)
            entropy = out.dist.entropy().float().reshape(T, B)
            new_values = out.value.float().reshape(T, B)

        with torch.no_grad():
            vtrace_targets, vtrace_advantages = self._compute_vtrace(
                behavior_log_probs=batch["behavior_log_probs"],
                target_log_probs=target_log_probs.detach(),
                rewards=batch["rewards"],
                values=new_values.detach(),
                bootstrap_value=batch["bootstrap_values"],
                dones=dones,
                gamma=cfg.gamma,
                rho_bar=cfg.vtrace_rho_bar,
                c_bar=cfg.vtrace_c_bar,
                lam=cfg.vtrace_lambda,
            )

        log_ratio = torch.clamp(target_log_probs - batch["behavior_log_probs"], -20.0, 20.0)
        ratio = torch.exp(log_ratio)
        adv = vtrace_advantages.detach()
        if cfg.normalize_advantages and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(new_values, vtrace_targets.detach())
        entropy_loss = -entropy.mean()
        total_loss = policy_loss + cfg.value_loss_coeff * value_loss + cfg.entropy_coeff * entropy_loss

        kickstart_loss = self._zero_loss
        if self._kickstart is not None and self._kickstart.current_lambda > 0:
            with amp_ctx:
                kickstart_loss = self._kickstart.compute(
                    student_dist=out.dist,
                    observations=batch["observations"],
                    dones=dones,
                    state0=state0,
                    action_mask=masks,
                )
            total_loss = total_loss + kickstart_loss

        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > cfg.eps_clip).float().mean()
            rho_mean = ratio.mean()                                              # (T4.6)
            rho_clip_frac = (ratio > cfg.vtrace_rho_bar).float().mean()          # (T4.6)
            explained_variance = _explained_variance(new_values, vtrace_targets)  # (T4.6)

        result = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": -entropy_loss,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "rho_mean": rho_mean,                                                # (T4.6)
            "rho_clip_frac": rho_clip_frac,                                      # (T4.6)
            "explained_variance": explained_variance,                            # (T4.6)
        }
        if self._kickstart is not None:
            result["kickstart_loss"] = kickstart_loss.detach()
            result["kickstart_lambda"] = torch.tensor(self._kickstart.current_lambda)
        return result
```

- [ ] **Step 7: `train_step`, `state_dict` and `load_state_dict`.**

Replace the whole `train_step` method. It keeps the T4.2 normalizer call and the T2.5 rule that the LR is set only by `set_progress`, so there is no `lr_scheduler.step()`.

```python
    def train_step(self, chunks: list[TrajectoryChunk]) -> dict[str, float]:
        """One training step: normalizer update, then num_epochs x minibatches of updates.

        Minibatching is over chunks (the batch dimension B); each chunk's [T]
        sequence stays intact, so stateful models unroll correctly. The LR is set
        by ``set_progress`` (called by the learner before each step).
        """
        cfg = self._config

        # Refresh observation-normalization statistics once per train step, from
        # this step's fresh samples only (never once per epoch/minibatch forward).
        obs_all = torch.cat([torch.as_tensor(c.observations) for c in chunks], dim=0)
        self._model.update_normalizers(obs_all.to(self._device))

        max_norm = cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf")
        sums: dict[str, float] = {}
        num_updates = 0
        for _epoch in range(cfg.num_epochs):
            indices = torch.randperm(len(chunks)).tolist()
            mb_size = cfg.minibatch_chunks if cfg.minibatch_chunks > 0 else len(chunks)
            for start in range(0, len(chunks), mb_size):
                mb_chunks = [chunks[i] for i in indices[start:start + mb_size]]
                if not mb_chunks:
                    continue
                losses = self.compute_loss(mb_chunks)
                self._optimizer.zero_grad()
                if self._scaler is not None:
                    self._scaler.scale(losses["total_loss"]).backward()
                    self._scaler.unscale_(self._optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm)
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    losses["total_loss"].backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm)
                    self._optimizer.step()
                losses["grad_norm"] = grad_norm.detach()     # total norm BEFORE clipping
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + float(value)
                num_updates += 1

        if self._kickstart is not None:
            self._kickstart.step()
        self._policy_version += 1
        self._consumed_samples += sum(c.chunk_length for c in chunks)

        metrics = {key: value / max(1, num_updates) for key, value in sums.items()}
        metrics["policy_version"] = float(self._policy_version)
        metrics["lr"] = float(self._optimizer.param_groups[0]["lr"])
        return metrics

    def state_dict(self) -> dict[str, Any]:
        """Deep CPU copy of the training state (model weights excluded)."""
        return deep_cpu_copy({
            "optimizer": self._optimizer.state_dict(),
            "progress": float(self._progress),
            "scaler": self._scaler.state_dict() if self._scaler is not None else None,
            "kickstart": self._kickstart.state_dict() if self._kickstart is not None else None,
            "policy_version": int(self._policy_version),
            "consumed_samples": int(self._consumed_samples),
        })

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore :meth:`state_dict` output; the input is copied, never aliased."""
        state = deep_cpu_copy(state)
        self._optimizer.load_state_dict(state["optimizer"])
        if self._scaler is not None and state.get("scaler") is not None:
            self._scaler.load_state_dict(state["scaler"])
        if self._kickstart is not None and state.get("kickstart") is not None:
            self._kickstart.load_state_dict(state["kickstart"])
        self._policy_version = int(state["policy_version"])
        self._consumed_samples = int(state.get("consumed_samples", 0))
        self.set_progress(float(state.get("progress", 0.0)))
```

If T2.5 already added a consumed-samples counter to APPO under another name, keep one counter (named `_consumed_samples`) and remove the duplicate.

- [ ] **Step 8: Rename the metric key in existing tests and consumers.**

Run `grep -rn "\"learning_rate\"\|'learning_rate'" src tests`. Two kinds of hits:
- Hits that read the **metric** (for example `assert "learning_rate" in metrics` in the old APPO tests): change them to `"lr"`.
- Hits that are the **config field** `AlgorithmConfig.learning_rate` (for example `{"learning_rate": 1e-2}` in a config dict): leave them unchanged.

In `src` there should be no consumer of the metric key. If T2.6 or the learner logs `metrics["learning_rate"]`, switch it to `metrics["lr"]`.

- [ ] **Step 9: Run the new tests.**

Run: `.venv/bin/python -m pytest tests/unit/test_algorithm_state.py tests/unit/test_appo_metrics.py -v`
Expected: all PASS. The GPU test is skipped without CUDA.

- [ ] **Step 10: Run the full fast suite.**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: PASS.

- [ ] **Step 11: Commit.**

```bash
git add src/colosseum/algorithms/base.py src/colosseum/algorithms/appo.py tests
git commit -m "feat: algorithm state_dict/load_state_dict and APPO diagnostics (R1-07, R1-10)"
```

---
### Task T7.1: Eval engine on `PolicyModel` with per-seat state and seat rotation

Spec block 7 (engine part); findings R4-11, ET-10, R4-14 (seat assignment, in-progress-episode bias).

**The problem today.** Eval:
- calls the network without state, so recurrent agents are evaluated as a different policy;
- assigns seats randomly;
- runs inference for every seat, active or not;
- stops at a global match count and drops in-progress episodes.

**After this task, `colosseum.eval` has a compact standalone engine:**
- `schedule_lineups` builds one seat→agent lineup per match:
  - pairwise: for pair (a, b), match m gives seat s to `(a, b)[(s + m) % 2]`, so every agent plays every seat equally often over an even number of matches;
  - solo: one agent, or a 1-player env.
- `play_matches` plays every lineup exactly once, to completion, on a `VectorEnv`:
  - inference is batched per agent through `act()`;
  - each (env, seat) has its own `State`, which starts at `initial_state(1)`, advances only when the seat acts (`info["active"]`, default True) and resets at episode end;
  - non-acting seats send the zero action;
  - an acting seat with no legal action raises `EnvContractError`;
  - the deterministic and stochastic modes are both available.
- The engine works for N-player games and solo.

Full unification with `RolloutLoop` is SP4. Reusing its internals would not be trivial (buffers, collection, transitions), so the engine is a separate loop with the same seat semantics. The old `evaluate_agents`/`EvalMatrix` API stays until T7.2 replaces the statistics and the CLI.

**Files:**
- Modify: `src/colosseum/eval.py` (add imports and an engine section; the existing T1.7 code stays)
- Modify: `tests/helpers.py` (append `AlternatingEnv`, `CounterModel`)
- Test: `tests/unit/test_eval_engine.py` (create), `tests/contract/test_eval_stateful.py` (create)

**Interfaces:**
- Consumes:
  - `act(model, obs, state, action_mask=None, deterministic=False) -> ActOutput(actions, log_probs, values, state)`, `PolicyModel.initial_state(1)`, `StepOutput` (T1.2);
  - `cat_batch`, `slice_batch` (int index keeps the batch dim) (T1.1);
  - `EnvContractError` (T1.4/T3.2);
  - `VectorEnv(env_fn, num_envs)` with `num_players`, `action_spec`, `reset_all(seed)`, `step(actions)`. With the T3.2 reset-info fix, the infos returned after an auto-reset carry the new episode's `active`/`action_mask` and the old episode's `terminal_info`;
  - `player_outcomes(total_rewards, terminal_infos, num_players)` (`core/outcomes.py`);
  - test kit (T4.1).
- Produces:
  - in `colosseum.eval`:
    - `MatchRecord(lineup: tuple[str, ...], outcomes: tuple[float, ...], returns: tuple[float, ...], length: int)` (frozen dataclass);
    - `schedule_lineups(agent_names, num_players, num_matches) -> list[tuple[str, ...]]`;
    - `play_matches(models: dict[str, PolicyModel], env_fn, lineups, num_envs=8, deterministic=False, seed=None) -> list[MatchRecord]`. Records come in completion order and there is exactly one per lineup.
  - in `tests/helpers.py`: `AlternatingEnv(num_actions=4)` and `CounterModel(num_actions=4)`.

- [ ] **Step 1: Append the turn-based fixtures to `tests/helpers.py`.**

Add `from colosseum.networks.model import PolicyModel, StepOutput` to the imports at the top. Then append:

```python
class AlternatingEnv(BaseEnv):
    """2-player turn-based toy: seats alternate (seat 0 first), 3 moves each, 6 env steps.

    ``info[p]["active"]`` marks the seat to move. The acting seat's mask allows
    every action; the waiting seat's mask allows none. Seat 0 always wins
    (+1 / -1 on the last step). ``self.episodes`` holds one list of
    ``(seat, action)`` per episode, in move order.
    """

    MOVES_PER_SEAT = 3

    def __init__(self, num_actions: int = 4) -> None:
        self._n = num_actions
        self._t = 0
        self.episodes: list[list[tuple[int, int]]] = []

    @property
    def num_players(self) -> int:
        return 2

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)

    @property
    def action_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(self._n)

    def _obs(self) -> dict[int, np.ndarray]:
        return {p: np.zeros(2, np.float32) for p in range(2)}

    def _infos(self) -> dict[int, dict]:
        current = self._t % 2
        return {
            p: {"active": p == current, "action_mask": np.full(self._n, p == current, dtype=bool)}
            for p in range(2)
        }

    def reset(self, seed=None):
        self._t = 0
        self.episodes.append([])
        return self._obs(), self._infos()

    def step(self, actions):
        current = self._t % 2
        self.episodes[-1].append((current, int(np.asarray(actions[current]).reshape(-1)[0])))
        self._t += 1
        done = self._t >= 2 * self.MOVES_PER_SEAT
        rewards = {0: 1.0, 1: -1.0} if done else {0: 0.0, 1: 0.0}
        return self._obs(), rewards, {0: done, 1: done}, {0: False, 1: False}, self._infos()


class CounterModel(PolicyModel):
    """Stateful toy policy: acts ``(number of own steps so far) % num_actions``.

    Its State is the step counter ``[B, 1]``; it ignores observations.
    """

    def __init__(self, num_actions: int = 4) -> None:
        super().__init__()
        self.num_actions = num_actions

    def initial_state(self, batch_size: int, device="cpu"):
        return torch.zeros(batch_size, 1, device=device)

    def step(self, obs, state, action_mask=None):
        batch = obs.shape[0]
        idx = state[:, 0].long() % self.num_actions
        logits = torch.full((batch, self.num_actions), -50.0, device=obs.device)
        logits[torch.arange(batch), idx] = 50.0
        dist = CategoricalDist(logits)
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        return StepOutput(dist, torch.zeros(batch, device=obs.device), state + 1)
```

- [ ] **Step 2: Write the failing unit test** `tests/unit/test_eval_engine.py`:

```python
"""Eval engine: scheduling, seat rotation, turn-based per-seat state (T7.1)."""
import itertools
from collections import Counter

import gymnasium
import numpy as np
import pytest

from colosseum.core.errors import EnvContractError
from colosseum.envs.base_env import BaseEnv
from colosseum.eval import MatchRecord, play_matches, schedule_lineups
from tests.helpers import AlternatingEnv, CounterModel

EXPECTED_EPISODE = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_pairwise_schedule_balances_seats(num_players):
    names = ["a", "b", "c"]
    lineups = schedule_lineups(names, num_players, num_matches=6)
    assert len(lineups) == 3 * 6
    for a, b in itertools.combinations(names, 2):
        pair_lineups = [lu for lu in lineups if set(lu) == {a, b}]
        assert len(pair_lineups) == 6
        for seat in range(num_players):
            counts = Counter(lu[seat] for lu in pair_lineups)
            assert counts[a] == counts[b] == 3


def test_solo_schedule():
    assert schedule_lineups(["a", "b"], 1, 3) == [("a",)] * 3 + [("b",)] * 3
    assert schedule_lineups(["a"], 2, 2) == [("a", "a"), ("a", "a")]
    with pytest.raises(ValueError):
        schedule_lineups(["a", "a"], 2, 2)
    with pytest.raises(ValueError):
        schedule_lineups(["a", "b"], 2, 0)


def test_turn_based_state_advances_only_on_own_moves_and_resets_per_episode():
    envs = []

    def env_fn():
        env = AlternatingEnv()
        envs.append(env)
        return env

    models = {"a": CounterModel(), "b": CounterModel()}
    lineups = schedule_lineups(["a", "b"], 2, 4)
    records = play_matches(models, env_fn, lineups, num_envs=2, deterministic=True)

    assert len(records) == 4
    assert sorted(r.lineup for r in records) == sorted(lineups)
    assert all(isinstance(r, MatchRecord) and r.length == 6 for r in records)
    assert all(r.outcomes == (1.0, 0.0) and r.returns == (1.0, -1.0) for r in records)
    complete = [ep for env in envs for ep in env.episodes if len(ep) == 6]
    assert len(complete) >= 4
    assert all(ep == EXPECTED_EPISODE for ep in complete)


def test_all_lineups_play_even_with_more_envs_than_matches():
    lineups = schedule_lineups(["a", "b", "c"], 2, 2)
    records = play_matches({n: CounterModel() for n in "abc"}, AlternatingEnv, lineups, num_envs=16)
    assert Counter(r.lineup for r in records) == Counter(lineups)


def test_unknown_agent_in_lineup_raises():
    with pytest.raises(ValueError, match="unknown"):
        play_matches({"a": CounterModel()}, AlternatingEnv, [("a", "z")])


class _NoLegalActionEnv(BaseEnv):
    """1-player env whose acting seat has an all-False mask."""

    @property
    def num_players(self):
        return 1

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(3)

    def reset(self, seed=None):
        return {0: np.zeros(2, np.float32)}, {0: {"action_mask": np.zeros(3, dtype=bool)}}

    def step(self, actions):
        return {0: np.zeros(2, np.float32)}, {0: 0.0}, {0: True}, {0: False}, {0: {}}


def test_acting_seat_without_legal_action_raises():
    with pytest.raises(EnvContractError, match="seat 0"):
        play_matches({"a": CounterModel(3)}, _NoLegalActionEnv, [("a",)])
```

- [ ] **Step 3: Write the failing contract test** `tests/contract/test_eval_stateful.py`:

```python
"""Eval runs a recurrent model exactly like a manual step loop with carried state (T7.1)."""
import torch

from colosseum.eval import play_matches
from colosseum.networks.model import act
from tests.helpers import MaskedToyEnv, tiny_model


def test_recurrent_agent_matches_manual_step_loop():
    model = tiny_model(core="lstm")
    envs = []

    def env_fn():
        env = MaskedToyEnv()
        envs.append(env)
        return env

    records = play_matches({"rnn": model}, env_fn, [("rnn",), ("rnn",)],
                           num_envs=1, deterministic=True, seed=7)
    assert len(records) == 2
    engine_actions = [a for env in envs for a in env.received]
    assert len(engine_actions) == 2 * MaskedToyEnv.EPISODE_LENGTH

    manual_env = MaskedToyEnv()
    obs, info = manual_env.reset(seed=7)
    state = model.initial_state(1)
    model.eval()
    for _ in range(2 * MaskedToyEnv.EPISODE_LENGTH):
        mask = torch.as_tensor(info[0]["action_mask"])[None]
        with torch.no_grad():
            out = act(model, torch.as_tensor(obs[0], dtype=torch.float32)[None], state, mask, deterministic=True)
        state = out.state
        obs, _reward, terminated, truncated, info = manual_env.step({0: int(out.actions[0])})
        if terminated[0] or truncated[0]:
            obs, info = manual_env.reset()
            state = model.initial_state(1)
    assert engine_actions == manual_env.received
```

- [ ] **Step 4: Run them and watch them fail.**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_engine.py tests/contract/test_eval_stateful.py -v`
Expected failure: `ImportError: cannot import name 'MatchRecord' from 'colosseum.eval'`.

- [ ] **Step 5: Add the engine to `src/colosseum/eval.py`.**

First make sure the module's import block contains all of the following, adding the missing lines and keeping it sorted. The T1.7 code may already import some of them; drop duplicates, and let `ruff check src/colosseum/eval.py` confirm the result.

```python
import itertools
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from colosseum.core.action_spec import ActionSpec
from colosseum.core.errors import EnvContractError
from colosseum.core.outcomes import player_outcomes
from colosseum.envs.base_env import BaseEnv
from colosseum.envs.vec_env import VectorEnv
from colosseum.networks.model import PolicyModel, act
from colosseum.networks.state import State, cat_batch, slice_batch
```

Make sure `logger = logging.getLogger(__name__)` exists at module level. Then append this section at the end of the file:

```python
# ---------------------------------------------------------------------------
# Engine (SP1 block 7): per-seat State, seat rotation, active-only inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchRecord:
    """One finished match: agent name, outcome in [0, 1] and return per seat."""

    lineup: tuple[str, ...]
    outcomes: tuple[float, ...]
    returns: tuple[float, ...]
    length: int


def schedule_lineups(
    agent_names: Sequence[str], num_players: int, num_matches: int,
) -> list[tuple[str, ...]]:
    """Seat -> agent lineups for an evaluation.

    - One agent, or a 1-player env (solo): ``num_matches`` lineups per agent with
      the agent in every seat.
    - Otherwise (pairwise): for every pair (a, b) in ``itertools.combinations``
      order, ``num_matches`` lineups where match m gives seat s to
      ``(a, b)[(s + m) % 2]``. With even ``num_matches`` every agent plays every
      seat equally often.
    """
    names = list(agent_names)
    if not names:
        raise ValueError("schedule_lineups: no agents")
    if len(set(names)) != len(names):
        raise ValueError(f"schedule_lineups: duplicate agent names in {names}")
    if num_players < 1:
        raise ValueError(f"schedule_lineups: num_players must be >= 1, got {num_players}")
    if num_matches < 1:
        raise ValueError(f"schedule_lineups: num_matches must be >= 1, got {num_matches}")
    if len(names) == 1 or num_players == 1:
        return [tuple([name] * num_players) for name in names for _ in range(num_matches)]
    if num_matches % 2:
        logger.warning(
            "eval: num_matches=%d is odd, so seats are balanced only up to one match per pair",
            num_matches,
        )
    lineups: list[tuple[str, ...]] = []
    for a, b in itertools.combinations(names, 2):
        pair = (a, b)
        for m in range(num_matches):
            lineups.append(tuple(pair[(s + m) % 2] for s in range(num_players)))
    return lineups


def _seat_mask(info: object, action_spec: ActionSpec) -> np.ndarray | None:
    """Flat bool action mask of one seat, or None when the env gives none."""
    if not isinstance(info, dict) or info.get("action_mask") is None:
        return None
    raw = info["action_mask"]
    return action_spec.flatten_mask(raw if isinstance(raw, dict) else np.asarray(raw))


def _check_acting_masks(
    masks: np.ndarray, seats: list[tuple[int, int]], action_spec: ActionSpec,
) -> None:
    for comp in action_spec.components:
        if comp.mask_size == 0:
            continue
        segment = masks[:, comp.mask_offset:comp.mask_offset + comp.mask_size]
        empty = np.flatnonzero(~segment.any(axis=1))
        if empty.size:
            env_idx, seat = seats[int(empty[0])]
            raise EnvContractError(
                f"eval: env {env_idx} seat {seat} is acting but its action mask allows "
                f"no action for component {comp.name!r}"
            )


def _act_group(
    model: PolicyModel,
    seats: list[tuple[int, int]],
    obs: np.ndarray,
    infos: list[dict],
    states: list[list[State]],
    action_spec: ActionSpec,
    deterministic: bool,
) -> tuple[np.ndarray, State]:
    """Batched ``act()`` for the acting seats of one agent; returns (actions, new state)."""
    obs_batch = torch.as_tensor(
        np.stack([np.asarray(obs[e, p]) for e, p in seats]), dtype=torch.float32,
    )
    seat_masks = [_seat_mask(infos[e].get(p, {}), action_spec) for e, p in seats]
    mask_batch = None
    if any(m is not None for m in seat_masks):
        allow_all = np.ones(action_spec.flat_mask_size, dtype=bool)
        stacked = np.stack([allow_all if m is None else m for m in seat_masks])
        _check_acting_masks(stacked, seats, action_spec)
        mask_batch = torch.as_tensor(stacked)
    state_batch = cat_batch([states[e][p] for e, p in seats])
    with torch.no_grad():
        out = act(model, obs_batch, state_batch, mask_batch, deterministic=deterministic)
    return out.actions.cpu().numpy(), out.state


def play_matches(
    models: dict[str, PolicyModel],
    env_fn: Callable[[], BaseEnv],
    lineups: Sequence[tuple[str, ...]],
    num_envs: int = 8,
    deterministic: bool = False,
    seed: int | None = None,
) -> list[MatchRecord]:
    """Play every lineup once, to completion; one record per lineup, in completion order.

    Each (env, seat) has its own model State: ``initial_state(1)`` at episode
    start, advanced only when the seat acts (``info[p]["active"]``, default True),
    reset at episode end. Non-acting seats send the zero action. Envs left without
    a scheduled match keep playing their last lineup until the rest finish; those
    extra episodes are discarded, so short episodes are not favoured.
    """
    lineups = [tuple(lu) for lu in lineups]
    if not lineups:
        return []
    unknown = sorted({name for lu in lineups for name in lu} - set(models))
    if unknown:
        raise ValueError(f"play_matches: lineups use unknown agents {unknown}")
    if num_envs < 1:
        raise ValueError(f"play_matches: num_envs must be >= 1, got {num_envs}")
    if seed is not None:
        torch.manual_seed(seed)
    for model in models.values():
        model.eval()
    vec_env = VectorEnv(env_fn, min(num_envs, len(lineups)))
    try:
        return _play(vec_env, models, lineups, deterministic, seed)
    finally:
        vec_env.close()


def _play(
    vec_env: VectorEnv,
    models: dict[str, PolicyModel],
    lineups: list[tuple[str, ...]],
    deterministic: bool,
    seed: int | None,
) -> list[MatchRecord]:
    n_envs, n_players, spec = vec_env.num_envs, vec_env.num_players, vec_env.action_spec
    bad = [lu for lu in lineups if len(lu) != n_players]
    if bad:
        raise ValueError(f"play_matches: lineup {bad[0]} does not have {n_players} seats")

    pending = deque(lineups)
    playing: list[tuple[str, ...]] = [pending.popleft() for _ in range(n_envs)]
    counted = [True] * n_envs   # False once an env has no scheduled match left
    states: list[list[State]] = [
        [models[playing[e][p]].initial_state(1) for p in range(n_players)]
        for e in range(n_envs)
    ]
    returns = np.zeros((n_envs, n_players), dtype=np.float64)
    lengths = np.zeros(n_envs, dtype=np.int64)
    records: list[MatchRecord] = []

    obs, infos = vec_env.reset_all(seed=seed)
    while len(records) < len(lineups):
        actions = np.zeros((n_envs, n_players, *spec.action_shape), dtype=spec.numpy_dtype)
        groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for e in range(n_envs):
            for p in range(n_players):
                if infos[e].get(p, {}).get("active", True):
                    groups[playing[e][p]].append((e, p))
        for name, seats in groups.items():
            acts, new_state = _act_group(
                models[name], seats, obs, infos, states, spec, deterministic,
            )
            for k, (e, p) in enumerate(seats):
                actions[e, p] = acts[k]
                states[e][p] = slice_batch(new_state, k)

        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        returns += rewards
        lengths += 1

        for e in range(n_envs):
            if not (terminated[e] or truncated[e]):
                continue
            if counted[e]:
                terminal_infos = {
                    p: infos[e].get(p, {}).get("terminal_info", {}) for p in range(n_players)
                }
                outcomes = player_outcomes(returns[e].tolist(), terminal_infos, n_players)
                records.append(MatchRecord(
                    lineup=playing[e],
                    outcomes=tuple(float(x) for x in outcomes),
                    returns=tuple(float(x) for x in returns[e]),
                    length=int(lengths[e]),
                ))
            if pending:
                playing[e] = pending.popleft()
            else:
                counted[e] = False   # keep the env busy with its last lineup; results ignored
            states[e] = [models[playing[e][p]].initial_state(1) for p in range(n_players)]
            returns[e] = 0.0
            lengths[e] = 0
    return records
```

- [ ] **Step 6: Run the new tests.**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_engine.py tests/contract/test_eval_stateful.py -v`
Expected: all PASS.

- [ ] **Step 7: Run the full fast suite.**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: PASS. The old `evaluate_agents` tests are untouched until T7.2.

- [ ] **Step 8: Commit.**

```bash
git add src/colosseum/eval.py tests/helpers.py tests/unit/test_eval_engine.py tests/contract/test_eval_stateful.py
git commit -m "feat: eval engine with per-seat model state, active-only inference and seat rotation (R4-11)"
```

---
### Task T7.2: Eval statistics and CLI: W/D/L, Wilson CIs, per-seat breakdown, solo, checkpoint-built models, JSON

Spec block 7 (statistics and CLI); findings R4-12 (partial), R4-13, R6-10, ET-11, ET-21.

**The problem today:**
- the CI of the mirrored row is inverted and can exclude its own point estimate;
- draws count as losses;
- `--num-matches` is a total, not a per-pair count;
- every agent must share the global architecture;
- solo games produce an empty report;
- there is no machine-readable output.

**After this task:**
- **Per pair:**
  - W/D/L;
  - the win rate with a Wilson interval;
  - the score (W + D/2)/n with a Wilson interval on the score. This is an approximation and the module docstring explains why it is conservative;
  - a per-seat breakdown and mean returns.
- **Reversed rows** are derived from the same counts.
- **Solo** (one agent, or a 1-player env): mean return and mean outcome, each with a 95% normal interval, plus a per-seat breakdown.
- **CLI:**
  - `--num-matches` is per pair (per agent in solo mode);
  - `-a/--agent name=path` accepts a checkpoint directory (built from its `meta.json` `networks` when present) or a `.pt` (built from `--config`);
  - `--output result.json` writes the report as JSON.
- **Removed:** the old `evaluate_agents`/`EvalMatrix`/`EvalResult`/`_wilson_ci`/`_run_matches` API.

**Files:**
- Modify: `src/colosseum/eval.py` (whole file; the T7.1 engine is repeated verbatim)
- Modify: `src/colosseum/cli.py` (`eval` command)
- Modify: `README.md` (eval command examples), `CLAUDE.md` (eval command line)
- Delete: the old eval tests (Step 6)
- Test: `tests/unit/test_eval_report.py` (create), `tests/integration/test_eval_cli.py` (create)

**Interfaces:**
- Consumes:
  - T7.1 engine names;
  - `build_model(config)` (T1.4);
  - `NetworkConfig` (T1.4; validated with `NetworkConfig.model_validate`, which accepts the `core.class` alias);
  - `load_config(path, overrides=None)` (T6.1);
  - `CheckpointManager(base_dir)`, with `.save(agent_id, policy_version, model_state: dict[str, np.ndarray], trainer_state=None, meta_extra=None) -> str` (the checkpoint dir) and `.load_model(agent_id, checkpoint_id) -> dict[str, np.ndarray]` (T5.3). The layout is `<base>/<agent>/ckpt_v<N>/{model.pt, trainer_state.pt, meta.json}`, and `meta_extra` keys are merged into `meta.json`; the launcher writes `meta["networks"]`;
  - `AlternatingEnv`, `CounterModel` (T7.1 helpers).
- Produces:
  - in `colosseum.eval`:
    - `wilson_interval(successes: float, n: int, z=1.96) -> (lo, hi)` (always contains the point estimate; `n == 0` gives `(0.0, 1.0)`);
    - `normal_interval(values, z=1.96) -> (mean, lo, hi)`;
    - `PairStats` (`.add(record)`, `.reversed()`, `.to_row()`) and `SoloStats` (`.add(record)`, `.to_row()`);
    - `EvalReport` (`.mode`, `.rows()`, `.to_dict()`, `.write_json(path)`, `.summary()`);
    - `summarize(records, agent_names, num_players, num_matches, deterministic=False) -> EvalReport`;
    - `evaluate(models, env_fn, num_matches=100, num_envs=8, deterministic=False, seed=None) -> EvalReport`;
    - `load_eval_model(path, config) -> PolicyModel`.
  - The JSON schema pinned in `tests/unit/test_eval_report.py`:
    - top-level keys: `mode`, `agents`, `num_players`, `num_matches_per_pair`, `deterministic`, `ci_level`, `score_ci_method`, `pairs`, `solo`;
    - each pair row: `agent_a`, `agent_b`, `n`, `wins`, `draws`, `losses`, `win_rate`, `win_rate_ci`, `score`, `score_ci`, `mean_return_a`, `mean_return_b`, `mean_episode_length`, `per_seat` (each cell: `n`, `wins`, `draws`, `losses`, `score`);
    - each solo row: `agent`, `n`, `mean_return`, `return_ci`, `mean_outcome`, `outcome_ci`, `mean_episode_length`, `per_seat` (each cell: `n`, `mean_return`, `mean_outcome`).
  - CLI: `colosseum eval -c CFG -a NAME=PATH [-a ...] [-n N] [--num-envs] [--deterministic] [--seed] [-o result.json]`.

- [ ] **Step 1: Write the failing unit test** `tests/unit/test_eval_report.py`:

```python
"""Eval statistics, reversed rows, solo mode and the pinned JSON schema (T7.2)."""
import json

import pytest

from colosseum.eval import MatchRecord, evaluate, normal_interval, summarize, wilson_interval
from tests.helpers import AlternatingEnv, CounterModel

TOP_KEYS = {"mode", "agents", "num_players", "num_matches_per_pair", "deterministic",
            "ci_level", "score_ci_method", "pairs", "solo"}
PAIR_ROW_KEYS = {"agent_a", "agent_b", "n", "wins", "draws", "losses", "win_rate", "win_rate_ci",
                 "score", "score_ci", "mean_return_a", "mean_return_b", "mean_episode_length", "per_seat"}
PAIR_SEAT_KEYS = {"n", "wins", "draws", "losses", "score"}
SOLO_ROW_KEYS = {"agent", "n", "mean_return", "return_ci", "mean_outcome", "outcome_ci",
                 "mean_episode_length", "per_seat"}
SOLO_SEAT_KEYS = {"n", "mean_return", "mean_outcome"}


def _rec(lineup, outcomes, returns=None, length=5):
    returns = returns if returns is not None else [0.0] * len(lineup)
    return MatchRecord(tuple(lineup), tuple(outcomes), tuple(returns), length)


def _pair_records(results):
    """'W'/'D'/'L' from agent a's side; lineups alternate (a, b), (b, a)."""
    out = []
    for i, r in enumerate(results):
        lineup = ("a", "b") if i % 2 == 0 else ("b", "a")
        score_a = {"W": 1.0, "D": 0.5, "L": 0.0}[r]
        outcomes = (score_a, 1.0 - score_a) if lineup[0] == "a" else (1.0 - score_a, score_a)
        out.append(_rec(lineup, outcomes))
    return out


def test_wilson_interval_contains_point_and_handles_edges():
    for successes, n in [(0, 10), (10, 10), (5, 10), (45.0, 100), (0.5, 1), (0, 1)]:
        lo, hi = wilson_interval(successes, n)
        assert 0.0 <= lo <= successes / n <= hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60


def test_normal_interval():
    mean, lo, hi = normal_interval([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5
    assert lo == pytest.approx(2.5 - 1.959963984540054 * 1.2909944487358056 / 2)
    assert hi == pytest.approx(2.5 + 1.959963984540054 * 1.2909944487358056 / 2)
    assert normal_interval([3.0]) == (3.0, 3.0, 3.0)
    assert normal_interval([]) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("w,d,l", [(10, 80, 10), (30, 10, 60), (0, 100, 0), (7, 0, 0)])
def test_pair_rows_are_consistent_and_contain_their_estimates(w, d, l):
    results = ["W"] * w + ["D"] * d + ["L"] * l
    report = summarize(_pair_records(results), ["a", "b"], num_players=2, num_matches=len(results))
    ab, ba = report.rows()
    n = len(results)
    assert (ab["agent_a"], ab["agent_b"], ba["agent_a"], ba["agent_b"]) == ("a", "b", "b", "a")
    assert (ab["n"], ab["wins"], ab["draws"], ab["losses"]) == (n, w, d, l)
    assert (ba["n"], ba["wins"], ba["draws"], ba["losses"]) == (n, l, d, w)
    for row in (ab, ba):
        assert row["win_rate_ci"][0] <= row["win_rate"] <= row["win_rate_ci"][1]
        assert row["score_ci"][0] <= row["score"] <= row["score_ci"][1]
    assert ab["win_rate"] == pytest.approx(w / n)
    assert ab["score"] == pytest.approx((w + d / 2) / n)
    assert ba["score"] == pytest.approx(1.0 - ab["score"])
    assert ba["score_ci"] == pytest.approx([1.0 - ab["score_ci"][1], 1.0 - ab["score_ci"][0]])
    # per-seat: a in seat 0 is b in seat 1 of the reversed row, with W and L swapped
    assert sum(cell["n"] for cell in ab["per_seat"].values()) == n
    for seat_a, seat_b in (("0", "1"), ("1", "0")):
        if seat_a in ab["per_seat"]:
            assert ab["per_seat"][seat_a]["wins"] == ba["per_seat"][seat_b]["losses"]
            assert ab["per_seat"][seat_a]["draws"] == ba["per_seat"][seat_b]["draws"]


def test_three_player_pair_uses_seat_groups():
    records = [_rec(("a", "b", "a"), (1.0, 0.0, 1.0)), _rec(("b", "a", "b"), (0.0, 1.0, 0.0))]
    ab, ba = summarize(records, ["a", "b"], num_players=3, num_matches=2).rows()
    assert set(ab["per_seat"]) == {"0,2", "1"}
    assert set(ba["per_seat"]) == {"0,2", "1"}
    assert ab["wins"] == 2 and ba["losses"] == 2
    assert ba["per_seat"]["1"] == {"n": 1, "wins": 0, "draws": 0, "losses": 1, "score": 0.0}


def test_solo_report_has_normal_intervals():
    records = [_rec(("a",), (0.5,), (r,), 10) for r in [1.0, 2.0, 3.0, 4.0]]
    report = summarize(records, ["a"], num_players=1, num_matches=4)
    assert report.mode == "solo"
    row = report.to_dict()["solo"][0]
    assert set(row) == SOLO_ROW_KEYS
    assert row["n"] == 4 and row["mean_return"] == 2.5
    assert row["return_ci"][0] < 2.5 < row["return_ci"][1]
    assert row["mean_episode_length"] == 10.0


def test_evaluate_end_to_end_per_seat_breakdown():
    # AlternatingEnv: seat 0 always wins -> each agent wins exactly its seat-0 matches.
    report = evaluate({"a": CounterModel(), "b": CounterModel()}, AlternatingEnv, num_matches=4, num_envs=2)
    ab, ba = report.rows()
    assert (ab["n"], ab["wins"], ab["draws"], ab["losses"]) == (4, 2, 0, 2)
    assert ab["per_seat"]["0"] == {"n": 2, "wins": 2, "draws": 0, "losses": 0, "score": 1.0}
    assert ab["per_seat"]["1"] == {"n": 2, "wins": 0, "draws": 0, "losses": 2, "score": 0.0}
    assert ba["per_seat"]["0"]["wins"] == 2
    text = report.summary()
    assert "win_rate" in text and "score CI" in text


def test_json_schema_is_pinned(tmp_path):
    report = evaluate({n: CounterModel() for n in "abc"}, AlternatingEnv, num_matches=2, num_envs=4)
    path = tmp_path / "result.json"
    report.write_json(path)
    data = json.loads(path.read_text())
    assert set(data) == TOP_KEYS
    assert data["mode"] == "pairwise"
    assert data["agents"] == ["a", "b", "c"]
    assert data["num_players"] == 2
    assert data["num_matches_per_pair"] == 2
    assert data["ci_level"] == 0.95
    assert data["solo"] == []
    assert [(r["agent_a"], r["agent_b"]) for r in data["pairs"]] == [
        ("a", "b"), ("b", "a"), ("a", "c"), ("c", "a"), ("b", "c"), ("c", "b"),
    ]
    for row in data["pairs"]:
        assert set(row) == PAIR_ROW_KEYS
        assert len(row["win_rate_ci"]) == 2 and len(row["score_ci"]) == 2
        assert row["n"] == 2
        for cell in row["per_seat"].values():
            assert set(cell) == PAIR_SEAT_KEYS

    solo = evaluate({"a": CounterModel()}, AlternatingEnv, num_matches=2, num_envs=2).to_dict()
    assert set(solo) == TOP_KEYS
    assert solo["mode"] == "solo" and solo["pairs"] == []
    assert set(solo["solo"][0]) == SOLO_ROW_KEYS
    assert set(solo["solo"][0]["per_seat"]) == {"0", "1"}
    for cell in solo["solo"][0]["per_seat"].values():
        assert set(cell) == SOLO_SEAT_KEYS
```

- [ ] **Step 2: Write the failing CLI test** `tests/integration/test_eval_cli.py`:

```python
"""`colosseum eval` CLI: .pt and checkpoint-dir agents, heterogeneous architectures, JSON (T7.2)."""
import json

import torch
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.config import NetworkConfig, load_config
from colosseum.core.registry import build_model
from colosseum.eval import load_eval_model

CONFIG = """\
env:
  env_class: examples.tic_tac_toe.env.TicTacToeEnv
  num_players: 2
networks:
  encoder_class: examples.tic_tac_toe.networks.TicTacToeEncoder
  policy_class: examples.tic_tac_toe.networks.TicTacToePolicy
  value_class: examples.tic_tac_toe.networks.TicTacToeValue
"""

GRU_NETWORKS = {
    "encoder_class": "examples.tic_tac_toe.networks.TicTacToeEncoder",
    "core": {"class": "colosseum.networks.cores.GRUCore", "kwargs": {"hidden_size": 64}},
    "policy_class": "examples.tic_tac_toe.networks.TicTacToePolicy",
    "value_class": "examples.tic_tac_toe.networks.TicTacToeValue",
}


def _setup(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(CONFIG)
    cfg = load_config(cfg_path)
    torch.manual_seed(0)
    pt_path = tmp_path / "a.pt"
    torch.save(build_model(cfg).state_dict(), pt_path)
    gru_model = build_model(cfg.model_copy(update={"networks": NetworkConfig.model_validate(GRU_NETWORKS)}))
    ckpt_dir = CheckpointManager(tmp_path / "checkpoints").save(
        "agent_b", 3,
        {k: v.detach().cpu().numpy() for k, v in gru_model.state_dict().items()},
        meta_extra={"networks": GRU_NETWORKS},
    )
    return cfg_path, cfg, pt_path, ckpt_dir


def test_load_eval_model_builds_from_meta_networks(tmp_path):
    _cfg_path, cfg, pt_path, ckpt_dir = _setup(tmp_path)
    assert not load_eval_model(pt_path, cfg).is_stateful          # global networks: no core
    assert load_eval_model(ckpt_dir, cfg).is_stateful             # meta.json networks: GRU core


def test_eval_cli_pairs_pt_and_heterogeneous_checkpoint(tmp_path):
    cfg_path, _cfg, pt_path, ckpt_dir = _setup(tmp_path)
    out = tmp_path / "result.json"
    result = CliRunner().invoke(main, [
        "eval", "-c", str(cfg_path), "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}",
        "-n", "4", "--num-envs", "2", "--seed", "0", "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "score" in result.output
    data = json.loads(out.read_text())
    assert data["mode"] == "pairwise"
    assert data["num_matches_per_pair"] == 4
    assert [(r["agent_a"], r["agent_b"]) for r in data["pairs"]] == [("a", "b"), ("b", "a")]
    assert data["pairs"][0]["n"] == 4
    assert set(data["pairs"][0]["per_seat"]) == {"0", "1"}


def test_eval_cli_rejects_bad_agent_specs(tmp_path):
    cfg_path, _cfg, pt_path, _ckpt_dir = _setup(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["eval", "-c", str(cfg_path), "-a", "no-equals-sign"])
    assert result.exit_code == 2 and "name=path" in result.output
    result = runner.invoke(main, ["eval", "-c", str(cfg_path), "-a", f"a={tmp_path / 'missing.pt'}"])
    assert result.exit_code == 2
    result = runner.invoke(main, ["eval", "-c", str(cfg_path), "-a", f"a={pt_path}", "-a", f"a={pt_path}"])
    assert result.exit_code == 2 and "duplicate" in result.output
```

- [ ] **Step 3: Run them and watch them fail.**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_report.py tests/integration/test_eval_cli.py -v`
Expected failure: `ImportError: cannot import name 'evaluate' from 'colosseum.eval'` (and `summarize`, `load_eval_model`).

- [ ] **Step 4: Replace `src/colosseum/eval.py` with the complete module below.**

The engine section repeats the T7.1 code verbatim. The old `EvalResult`, `EvalMatrix`, `_wilson_ci`, `evaluate_agents` and `_run_matches` are gone.

```python
"""Evaluation: inference-only matches between agents and checkpoints.

Engine
------
``schedule_lineups`` turns agent names into a list of matches (one seat -> agent
lineup per match). ``play_matches`` plays them on a ``VectorEnv`` with batched
``act()`` inference per agent and one model ``State`` per (env, seat):

- the state starts at ``initial_state(1)`` at every episode start;
- it advances only when that seat acts (``info[p]["active"]``, default True);
  non-acting seats send the all-zeros action and keep their state;
- an acting seat whose action mask allows nothing raises ``EnvContractError``.

These are the same semantics as training rollouts, so stateful models (LSTM,
GRU, attention) are evaluated as the policy they were trained as.

Seat rotation: for a pair (a, b), match m gives seat s to ``(a, b)[(s + m) % 2]``.
With an even number of matches per pair each agent plays every seat equally
often. Every scheduled match is played to completion; extra episodes that idle
envs play while others finish are discarded, so short episodes are not favoured.

Statistics
----------
Pairwise (two or more agents in an N-player game, N >= 2). From agent a's side a
match is a win when a's mean seat outcome (``core.outcomes``) beats b's, a draw
when equal. Each pair reports:

- W/D/L and the win rate W/n with a 95% Wilson score interval;
- the score (W + D/2)/n with a 95% Wilson interval computed on the score as if it
  were a Bernoulli proportion. This is an approximation: a draw counts as half a
  win, and the per-match score variance (W + D/4)/n - s^2 never exceeds the
  Bernoulli variance s(1 - s), so the interval is conservative (never too narrow).
  Paired/rating-based selection (Bradley-Terry with bootstrap) is SP4;
- a per-seat breakdown keyed by the seats agent a occupied (e.g. "0", or "0,2");
- mean returns and mean episode length.

The reversed row (b vs a) is derived from the same counts, so the two rows never
contradict each other.

Solo (one agent, or a 1-player env): per agent, the mean episode return and the
mean outcome with 95% normal-approximation intervals ``mean +- 1.96 * sd / sqrt(n)``,
plus a per-seat breakdown when the env has several seats.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from colosseum.core.action_spec import ActionSpec
from colosseum.core.config import ColosseumConfig, NetworkConfig
from colosseum.core.errors import EnvContractError
from colosseum.core.outcomes import player_outcomes
from colosseum.core.registry import build_model
from colosseum.envs.base_env import BaseEnv
from colosseum.envs.vec_env import VectorEnv
from colosseum.networks.model import PolicyModel, act
from colosseum.networks.state import State, cat_batch, slice_batch

logger = logging.getLogger(__name__)

Z_95 = 1.959963984540054
SCORE_CI_METHOD = "wilson-on-score (draw = half win; conservative approximation)"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchRecord:
    """One finished match: agent name, outcome in [0, 1] and return per seat."""

    lineup: tuple[str, ...]
    outcomes: tuple[float, ...]
    returns: tuple[float, ...]
    length: int


def schedule_lineups(
    agent_names: Sequence[str], num_players: int, num_matches: int,
) -> list[tuple[str, ...]]:
    """Seat -> agent lineups for an evaluation.

    - One agent, or a 1-player env (solo): ``num_matches`` lineups per agent with
      the agent in every seat.
    - Otherwise (pairwise): for every pair (a, b) in ``itertools.combinations``
      order, ``num_matches`` lineups where match m gives seat s to
      ``(a, b)[(s + m) % 2]``.
    """
    names = list(agent_names)
    if not names:
        raise ValueError("schedule_lineups: no agents")
    if len(set(names)) != len(names):
        raise ValueError(f"schedule_lineups: duplicate agent names in {names}")
    if num_players < 1:
        raise ValueError(f"schedule_lineups: num_players must be >= 1, got {num_players}")
    if num_matches < 1:
        raise ValueError(f"schedule_lineups: num_matches must be >= 1, got {num_matches}")
    if len(names) == 1 or num_players == 1:
        return [tuple([name] * num_players) for name in names for _ in range(num_matches)]
    if num_matches % 2:
        logger.warning(
            "eval: num_matches=%d is odd, so seats are balanced only up to one match per pair",
            num_matches,
        )
    lineups: list[tuple[str, ...]] = []
    for a, b in itertools.combinations(names, 2):
        pair = (a, b)
        for m in range(num_matches):
            lineups.append(tuple(pair[(s + m) % 2] for s in range(num_players)))
    return lineups


def _seat_mask(info: object, action_spec: ActionSpec) -> np.ndarray | None:
    """Flat bool action mask of one seat, or None when the env gives none."""
    if not isinstance(info, dict) or info.get("action_mask") is None:
        return None
    raw = info["action_mask"]
    return action_spec.flatten_mask(raw if isinstance(raw, dict) else np.asarray(raw))


def _check_acting_masks(
    masks: np.ndarray, seats: list[tuple[int, int]], action_spec: ActionSpec,
) -> None:
    for comp in action_spec.components:
        if comp.mask_size == 0:
            continue
        segment = masks[:, comp.mask_offset:comp.mask_offset + comp.mask_size]
        empty = np.flatnonzero(~segment.any(axis=1))
        if empty.size:
            env_idx, seat = seats[int(empty[0])]
            raise EnvContractError(
                f"eval: env {env_idx} seat {seat} is acting but its action mask allows "
                f"no action for component {comp.name!r}"
            )


def _act_group(
    model: PolicyModel,
    seats: list[tuple[int, int]],
    obs: np.ndarray,
    infos: list[dict],
    states: list[list[State]],
    action_spec: ActionSpec,
    deterministic: bool,
) -> tuple[np.ndarray, State]:
    obs_batch = torch.as_tensor(
        np.stack([np.asarray(obs[e, p]) for e, p in seats]), dtype=torch.float32,
    )
    seat_masks = [_seat_mask(infos[e].get(p, {}), action_spec) for e, p in seats]
    mask_batch = None
    if any(m is not None for m in seat_masks):
        allow_all = np.ones(action_spec.flat_mask_size, dtype=bool)
        stacked = np.stack([allow_all if m is None else m for m in seat_masks])
        _check_acting_masks(stacked, seats, action_spec)
        mask_batch = torch.as_tensor(stacked)
    state_batch = cat_batch([states[e][p] for e, p in seats])
    with torch.no_grad():
        out = act(model, obs_batch, state_batch, mask_batch, deterministic=deterministic)
    return out.actions.cpu().numpy(), out.state


def play_matches(
    models: dict[str, PolicyModel],
    env_fn: Callable[[], BaseEnv],
    lineups: Sequence[tuple[str, ...]],
    num_envs: int = 8,
    deterministic: bool = False,
    seed: int | None = None,
) -> list[MatchRecord]:
    """Play every lineup once (to completion) and return one record per match.

    Records are in completion order, not schedule order.
    """
    lineups = [tuple(lu) for lu in lineups]
    if not lineups:
        return []
    unknown = sorted({name for lu in lineups for name in lu} - set(models))
    if unknown:
        raise ValueError(f"play_matches: lineups use unknown agents {unknown}")
    if num_envs < 1:
        raise ValueError(f"play_matches: num_envs must be >= 1, got {num_envs}")
    if seed is not None:
        torch.manual_seed(seed)
    for model in models.values():
        model.eval()
    vec_env = VectorEnv(env_fn, min(num_envs, len(lineups)))
    try:
        return _play(vec_env, models, lineups, deterministic, seed)
    finally:
        vec_env.close()


def _play(
    vec_env: VectorEnv,
    models: dict[str, PolicyModel],
    lineups: list[tuple[str, ...]],
    deterministic: bool,
    seed: int | None,
) -> list[MatchRecord]:
    n_envs, n_players, spec = vec_env.num_envs, vec_env.num_players, vec_env.action_spec
    bad = [lu for lu in lineups if len(lu) != n_players]
    if bad:
        raise ValueError(f"play_matches: lineup {bad[0]} does not have {n_players} seats")

    pending = deque(lineups)
    playing: list[tuple[str, ...]] = [pending.popleft() for _ in range(n_envs)]
    counted = [True] * n_envs   # False once an env has no scheduled match left
    states: list[list[State]] = [
        [models[playing[e][p]].initial_state(1) for p in range(n_players)]
        for e in range(n_envs)
    ]
    returns = np.zeros((n_envs, n_players), dtype=np.float64)
    lengths = np.zeros(n_envs, dtype=np.int64)
    records: list[MatchRecord] = []

    obs, infos = vec_env.reset_all(seed=seed)
    while len(records) < len(lineups):
        actions = np.zeros((n_envs, n_players, *spec.action_shape), dtype=spec.numpy_dtype)
        groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for e in range(n_envs):
            for p in range(n_players):
                if infos[e].get(p, {}).get("active", True):
                    groups[playing[e][p]].append((e, p))
        for name, seats in groups.items():
            acts, new_state = _act_group(
                models[name], seats, obs, infos, states, spec, deterministic,
            )
            for k, (e, p) in enumerate(seats):
                actions[e, p] = acts[k]
                states[e][p] = slice_batch(new_state, k)

        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        returns += rewards
        lengths += 1

        for e in range(n_envs):
            if not (terminated[e] or truncated[e]):
                continue
            if counted[e]:
                terminal_infos = {
                    p: infos[e].get(p, {}).get("terminal_info", {}) for p in range(n_players)
                }
                outcomes = player_outcomes(returns[e].tolist(), terminal_infos, n_players)
                records.append(MatchRecord(
                    lineup=playing[e],
                    outcomes=tuple(float(x) for x in outcomes),
                    returns=tuple(float(x) for x in returns[e]),
                    length=int(lengths[e]),
                ))
            if pending:
                playing[e] = pending.popleft()
            else:
                counted[e] = False   # keep the env busy with its last lineup; results ignored
            states[e] = [models[playing[e][p]].initial_state(1) for p in range(n_players)]
            returns[e] = 0.0
            lengths[e] = 0
    return records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_interval(successes: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion ``successes / n``.

    ``successes`` may be fractional (score with draws as half wins). The result
    always contains the point estimate; ``n == 0`` gives ``(0.0, 1.0)``.
    """
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(0.0, p * (1.0 - p)) / n + z * z / (4.0 * n * n)) / denom
    low = min(max(0.0, center - half), p)
    high = max(min(1.0, center + half), p)
    return low, high


def normal_interval(values: Sequence[float], z: float = Z_95) -> tuple[float, float, float]:
    """``(mean, low, high)`` with ``mean +- z * sd / sqrt(n)`` (sd with ddof=1).

    Fewer than two values give a zero-width interval; no values give zeros.
    """
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(x.mean())
    if x.size < 2:
        return mean, mean, mean
    half = z * float(x.std(ddof=1)) / math.sqrt(x.size)
    return mean, mean - half, mean + half


def _seat_key(seats: Sequence[int]) -> str:
    return ",".join(str(s) for s in seats)


@dataclass
class PairStats:
    """Counts for one unordered pair, from ``agent_a``'s side."""

    agent_a: str
    agent_b: str
    num_players: int
    wins: int = 0
    draws: int = 0
    losses: int = 0
    return_a: float = 0.0
    return_b: float = 0.0
    total_length: int = 0
    per_seat: dict[tuple[int, ...], list[int]] = field(default_factory=dict)  # a's seats -> [W, D, L]

    @property
    def n(self) -> int:
        return self.wins + self.draws + self.losses

    def add(self, record: MatchRecord) -> None:
        seats_a = tuple(s for s, name in enumerate(record.lineup) if name == self.agent_a)
        seats_b = tuple(s for s, name in enumerate(record.lineup) if name == self.agent_b)
        if not seats_a or not seats_b or len(seats_a) + len(seats_b) != len(record.lineup):
            raise ValueError(
                f"PairStats({self.agent_a}, {self.agent_b}): record lineup {record.lineup} "
                f"is not a lineup of this pair"
            )
        score_a = float(np.mean([record.outcomes[s] for s in seats_a]))
        score_b = float(np.mean([record.outcomes[s] for s in seats_b]))
        cell = self.per_seat.setdefault(seats_a, [0, 0, 0])
        if score_a > score_b:
            self.wins += 1
            cell[0] += 1
        elif score_a < score_b:
            self.losses += 1
            cell[2] += 1
        else:
            self.draws += 1
            cell[1] += 1
        self.return_a += float(np.mean([record.returns[s] for s in seats_a]))
        self.return_b += float(np.mean([record.returns[s] for s in seats_b]))
        self.total_length += record.length

    def reversed(self) -> PairStats:
        """The same counts seen from ``agent_b``'s side."""
        def complement(seats: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(s for s in range(self.num_players) if s not in seats)

        return PairStats(
            agent_a=self.agent_b, agent_b=self.agent_a, num_players=self.num_players,
            wins=self.losses, draws=self.draws, losses=self.wins,
            return_a=self.return_b, return_b=self.return_a, total_length=self.total_length,
            per_seat={complement(k): [v[2], v[1], v[0]] for k, v in self.per_seat.items()},
        )

    def to_row(self) -> dict[str, Any]:
        n = self.n
        points = self.wins + 0.5 * self.draws
        per_seat = {}
        for seats, (w, d, l) in sorted(self.per_seat.items()):
            m = w + d + l
            per_seat[_seat_key(seats)] = {
                "n": m, "wins": w, "draws": d, "losses": l,
                "score": (w + 0.5 * d) / m if m else 0.0,
            }
        return {
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "n": n,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "win_rate": self.wins / n if n else 0.0,
            "win_rate_ci": list(wilson_interval(self.wins, n)),
            "score": points / n if n else 0.0,
            "score_ci": list(wilson_interval(points, n)),
            "mean_return_a": self.return_a / n if n else 0.0,
            "mean_return_b": self.return_b / n if n else 0.0,
            "mean_episode_length": self.total_length / n if n else 0.0,
            "per_seat": per_seat,
        }


@dataclass
class SoloStats:
    """Per-agent episode statistics for solo evaluation."""

    agent: str
    num_players: int
    returns: list[float] = field(default_factory=list)    # per match: mean over seats
    outcomes: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    seat_returns: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    seat_outcomes: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, record: MatchRecord) -> None:
        if set(record.lineup) != {self.agent}:
            raise ValueError(f"SoloStats({self.agent}): unexpected lineup {record.lineup}")
        self.returns.append(float(np.mean(record.returns)))
        self.outcomes.append(float(np.mean(record.outcomes)))
        self.lengths.append(record.length)
        for seat in range(len(record.lineup)):
            self.seat_returns[seat].append(record.returns[seat])
            self.seat_outcomes[seat].append(record.outcomes[seat])

    def to_row(self) -> dict[str, Any]:
        mean_ret, ret_lo, ret_hi = normal_interval(self.returns)
        mean_out, out_lo, out_hi = normal_interval(self.outcomes)
        per_seat = {
            str(seat): {
                "n": len(self.seat_returns[seat]),
                "mean_return": float(np.mean(self.seat_returns[seat])),
                "mean_outcome": float(np.mean(self.seat_outcomes[seat])),
            }
            for seat in sorted(self.seat_returns)
        }
        return {
            "agent": self.agent,
            "n": len(self.returns),
            "mean_return": mean_ret,
            "return_ci": [ret_lo, ret_hi],
            "mean_outcome": mean_out,
            "outcome_ci": [out_lo, out_hi],
            "mean_episode_length": float(np.mean(self.lengths)) if self.lengths else 0.0,
            "per_seat": per_seat,
        }


@dataclass
class EvalReport:
    """Result of an evaluation; ``to_dict()`` is the JSON schema written by ``--output``."""

    agents: list[str]
    num_players: int
    num_matches: int
    deterministic: bool
    pairs: list[PairStats] = field(default_factory=list)
    solo: list[SoloStats] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return "solo" if self.solo else "pairwise"

    def rows(self) -> list[dict[str, Any]]:
        """Both directions of every pair, derived from one set of counts."""
        out: list[dict[str, Any]] = []
        for pair in self.pairs:
            out.append(pair.to_row())
            out.append(pair.reversed().to_row())
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "agents": list(self.agents),
            "num_players": self.num_players,
            "num_matches_per_pair": self.num_matches,
            "deterministic": self.deterministic,
            "ci_level": 0.95,
            "score_ci_method": SCORE_CI_METHOD,
            "pairs": self.rows(),
            "solo": [s.to_row() for s in self.solo],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def summary(self) -> str:
        lines: list[str] = []
        if self.mode == "solo":
            header = (f"{'agent':<16} {'n':>6}  {'mean_return [95% CI]':<30} "
                      f"{'mean_outcome [95% CI]':<30} {'mean_len':>8}")
            lines += [header, "-" * len(header)]
            for s in self.solo:
                r = s.to_row()
                lines.append(
                    f"{r['agent']:<16} {r['n']:>6}  "
                    f"{r['mean_return']:>9.3f} [{r['return_ci'][0]:.3f}, {r['return_ci'][1]:.3f}]  "
                    f"{r['mean_outcome']:>6.3f} [{r['outcome_ci'][0]:.3f}, {r['outcome_ci'][1]:.3f}]  "
                    f"{r['mean_episode_length']:>8.1f}"
                )
            return "\n".join(lines)
        header = (f"{'agent_a':<16} {'agent_b':<16} {'n':>6} {'W':>6} {'D':>6} {'L':>6}  "
                  f"{'win_rate [95% CI]':<24} {'score [95% CI]':<24}")
        lines += [header, "-" * len(header)]
        for r in self.rows():
            lines.append(
                f"{r['agent_a']:<16} {r['agent_b']:<16} {r['n']:>6} {r['wins']:>6} "
                f"{r['draws']:>6} {r['losses']:>6}  "
                f"{r['win_rate']:.3f} [{r['win_rate_ci'][0]:.3f}, {r['win_rate_ci'][1]:.3f}]   "
                f"{r['score']:.3f} [{r['score_ci'][0]:.3f}, {r['score_ci'][1]:.3f}]"
            )
            for seats, cell in r["per_seat"].items():
                lines.append(
                    f"{'':<16}   seats {seats:<8} n={cell['n']:<5} W={cell['wins']:<5} "
                    f"D={cell['draws']:<5} L={cell['losses']:<5} score={cell['score']:.3f}"
                )
        lines.append(f"(score CI: {SCORE_CI_METHOD})")
        return "\n".join(lines)


def summarize(
    records: Sequence[MatchRecord],
    agent_names: Sequence[str],
    num_players: int,
    num_matches: int,
    deterministic: bool = False,
) -> EvalReport:
    """Aggregate match records into an :class:`EvalReport`."""
    names = list(agent_names)
    report = EvalReport(agents=names, num_players=num_players,
                        num_matches=num_matches, deterministic=deterministic)
    if len(names) == 1 or num_players == 1:
        solo = {name: SoloStats(name, num_players) for name in names}
        for record in records:
            solo[record.lineup[0]].add(record)
        report.solo = [solo[name] for name in names]
        return report
    pairs = {(a, b): PairStats(a, b, num_players) for a, b in itertools.combinations(names, 2)}
    for record in records:
        present = sorted(set(record.lineup), key=names.index)
        if len(present) != 2:
            raise ValueError(f"summarize: lineup {record.lineup} is not a pair lineup")
        pairs[(present[0], present[1])].add(record)
    report.pairs = list(pairs.values())
    return report


def evaluate(
    models: dict[str, PolicyModel],
    env_fn: Callable[[], BaseEnv],
    num_matches: int = 100,
    num_envs: int = 8,
    deterministic: bool = False,
    seed: int | None = None,
) -> EvalReport:
    """Schedule, play and summarize an evaluation.

    ``num_matches`` is per pair (pairwise) or per agent (solo).
    """
    names = list(models)
    probe = env_fn()
    try:
        num_players = probe.num_players
    finally:
        probe.close()
    lineups = schedule_lineups(names, num_players, num_matches)
    records = play_matches(models, env_fn, lineups, num_envs=num_envs,
                           deterministic=deterministic, seed=seed)
    return summarize(records, names, num_players, num_matches, deterministic)


# ---------------------------------------------------------------------------
# Loading agents
# ---------------------------------------------------------------------------


def load_eval_model(path: str | Path, config: ColosseumConfig) -> PolicyModel:
    """Build a model and load its weights for evaluation.

    ``path`` is either
    - a checkpoint directory ``.../<agent>/ckpt_v<N>/`` containing ``model.pt``
      (layout of ``CheckpointManager``). If its ``meta.json`` has a ``networks``
      section, the model is built from it (so agents with different
      architectures can be compared); otherwise from ``config.networks``;
    - a ``.pt`` file with a plain ``state_dict`` (e.g. ``colosseum bc`` output),
      built from ``config.networks``.
    """
    p = Path(path)
    if p.is_dir():
        model_config = config
        meta_path = p / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            if meta.get("networks") is not None:
                networks = NetworkConfig.model_validate(meta["networks"])
                model_config = config.model_copy(update={"networks": networks})
        model = build_model(model_config)
        if not (p / "model.pt").is_file():
            raise FileNotFoundError(f"{p} is a directory but has no model.pt")
        # Imported here: the checkpoint layout (T5.3) is owned by the coordinator package.
        from colosseum.coordinator.checkpoint_manager import CheckpointManager

        raw = CheckpointManager(p.parent.parent).load_model(p.parent.name, p.name)
    elif p.is_file() and p.suffix == ".pt":
        model = build_model(config)
        raw = torch.load(p, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"{p}: expected a checkpoint directory or a .pt file")
    state_dict = {k: torch.as_tensor(np.asarray(v) if isinstance(v, np.ndarray) else v)
                  for k, v in raw.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model
```

- [ ] **Step 5: Replace the CLI `eval` command.**

In `src/colosseum/cli.py`, add this helper next to the existing module-level helpers (for example below the `--set` parsing helper):

```python
def _parse_agent_spec(spec: str) -> tuple[str, str]:
    """``name=path`` -> (name, path)."""
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise click.BadParameter(f"expected name=path, got {spec!r}", param_hint="'--agent'")
    return name, path
```

Then replace the whole `eval_cmd` command (decorators and function) with the code below. If T6.2/T6.5 changed how CLI commands set up logging, keep that setup in place of the `logging.basicConfig` line.

```python
@main.command("eval")
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Config YAML (env section; networks for .pt agents)")
@click.option("--agent", "-a", "agents", required=True, multiple=True,
              help="name=path. path is a checkpoint dir (model built from its meta.json 'networks', "
                   "else from --config) or a .pt state_dict (model built from --config). Repeatable.")
@click.option("--num-matches", "-n", default=100, type=int, show_default=True,
              help="Matches per agent pair (solo: episodes per agent). Use an even number for exact seat balance.")
@click.option("--num-envs", default=8, type=int, show_default=True, help="Parallel environments")
@click.option("--deterministic", is_flag=True, default=False, help="Act greedily (distribution mode)")
@click.option("--seed", default=None, type=int, help="Seed for env resets and sampling")
@click.option("--output", "-o", default=None, type=click.Path(dir_okay=False),
              help="Write the machine-readable result as JSON")
def eval_cmd(
    config: str,
    agents: tuple[str, ...],
    num_matches: int,
    num_envs: int,
    deterministic: bool,
    seed: int | None,
    output: str | None,
) -> None:
    """Evaluate agents/checkpoints against each other (no training)."""
    import logging

    from colosseum.core.config import load_config
    from colosseum.core.registry import import_class
    from colosseum.eval import evaluate, load_eval_model

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = load_config(config)
    models = {}
    for spec in agents:
        name, path = _parse_agent_spec(spec)
        if name in models:
            raise click.BadParameter(f"duplicate agent name {name!r}", param_hint="'--agent'")
        try:
            models[name] = load_eval_model(path, cfg)
        except (FileNotFoundError, ValueError) as exc:
            raise click.BadParameter(str(exc), param_hint="'--agent'") from exc

    env_cls = import_class(cfg.env.env_class)

    def env_fn():
        return env_cls(**cfg.env.kwargs)

    report = evaluate(models, env_fn, num_matches=num_matches, num_envs=num_envs,
                      deterministic=deterministic, seed=seed)
    click.echo("\n" + report.summary())
    if output:
        report.write_json(output)
        click.echo(f"Result written to {output}")
```

- [ ] **Step 6: Remove the old eval API's tests and callers.**

Run `grep -rln "evaluate_agents\|EvalMatrix\|EvalResult\|_wilson_ci\|_run_matches" src tests examples scripts`:
- Delete every test that uses those names. Before this part they all lived in `test_eval.py`; delete the file if nothing else is left in it.
- `src` must have no hits left. The only former caller was `cli.py`, replaced in Step 5.
- Leave `review/` untouched.

- [ ] **Step 7: Docs.**

In `README.md`, both `colosseum eval` examples (the quick-start one and the section one) use `-a name:path/model.pt`. Rewrite each `-a` line as `-a name=<checkpoint dir>`, i.e. drop `/model.pt`, and add `--output result.json` to the second example. For example:

```
colosseum eval -c configs/examples/tic_tac_toe.yaml \
  -a agent_a=runs/<run_name>/checkpoints/agent_0/ckpt_v100 \
  -a agent_b=runs/<run_name>/checkpoints/agent_0/ckpt_v200 \
  --num-matches 1000 --output result.json
```

Below the second example, add:

```
`--num-matches` is per pair. Seats rotate, so each agent plays every seat equally often. The report gives W/D/L,
the win rate and the score (W + D/2)/n, each with a 95% Wilson interval (the score interval is a conservative
approximation), plus a per-seat breakdown. A checkpoint directory is built from its `meta.json` `networks`, so
different architectures can be compared. A `.pt` file uses the config's `networks`. With one agent or a 1-player
env the report is solo: mean return and outcome with 95% normal intervals.
```

In `CLAUDE.md`, replace `` - Command: `colosseum eval --agents A_ckpt_100 B_ckpt_200 --num_matches 1000 --env my_game` `` with `` - Command: `colosseum eval -c cfg.yaml -a A=<ckpt_dir> -a B=<ckpt_dir> --num-matches 1000 --output result.json` ``.

- [ ] **Step 8: Run the new tests.**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_report.py tests/integration/test_eval_cli.py tests/unit/test_eval_engine.py tests/contract/test_eval_stateful.py -v`
Expected: all PASS.

- [ ] **Step 9: Run the full fast suite and ruff.**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q && .venv/bin/ruff check src tests`
Expected: PASS, no ruff findings.

- [ ] **Step 10: Commit.**

```bash
git add src/colosseum/eval.py src/colosseum/cli.py README.md CLAUDE.md tests
git commit -m "feat: eval report with draw-aware Wilson CIs, per-seat stats, solo mode, checkpoint-built models, JSON (R4-13, R6-10, ET-11)"
```

---
## Contract notes

1. **`PolicyModel.update_normalizers` default (T1.2/T1.4 contract: "default: no-op; ComposedModel delegates").**
   - **Issue.** With a no-op base default, a monolithic `PolicyModel` that contains `NormalizeObs` would never update its statistics, and nothing would warn about it.
   - **Change (T4.2).** The base default calls `update(obs)` on every `NormalizeObs` submodule and stays a no-op when there are none. `ComposedModel` inherits it and has no override. Models without normalizers behave as before. Models whose normalizer sees a transformed observation override the method; `NormalizeObs.update` raises a precise `ValueError` on a shape mismatch.
   - **Proposed contract text:** `def update_normalizers(self, obs: Tensor) -> None: ...  # default: update() every NormalizeObs submodule (no-op if none)`.

2. **`UnrollOutput` has no final state.**
   - **Issue.** Stateful BC (T4.4) cannot carry state from one window to the next. Each window starts from `initial_state`. A random window offset per epoch moves the points where a window starts mid-episode. Episodes no longer than `seq_len` and aligned with window starts are exact.
   - **SP1:** no contract change.
   - **Proposal for SP6** (together with the fast unroll paths): add an optional `UnrollOutput.final_state: State = None` so that BC and R2D2-style training can carry state across windows.

3. **Algorithm state keys (T4.6).**
   - **Issue.** The spec lists an "LR-scheduler" state, but since T2.5 the LR is a pure function of `progress`.
   - **Change.** `APPO.state_dict()` stores `progress` instead. Its keys are exactly `optimizer`, `progress`, `scaler`, `kickstart`, `policy_version` and `consumed_samples`, all deep CPU copies. `load_state_dict()` deep-copies its input and re-applies `set_progress(progress)`.
   - **T5.3 should:**
     - save `algorithm.state_dict()` as `trainer_state.pt`;
     - on resume, load the model weights and then call `algorithm.load_state_dict(trainer_state)`;
     - remove the learner's use of `algorithm.optimizer_state_dict`, `algorithm._optimizer` and `algorithm._policy_version`, together with `BaseAlgorithm.optimizer_state_dict`. T4.6 keeps that property only so the learner works until T5.3.
   - `colosseum.algorithms.base.deep_cpu_copy` is available for T5.3.

4. **Kickstart teacher state.**
   - **Design.** The teacher is unrolled from the chunk's `initial_state`, which the student's behavior policy recorded. This is exact when teacher == student (the BC → RL start) and an approximation afterwards. It requires the teacher to have the same state layout as the student; `APPO.__init__` checks this and raises `ValueError` otherwise.
   - **Spec fit.** In SP1 this always holds, because the spec builds the teacher from the student's config. A separate teacher config with its own state is SP3.

5. **Metric key rename.** APPO's `learning_rate` metric is now `lr`, as the spec's metric list says. T6.3 (console/metrics.jsonl) and T6.4 (WandB) must read `lr`.

6. **Eval checkpoint loading depends on T5.3:**
   - `CheckpointManager(base_dir)` has no side effects beyond creating directories;
   - `.load_model(agent_id, checkpoint_id)` returns `dict[str, np.ndarray]`;
   - `meta_extra` keys are merged into `meta.json`;
   - the launcher writes `meta.json["networks"] = agent_config.networks.model_dump(by_alias=True)`.
   
   If T5.3 stores the networks section under another key, `load_eval_model` and `tests/integration/test_eval_cli.py` must follow it.

7. **New cross-task names this part adds** (additions, not deviations; worth copying into the overview's contract section):
   - `compute_vtrace(..., lam=1.0)`;
   - `CategoricalDist.mask`;
   - `CompositeDist` in insertion order, with `.keys`, `.components` and `.flat_mask_size`;
   - `ActionSpec.component_names` and `ActionSpec.check_distribution(dist)`, called by `validate_config`;
   - `KickstartLoss(teacher, initial_lambda, decay_steps, direction)` with `.compute(student_dist, observations, dones, state0, action_mask)` and `.state_dict()`/`.load_state_dict()`;
   - `BCConfig.seq_len` / `ColosseumConfig.bc`;
   - `OfflineBCTrainer(model, lr, device, seq_len)`;
   - `APPO.consumed_samples`;
   - the `colosseum.eval` API: `MatchRecord`, `schedule_lineups`, `play_matches`, `wilson_interval`, `normal_interval`, `PairStats`, `SoloStats`, `EvalReport`, `summarize`, `evaluate`, `load_eval_model`;
   - the CLI flag `-a/--agent name=path` (was `--agents name:path`).
