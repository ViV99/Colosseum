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
