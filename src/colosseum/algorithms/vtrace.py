"""V-trace off-policy correction for IMPALA-style training.

Implements the V-trace algorithm from:
  "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
  Espeholt et al., 2018.

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute V-trace targets and policy gradient advantages.

    Args:
        behavior_log_probs: [T, B] log π_μ(a_t|s_t) from the worker's (behavior) policy
        target_log_probs:   [T, B] log π(a_t|s_t) from the learner's (target) policy
        rewards:            [T, B] rewards r_t
        values:             [T, B] V(s_t) from the target (current) network
        bootstrap_value:    [B]    V(s_T) bootstrap value at the end of the chunk
        dones:              [T, B] episode termination flags (1.0 = done)
        gamma:              discount factor
        rho_bar:            importance weight clipping for ρ̄
        c_bar:              trace-cutting coefficient clipping for c̄

    Returns:
        vs:         [T, B] V-trace corrected value targets
        advantages: [T, B] advantages for policy gradient

    Math:
        ρ_t = min(ρ̄, π(a_t|x_t) / μ(a_t|x_t))
        c_t = min(c̄, π(a_t|x_t) / μ(a_t|x_t))
        δ_t = ρ_t * (r_t + γ * V(x_{t+1}) - V(x_t))
        v_s - V(x_s) = Σ_{t=s}^{s+n-1} γ^{t-s} * (Π_{i=s}^{t-1} c_i) * δ_t

    The advantage for policy gradient is:
        A_t = ρ_t * (r_t + γ * v_{t+1} - V(x_t))
    where v_{t+1} is the V-trace target (not the raw value prediction).
    """
    T, B = behavior_log_probs.shape

    # Importance sampling ratios
    log_rhos = torch.clamp(target_log_probs - behavior_log_probs, -20.0, 20.0)
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)

    # Not-done mask: 1 where episode continues, 0 at episode boundaries
    not_done = 1.0 - dones.float()

    # Append bootstrap value for indexing: values_plus[t] = V(s_t), values_plus[T] = V(s_T)
    values_plus = torch.cat([values, bootstrap_value.unsqueeze(0)], dim=0)  # [T+1, B]

    # TD errors: δ_t = ρ_t * (r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t))
    deltas = clipped_rhos * (rewards + gamma * not_done * values_plus[1:] - values_plus[:-1])

    # Backward recursion to compute v_s - V(s_s)
    # vs_minus_v[t] = δ_t + γ * (1-done_t) * c_t * vs_minus_v[t+1]
    vs_minus_v = torch.zeros(T + 1, B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        vs_minus_v[t] = deltas[t] + gamma * not_done[t] * cs[t] * vs_minus_v[t + 1]

    # V-trace targets: v_s = V(s_s) + (v_s - V(s_s))
    vs = values_plus[:-1] + vs_minus_v[:-1]  # [T, B]

    # Advantages for policy gradient
    # A_t = ρ_t * (r_t + γ * v_{t+1} - V(x_t))
    # where v_{t+1} is the V-trace target for next step
    vs_plus = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)  # [T, B]
    advantages = clipped_rhos * (rewards + gamma * not_done * vs_plus - values_plus[:-1])

    return vs, advantages
