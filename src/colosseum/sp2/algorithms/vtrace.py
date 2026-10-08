"""V-trace(lambda) over chunk v2 slots (spec block 6).

Generalizes SP1's ``compute_vtrace`` (IMPALA, Espeholt et al. 2018, with the lambda variant
of Remark 2) to chunks of ACT / BOOT / PAD slots:

- values ``V`` come from the learner's network on every slot (``PolicyModel.unroll``);
- non-ACT slots have rho = c = 0, reward 0 and ``vs = V``, so a trace stops at a BOOT;
- the next value of a non-terminal ACT at slot t is ``V`` of slot t+1 (an ACT of the same
  episode or its BOOT); a terminal ACT bootstraps 0; a PAD follows only episode ends;
- everything is selected with ``torch.where`` (never by multiplying with a mask), so NaN in
  the values of slots that are not used (e.g. a PAD) never reaches the targets.

Math, on ACT slots (``cont_t`` = ACT and not terminal):
    rho_t = min(rho_bar, exp(log_rho_t)),  c_t = lam * min(c_bar, exp(log_rho_t))
    delta_t = rho_t * (r_t + gamma * cont_t * V_{t+1} - V_t)
    vs_t - V_t = delta_t + gamma * cont_t * c_t * (vs_{t+1} - V_{t+1})
    td_t = r_t + gamma * cont_t * vs_{t+1} - V_t           (the policy-gradient advantage)
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class VTraceOut(NamedTuple):
    vs: Tensor           # [S, B] value targets (== values on non-ACT slots)
    td: Tensor           # [S, B] r_t + gamma * (1 - terminal_t) * vs_{t+1} - V_t on ACT slots; 0 elsewhere
    clipped_rho: Tensor  # [S, B] min(rho_bar, rho_t) on ACT slots; 0 elsewhere


def compute_vtrace_slots(
    *,
    log_rhos: Tensor,
    rewards: Tensor,
    values: Tensor,
    is_act: Tensor,
    terminal: Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    lam: float = 1.0,
) -> VTraceOut:
    """V-trace(lambda) targets and TD advantages over ``[S, B]`` slots (module docstring).

    ``log_rhos`` is the scalar log importance ratio of each slot (from
    ``algorithm.unit_trace``; ignored where ``is_act`` is False). ``is_act`` and ``terminal``
    are bool. An ACT never sits in the last slot.
    """
    S = values.shape[0]
    zeros = torch.zeros_like(values)
    is_act = is_act.bool()
    cont = is_act & ~terminal.bool()
    log_rhos = torch.where(is_act, torch.clamp(log_rhos.to(values.dtype), -20.0, 20.0), zeros)
    rhos = torch.exp(log_rhos)
    clipped_rho = torch.where(is_act, torch.clamp(rhos, max=rho_bar), zeros)
    cs = torch.where(is_act, lam * torch.clamp(rhos, max=c_bar), zeros)
    rewards = torch.where(is_act, rewards.to(values.dtype), zeros)
    v_t = torch.where(is_act, values, zeros)
    next_values = torch.cat([values[1:], zeros[:1]], dim=0)
    v_next = torch.where(cont, next_values, zeros)
    deltas = torch.where(is_act, clipped_rho * (rewards + gamma * v_next - v_t), zeros)

    vs_minus_v = torch.zeros_like(values)
    acc = zeros[0]
    for t in reversed(range(S)):
        acc = torch.where(is_act[t], deltas[t] + gamma * cs[t] * torch.where(cont[t], acc, zeros[t]), zeros[t])
        vs_minus_v[t] = acc
    vs = torch.where(is_act, values + vs_minus_v, values)

    vs_next = torch.where(cont, torch.cat([vs[1:], zeros[:1]], dim=0), zeros)
    td = torch.where(is_act, rewards + gamma * vs_next - v_t, zeros)
    return VTraceOut(vs=vs, td=td, clipped_rho=clipped_rho)
