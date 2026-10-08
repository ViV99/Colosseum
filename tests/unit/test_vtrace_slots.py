"""V-trace(lambda) over act/boot/pad slots against a slot-by-slot reference (T4.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from colosseum.sp2.algorithms.vtrace import VTraceOut, compute_vtrace_slots
from colosseum.sp2.core.types import SLOT_ACT, SLOT_BOOT, SLOT_PAD

LETTER_KIND = {"A": SLOT_ACT, "T": SLOT_ACT, "B": SLOT_BOOT, "R": SLOT_BOOT, "P": SLOT_PAD}


def random_pattern(S: int, rng: np.random.Generator) -> str:
    """A valid slot pattern: A open ACT, T terminal ACT, B boot (chunk end), R boot with reset, P pad."""
    out: list[str] = []
    while len(out) < S:
        if S - len(out) == 1:
            out.append("P")                       # only reached at an episode boundary
            break
        n, end = int(rng.integers(1, 5)), str(rng.choice(["T", "R"]))
        for _ in range(n):
            if S - len(out) == 1:
                out.append("B")                   # the open ACT continues in the next chunk
                break
            out.append("A")
        else:
            if end == "T":
                out[-1] = "T"
            else:
                out.append("R")
    return "".join(out)


def tensors(patterns: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    kind = torch.tensor([[LETTER_KIND[c] for c in p] for p in patterns], dtype=torch.int8).T
    terminal = torch.tensor([[c == "T" for c in p] for p in patterns]).T
    return kind, terminal


def reference(log_rho, r, v, pattern, gamma, rho_bar, c_bar, lam):
    """Plain-python V-trace over one column of slots."""
    S = len(pattern)
    vs, td, crho = list(v), [0.0] * S, [0.0] * S
    for t in reversed(range(S)):
        if pattern[t] not in "AT":
            continue
        rho = math.exp(min(20.0, max(-20.0, log_rho[t])))
        crho[t] = min(rho_bar, rho)
        c = lam * min(c_bar, rho)
        if pattern[t] == "T":
            v_next, acc_next = 0.0, 0.0
        else:
            v_next = v[t + 1]
            acc_next = vs[t + 1] - v[t + 1] if pattern[t + 1] in "AT" else 0.0
        vs[t] = v[t] + crho[t] * (r[t] + gamma * v_next - v[t]) + gamma * c * acc_next
    for t in range(S):
        if pattern[t] in "AT":
            nxt = 0.0 if pattern[t] == "T" else vs[t + 1]
            td[t] = r[t] + gamma * nxt - v[t]
    return vs, td, crho


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("lam,rho_bar,c_bar", [(1.0, 1.0, 1.0), (0.9, 1.0, 1.0), (0.95, 0.8, 1.2), (0.5, 2.0, 0.7)])
def test_matches_the_slot_reference(seed, lam, rho_bar, c_bar):
    rng = np.random.default_rng(seed)
    S, B, gamma = 12, 5, 0.97
    patterns = [random_pattern(S, rng) for _ in range(B)]
    kind, terminal = tensors(patterns)
    log_rhos = torch.tensor(rng.normal(0, 0.7, (S, B)), dtype=torch.float32)
    rewards = torch.tensor(rng.normal(0, 1, (S, B)), dtype=torch.float32)
    values = torch.tensor(rng.normal(0, 1, (S, B)), dtype=torch.float32)
    out = compute_vtrace_slots(log_rhos=log_rhos, rewards=rewards, values=values, is_act=kind == SLOT_ACT,
                               terminal=terminal, gamma=gamma, rho_bar=rho_bar, c_bar=c_bar, lam=lam)
    assert isinstance(out, VTraceOut)
    for b, pattern in enumerate(patterns):
        vs, td, crho = reference(log_rhos[:, b].tolist(), rewards[:, b].tolist(), values[:, b].tolist(),
                                 pattern, gamma, rho_bar, c_bar, lam)
        assert out.vs[:, b].tolist() == pytest.approx(vs, abs=1e-5), pattern
        assert out.td[:, b].tolist() == pytest.approx(td, abs=1e-5), pattern
        assert out.clipped_rho[:, b].tolist() == pytest.approx(crho, abs=1e-6), pattern


def _gae(pattern, r, v, gamma, lam):
    """GAE(lambda) over the ACT slots of one column; a BOOT gives the bootstrap value."""
    adv = [0.0] * len(pattern)
    running = 0.0
    for t in reversed(range(len(pattern))):
        if pattern[t] not in "AT":
            running = 0.0
            continue
        if pattern[t] == "T":
            delta, running = r[t] - v[t], 0.0
        else:
            delta = r[t] + gamma * v[t + 1] - v[t]
        running = delta + gamma * lam * running
        adv[t] = running
    return adv


@pytest.mark.parametrize("lam", [1.0, 0.95, 0.5, 0.0])
def test_zero_lag_equals_gae_over_act_slots(lam):
    rng = np.random.default_rng(7)
    S, B, gamma = 16, 6, 0.9
    patterns = [random_pattern(S, rng) for _ in range(B)]
    kind, terminal = tensors(patterns)
    rewards = torch.tensor(rng.normal(0, 1, (S, B)), dtype=torch.float32)
    values = torch.tensor(rng.normal(0, 1, (S, B)), dtype=torch.float32)
    out = compute_vtrace_slots(log_rhos=torch.zeros(S, B), rewards=rewards, values=values,
                               is_act=kind == SLOT_ACT, terminal=terminal, gamma=gamma, lam=lam)
    for b, pattern in enumerate(patterns):
        gae = _gae(pattern, rewards[:, b].tolist(), values[:, b].tolist(), gamma, lam)
        is_act = [c in "AT" for c in pattern]
        got = (out.vs[:, b] - values[:, b]).tolist()
        assert [g for g, a in zip(got, is_act) if a] == pytest.approx([g for g, a in zip(gae, is_act) if a],
                                                                       abs=1e-5)


def test_boot_cuts_the_trace_and_terminal_bootstraps_zero():
    # column 0: A A B (chunk end); column 1: A T P
    kind, terminal = tensors(["AAB", "ATP"])
    values = torch.tensor([[1.0, 1.0], [2.0, 2.0], [5.0, 100.0]])
    rewards = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    out = compute_vtrace_slots(log_rhos=torch.zeros(3, 2), rewards=rewards, values=values,
                               is_act=kind == SLOT_ACT, terminal=terminal, gamma=0.5)
    # column 0: vs_1 = 1 + 0.5 * 5 = 3.5; vs_0 = 1 + 0.5 * 3.5 = 2.75; the BOOT keeps its value
    assert out.vs[:, 0].tolist() == pytest.approx([2.75, 3.5, 5.0])
    # column 1: the terminal ACT ignores the PAD's value: vs_1 = 1, vs_0 = 1 + 0.5 * 1
    assert out.vs[:2, 1].tolist() == pytest.approx([1.5, 1.0])
    assert out.td[:, 1].tolist() == pytest.approx([1.0 + 0.5 * 1.0 - 1.0, 1.0 - 2.0, 0.0])
    assert out.clipped_rho[:, 0].tolist() == [1.0, 1.0, 0.0]


def test_nan_in_unused_slots_never_reaches_the_targets():
    rng = np.random.default_rng(3)
    S, B = 10, 8
    patterns = [random_pattern(S, rng) for _ in range(B - 2)] + ["AAATAAAATP", "AARAAAAARP"]
    kind, terminal = tensors(patterns)
    rewards = torch.tensor(rng.normal(0, 1, (S, B)), dtype=torch.float32)
    values = torch.tensor(rng.normal(0, 1, (S, B)), dtype=torch.float32)
    pad = kind == SLOT_PAD
    poisoned_values = torch.where(pad, torch.full_like(values, float("nan")), values)
    poisoned_rewards = torch.where(kind != SLOT_ACT, torch.full_like(rewards, float("nan")), rewards)
    poisoned_rhos = torch.where(kind != SLOT_ACT, torch.full_like(values, float("nan")), torch.zeros_like(values))
    kwargs = dict(is_act=kind == SLOT_ACT, terminal=terminal, gamma=0.99, lam=0.9)
    clean = compute_vtrace_slots(log_rhos=torch.zeros(S, B), rewards=rewards, values=values, **kwargs)
    dirty = compute_vtrace_slots(log_rhos=poisoned_rhos, rewards=poisoned_rewards, values=poisoned_values,
                                 **kwargs)
    act = kind == SLOT_ACT
    assert torch.isfinite(dirty.vs[act]).all() and torch.isfinite(dirty.td).all()
    assert torch.equal(dirty.vs[act], clean.vs[act])
    assert torch.equal(dirty.td, clean.td) and torch.equal(dirty.clipped_rho, clean.clipped_rho)


def test_random_patterns_respect_the_slot_invariants():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = random_pattern(int(rng.integers(2, 12)), rng)
        assert p[-1] in "BRP"                               # an ACT never takes the last slot
        for i, c in enumerate(p[1:], start=1):
            if c == "B":
                assert i == len(p) - 1 and p[i - 1] == "A"
            if c == "R":
                assert p[i - 1] == "A"
            if c == "P":
                assert i == len(p) - 1 and p[i - 1] in "TR"
