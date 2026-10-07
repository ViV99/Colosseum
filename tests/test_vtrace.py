"""Unit tests for V-trace computation."""
import pytest
import torch

from colosseum.algorithms.vtrace import compute_vtrace


def test_on_policy_reduces_to_nstep():
    """When behavior == target (on-policy), V-trace should reduce to n-step returns."""
    T, B = 4, 2
    gamma = 0.99

    behavior_lp = torch.zeros(T, B)
    target_lp = torch.zeros(T, B)
    rewards = torch.ones(T, B)
    values = torch.zeros(T, B)
    bootstrap = torch.zeros(B)
    dones = torch.zeros(T, B)

    vs, adv = compute_vtrace(behavior_lp, target_lp, rewards, values, bootstrap, dones, gamma=gamma)

    # On-policy: v_s should be discounted sum of rewards
    # v_3 = r_3 + gamma * bootstrap = 1.0
    # v_2 = r_2 + gamma * v_3 = 1 + 0.99 = 1.99
    # v_1 = r_1 + gamma * v_2 = 1 + 0.99*1.99 = 2.9701
    # v_0 = r_0 + gamma * v_1 = 1 + 0.99*2.9701 = 3.940399
    expected_v3 = 1.0
    expected_v2 = 1 + gamma * expected_v3
    expected_v1 = 1 + gamma * expected_v2
    expected_v0 = 1 + gamma * expected_v1

    assert torch.allclose(vs[3, 0], torch.tensor(expected_v3), atol=1e-4)
    assert torch.allclose(vs[0, 0], torch.tensor(expected_v0), atol=1e-4)


def test_episode_boundary_resets():
    """V-trace should reset at episode boundaries (dones=1)."""
    T, B = 4, 1
    gamma = 0.99

    behavior_lp = torch.zeros(T, B)
    target_lp = torch.zeros(T, B)
    rewards = torch.ones(T, B)
    values = torch.zeros(T, B)
    bootstrap = torch.zeros(B)
    dones = torch.zeros(T, B)
    dones[1, 0] = 1.0  # Episode ends at t=1

    vs, adv = compute_vtrace(behavior_lp, target_lp, rewards, values, bootstrap, dones, gamma=gamma)

    # After done at t=1, t=2 starts a new episode
    # v_3 = r_3 + gamma * bootstrap = 1.0
    # v_2 = r_2 + gamma * v_3 = 1 + 0.99 = 1.99
    # v_1 = r_1 + 0 (done, so gamma * next is zeroed) = 1.0
    # v_0 = r_0 + gamma * v_1 = 1 + 0.99 * 1.0 = 1.99
    assert torch.allclose(vs[1, 0], torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(vs[0, 0], torch.tensor(1.99), atol=1e-4)
    assert torch.allclose(vs[3, 0], torch.tensor(1.0), atol=1e-4)


def test_importance_weight_clipping():
    """Large importance ratios should be clipped."""
    T, B = 2, 1
    gamma = 0.99
    rho_bar = 1.0

    # Large ratio: target much more likely than behavior
    behavior_lp = torch.tensor([[-5.0], [-5.0]])
    target_lp = torch.tensor([[0.0], [0.0]])
    rewards = torch.ones(T, B)
    values = torch.zeros(T, B)
    bootstrap = torch.zeros(B)
    dones = torch.zeros(T, B)

    vs, adv = compute_vtrace(
        behavior_lp, target_lp, rewards, values, bootstrap, dones,
        gamma=gamma, rho_bar=rho_bar, c_bar=rho_bar,
    )

    # With rho_bar=1.0 and huge raw ratio, the clipped ratio is 1.0
    # So result should be same as on-policy
    vs_onpol, _ = compute_vtrace(
        torch.zeros(T, B), torch.zeros(T, B), rewards, values, bootstrap, dones,
        gamma=gamma, rho_bar=rho_bar, c_bar=rho_bar,
    )
    assert torch.allclose(vs, vs_onpol, atol=1e-4)


def test_output_shapes():
    """Check output shapes are correct."""
    T, B = 16, 8
    behavior_lp = torch.randn(T, B)
    target_lp = torch.randn(T, B)
    rewards = torch.randn(T, B)
    values = torch.randn(T, B)
    bootstrap = torch.randn(B)
    dones = torch.zeros(T, B)

    vs, adv = compute_vtrace(behavior_lp, target_lp, rewards, values, bootstrap, dones)

    assert vs.shape == (T, B)
    assert adv.shape == (T, B)


# ---------------------------------------------------------------
# T4.7: Edge cases for episode boundaries
# ---------------------------------------------------------------

def test_vtrace_all_done():
    """Every step is an episode boundary — no discounting across steps."""
    T, B = 4, 2
    behavior_lp = torch.zeros(T, B)
    target_lp = torch.zeros(T, B)
    rewards = torch.ones(T, B) * 2.0
    values = torch.zeros(T, B)
    bootstrap = torch.zeros(B)
    dones = torch.ones(T, B)  # all done

    vs, adv = compute_vtrace(behavior_lp, target_lp, rewards, values, bootstrap, dones, gamma=0.99)

    # With done=1 at every step, no future is discounted: v_t = rho * (r_t + 0 - V(t)) = r_t
    # (on-policy, rho=1, values=0, so vs = rewards)
    assert torch.allclose(vs, rewards, atol=1e-4)


def test_vtrace_first_step_done():
    """Done at t=0 — first step is terminal."""
    T, B = 4, 1
    behavior_lp = torch.zeros(T, B)
    target_lp = torch.zeros(T, B)
    rewards = torch.ones(T, B)
    values = torch.zeros(T, B)
    bootstrap = torch.zeros(B)
    dones = torch.zeros(T, B)
    dones[0, 0] = 1.0

    vs, adv = compute_vtrace(behavior_lp, target_lp, rewards, values, bootstrap, dones, gamma=0.99)

    # v_0 = r_0 (done, no future) = 1.0
    assert torch.allclose(vs[0, 0], torch.tensor(1.0), atol=1e-4)
    # v_1..v_3 should be normal discounted returns
    assert vs[1, 0].item() > 1.0  # has future rewards


def test_vtrace_last_step_done():
    """Done at t=T-1 — last step is terminal."""
    T, B = 4, 1
    behavior_lp = torch.zeros(T, B)
    target_lp = torch.zeros(T, B)
    rewards = torch.ones(T, B)
    values = torch.zeros(T, B)
    bootstrap = torch.zeros(B)
    dones = torch.zeros(T, B)
    dones[T - 1, 0] = 1.0

    vs, adv = compute_vtrace(behavior_lp, target_lp, rewards, values, bootstrap, dones, gamma=0.99)

    # v_{T-1} = r_{T-1} (done) = 1.0
    assert torch.allclose(vs[T - 1, 0], torch.tensor(1.0), atol=1e-4)
    # All steps should have finite values
    assert torch.isfinite(vs).all()
    assert torch.isfinite(adv).all()


def test_vtrace_multiple_episodes():
    """Multiple short episodes in one chunk (done at t=1 and t=3)."""
    T, B = 6, 1
    behavior_lp = torch.zeros(T, B)
    target_lp = torch.zeros(T, B)
    rewards = torch.ones(T, B)
    values = torch.zeros(T, B)
    bootstrap = torch.zeros(B)
    dones = torch.zeros(T, B)
    dones[1, 0] = 1.0  # episode 1: t=0,1
    dones[3, 0] = 1.0  # episode 2: t=2,3

    vs, adv = compute_vtrace(behavior_lp, target_lp, rewards, values, bootstrap, dones, gamma=0.99)

    # Episode 1: v_1 = 1.0 (done), v_0 = 1 + 0.99*1.0 = 1.99
    assert torch.allclose(vs[1, 0], torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(vs[0, 0], torch.tensor(1.99), atol=1e-4)
    # Episode 2: v_3 = 1.0 (done), v_2 = 1 + 0.99*1.0 = 1.99
    assert torch.allclose(vs[3, 0], torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(vs[2, 0], torch.tensor(1.99), atol=1e-4)
