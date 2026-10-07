"""Tests for VectorEnv: shapes, auto-reset, terminal info, seeding."""

import os
import sys


import numpy as np

from colosseum.envs.vec_env import VectorEnv


def _make_ttt_env():
    from examples.tic_tac_toe.env import TicTacToeEnv
    return TicTacToeEnv()


def test_vec_env_shapes():
    """Verify observation/reward/done shapes after reset and step."""
    vec_env = VectorEnv(_make_ttt_env, num_envs=4)
    obs, infos = vec_env.reset_all()

    assert obs.shape[0] == 4  # num_envs
    assert obs.shape[1] == 2  # num_players
    assert len(obs.shape) >= 3  # at least [K, N, *obs_shape]

    # Random actions
    actions = np.zeros((4, 2), dtype=np.int64)
    for k in range(4):
        for p in range(2):
            actions[k, p] = np.random.randint(0, 9)

    next_obs, rewards, terminated, truncated, step_infos = vec_env.step(actions)

    assert next_obs.shape == obs.shape
    assert rewards.shape == (4, 2)
    assert terminated.shape == (4,)
    assert truncated.shape == (4,)
    assert len(step_infos) == 4

    vec_env.close()


def test_vec_env_auto_reset():
    """When env is done, auto-resets and stores terminal_observation in info."""
    vec_env = VectorEnv(_make_ttt_env, num_envs=2)
    obs, _ = vec_env.reset_all()

    # Play until at least one env is done
    done_found = False
    for _ in range(100):
        actions = np.random.randint(0, 9, size=(2, 2))
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)

        for k in range(2):
            if terminated[k] or truncated[k]:
                done_found = True
                # Auto-reset should store terminal observation
                assert "terminal_observation" in infos[k][0]
                # Observation returned should be from the NEW episode (after reset)
                # terminal_observation should be the FINAL observation from the ended episode
                break
        if done_found:
            break

    assert done_found, "No env completed within 100 steps"
    vec_env.close()


def test_vec_env_terminal_info_no_circular_ref():
    """terminal_info should not contain a circular reference (T0.1 fix)."""
    vec_env = VectorEnv(_make_ttt_env, num_envs=1)
    obs, _ = vec_env.reset_all()

    for _ in range(200):
        actions = np.random.randint(0, 9, size=(1, 2))
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)

        if terminated[0] or truncated[0]:
            info = infos[0][0]
            assert "terminal_info" in info
            terminal_info = info["terminal_info"]
            # terminal_info should NOT contain itself
            assert "terminal_info" not in terminal_info
            assert "terminal_observation" not in terminal_info
            break

    vec_env.close()


def test_vec_env_reset_all_with_seed():
    """Seeded reset produces deterministic observations."""
    vec_env1 = VectorEnv(_make_ttt_env, num_envs=3)
    vec_env2 = VectorEnv(_make_ttt_env, num_envs=3)

    obs1, _ = vec_env1.reset_all(seed=42)
    obs2, _ = vec_env2.reset_all(seed=42)

    assert np.array_equal(obs1, obs2)

    vec_env1.close()
    vec_env2.close()


def test_vec_env_close():
    """All envs should be closed properly."""
    vec_env = VectorEnv(_make_ttt_env, num_envs=3)
    vec_env.reset_all()
    vec_env.close()
    # Should not raise on double close
    vec_env.close()
