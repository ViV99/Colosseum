"""Tests for SubprocessVectorEnv: parity with in-process VectorEnv, clean close.

Runnable both as ``python -m pytest tests/test_subproc_vec_env.py`` and
standalone ``python tests/test_subproc_vec_env.py`` (the latter sets the spawn
start method under ``__main__``).
"""

import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import numpy as np

from colosseum.envs.vec_env import VectorEnv
from colosseum.envs.subproc_vec_env import SubprocessVectorEnv


# Top-level (picklable) env factory — required for the spawn start method.
def make_ttt_env():
    from examples.tic_tac_toe.env import TicTacToeEnv
    return TicTacToeEnv()


def _fixed_actions(num_envs: int, num_players: int, num_actions: int, seed: int):
    """Deterministic action arrays so both vec envs are driven identically."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_actions, size=(num_envs, num_players)).astype(np.int64)


def test_subproc_reset_all_parity():
    """reset_all with the same seed yields identical obs shape/dtype/values."""
    num_envs = 6
    ref = VectorEnv(make_ttt_env, num_envs=num_envs)
    sub = SubprocessVectorEnv(make_ttt_env, num_envs=num_envs, num_workers=3)
    try:
        assert sub.num_envs == ref.num_envs
        assert sub.num_players == ref.num_players
        assert sub.action_spec.space_type == ref.action_spec.space_type
        assert sub.observation_space == ref.observation_space

        ref_obs, ref_infos = ref.reset_all(seed=123)
        sub_obs, sub_infos = sub.reset_all(seed=123)

        assert sub_obs.shape == ref_obs.shape
        assert sub_obs.dtype == ref_obs.dtype
        assert len(sub_infos) == len(ref_infos) == num_envs
        # TicTacToe reset is deterministic given seed -> values must match too.
        assert np.array_equal(sub_obs, ref_obs)
    finally:
        ref.close()
        sub.close()


def test_subproc_step_parity():
    """Stepping K times with identical fixed actions matches VectorEnv."""
    num_envs = 8
    num_steps = 30
    ref = VectorEnv(make_ttt_env, num_envs=num_envs)
    sub = SubprocessVectorEnv(make_ttt_env, num_envs=num_envs, num_workers=4)
    try:
        ref_obs, _ = ref.reset_all(seed=7)
        sub_obs, _ = sub.reset_all(seed=7)
        assert np.array_equal(ref_obs, sub_obs)

        num_actions = 9  # TicTacToe Discrete(9)
        saw_done = False
        for k in range(num_steps):
            actions = _fixed_actions(num_envs, 2, num_actions, seed=1000 + k)

            r_obs, r_rew, r_term, r_trunc, r_infos = ref.step(actions)
            s_obs, s_rew, s_term, s_trunc, s_infos = sub.step(actions)

            # Shapes / structure parity.
            assert s_obs.shape == r_obs.shape == (num_envs, 2, 3, 3, 3)
            assert s_obs.dtype == r_obs.dtype
            assert s_rew.shape == r_rew.shape == (num_envs, 2)
            assert s_term.shape == r_term.shape == (num_envs,)
            assert s_trunc.shape == r_trunc.shape == (num_envs,)
            assert len(s_infos) == len(r_infos) == num_envs

            # Value parity (TicTacToe is deterministic given identical actions).
            assert np.array_equal(s_obs, r_obs)
            assert np.array_equal(s_rew, r_rew)
            assert np.array_equal(s_term, r_term)
            assert np.array_equal(s_trunc, r_trunc)

            # Auto-reset / terminal-info structure parity for any done env.
            for env_idx in range(num_envs):
                if r_term[env_idx] or r_trunc[env_idx]:
                    saw_done = True
                    for p in range(2):
                        assert "terminal_observation" in s_infos[env_idx][p]
                        assert "terminal_info" in s_infos[env_idx][p]
                        # No circular reference in terminal_info.
                        assert "terminal_info" not in s_infos[env_idx][p]["terminal_info"]
                    # Fresh obs after auto-reset should be the empty board obs.
                    assert np.array_equal(s_obs[env_idx], r_obs[env_idx])

        assert saw_done, "Expected at least one env to finish within the rollout"
    finally:
        ref.close()
        sub.close()


def test_subproc_auto_reset_gives_fresh_obs():
    """An env that finishes is auto-reset to a fresh (empty-board) observation."""
    sub = SubprocessVectorEnv(make_ttt_env, num_envs=2, num_workers=2)
    try:
        sub.reset_all(seed=0)
        # One env's fresh obs is [num_players, 3, 3, 3]: empty board for both
        # players -> channel 2 (empty squares) all ones, channels 0/1 zero.
        fresh = np.zeros((2, 3, 3, 3), dtype=np.float32)
        fresh[:, 2] = 1.0

        done_found = False
        for k in range(100):
            actions = _fixed_actions(2, 2, 9, seed=500 + k)
            obs, rew, term, trunc, infos = sub.step(actions)
            for env_idx in range(2):
                if term[env_idx] or trunc[env_idx]:
                    done_found = True
                    # obs[env_idx] is [num_players, 3, 3, 3]; after auto-reset
                    # it must be a fresh empty board AND differ from the stored
                    # terminal_observation (proving a NEW episode started).
                    assert np.array_equal(obs[env_idx], fresh)
                    term_obs = infos[env_idx][0]["terminal_observation"]
                    assert not np.array_equal(obs[env_idx][0], term_obs)
            if done_found:
                break
        assert done_found, "No env completed within 100 steps"
    finally:
        sub.close()


def test_subproc_reset_done_parity():
    """reset_done resets only flagged envs (zeros elsewhere), matching VectorEnv."""
    num_envs = 4
    ref = VectorEnv(make_ttt_env, num_envs=num_envs)
    sub = SubprocessVectorEnv(make_ttt_env, num_envs=num_envs, num_workers=2)
    try:
        ref.reset_all(seed=11)
        sub.reset_all(seed=11)

        terminated = np.array([True, False, False, True])
        truncated = np.array([False, False, True, False])

        r_obs, r_infos = ref.reset_done(terminated, truncated)
        s_obs, s_infos = sub.reset_done(terminated, truncated)

        assert s_obs.shape == r_obs.shape == (num_envs, 2, 3, 3, 3)
        assert len(s_infos) == len(r_infos) == num_envs

        # Not-done envs (index 1) must be all zeros in both, with empty info.
        assert np.array_equal(s_obs[1], np.zeros_like(s_obs[1]))
        assert s_infos[1] == {} == r_infos[1]

        # Done envs get a fresh empty board (deterministic) -> obs must match ref.
        for env_idx in (0, 2, 3):
            assert np.array_equal(s_obs[env_idx], r_obs[env_idx])
            assert s_infos[env_idx] != {}
    finally:
        ref.close()
        sub.close()


def test_subproc_default_num_workers():
    """Default num_workers = min(num_envs, cpu_count); never exceeds num_envs."""
    sub = SubprocessVectorEnv(make_ttt_env, num_envs=3)
    try:
        expected = min(3, os.cpu_count() or 1)
        assert sub.num_workers == expected
        assert sub.num_workers <= sub.num_envs
        obs, _ = sub.reset_all(seed=1)
        assert obs.shape[0] == 3
    finally:
        sub.close()


def test_subproc_uneven_split():
    """Workers that don't divide num_envs evenly still cover all envs in order."""
    num_envs = 7
    ref = VectorEnv(make_ttt_env, num_envs=num_envs)
    sub = SubprocessVectorEnv(make_ttt_env, num_envs=num_envs, num_workers=3)
    try:
        assert sub.num_envs == num_envs
        r_obs, _ = ref.reset_all(seed=99)
        s_obs, _ = sub.reset_all(seed=99)
        assert s_obs.shape == r_obs.shape
        assert np.array_equal(s_obs, r_obs)
    finally:
        ref.close()
        sub.close()


def test_subproc_close_is_prompt_and_no_zombies():
    """close() returns promptly and leaves no live subprocesses behind."""
    sub = SubprocessVectorEnv(make_ttt_env, num_envs=6, num_workers=3)
    sub.reset_all(seed=3)
    sub.step(_fixed_actions(6, 2, 9, seed=0))

    procs = list(sub._procs)
    assert all(p.is_alive() for p in procs)

    start = time.time()
    sub.close()
    elapsed = time.time() - start

    assert elapsed < 10.0, f"close() took too long: {elapsed:.2f}s"
    # Give the OS a brief moment to reap, then confirm no zombies.
    for p in procs:
        p.join(timeout=2.0)
        assert not p.is_alive(), "Subprocess still alive after close()"

    # Double close must be a no-op (no raise, no hang).
    sub.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    print("=" * 60)
    print("SubprocessVectorEnv tests")
    print("=" * 60)

    test_subproc_reset_all_parity()
    print("  reset_all parity PASSED")

    test_subproc_step_parity()
    print("  step parity PASSED")

    test_subproc_auto_reset_gives_fresh_obs()
    print("  auto-reset fresh obs PASSED")

    test_subproc_reset_done_parity()
    print("  reset_done parity PASSED")

    test_subproc_default_num_workers()
    print("  default num_workers PASSED")

    test_subproc_uneven_split()
    print("  uneven split PASSED")

    test_subproc_close_is_prompt_and_no_zombies()
    print("  prompt close / no zombies PASSED")

    print()
    print("All SubprocessVectorEnv tests PASSED")
