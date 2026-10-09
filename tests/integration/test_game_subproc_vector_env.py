"""SubprocessVectorEnv v2: parity with VectorEnv, batched requests, forwarded errors, close (SP2 T1.6)."""
import functools
import os
import time

import numpy as np
import pytest

from colosseum.core.errors import EnvContractError
from colosseum.envs.game import StepResult
from colosseum.envs.vector import SubprocessVectorEnv, VectorEnv
from game_helpers import EliminationFFA, ScriptedGame, SoloCounterGame, UnitsGame


def _ffa():
    return EliminationFFA(max_players=3)


def _short_script():
    """Its first step indexes past the script: an IndexError raised inside the child."""
    first = StepResult(acting={0}, obs={0: np.zeros(2, dtype=np.float32)})
    return ScriptedGame(SoloCounterGame().spec, [first])


class _RaisingGame(SoloCounterGame):
    def step(self, actions):
        raise EnvContractError("bad step from the env")


class _DyingGame(SoloCounterGame):
    """The process dies on the first step (like a segfault or an OOM kill in a native env)."""

    def step(self, actions):
        os._exit(7)


def _not_an_env():
    return object()


def _units():
    return UnitsGame(max_units=3)


def _drive(vec, num_steps):
    """Same actions in both vectors; returns the observed (acting, rewards, episode_over) per env and step."""
    trace = []
    vec.reset({e: (e, "3p" if e % 2 else "2p") for e in range(vec.num_envs)})
    acting = {e: ({0, 1, 2} if e % 2 else {0, 1}) for e in range(vec.num_envs)}
    for _ in range(num_steps):
        live = {e: a for e, a in acting.items() if a}
        out = vec.step({e: {p: 0 for p in seats} for e, seats in live.items()})
        trace.append({e: (sorted(r.acting), r.rewards, r.episode_over) for e, r in sorted(out.items())})
        acting = {e: r.acting for e, r in out.items()}
    return trace


def test_parity_with_vector_env():
    ref = VectorEnv(_ffa, num_envs=4)
    sub = SubprocessVectorEnv(_ffa, num_envs=4, num_workers=2)
    try:
        assert sub.spec == ref.spec and sub.num_workers == 2
        assert _drive(sub, 3) == _drive(ref, 3)
    finally:
        ref.close()
        sub.close()


def test_default_num_workers_is_min_of_envs_and_cpus():
    """Default ``num_workers = min(num_envs, cpu_count)``; never more than ``num_envs`` (SP1 port)."""
    sub = SubprocessVectorEnv(SoloCounterGame, num_envs=2)
    try:
        assert sub.num_workers == min(2, os.cpu_count() or 1)
        assert sub.num_workers <= sub.num_envs
        assert sorted(sub.reset({0: (1, "solo"), 1: (2, "solo")})) == [0, 1]
    finally:
        sub.close()


def test_requests_only_reach_the_owning_child_and_keep_dtypes():
    sub = SubprocessVectorEnv(_units, num_envs=3, num_workers=2)   # slices [0, 2) and [2, 3)
    try:
        out = sub.reset({2: (None, "solo")})
        assert list(out) == [2]
        obs = out[2].obs[0]
        assert obs["grid"].dtype == np.uint8 and obs["entity_mask"].dtype == np.int8
        both = sub.reset({0: (None, "solo"), 2: (None, "solo")})   # one round for both children
        assert sorted(both) == [0, 2]
        action = {"base": np.int64(1), "units": {"move": np.zeros(3, np.int64), "target": np.zeros(3, np.int64)}}
        stepped = sub.step({0: {0: action}})
        assert list(stepped) == [0] and stepped[0].rewards == {0: 0.75}
    finally:
        sub.close()


def test_child_errors_are_reraised_with_the_child_traceback():
    sub = SubprocessVectorEnv(_RaisingGame, num_envs=2, num_workers=2)
    try:
        sub.reset({0: (None, "solo"), 1: (None, "solo")})
        with pytest.raises(EnvContractError, match="bad step from the env") as info:
            sub.step({0: {0: 1}, 1: {0: 1}})
        assert any("SubprocessVectorEnv child" in note for note in info.value.__notes__)
        # Both children answered, so the protocol stays in sync: a reset still works.
        assert sub.reset({1: (None, "solo")})[1].acting == {0}
    finally:
        sub.close()


def test_any_env_exception_is_forwarded():
    sub = SubprocessVectorEnv(_short_script, num_envs=1, num_workers=1)
    try:
        sub.reset({0: (None, "solo")})
        with pytest.raises(IndexError):
            sub.step({0: {0: 1}})
    finally:
        sub.close()


def test_unknown_layout_is_rejected_in_the_parent():
    sub = SubprocessVectorEnv(SoloCounterGame, num_envs=1, num_workers=1)
    try:
        with pytest.raises(ValueError, match="unknown layout"):
            sub.reset({0: (None, "4p")})
    finally:
        sub.close()


def test_close_is_bounded_and_idempotent():
    sub = SubprocessVectorEnv(functools.partial(SoloCounterGame, length=3), num_envs=2, num_workers=2)
    procs = list(sub._procs)
    start = time.monotonic()
    sub.close()
    sub.close()
    assert time.monotonic() - start < 10
    assert not any(p.is_alive() for p in procs)


def test_spec_is_checked_in_the_parent_before_spawning():
    with pytest.raises(EnvContractError, match="must be a GameSpec"):
        SubprocessVectorEnv(functools.partial(ScriptedGame, None, []), num_envs=1, num_workers=1)


def test_env_fn_must_build_multi_agent_envs_in_the_parent():
    with pytest.raises(EnvContractError, match="must return a MultiAgentEnv"):
        SubprocessVectorEnv(_not_an_env, num_envs=1, num_workers=1)


def test_a_dead_child_is_reported_with_its_env_range_and_exit_code():
    sub = SubprocessVectorEnv(_DyingGame, num_envs=3, num_workers=2)   # slices [0, 2) and [2, 3)
    try:
        sub.reset({0: (None, "solo"), 2: (None, "solo")})
        with pytest.raises(RuntimeError, match=r"env child 0 \(envs 0\.\.1\) died \(exit code 7\)"):
            sub.step({0: {0: 1}})
        with pytest.raises(RuntimeError, match=r"env child 0 \(envs 0\.\.1\) died \(exit code 7\)"):
            sub.step({1: {0: 1}})                                      # the pipe is gone: the send fails
        assert sub.reset({2: (None, "solo")})[2].acting == {0}         # the other child still answers
    finally:
        sub.close()
