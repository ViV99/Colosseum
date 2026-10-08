"""VectorEnv v2: no auto-reset, per-env reset requests, spec equality (SP2 T1.6)."""
import functools

import numpy as np
import pytest

from colosseum.core.errors import EnvContractError
from colosseum.sp2.envs.vector import VectorEnv
from game_helpers import EliminationFFA, SoloCounterGame


class _ClosingGame(SoloCounterGame):
    closed: list[int] = []

    def close(self) -> None:
        _ClosingGame.closed.append(id(self))


class _CountingGame(SoloCounterGame):
    calls: list[str] = []

    def reset(self, seed, layout):
        _CountingGame.calls.append("reset")
        return super().reset(seed, layout)

    def step(self, actions):
        _CountingGame.calls.append("step")
        return super().step(actions)


class _OtherLengthSpec(SoloCounterGame):
    """Every second instance has a different observation space."""

    count = 0

    def __init__(self) -> None:
        super().__init__()
        _OtherLengthSpec.count += 1
        if _OtherLengthSpec.count % 2 == 0:
            self.spec = EliminationFFA().spec


def test_reset_and_step_only_the_requested_envs():
    vec = VectorEnv(functools.partial(EliminationFFA, max_players=3), num_envs=3)
    assert vec.num_envs == 3 and list(vec.spec.layouts) == ["2p", "3p"]
    first = vec.reset({0: (1, "2p"), 2: (2, "3p")})
    assert sorted(first) == [0, 2]
    assert first[0].acting == {0, 1} and first[2].acting == {0, 1, 2}
    stepped = vec.step({2: {0: 0, 1: 1, 2: 0}})
    assert list(stepped) == [2] and stepped[2].terminated == {2}
    vec.close()


def test_no_auto_reset_a_finished_env_returns_its_final_result():
    vec = VectorEnv(functools.partial(SoloCounterGame, length=2), num_envs=2)
    vec.reset({0: (None, "solo"), 1: (None, "solo")})
    vec.step({0: {0: 1}, 1: {0: 1}})
    final = vec.step({0: {0: 1}, 1: {0: 0}})
    assert final[0].episode_over and final[1].episode_over and not final[0].acting
    again = vec.reset({1: (None, "solo")})   # only env 1 starts a new episode
    assert list(again) == [1] and again[1].acting == {0}
    vec.close()


def test_unknown_layout_and_bad_index():
    vec = VectorEnv(SoloCounterGame, num_envs=1)
    with pytest.raises(ValueError, match="env 0: unknown layout '2p'"):
        vec.reset({0: (None, "2p")})
    with pytest.raises(IndexError):
        vec.step({3: {0: 1}})
    vec.close()


def test_every_env_must_report_the_same_spec():
    _OtherLengthSpec.count = 0
    with pytest.raises(EnvContractError, match="env 1 reports a different GameSpec than env 0"):
        VectorEnv(_OtherLengthSpec, num_envs=2)


def test_env_fn_must_build_multi_agent_envs():
    with pytest.raises(EnvContractError, match="must return a MultiAgentEnv"):
        VectorEnv(lambda: object(), num_envs=1)


def test_close_closes_every_env():
    _ClosingGame.closed = []
    vec = VectorEnv(_ClosingGame, num_envs=3)
    vec.close()
    assert len(_ClosingGame.closed) == 3
    vec.close()                                # idempotent
    assert len(_ClosingGame.closed) == 3


def test_results_are_numpy():
    vec = VectorEnv(SoloCounterGame, num_envs=1)
    result = vec.reset({0: (None, "solo")})[0]
    assert isinstance(result.obs[0], np.ndarray)
    vec.close()


def test_bad_requests_are_rejected_before_any_env_is_touched():
    _CountingGame.calls = []
    vec = VectorEnv(_CountingGame, num_envs=2)
    with pytest.raises(ValueError, match="env 1: unknown layout"):
        vec.reset({0: (None, "solo"), 1: (None, "2p")})
    with pytest.raises(IndexError):
        vec.reset({0: (None, "solo"), 5: (None, "solo")})
    assert _CountingGame.calls == []
    vec.reset({0: (None, "solo")})
    with pytest.raises(IndexError):
        vec.step({0: {0: 1}, 5: {0: 1}})
    assert _CountingGame.calls == ["reset"]
    vec.close()
