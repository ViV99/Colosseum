"""Open transitions: sealing on the slot's next action, no bootstrap forward (T3.1).

Real RolloutLoop in-process; ProbeModel's value is a known function of the
observation, and GridStepEnv logs everything it saw, so every chunk can be
compared with a straightforward reference built from the env log.
"""
import math

import numpy as np
import pytest

from dataflow_helpers import EnvFactory, GridStepEnv, ProbeModel, make_loop, reference_transitions


def test_no_bootstrap_forward():
    created = []

    def model_factory():
        model = ProbeModel(obs_coeffs=(0.0, 0.0, 0.5, 0.25))
        created.append(model)
        return model

    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(3, 5)), model_factory,
                          num_envs=2, chunk_length=4)
    for _ in range(30):
        loop.step()
    assert len(col.chunks) >= 8
    # exactly one batched forward per env step (2 envs x 2 seats), nothing else
    assert created[0].calls == [4] * 30


def test_full_buffer_is_sealed_when_the_slot_acts_again():
    coeffs = (0.0, 0.0, 0.5, 0.25)
    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(10,)), lambda: ProbeModel(obs_coeffs=coeffs),
                          num_envs=1, chunk_length=4)
    for _ in range(4):
        loop.step()
    assert col.chunks == []                  # 4 transitions recorded, the 4th still open
    loop.step()
    assert len(col.chunks) == 2              # sealed by each seat's 5th action
    for chunk in col.chunks:
        p = int(chunk.observations[0, 3])
        assert float(chunk.bootstrap_value) == pytest.approx(0.5 * 4 + 0.25 * p)   # V(obs at t=4)


def test_full_buffer_ending_an_episode_is_sealed_at_once_with_zero_bootstrap():
    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(4,)), ProbeModel, num_envs=1, chunk_length=4)
    for _ in range(4):
        loop.step()
    assert len(col.chunks) == 2
    assert all(float(c.bootstrap_value) == 0.0 and bool(c.dones[-1]) for c in col.chunks)


def test_simultaneous_chunks_match_the_reference_construction():
    coeffs = (0.0, 0.1, 0.5, 0.25)          # value = 0.1*ep + 0.5*t + 0.25*player
    envs = EnvFactory(GridStepEnv, lengths=(3, 6, 2))
    loop, col = make_loop(envs, lambda: ProbeModel(obs_coeffs=coeffs), num_envs=2, chunk_length=4)
    for _ in range(40):
        loop.step()

    def value(tr, p):
        return coeffs[1] * tr["ep"] + coeffs[2] * tr["t"] + coeffs[3] * p

    by_slot = {}
    for chunk in col.chunks:
        by_slot.setdefault((int(chunk.observations[0, 0]), int(chunk.observations[0, 3])), []).append(chunk)
    assert set(by_slot) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for (env_id, p), chunks in by_slot.items():
        ref = reference_transitions(envs.created[env_id].log, 2)[p]
        for k, chunk in enumerate(chunks):
            rows = ref[4 * k: 4 * k + 4]
            assert [r["ep"] for r in rows] == chunk.observations[:, 1].int().tolist()
            assert [r["t"] for r in rows] == chunk.observations[:, 2].int().tolist()
            assert [r["action"] for r in rows] == chunk.actions.tolist()
            assert np.allclose([r["reward"] for r in rows], chunk.rewards.numpy())
            assert [r["done"] for r in rows] == chunk.dones.tolist()
            assert np.allclose([value(r, p) for r in rows], chunk.values.numpy(), atol=1e-5)
            assert np.allclose(chunk.action_log_probs.numpy(), -math.log(3), atol=1e-5)
            if rows[-1]["done"]:
                assert float(chunk.bootstrap_value) == 0.0
            else:
                assert float(chunk.bootstrap_value) == pytest.approx(value(ref[4 * k + 4], p), abs=1e-5)
            assert chunk.initial_state is None
