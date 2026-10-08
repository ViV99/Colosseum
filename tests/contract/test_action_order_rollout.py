"""12-unit MultiDiscrete through the real RolloutLoop: unit i gets head i's action (T4.5)."""
import math

import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from helpers import TWELVE_NVEC, TwelveUnitEnv, rollout_chunks, twelve_unit_model


@pytest.mark.parametrize("use_mask", [False, True])
def test_twelve_units_receive_their_own_head_through_rollout_loop(use_mask):
    envs = []

    def env_fn():
        env = TwelveUnitEnv(use_mask=use_mask)
        envs.append(env)
        return env

    # Unmasked: head i is peaked on action i. Masked: uniform heads, the mask leaves only action i.
    model = twelve_unit_model(peaked=not use_mask)
    chunks = rollout_chunks(model, env_fn, num_chunks=2, chunk_length=4, num_envs=1)

    received = [a for env in envs for a in env.received]
    assert received
    for action in received:
        np.testing.assert_array_equal(action, np.arange(12))
    for chunk in chunks:
        expected = np.tile(np.arange(12, dtype=np.float32), (chunk.chunk_length, 1))
        np.testing.assert_array_equal(np.asarray(chunk.actions), expected)
        assert torch.isfinite(torch.as_tensor(chunk.action_log_probs)).all()
        if use_mask:
            # Natural-order one-hot rows: unit i's segment allows only action i.
            one_hot = np.concatenate([np.arange(n) == i for i, n in enumerate(TWELVE_NVEC)])
            masks = np.asarray(chunk.action_masks)
            assert masks.shape == (chunk.chunk_length, one_hot.size)
            np.testing.assert_array_equal(masks, np.tile(one_hot, (chunk.chunk_length, 1)))

    metrics = APPO(model, AlgorithmConfig(), device="cpu").train_step(chunks)
    assert all(math.isfinite(v) for v in metrics.values())
