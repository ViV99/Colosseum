"""SP3 warm start on CUDA under AMP (docs/GPU_CHECKS.md): the critic warm-up (the policy path has no
gradients, so GradScaler.unscale_/step must skip it) and the DAgger label loss under autocast. The CPU
cases (no AMP: APPO enables AMP only on CUDA) run in the fast suite and keep the test logic honest."""
from __future__ import annotations

import math

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.bc.kickstart import KickstartLoss
from colosseum.core.config import AlgorithmConfig
from colosseum.core.specs import ActionSpec
from colosseum.core.types import SLOT_ACT, TrajectoryChunk
from colosseum.envs.game import RoleSpec
from game_helpers import make_test_model, synthetic_chunk

ROLE = RoleSpec(gymnasium.spaces.Box(-5.0, 5.0, (4,), np.float32), gymnasium.spaces.Discrete(3),
                gymnasium.spaces.Box(-5.0, 5.0, (2,), np.float32))
SPEC = ActionSpec.from_space(ROLE.action_space)
CASES = [
    pytest.param("cpu", None, id="cpu"),
    pytest.param("cuda", "float16", id="cuda-float16", marks=pytest.mark.gpu),
    pytest.param("cuda", "bfloat16", id="cuda-bfloat16", marks=pytest.mark.gpu),
]


def _config(amp_dtype: str | None, **kwargs) -> AlgorithmConfig:
    if amp_dtype is None:
        return AlgorithmConfig(**kwargs)
    return AlgorithmConfig(use_amp=True, amp_dtype=amp_dtype, **kwargs)


def _weights(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _labeled(chunk: TrajectoryChunk, action: int) -> TrajectoryChunk:
    """``chunk`` with the teacher action ``action`` on every ACT slot (as the workers write DAgger labels)."""
    payload = chunk.to_payload()
    is_act = np.asarray(chunk.kind) == SLOT_ACT
    payload["teacher_action"] = np.where(is_act, action, 0).astype(np.int64)
    payload["has_teacher"] = is_act
    return TrajectoryChunk.from_payload(payload)


@pytest.mark.parametrize(("device", "amp_dtype"), CASES)
def test_critic_warmup_under_amp_updates_only_the_value_path(device, amp_dtype):
    torch.manual_seed(0)
    model = make_test_model(ROLE)
    batch = [synthetic_chunk(model, ROLE, "AAAAAAAB", seed=s) for s in range(4)]
    algo = APPO(model, _config(amp_dtype, learning_rate=1e-2), SPEC, device=device, critic_warmup_steps=3)
    assert algo._use_amp == (amp_dtype is not None)
    value_ids = {id(p) for p in model.value_parameters()}
    value_keys = {name for name, p in model.named_parameters() if id(p) in value_ids}
    assert value_keys and any(name.startswith("critic_encoder.") for name in value_keys)
    before = _weights(model)
    skipped = 0
    for _ in range(3):
        metrics = algo.train_step(batch)
        assert metrics["critic_warmup"] == 1.0
        assert math.isfinite(metrics["value_loss"]) and math.isfinite(metrics["total_loss"])
        skipped += int(metrics.get("skipped_updates", 0))
    after = _weights(model)
    for key, value in before.items():
        if key not in value_keys:
            assert torch.equal(value, after[key]), key                   # the workers' policy, bit for bit
    if skipped < 3:                                                        # GradScaler may skip an fp16 step
        assert any(not torch.equal(before[k], after[k]) for k in value_keys)
    for p in model.parameters():
        if id(p) not in value_ids:
            assert p.grad is None                                          # nothing for GradScaler to unscale
    assert {id(p) for p in algo._optimizer.state} <= value_ids


@pytest.mark.parametrize(("device", "amp_dtype"), CASES)
def test_dagger_label_loss_under_amp_is_finite_and_trains(device, amp_dtype):
    torch.manual_seed(0)
    model = make_test_model(ROLE)
    chunks = [_labeled(synthetic_chunk(model, ROLE, "AAAAAAAB", seed=s), 2) for s in range(4)]
    kick = KickstartLoss(None, initial_lambda=1.0, decay_steps=100)
    algo = APPO(model, _config(amp_dtype, learning_rate=1e-2), SPEC, device=device, kickstart=kick)
    assert algo.teacher_active
    before = _weights(model)
    metrics = algo.train_step(chunks)
    assert metrics["kickstart_loss"] > 0 and math.isfinite(metrics["kickstart_loss"])
    assert metrics["kickstart_label_frac"] == pytest.approx(1.0)
    for key in ("total_loss", "policy_loss", "value_loss", "entropy"):
        assert math.isfinite(metrics[key]), key
    assert math.isfinite(metrics["grad_norm"]) or metrics["skipped_updates"] > 0
    if not metrics.get("skipped_updates", 0):                              # the policy path learned
        policy_keys = [k for k in before if k.startswith("policy.")]
        after = _weights(model)
        assert policy_keys and any(not torch.equal(before[k], after[k]) for k in policy_keys)
    assert kick.current_lambda == pytest.approx(0.99)
