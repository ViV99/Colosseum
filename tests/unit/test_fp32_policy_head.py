"""``ComposedModel`` runs the policy head in float32 under CUDA autocast (GPU checks, M-6, ruling G-R6)."""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.specs import ActionSpec
from colosseum.core.tree import tree_leaves
from colosseum.core.types import SLOT_ACT
from colosseum.envs.game import RoleSpec
from colosseum.envs.spaces import Units
from colosseum.networks.base import BaseEncoder, BasePolicy, EncoderOutput
from game_helpers import GenericEncoder, make_test_model, synthetic_chunk

ROLE = RoleSpec(gymnasium.spaces.Box(-1.0, 1.0, (8,), np.float32),
                Units(4, gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32)))


class AuxEncoder(BaseEncoder):
    """``GenericEncoder`` plus an ``aux`` with a float and an integer leaf."""

    def __init__(self, inner: GenericEncoder) -> None:
        super().__init__()
        self.inner = inner

    @property
    def latent_dim(self) -> int:
        return self.inner.latent_dim

    def forward(self, obs):
        latent = self.inner(obs).latent
        return EncoderOutput(latent, {"emb": latent * 2.0, "idx": torch.zeros(latent.shape[0], dtype=torch.long,
                                                                               device=latent.device)})


class SpyPolicy(BasePolicy):
    """Wraps a real head; records what it was called with."""

    def __init__(self, inner: BasePolicy) -> None:
        super().__init__()
        self.inner = inner
        self.calls: list[dict] = []

    def forward(self, features, aux):
        self.calls.append({"features": features, "aux": aux,
                           "autocast": torch.is_autocast_enabled("cuda") if torch.cuda.is_available() else False})
        return self.inner(features, aux)


def _spy_model(core: str = "lstm"):
    model = make_test_model(ROLE, core=core, hidden=16)
    model.encoder = AuxEncoder(model.encoder)
    model.policy = SpyPolicy(model.policy)
    return model


def _unroll(model, device: str, S: int = 3, B: int = 2):
    obs = torch.randn(S, B, 8, device=device)
    return model.unroll(obs, model.initial_state(B, device), torch.zeros(S, B, dtype=torch.bool, device=device))


def test_without_autocast_the_head_gets_the_very_same_tensors():
    model = _spy_model()
    seen: dict = {}

    def recorder(name, fn):
        def wrapped(*args):
            seen[name] = out = fn(*args)
            return out
        return wrapped

    model.encoder.forward = recorder("enc", model.encoder.forward)
    model.core.step = recorder("core_step", model.core.step)
    value_inputs: list = []
    orig_value = model.value.forward
    model.value.forward = lambda x: (value_inputs.append(x), orig_value(x))[1]

    model.step(torch.randn(2, 8), model.initial_state(2))
    call = model.policy.calls[-1]
    assert call["features"] is seen["core_step"][0]
    assert call["aux"] is seen["enc"].aux

    _unroll(model, "cpu")
    call = model.policy.calls[-1]
    assert call["features"] is value_inputs[-1]  # the value head (no critic) gets the same features object
    assert call["aux"] is seen["enc"].aux
    assert call["aux"]["idx"].dtype == torch.long


@pytest.mark.gpu
@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16], ids=["float16", "bfloat16"])
def test_under_cuda_autocast_the_policy_head_runs_in_float32(amp_dtype):
    model = _spy_model().to("cuda")
    with torch.autocast("cuda", dtype=amp_dtype):
        step = model.step(torch.randn(2, 8, device="cuda"), model.initial_state(2, "cuda"))
        out = _unroll(model, "cuda")
    assert len(model.policy.calls) == 2  # step + unroll
    for call in model.policy.calls:
        assert call["features"].dtype == torch.float32 and call["autocast"] is False
        assert call["aux"]["emb"].dtype == torch.float32 and call["aux"]["idx"].dtype == torch.long
    assert out.value.dtype == amp_dtype  # the value head still runs under autocast
    for dist in (step.dist, out.dist):
        params = [leaf for part in dist.parts.values() for leaf in tree_leaves(part.params)]
        assert params and all(p.dtype == torch.float32 for p in params)


@pytest.mark.gpu
def test_bf16_zero_lag_log_ratio_with_a_small_sigma_stays_small():
    role = RoleSpec(gymnasium.spaces.Box(-1.0, 1.0, (16,), np.float32),
                    Units(6, gymnasium.spaces.Box(-1.0, 1.0, (4,), np.float32)))
    torch.manual_seed(0)
    model = make_test_model(role, core="lstm", hidden=256)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "log_std" in name:
                p.fill_(-4.0)
    chunks = [synthetic_chunk(model, role, "A" * 63 + "B", seed=i, logp_noise=0.0) for i in range(8)]
    algo = APPO(model, AlgorithmConfig(use_amp=True, amp_dtype="bfloat16"), ActionSpec.from_space(role.action_space),
                device="cuda")
    with torch.no_grad():
        _, ev = algo._evaluate(algo._prepare_batch(chunks))
    behaviour = torch.stack([c.behavior_unit_logp for c in chunks], dim=1)
    valid = ev["unit_valid"].cpu() & (torch.stack([c.kind for c in chunks], dim=1) == SLOT_ACT).unsqueeze(-1)
    p95 = (ev["unit_log_probs"].cpu() - behaviour)[valid].abs().quantile(0.95).item()
    assert p95 < 0.05  # 0.137 with the head under bf16 autocast, 0.0166 in float32 (M-6)
