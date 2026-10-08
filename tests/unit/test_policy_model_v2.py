"""PolicyModel v2: act() with per-decider log-probs, unroll == step for 4 cores, critic, aux, normalizers (T2.3)."""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from colosseum.networks.cores import LSTMCore, NoCore
from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.tree import tree_index, tree_map, tree_stack, tree_to_torch
from colosseum.sp2.networks.base import BaseEncoder, BasePolicy, EncoderOutput
from colosseum.sp2.networks.composed import ComposedModel
from colosseum.sp2.networks.dist import make_distribution
from colosseum.sp2.networks.heads import UnitsHead
from colosseum.sp2.networks.model import act
from colosseum.sp2.networks.normalization import NormalizeObs
from game_helpers import (
    CORE_KINDS,
    GenericValue,
    GlobalStateGame,
    RandomPolicy,
    TreePolicyHead,
    TurnTakingGame,
    UnitsGame,
    make_test_model,
)

UNITS_ROLE = UnitsGame(max_units=3).spec.roles["player"]
S, B = 5, 3


def _slots(role, seed=0):
    """Random observations [S, B, ...] and masks with some absent units, as torch trees."""
    space = role.observation_space
    space.seed(seed)
    obs = tree_stack([tree_stack([space.sample() for _ in range(B)]) for _ in range(S)])
    mask = ActionSpec.from_space(role.action_space).full_mask((S, B))
    rng = np.random.default_rng(seed)
    mask["units"]["unit"][...] = rng.random((S, B, 3)) < 0.7
    return tree_to_torch(obs), tree_to_torch(mask)


def _reset_after():
    reset_after = torch.zeros(S, B, dtype=torch.bool)
    reset_after[1, 0] = reset_after[3, 2] = True
    return reset_after


def _step_through(model, obs, mask, reset_after):
    state = model.initial_state(B)
    actions, unit_lps, joint = [], [], []
    for s in range(S):
        out = act(model, tree_index(obs, s), state, tree_index(mask, s))
        actions.append(out.actions)
        unit_lps.append(out.unit_log_probs)
        joint.append(out.log_probs)
        state = model.reset_state(out.state, reset_after[s])
    flat_actions = tree_map(lambda *xs: torch.cat(xs, dim=0), actions[0], *actions[1:])
    return flat_actions, torch.cat(unit_lps), torch.cat(joint)


@pytest.mark.parametrize("core", CORE_KINDS)
def test_unroll_reproduces_step_per_decider(core):
    torch.manual_seed(0)
    model = make_test_model(UNITS_ROLE, core=core)
    obs, mask = _slots(UNITS_ROLE)
    reset_after = _reset_after()
    actions, unit_lps, joint = _step_through(model, obs, mask, reset_after)
    assert unit_lps.shape == (S * B, 1 + 3)                         # decider 0 (base) + 3 units
    assert torch.allclose(joint, unit_lps.sum(-1))
    out = model.unroll(obs, model.initial_state(B), reset_after, mask, with_value=True)
    assert torch.allclose(out.dist.unit_log_prob(actions), unit_lps, atol=1e-5)
    assert out.value.shape == (S * B,)
    assert model.unroll(obs, model.initial_state(B), reset_after, mask, with_value=False).value is None


def test_act_returns_env_format_actions():
    model = make_test_model(UNITS_ROLE)
    obs, mask = _slots(UNITS_ROLE)
    out = act(model, tree_index(obs, 0), None, tree_index(mask, 0), deterministic=True)
    assert out.actions["base"].shape == (B,) and out.actions["units"]["move"].shape == (B, 3)
    assert out.unit_log_probs.shape == (B, 4) and out.state is None
    absent = ~tree_index(mask, 0)["units"]["unit"]
    assert torch.all(out.unit_log_probs[:, 1:][absent] == 0)


def test_unroll_argument_errors():
    model = make_test_model(UNITS_ROLE, core="lstm")
    obs, mask = _slots(UNITS_ROLE)
    with pytest.raises(ValueError, match="state0 is None"):
        model.unroll(obs, None, _reset_after(), mask)
    assert model.is_stateful and not make_test_model(UNITS_ROLE).is_stateful


def test_critic_encoder_feeds_the_value_only():
    role = GlobalStateGame().spec.roles["player"]
    model = make_test_model(role, core="gru")
    assert model.critic_encoder is not None
    obs = torch.randn(S, B, 2)
    reset_after = torch.zeros(S, B, dtype=torch.bool)
    with pytest.raises(ValueError, match="needs global_state"):
        model.unroll(obs, model.initial_state(B), reset_after)
    gs_a, gs_b = torch.zeros(S, B, 4), torch.ones(S, B, 4)
    a = model.unroll(obs, model.initial_state(B), reset_after, global_state=gs_a)
    b = model.unroll(obs, model.initial_state(B), reset_after, global_state=gs_b)
    actions = torch.zeros(S * B, dtype=torch.long)
    assert torch.allclose(a.dist.log_prob(actions), b.dist.log_prob(actions))
    assert not torch.allclose(a.value, b.value)
    assert model.unroll(obs, model.initial_state(B), reset_after, with_value=False).value is None


class _EntityEncoder(BaseEncoder):
    """Latent from the grid; per-unit embeddings of the entity list go around the core in ``aux``."""

    def __init__(self, hidden=16, emb=8):
        super().__init__()
        self.grid = nn.Linear(16, hidden)
        self.entities = nn.Linear(3, emb)

    @property
    def latent_dim(self):
        return self.grid.out_features

    def forward(self, obs):
        latent = torch.relu(self.grid(obs["grid"].float().flatten(1) / 255.0))
        return EncoderOutput(latent, {"units": torch.relu(self.entities(obs["entities"]))})


class _AuxPolicy(BasePolicy):
    def __init__(self, in_dim, spec, emb=8):
        super().__init__()
        self.spec = spec
        self.base = nn.Linear(in_dim, 3)
        self.units = UnitsHead(spec.groups[1], in_dim + emb, hidden=8)

    def forward(self, features, aux):
        per_unit = torch.cat([features.unsqueeze(1).expand(-1, aux["units"].shape[1], -1), aux["units"]], dim=-1)
        return make_distribution(self.spec, {"base": self.base(features), "units": self.units(per_unit)})


def test_encoder_aux_bypasses_the_core():
    torch.manual_seed(0)
    spec = ActionSpec.from_space(UNITS_ROLE.action_space)
    encoder = _EntityEncoder()
    core = LSTMCore(16, hidden_size=16)
    model = ComposedModel(encoder, core, _AuxPolicy(16, spec), GenericValue(16))
    obs, mask = _slots(UNITS_ROLE)
    reset_after = _reset_after()
    actions, unit_lps, _ = _step_through(model, obs, mask, reset_after)
    out = model.unroll(obs, model.initial_state(B), reset_after, mask)
    assert torch.allclose(out.dist.unit_log_prob(actions), unit_lps, atol=1e-5)
    out.dist.log_prob(actions).sum().backward()
    assert encoder.entities.weight.grad is not None and encoder.entities.weight.grad.abs().sum() > 0


def test_uint8_leaves_reach_the_encoder_unchanged():
    model = make_test_model(UNITS_ROLE)
    obs, mask = _slots(UNITS_ROLE)
    model.step(tree_index(obs, 0), None, tree_index(mask, 0))
    assert model.encoder.seen_dtypes == [torch.uint8, torch.float32, torch.int8]


class _NormEncoder(BaseEncoder):
    def __init__(self):
        super().__init__()
        self.norm = NormalizeObs(shape=(3,), path=("entities",))
        self.gs_norm = NormalizeObs(shape=(4,), source="global_state")
        self.fc = nn.Linear(3, 4)

    @property
    def latent_dim(self):
        return 4

    def forward(self, obs):
        return self.fc(self.norm(obs["entities"]).mean(dim=1))


def test_update_normalizers_reads_each_leaf_path_and_source():
    encoder = _NormEncoder()
    model = ComposedModel(encoder, NoCore(4), TreePolicyHead(4, ActionSpec.from_space(UNITS_ROLE.action_space)),
                          GenericValue(4))
    obs = {"grid": torch.zeros(6, 4, 4, dtype=torch.uint8), "entities": torch.full((6, 3, 3), 2.0),
           "entity_mask": torch.ones(6, 3, dtype=torch.int8)}
    model.update_normalizers(obs)
    assert torch.allclose(encoder.norm.rms.mean, torch.full((3,), 2.0), atol=1e-3)
    assert torch.allclose(encoder.gs_norm.rms.mean, torch.zeros(4))         # no global_state given: skipped
    model.update_normalizers(obs, global_state=torch.full((5, 4), 3.0))
    assert torch.allclose(encoder.gs_norm.rms.mean, torch.full((4,), 3.0), atol=1e-3)


def test_random_policy_is_uniform_over_legal_actions():
    role = TurnTakingGame().spec.roles["player"]
    policy = RandomPolicy(role)
    mask = torch.tensor([[True, True, False]] * 4)
    out = act(policy, torch.zeros(4, 3), None, mask)
    assert set(out.actions.tolist()) <= {0, 1}
    assert torch.allclose(out.log_probs, torch.full((4,), -math.log(2)))
    obs, mask = _slots(UNITS_ROLE)
    units = act(RandomPolicy(UNITS_ROLE), tree_index(obs, 0), None, tree_index(mask, 0))
    assert units.unit_log_probs.shape == (B, 4) and units.actions["units"]["target"].shape == (B, 3)
    unrolled = RandomPolicy(UNITS_ROLE).unroll(obs, None, _reset_after(), mask)
    assert unrolled.value.shape == (S * B,) and torch.all(unrolled.value == 0)
