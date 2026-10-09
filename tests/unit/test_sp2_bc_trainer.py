"""Offline BC on observation/action trees (T6.3)."""
from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.core.errors import DataError
from colosseum.sp2.bc import offline_bc as bc_module
from colosseum.sp2.bc.offline_bc import OfflineBCTrainer, _window_index, per_sample_nll
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.tree import tree_stack
from colosseum.sp2.envs.game import RoleSpec
from game_helpers import UnitsGame, make_test_model, sample_legal_action

pytestmark = pytest.mark.usefixtures("restore_global_rng")

BOX = gymnasium.spaces.Box(0.0, 1.0, (3,), np.float32)
DICT_OBS = gymnasium.spaces.Dict({"grid": gymnasium.spaces.Box(0, 255, (2, 2), np.uint8),
                                  "vec": gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)})
FOUR = gymnasium.spaces.Discrete(4)


def trainer_for(role: RoleSpec, core: str = "none", seq_len: int = 4, lr: float = 1e-2,
                device: str = "cpu") -> OfflineBCTrainer:
    torch.manual_seed(0)
    return OfflineBCTrainer(make_test_model(role, core=core), ActionSpec.from_space(role.action_space),
                            ObsSpec.from_space(role.observation_space), lr=lr, device=device, seq_len=seq_len)


def samples(space, n: int, seed: int = 0):
    space.seed(seed)
    return tree_stack([space.sample() for _ in range(n)])


def units_decisions(n: int, seed: int = 0) -> dict:
    """``n`` random legal decisions of ``UnitsGame`` (absent units: ``unit=False``) as BC trees."""
    env = UnitsGame()
    role = env.spec.roles["player"]
    spec = ActionSpec.from_space(role.action_space)
    rng = np.random.default_rng(seed)
    data = {"observations": [], "actions": [], "action_masks": [], "dones": []}
    result = env.reset(seed, "solo")
    while len(data["dones"]) < n:
        mask = spec.normalize_mask(result.action_masks.get(0), "UnitsGame")
        action = sample_legal_action(spec, mask, rng)
        data["observations"].append(result.obs[0])
        data["actions"].append(action)
        data["action_masks"].append(mask)
        result = env.step({0: action})
        data["dones"].append(result.episode_over)
        if result.episode_over:
            result = env.reset(None, "solo")
    out = {key: tree_stack(value) for key, value in data.items() if key != "dones"}
    out["dones"] = np.asarray(data["dones"])
    return out


def test_window_index_tiles_with_offset_and_padding():
    assert _window_index(5, 2).tolist() == [[0, 1], [2, 3], [4, -1]]
    assert _window_index(5, 2, offset=1).tolist() == [[0, -1], [1, 2], [3, 4]]
    assert _window_index(0, 3).shape == (0, 3)


class _FixedDist:
    def __init__(self, unit_lp, unit_valid):
        self._lp, self._valid = unit_lp, unit_valid

    def log_prob(self, actions):
        return torch.where(self._valid, self._lp, torch.zeros_like(self._lp)).sum(-1)

    def unit_log_prob(self, actions):
        return self._lp

    def unit_valid(self, actions):
        return self._valid


def test_nll_is_joint_for_one_decider_and_a_mean_over_valid_deciders_otherwise():
    lp = torch.tensor([[-1.0, -3.0, -5.0], [-2.0, -4.0, -6.0]])
    valid = torch.tensor([[True, True, False], [False, False, False]])
    nll, ok = per_sample_nll(_FixedDist(lp, valid), None, num_deciders=3)
    assert nll[0].item() == pytest.approx(2.0) and ok.tolist() == [True, False]
    joint, ok1 = per_sample_nll(_FixedDist(lp[:, :1], torch.ones(2, 1, dtype=torch.bool)), None, num_deciders=1)
    assert joint.tolist() == [1.0, 2.0] and ok1.all()


def test_stateless_training_on_dict_observations_keeps_dtypes():
    role = RoleSpec(DICT_OBS, FOUR)
    trainer = trainer_for(role)
    obs = samples(DICT_OBS, 64)
    actions = np.full(64, 2, dtype=np.int64)
    trainer.add_data(obs, actions, dones=np.arange(64) % 8 == 7)
    assert trainer._observations[0]["grid"].dtype == torch.uint8  # kept until the model casts
    metrics = trainer.train(num_epochs=20, batch_size=16)
    assert metrics["bc_loss"] < metrics["bc_loss_first_epoch"] and metrics["accuracy"] > 0.9
    assert trainer.model.encoder.seen_dtypes == [torch.uint8, torch.float32]


def test_stateful_training_uses_windows_with_resets():
    role = RoleSpec(BOX, FOUR)
    trainer = trainer_for(role, core="lstm", seq_len=4)
    obs = samples(BOX, 40)
    trainer.add_data(obs, np.arange(40) % 4, np.ones((40, 4), dtype=bool), np.arange(40) % 5 == 4)
    metrics = trainer.train(num_epochs=3, batch_size=8)
    assert np.isfinite(metrics["bc_loss"]) and "accuracy" in metrics


def test_stateful_windows_reset_the_state_after_dones():
    """SP1 guarantee: after ``dones[t]`` decision t + 1 sees what a fresh episode start sees."""
    role = RoleSpec(BOX, FOUR)
    trainer = trainer_for(role, core="lstm", seq_len=8)
    obs = torch.as_tensor(samples(BOX, 8, seed=1))
    index = _window_index(8, 8).t()                                   # [L=8, b=1]
    dones = torch.zeros(8, dtype=torch.bool)
    dones[3] = True
    every_action = torch.arange(4)
    model = trainer.model
    with torch.no_grad():
        reset_dist, _ = trainer._sequence_forward(index, obs, None, dones)
        carried_dist, _ = trainer._sequence_forward(index, obs, None, torch.zeros(8, dtype=torch.bool))
        fresh = model.step(obs[4:5].expand(4, -1), model.initial_state(4)).dist.log_prob(every_action)

        def at_step_4(dist):
            return torch.stack([dist.log_prob(torch.full((8,), a))[4] for a in range(4)])

        assert torch.allclose(at_step_4(reset_dist), fresh, atol=1e-6)
        assert not torch.allclose(at_step_4(carried_dist), fresh, atol=1e-4)


def test_units_actions_train_on_the_mean_over_valid_deciders():
    role = UnitsGame().spec.roles["player"]
    trainer = trainer_for(role, lr=1e-2)
    assert trainer._num_deciders == 5
    trainer.add_data(**units_decisions(48))
    metrics = trainer.train(num_epochs=15, batch_size=16)
    assert metrics["bc_loss"] < metrics["bc_loss_first_epoch"]
    assert "accuracy" not in metrics  # no accuracy for Units actions


def test_illegal_units_actions_under_their_masks_are_data_errors():
    trainer = trainer_for(UnitsGame().spec.roles["player"])
    data = units_decisions(12)
    data["actions"]["units"]["move"][:, 0] = 3            # unit 0 always exists; the target counts (move == 3)
    data["actions"]["units"]["target"][:, 0] = 3          # unit 3 exists only at t = 3 (decision 3, 9)
    with pytest.raises(DataError, match=r"BC decision 0: unit 0 component 'target': action 3 at units is illegal"):
        trainer.add_data(**data)


def test_a_gated_off_units_component_need_not_be_legal():
    """``target`` counts only where ``move == 3``: elsewhere its recorded value is free."""
    trainer = trainer_for(UnitsGame().spec.roles["player"])
    data = units_decisions(12)
    data["actions"]["units"]["move"][:, 0] = 0
    data["actions"]["units"]["target"][:, 0] = 3          # illegal row value, but the gate is off
    absent = ~data["action_masks"]["units"]["unit"]
    data["actions"]["units"]["move"][absent] = 3          # absent units: nothing is checked
    data["actions"]["units"]["target"][absent] = 3
    trainer.add_data(**data)
    assert trainer.num_samples == 12


def test_first_illegal_action_reports_the_first_bad_decision():
    spec = ActionSpec.from_space(gymnasium.spaces.Dict({"a": FOUR, "m": gymnasium.spaces.MultiDiscrete([3, 2])}))
    actions = {"a": torch.zeros(4, dtype=torch.long), "m": torch.zeros(4, 2, dtype=torch.long)}
    mask = {k: torch.as_tensor(v) for k, v in spec.full_mask((4,)).items()}
    assert spec.first_illegal_action(actions, mask) is None and spec.first_illegal_action(actions, None) is None
    mask["m"][2, 3] = False                                           # sub-action 1, value 0 at decision 2
    mask["a"][3] = False                                              # empty row at decision 3
    assert spec.first_illegal_action(actions, mask) == (2, "action 0 at m (sub-action 1) is illegal under its "
                                                           "action mask")
    mask["a"][1] = False
    assert spec.first_illegal_action(actions, mask) == (1, "action mask a has no legal action")


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"actions": np.zeros(8, np.float32)}, "floating point"),
    ({"actions": np.zeros((8, 2), np.int64)}, "actions leaf"),
    ({"actions": np.full(8, 4, np.int64)}, "outside"),
    ({"observations": {"x": np.zeros((8, 3), np.float32)}}, "observations has leaves"),
    ({"observations": np.zeros((8, 4), np.float32)}, "observations leaf"),
    ({"observations": np.array(["a"] * 8)}, "observations"),
    ({"observations": np.where(np.arange(24).reshape(8, 3) == 7, np.nan, 0.5).astype(np.float32)},
     "observations leaf <root> has NaN/inf values \\(first at decision 2\\)"),
    ({"action_masks": np.ones((8, 3), dtype=bool)}, "action_masks leaf"),
    ({"action_masks": np.arange(8)[:, None] != np.ones((1, 4))}, "decision 1: action mask <root> has no"),
    ({"action_masks": np.arange(32).reshape(8, 4) != 16}, "decision 4: action 0 at <root> is illegal under its"),
    ({"dones": np.zeros(7, dtype=bool)}, "dones"),
], ids=["float-actions", "action-shape", "action-range", "obs-structure", "obs-shape", "obs-strings",
        "obs-nan", "mask-shape", "mask-empty-row", "illegal-action", "dones"])
def test_data_that_does_not_fit_the_agent_is_a_data_error(kwargs, message):
    trainer = trainer_for(RoleSpec(BOX, FOUR))
    data = {"observations": samples(BOX, 8), "actions": np.zeros(8, np.int64),
            "action_masks": np.ones((8, 4), dtype=bool), "dones": None, **kwargs}
    with pytest.raises(DataError, match=message):
        trainer.add_data(data["observations"], data["actions"], data["action_masks"], data["dones"])


def test_masks_must_be_given_for_every_file_or_none():
    trainer = trainer_for(RoleSpec(BOX, FOUR))
    trainer.add_data(samples(BOX, 4), np.zeros(4, np.int64), np.ones((4, 4), dtype=bool))
    with pytest.raises(DataError, match="every batch/file"):
        trainer.add_data(samples(BOX, 4), np.zeros(4, np.int64))


def test_masks_on_a_box_action_space_are_rejected():
    box_actions = gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32)
    trainer = trainer_for(RoleSpec(BOX, box_actions))
    with pytest.raises(DataError, match="nothing to mask"):
        trainer.add_data(samples(BOX, 4), np.zeros((4, 2), np.float32), np.ones((4, 2), dtype=bool))


def test_box_actions_must_be_finite():
    box_actions = gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32)
    trainer = trainer_for(RoleSpec(BOX, box_actions))
    actions = np.zeros((4, 2), np.float32)
    actions[3, 1] = np.inf
    with pytest.raises(DataError, match="actions leaf <root> has NaN/inf"):
        trainer.add_data(samples(BOX, 4), actions)


def test_illegal_expert_actions_fail_in_add_data_naming_the_file(tmp_path):
    trainer = trainer_for(RoleSpec(BOX, FOUR))
    masks = np.zeros((8, 4), dtype=bool)
    masks[:, 1] = True
    masks[:3, 0] = True
    path = tmp_path / "expert.pt"
    torch.save({"observations": torch.as_tensor(samples(BOX, 8)), "actions": torch.zeros(8, dtype=torch.long),
                "action_masks": torch.as_tensor(masks)}, path)
    with pytest.raises(DataError, match=r"expert\.pt: BC decision 3: action 0 at <root> is illegal under its "
                                        r"action mask"):
        trainer.load_data(path)
    assert trainer.num_samples == 0  # nothing trained, nothing kept


def test_unreadable_files_are_data_errors(tmp_path, monkeypatch):
    trainer = trainer_for(RoleSpec(BOX, FOUR))
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a torch file")
    with pytest.raises(DataError, match="garbage.pt"):
        trainer.load_data(garbage)

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(bc_module.torch, "load", denied)
    with pytest.raises(DataError, match="PermissionError"):
        trainer.load_data(garbage)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DataError, match="no .pt files"):
        trainer.load_data(empty)


def test_files_load_as_trees(tmp_path):
    role = RoleSpec(DICT_OBS, FOUR)
    path = tmp_path / "data.pt"
    obs = {k: torch.as_tensor(v) for k, v in samples(DICT_OBS, 16).items()}
    torch.save({"observations": obs, "actions": torch.zeros(16, dtype=torch.long), "extra": torch.zeros(1)}, path)
    trainer = trainer_for(role)
    assert trainer.load_data(path) == 16 and trainer.num_samples == 16


@pytest.mark.gpu
@pytest.mark.parametrize("core", ["none", "lstm"])
def test_bc_with_tree_data_on_cuda(core):
    """R12 successor of SP1's ``test_bc_trains_on_cuda``: Dict + uint8 observations, masks, dones."""
    trainer = trainer_for(RoleSpec(DICT_OBS, FOUR), core=core, seq_len=8, device="cuda")
    n = 64
    actions = np.arange(n) % 4
    masks = np.zeros((n, 4), dtype=bool)
    masks[np.arange(n), actions] = True
    masks[:, 0] = True
    trainer.add_data(samples(DICT_OBS, n), actions, masks, np.arange(n) % 8 == 7)
    metrics = trainer.train(num_epochs=3, batch_size=16)
    assert all(p.device.type == "cuda" for p in trainer.model.parameters())
    assert np.isfinite(metrics["bc_loss"]) and 0.0 <= metrics["accuracy"] <= 1.0
    assert trainer.model.encoder.seen_dtypes == [torch.uint8, torch.float32]
