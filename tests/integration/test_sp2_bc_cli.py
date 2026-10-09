"""`python -m colosseum.sp2 bc --agent A` (T6.3)."""
from __future__ import annotations

import pytest
import torch
from click.testing import CliRunner

from colosseum.sp2.cli import main
from colosseum.sp2.core.registry import build_model
from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.tree import tree_map, tree_stack
from game_helpers import agent_role_of, make_test_config, write_test_config


def write_data(path, config, agent_id: str = "agent_0", n: int = 64) -> None:
    """Random decisions in the agent's spaces, all-legal masks, an episode end every 5."""
    _roles, role = agent_role_of(config, agent_id)
    role.observation_space.seed(0)
    role.action_space.seed(0)
    obs = tree_stack([role.observation_space.sample() for _ in range(n)])
    actions = tree_stack([role.action_space.sample() for _ in range(n)])
    spec = ActionSpec.from_space(role.action_space)
    data = {"observations": tree_map(torch.as_tensor, obs), "actions": tree_map(torch.as_tensor, actions),
            "dones": torch.arange(n) % 5 == 4}
    if spec.has_masks:
        data["action_masks"] = tree_map(torch.as_tensor, spec.full_mask((n,)))
    torch.save(data, path)


def bc(*args):
    return CliRunner().invoke(main, ["bc", *map(str, args)])


def test_bc_trains_and_saves_weights_that_load_into_the_agent_model(tmp_path):
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns")
    cfg = make_test_config("turns")
    write_data(tmp_path / "data.pt", cfg)
    out = tmp_path / "bc.pt"
    result = bc("-c", cfg_path, "-d", tmp_path / "data.pt", "-o", out, "--epochs", 2, "--batch-size", 16)
    assert result.exit_code == 0, result.output
    assert "final-epoch NLL" in result.output and "(agent_0)" in result.output
    _roles, role = agent_role_of(cfg, "agent_0")
    build_model(cfg.get_agent_config("agent_0"), role).load_state_dict(torch.load(out, weights_only=True))


def test_agent_selects_networks_and_roles(tmp_path):
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "asymmetric")
    cfg = make_test_config("asymmetric")
    write_data(tmp_path / "prey.pt", cfg, agent_id="prey")
    out = tmp_path / "prey_bc.pt"
    result = bc("-c", cfg_path, "-d", tmp_path / "prey.pt", "-o", out, "--agent", "prey", "--epochs", 1)
    assert result.exit_code == 0, result.output
    _roles, prey_role = agent_role_of(cfg, "prey")
    build_model(cfg.get_agent_config("prey"), prey_role).load_state_dict(torch.load(out, weights_only=True))

    result = bc("-c", cfg_path, "-d", tmp_path / "prey.pt", "-o", out)
    assert result.exit_code == 1 and "--agent" in result.stderr
    result = bc("-c", cfg_path, "-d", tmp_path / "prey.pt", "-o", out, "--agent", "nobody")
    assert result.exit_code == 1 and "nobody" in result.stderr
    result = bc("-c", cfg_path, "-d", tmp_path / "prey.pt", "-o", tmp_path / "x.pt", "--agent", "hunter")
    assert result.exit_code == 1 and result.stderr.startswith("Config error:")  # prey data for the hunter


def test_bc_rejects_non_positive_counts(tmp_path):
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns")
    write_data(tmp_path / "data.pt", make_test_config("turns"))
    result = bc("-c", cfg_path, "-d", tmp_path / "data.pt", "-o", tmp_path / "x.pt", "--seq-len", 0)
    assert result.exit_code == 2 and "--seq-len" in result.output
    for option in ("--epochs", "--batch-size"):
        result = bc("-c", cfg_path, "-d", tmp_path / "data.pt", "-o", tmp_path / "x.pt", option, 0)
        assert result.exit_code == 2 and option in result.output


def test_bc_cli_has_no_action_type_option(tmp_path):
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns")
    write_data(tmp_path / "data.pt", make_test_config("turns"))
    result = bc("-c", cfg_path, "-d", tmp_path / "data.pt", "-o", tmp_path / "x.pt", "--action-type", "discrete")
    assert result.exit_code == 2 and "No such option" in result.output


def _garbage(tmp_path):
    path = tmp_path / "data.pt"
    path.write_bytes(b"not a torch file")
    return path


def _missing_actions(tmp_path):
    path = tmp_path / "data.pt"
    torch.save({"observations": torch.rand(8, 3)}, path)
    return path


def _empty_dir(tmp_path):
    path = tmp_path / "no_data"
    path.mkdir()
    return path


def _float_actions(tmp_path):
    path = tmp_path / "data.pt"
    write_data(path, make_test_config("turns"))
    data = torch.load(path, weights_only=True)
    data["actions"] = tree_map(lambda x: x.float() + 0.5, data["actions"])
    torch.save(data, path)
    return path


def _illegal_actions(tmp_path):
    """Every recorded action is forbidden by its own mask (P7: a DataError, not a traceback)."""
    path = tmp_path / "data.pt"
    write_data(path, make_test_config("turns"))
    data = torch.load(path, weights_only=True)
    data["action_masks"] = torch.nn.functional.one_hot((data["actions"] + 1) % 3, 3).bool()
    torch.save(data, path)
    return path


def _empty_mask_row(tmp_path):
    path = tmp_path / "data.pt"
    write_data(path, make_test_config("turns"))
    data = torch.load(path, weights_only=True)
    data["action_masks"][5] = False
    torch.save(data, path)
    return path


def _nan_observations(tmp_path):
    path = tmp_path / "data.pt"
    write_data(path, make_test_config("turns"))
    data = torch.load(path, weights_only=True)
    data["observations"][7, 1] = float("nan")
    torch.save(data, path)
    return path


@pytest.mark.parametrize(("make_data", "message"), [
    (_garbage, "data.pt"),
    (_missing_actions, "missing BC data keys ['actions']"),
    (_float_actions, "floating point"),
    (_empty_dir, "no .pt files"),
    (_illegal_actions, "data.pt: BC decision 0: action"),
    (_empty_mask_row, "data.pt: BC decision 5: action mask <root> has no legal action"),
    (_nan_observations, "has NaN/inf values (first at decision 7)"),
], ids=["unreadable", "missing-key", "float-actions", "empty-dir", "illegal-actions", "empty-mask-row",
        "nan-observations"])
def test_bad_data_is_a_one_line_config_error(make_data, message, tmp_path):
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns")
    out = tmp_path / "bc.pt"
    result = bc("-c", cfg_path, "-d", make_data(tmp_path), "-o", out)
    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("Config error:") and message in result.stderr, result.stderr
    assert len(result.stderr.strip().splitlines()) == 1 and "Traceback" not in result.output
    assert not out.exists()


def test_a_data_file_that_cannot_be_opened_is_a_one_line_config_error(tmp_path, monkeypatch):
    """SP1 residual: PermissionError / OSError from torch.load used to print a traceback."""
    import colosseum.sp2.bc.offline_bc as bc_module

    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns")
    data = tmp_path / "data.pt"
    write_data(data, make_test_config("turns"))

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(data))

    monkeypatch.setattr(bc_module.torch, "load", denied)
    result = bc("-c", cfg_path, "-d", data, "-o", tmp_path / "bc.pt")
    assert result.exit_code == 1 and result.stderr.startswith("Config error:")
    assert "PermissionError" in result.stderr and len(result.stderr.strip().splitlines()) == 1


def test_an_illegal_units_action_is_a_one_line_config_error(tmp_path):
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "units")
    path = tmp_path / "units.pt"
    write_data(path, make_test_config("units"))
    data = torch.load(path, weights_only=True)
    move = data["actions"]["units"]["move"][:, 1]                     # unit 1 is present (full masks)
    row = data["action_masks"]["units"]["action"]
    row[torch.arange(len(move)), 1, move] = False                      # forbid its recorded move
    torch.save(data, path)
    result = bc("-c", cfg_path, "-d", path, "-o", tmp_path / "bc.pt", "--epochs", 1)
    assert result.exit_code == 1 and result.stderr.startswith("Config error:"), result.output
    assert "unit 1 component 'move'" in result.stderr and len(result.stderr.strip().splitlines()) == 1
