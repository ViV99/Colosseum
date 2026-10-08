"""`colosseum bc` CLI (T4.4)."""
import pytest
import torch
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.config import load_config
from colosseum.core.registry import build_model

CONFIG = """\
env:
  env_class: examples.tic_tac_toe.env.TicTacToeEnv
  num_players: 2
networks:
  encoder_class: examples.tic_tac_toe.networks.TicTacToeEncoder
  policy_class: examples.tic_tac_toe.networks.TicTacToePolicy
  value_class: examples.tic_tac_toe.networks.TicTacToeValue
"""


def _write_data(path, n=64):
    actions = torch.randint(0, 9, (n,))
    masks = torch.rand(n, 9) < 0.5
    masks[torch.arange(n), actions] = True
    torch.save({
        "observations": torch.rand(n, 3, 3, 3),
        "actions": actions,
        "action_masks": masks,
        "dones": torch.arange(n) % 5 == 4,
    }, path)


def test_bc_cli_trains_and_saves_loadable_weights(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(CONFIG)
    data = tmp_path / "data.pt"
    _write_data(data)
    out = tmp_path / "bc.pt"
    result = CliRunner().invoke(main, [
        "bc", "-c", str(cfg), "-d", str(data), "-o", str(out),
        "--epochs", "2", "--batch-size", "16", "--seq-len", "8",
    ])
    assert result.exit_code == 0, result.output
    assert "final-epoch NLL" in result.output
    model = build_model(load_config(cfg))
    model.load_state_dict(torch.load(out, weights_only=True))


def test_bc_cli_has_no_action_type_option(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(CONFIG)
    data = tmp_path / "data.pt"
    _write_data(data)
    result = CliRunner().invoke(main, [
        "bc", "-c", str(cfg), "-d", str(data), "-o", str(tmp_path / "x.pt"), "--action-type", "discrete",
    ])
    assert result.exit_code == 2
    assert "No such option" in result.output


def test_bc_cli_rejects_non_positive_seq_len(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(CONFIG)
    data = tmp_path / "data.pt"
    _write_data(data)
    result = CliRunner().invoke(main, [
        "bc", "-c", str(cfg), "-d", str(data), "-o", str(tmp_path / "x.pt"), "--seq-len", "0",
    ])
    assert result.exit_code == 2
    assert "--seq-len" in result.output


def _garbage_file(tmp_path):
    path = tmp_path / "data.pt"
    path.write_bytes(b"not a torch file")
    return path


def _missing_actions_file(tmp_path):
    path = tmp_path / "data.pt"
    torch.save({"observations": torch.rand(8, 3, 3, 3)}, path)
    return path


def _float_actions_file(tmp_path):
    path = tmp_path / "data.pt"
    torch.save({"observations": torch.rand(8, 3, 3, 3), "actions": torch.rand(8)}, path)
    return path


def _empty_dir(tmp_path):
    path = tmp_path / "no_data"
    path.mkdir()
    return path


@pytest.mark.parametrize(("make_data", "message"), [
    (_garbage_file, "data.pt"),
    (_missing_actions_file, "missing BC data keys ['actions']"),
    (_float_actions_file, "floating point"),
    (_empty_dir, "no .pt files"),
], ids=["unreadable", "missing-key", "action-check", "empty-dir"])
def test_bc_cli_reports_bad_data_as_one_line_config_error(make_data, message, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(CONFIG)
    out = tmp_path / "bc.pt"
    result = CliRunner().invoke(main, ["bc", "-c", str(cfg), "-d", str(make_data(tmp_path)), "-o", str(out)])
    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("Config error:") and message in result.stderr, result.stderr
    assert len(result.stderr.strip().splitlines()) == 1
    assert "Traceback" not in result.output and isinstance(result.exception, SystemExit)
    assert not out.exists()
