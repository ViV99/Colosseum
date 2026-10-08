"""`colosseum bc` CLI (T4.4)."""
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
