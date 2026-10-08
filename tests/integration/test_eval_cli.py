"""`colosseum eval` CLI: .pt and checkpoint-dir agents, heterogeneous architectures, JSON (T7.2)."""
import json
import os
import time

import pytest
import torch
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.coordinator.checkpoint_manager import CheckpointManager, load_checkpoint_dir
from colosseum.core.config import NetworkConfig, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model
from colosseum.eval import load_eval_model

CONFIG = """\
env:
  env_class: examples.tic_tac_toe.env.TicTacToeEnv
  num_players: 2
networks:
  encoder_class: examples.tic_tac_toe.networks.TicTacToeEncoder
  policy_class: examples.tic_tac_toe.networks.TicTacToePolicy
  value_class: examples.tic_tac_toe.networks.TicTacToeValue
"""

GRU_NETWORKS = {
    "encoder_class": "examples.tic_tac_toe.networks.TicTacToeEncoder",
    "core": {"class": "colosseum.networks.cores.GRUCore", "kwargs": {"hidden_size": 64}},
    "policy_class": "examples.tic_tac_toe.networks.TicTacToePolicy",
    "value_class": "examples.tic_tac_toe.networks.TicTacToeValue",
}


def _setup(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(CONFIG)
    cfg = load_config(cfg_path)
    torch.manual_seed(0)
    pt_path = tmp_path / "a.pt"
    torch.save(build_model(cfg).state_dict(), pt_path)
    gru_model = build_model(cfg.model_copy(update={"networks": NetworkConfig.model_validate(GRU_NETWORKS)}))
    base_dir = tmp_path / "checkpoints"
    ckpt_id = CheckpointManager(base_dir).save(
        "agent_b", 3,
        {k: v.detach().cpu().numpy() for k, v in gru_model.state_dict().items()},
        meta_extra={"networks": GRU_NETWORKS},
    )
    return cfg_path, cfg, pt_path, base_dir / "agent_b" / ckpt_id


def test_load_eval_model_builds_from_meta_networks(tmp_path):
    _cfg_path, cfg, pt_path, ckpt_dir = _setup(tmp_path)
    assert not load_eval_model(pt_path, cfg).is_stateful          # global networks: no core
    assert load_eval_model(ckpt_dir, cfg).is_stateful             # meta.json networks: GRU core


def test_eval_cli_pairs_pt_and_heterogeneous_checkpoint(tmp_path):
    cfg_path, _cfg, pt_path, ckpt_dir = _setup(tmp_path)
    out = tmp_path / "result.json"
    result = CliRunner().invoke(main, [
        "eval", "-c", str(cfg_path), "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}",
        "-n", "4", "--num-envs", "2", "--seed", "0", "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "score" in result.output
    data = json.loads(out.read_text())
    assert data["mode"] == "pairwise"
    assert data["num_matches_per_pair"] == 4
    assert [(r["agent_a"], r["agent_b"]) for r in data["pairs"]] == [("a", "b"), ("b", "a")]
    assert data["pairs"][0]["n"] == 4
    assert set(data["pairs"][0]["per_seat"]) == {"0", "1"}


def test_eval_cli_rejects_bad_agent_specs(tmp_path):
    cfg_path, _cfg, pt_path, _ckpt_dir = _setup(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["eval", "-c", str(cfg_path), "-a", "no-equals-sign"])
    assert result.exit_code == 2 and "name=path" in result.output
    result = runner.invoke(main, ["eval", "-c", str(cfg_path), "-a", f"a={tmp_path / 'missing.pt'}"])
    assert result.exit_code == 2
    result = runner.invoke(main, ["eval", "-c", str(cfg_path), "-a", f"a={pt_path}", "-a", f"a={pt_path}"])
    assert result.exit_code == 2 and "duplicate" in result.output


def test_eval_cli_rounds_an_odd_pairwise_num_matches_up(tmp_path):
    cfg_path, _cfg, pt_path, ckpt_dir = _setup(tmp_path)
    out = tmp_path / "result.json"
    result = CliRunner().invoke(main, [
        "eval", "-c", str(cfg_path), "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}",
        "-n", "3", "--num-envs", "2", "--seed", "0", "-o", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "using 4" in result.stderr
    data = json.loads(out.read_text())
    assert data["num_matches_per_pair"] == 4 and data["pairs"][0]["n"] == 4
    # Solo mode has no seat balance to keep: an odd count is used as given.
    result = CliRunner().invoke(main, [
        "eval", "-c", str(cfg_path), "-a", f"a={pt_path}", "-n", "3", "--num-envs", "2", "-o", str(out),
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["mode"] == "solo" and data["num_matches_per_pair"] == 3 and data["solo"][0]["n"] == 3


def test_eval_cli_malformed_checkpoint_dir_is_a_config_error(tmp_path):
    cfg_path, _cfg, pt_path, ckpt_dir = _setup(tmp_path)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    (ckpt_dir / "meta.json").write_text(json.dumps({**meta, "policy_version": 7}))
    result = CliRunner().invoke(main, ["eval", "-c", str(cfg_path), "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}"])
    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("Config error:") and "policy_version" in result.stderr
    assert "Traceback" not in result.output


def test_eval_cli_invalid_meta_networks_is_a_config_error(tmp_path):
    cfg_path, _cfg, pt_path, ckpt_dir = _setup(tmp_path)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    bad = {**GRU_NETWORKS, "core": {"class": "colosseum.networks.cores.GRUCore", "kwargs": {"hidden_size": 0}}}
    (ckpt_dir / "meta.json").write_text(json.dumps({**meta, "networks": bad}))
    result = CliRunner().invoke(main, ["eval", "-c", str(cfg_path), "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}"])
    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("Config error:") and str(ckpt_dir) in result.stderr

    (ckpt_dir / "meta.json").write_text(json.dumps({**meta, "networks": {"bogus_key": 1}}))
    result = CliRunner().invoke(main, ["eval", "-c", str(cfg_path), "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}"])
    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("Config error:") and "networks" in result.stderr


def test_eval_cli_weights_of_another_architecture_are_a_config_error(tmp_path):
    cfg_path, _cfg, _pt_path, ckpt_dir = _setup(tmp_path)
    gru_pt = tmp_path / "gru.pt"
    torch.save(torch.load(ckpt_dir / "model.pt", weights_only=True), gru_pt)
    result = CliRunner().invoke(main, ["eval", "-c", str(cfg_path), "-a", f"a={gru_pt}"])
    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("Config error:") and "do not match" in result.stderr


def test_load_eval_model_never_modifies_the_run(tmp_path):
    """A live run's checkpoint dir is only read: no scan, no tmp-dir cleanup or restore."""
    _cfg_path, cfg, _pt_path, ckpt_dir = _setup(tmp_path)
    agent_dir = ckpt_dir.parent
    stale = agent_dir / ".tmp-ckpt_v9-0123abcd"
    stale.mkdir()
    old = time.time() - 10 * 3600
    os.utime(stale, (old, old))
    interrupted = agent_dir / ".tmp-old-ckpt_v5-0123abcd"
    interrupted.mkdir()
    before = sorted(p.name for p in agent_dir.iterdir())

    load_eval_model(ckpt_dir, cfg)

    assert sorted(p.name for p in agent_dir.iterdir()) == before
    assert stale.is_dir() and interrupted.is_dir() and not (agent_dir / "ckpt_v5").exists()


def test_load_checkpoint_dir_is_strict(tmp_path):
    _cfg_path, _cfg, _pt_path, ckpt_dir = _setup(tmp_path)
    loaded = load_checkpoint_dir(ckpt_dir)
    assert loaded["meta"]["policy_version"] == 3 and loaded["meta"]["networks"] == GRU_NETWORKS
    assert set(loaded["model_state"]) == set(torch.load(ckpt_dir / "model.pt", weights_only=True))
    (ckpt_dir / "meta.json").unlink()
    with pytest.raises(ConfigError, match="meta.json"):
        load_checkpoint_dir(ckpt_dir)
    with pytest.raises(ConfigError, match="not a directory"):
        load_checkpoint_dir(tmp_path / "nowhere")
