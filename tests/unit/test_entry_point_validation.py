"""Entry points that build models run validate_config first (T4.5 fix round 1)."""
import multiprocessing as mp

import pytest
import yaml
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.errors import ConfigError


def _misordered_config(tmp_path):
    path = tmp_path / "misordered.yaml"
    path.write_text(yaml.safe_dump({
        "env": {"env_class": "helpers.TwelveUnitEnv", "num_players": 1, "kwargs": {"use_mask": False}},
        "networks": {
            "encoder_class": "helpers.SimpleEncoder",
            "policy_class": "helpers.MisorderedTwelveHeadPolicy",
            "value_class": "helpers.SimpleValue",
        },
    }))
    return path


def _fail(name):
    def _boom(*_args, **_kwargs):
        raise AssertionError(f"{name} ran before validate_config rejected the config")
    return _boom


def test_bc_cli_rejects_misordered_policy(tmp_path):
    data = tmp_path / "data.pt"
    data.write_bytes(b"")  # never read: validation fails first
    out = tmp_path / "out.pt"
    result = CliRunner().invoke(main, ["bc", "-c", str(_misordered_config(tmp_path)),
                                       "-d", str(data), "-o", str(out)])
    assert isinstance(result.exception, ConfigError), result.output
    assert "action space" in str(result.exception)
    assert not out.exists()


def test_eval_cli_rejects_misordered_policy(tmp_path):
    missing = tmp_path / "never_loaded.pt"
    result = CliRunner().invoke(main, ["eval", "-c", str(_misordered_config(tmp_path)),
                                       "-a", f"x:{missing}", "-a", f"y:{missing}"])
    assert isinstance(result.exception, ConfigError), result.output
    assert "action space" in str(result.exception)


def test_distributed_learner_rejects_misordered_policy_before_serving(tmp_path, monkeypatch):
    from colosseum import distributed
    from colosseum.transport import grpc_transport

    monkeypatch.setattr(grpc_transport, "serve_trajectory_receiver", _fail("serve_trajectory_receiver"))
    with pytest.raises(ConfigError, match="action space"):
        distributed.run_distributed_learner(str(_misordered_config(tmp_path)), "agent_0", 0, "localhost:1")


def test_distributed_workers_reject_misordered_policy_before_spawning(tmp_path, monkeypatch):
    from colosseum import distributed

    monkeypatch.setattr(mp, "set_start_method", _fail("mp.set_start_method"))
    monkeypatch.setattr(distributed.mp, "Process", _fail("mp.Process"))
    with pytest.raises(ConfigError, match="action space"):
        distributed.run_distributed_workers(str(_misordered_config(tmp_path)), "localhost:1",
                                            {"agent_0": "localhost:2"})
