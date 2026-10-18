"""SP5 config (T1.4, spec blocks 9.1 and 4.10): the distributed section's defaults and checks (global only), the new
learner keys with their per-agent override, learner.device values, the transport translation (one WARNING; old +
new key = ConfigError), --set, and resolved configs of SP2-SP4 runs."""
from __future__ import annotations

import json
import logging

import pytest
import yaml

from colosseum.core.config import (
    ColosseumConfig,
    DistributedBundleConfig,
    DistributedConfig,
    LearnerPlacementConfig,
    load_config,
)
from colosseum.core.errors import ConfigError
from game_helpers import SP3_RUN

SPEC_DEFAULTS = {
    "port": 47100, "data_port": 47102, "insecure": False, "heartbeat_sec": 1.0, "node_lost_after_sec": 10.0,
    "learner_relocate_after_sec": 120.0, "hub_lost_timeout_sec": 1800.0, "hub_lost_collect_sec": 300.0,
    "shutdown_grace_sec": 20.0, "connect_timeout_sec": 5.0, "block_lease_sec": 600.0, "relay": True,
    "results_batch": 256, "results_flush_sec": 0.25, "node_result_buffer_mb": 256.0, "max_message_mb": 64,
    "bundle": {"include": [], "exclude": [], "max_file_mb": 10.0}, "requirements": None,
}


def _data(**sections) -> dict:
    return {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}, **sections}


def _load(tmp_path, data: dict, overrides: dict | None = None) -> ColosseumConfig:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(path, overrides)


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and r.name.startswith("colosseum")]


def test_the_distributed_defaults_are_the_specs():
    cfg = ColosseumConfig.model_validate(_data())
    assert cfg.distributed.model_dump() == SPEC_DEFAULTS
    assert cfg.distributed.bundle == DistributedBundleConfig()
    assert not hasattr(cfg, "transport") and "transport" not in ColosseumConfig.model_fields


@pytest.mark.parametrize("bad, match", [
    ({"data_port": 47100}, "must differ"),
    ({"port": -1}, "port"),
    ({"data_port": 70000}, "data_port"),
    ({"heartbeat_sec": 0}, "heartbeat_sec"),
    ({"node_lost_after_sec": 1.0}, "larger than distributed.heartbeat_sec"),
    ({"learner_relocate_after_sec": 5.0}, "at least distributed.node_lost_after_sec"),
    ({"results_batch": 0}, "results_batch"),
    ({"max_message_mb": 0}, "max_message_mb"),
    ({"bundle": {"max_file_mb": 0}}, "max_file_mb"),
    ({"hub_port": 1}, "hub_port"),
])
def test_the_distributed_section_is_checked(bad, match):
    with pytest.raises(ValueError, match=match):
        ColosseumConfig.model_validate(_data(distributed=bad))


@pytest.mark.parametrize("ports", [{"port": 0}, {"data_port": 0}, {"port": 0, "data_port": 0},
                                   {"port": 0, "data_port": 47100}, {"port": 65535, "data_port": 0}])
def test_port_zero_is_an_ephemeral_port_and_skips_the_port_set_check(ports):
    """Pre-flight ruling P-PF2-8: 0 = an ephemeral port chosen at bind (tests, train --listen); port != data_port is
    checked only when both are fixed."""
    cfg = ColosseumConfig.model_validate(_data(distributed=ports))
    assert {k: getattr(cfg.distributed, k) for k in ports} == ports


def test_the_distributed_section_is_global_only_and_set_reaches_it(tmp_path):
    cfg = _load(tmp_path, _data(), {"distributed.port": 50000, "distributed.bundle.exclude": ["data/"],
                                    "distributed.relay": False})
    assert (cfg.distributed.port, cfg.distributed.bundle.exclude, cfg.distributed.relay) == (50000, ["data/"], False)
    with pytest.raises(ConfigError):
        _load(tmp_path, _data(agents={"a": {}}), {"agents.a.distributed.port": 1})
    with pytest.raises(ConfigError, match="Unknown config key 'distributed.nonsense'"):
        _load(tmp_path, _data(), {"distributed.nonsense": 1})


def test_the_new_learner_keys_have_defaults_and_a_per_agent_override(tmp_path):
    cfg = _load(tmp_path, _data(agents={"a": {}, "b": {"learner": {"device": "cpu", "max_policy_lag": 4,
                                                                    "placement": {"node": "laptop"}}}}))
    a, b = cfg.get_agent_config("a").learner, cfg.get_agent_config("b").learner
    assert (a.max_policy_lag, a.recovery_interval_sec, a.placement) == (None, 300.0, LearnerPlacementConfig())
    assert a.placement.model_dump() == {"node": None, "tags": [], "strict": False}
    assert (b.device, b.max_policy_lag, b.placement.node, b.placement.strict) == ("cpu", 4, "laptop", False)
    over = _load(tmp_path, _data(agents={"a": {}}), {"agents.a.learner.placement.tags": ["gpu", "big"],
                                                     "agents.a.learner.placement.strict": True,
                                                     "learner.recovery_interval_sec": 60})
    learner = over.get_agent_config("a").learner
    assert (learner.placement.tags, learner.placement.strict, learner.recovery_interval_sec) == (["gpu", "big"], True,
                                                                                                 60.0)
    for bad in ({"max_policy_lag": -1}, {"recovery_interval_sec": 0}, {"placement": {"host": "x"}}):
        with pytest.raises(ValueError):
            ColosseumConfig.model_validate(_data(learner=bad))


@pytest.mark.parametrize("device, ok", [("auto", True), ("cpu", True), ("cuda", True), ("cuda:1", True),
                                         ("mps", True), ("gpu", False), ("cuda:x", False), ("CPU", False)])
def test_learner_device_values(device, ok):
    if ok:
        assert ColosseumConfig.model_validate(_data(learner={"device": device})).learner.device == device
    else:
        with pytest.raises(ValueError, match="auto, cpu, cuda, cuda:N or mps"):
            ColosseumConfig.model_validate(_data(learner={"device": device}))


def test_the_transport_section_is_translated_with_one_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = _load(tmp_path, _data(transport={"mode": "grpc", "grpc_port": 50051, "grpc_max_message_mb": 128}))
    assert cfg.distributed.max_message_mb == 128 and cfg.distributed.port == 47100
    (warning,) = _warnings(caplog)
    assert "transport.mode, transport.grpc_port ignored" in warning
    assert "transport.grpc_max_message_mb read as distributed.max_message_mb: 128" in warning
    assert "transport" not in cfg.model_dump() and "transport" not in json.dumps(cfg.model_dump(mode="json"))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        only_mode = _load(tmp_path, _data(transport={"mode": "local"}))
    assert only_mode.distributed.max_message_mb == 64 and len(_warnings(caplog)) == 1
    assert "grpc_max_message_mb" not in _warnings(caplog)[0]


def test_old_and_new_message_size_together_are_an_error(tmp_path):
    with pytest.raises(ConfigError, match="keep only distributed.max_message_mb"):
        _load(tmp_path, _data(transport={"grpc_max_message_mb": 32}, distributed={"max_message_mb": 16}))
    with pytest.raises(ConfigError, match="unknown keys"):
        _load(tmp_path, _data(transport={"grpc_host": "x"}))


def test_set_still_accepts_the_old_transport_keys(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = _load(tmp_path, _data(), {"transport.grpc_max_message_mb": 32, "transport.mode": "local"})
    assert cfg.distributed.max_message_mb == 32 and len(_warnings(caplog)) == 1


def test_a_resolved_config_of_an_sp3_run_loads_and_is_written_back_without_transport(tmp_path, caplog):
    """SP3 / SP4 run dirs hold config.resolved.yaml with a transport section: it loads (one WARNING) and the new
    resolved config has a distributed section instead."""
    with caplog.at_level(logging.WARNING):
        cfg = load_config(SP3_RUN / "config.resolved.yaml")
    assert len(_warnings(caplog)) == 1 and cfg.distributed.max_message_mb == 64
    dumped = cfg.model_dump(mode="json", by_alias=True)
    assert "transport" not in dumped and dumped["distributed"] == SPEC_DEFAULTS
    assert ColosseumConfig.model_validate(dumped).distributed == DistributedConfig()
