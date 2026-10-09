"""Distributed roles on SP2: scope check, fixed lineups, run dirs, exit codes (T6.4)."""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import re
import signal
from collections import Counter

import pytest

from colosseum.core.config import load_config
from colosseum.core.errors import ConfigError
from game_helpers import make_test_config, write_test_config


class _ExitedProcess:
    """mp.Process stand-in that has already exited with ``exitcode``."""

    exitcode = 0

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


@pytest.mark.parametrize(("host", "role"), [
    ("node-1", "workers-node-1"),
    ("my host.local/x", "workers-my-host.local-x"),
    ("-.odd", "workers-odd"),
    ("", "workers-host"),
])
def test_workers_role_includes_the_sanitized_hostname(monkeypatch, host, role):
    import colosseum.distributed as distributed

    monkeypatch.setattr(distributed.socket, "gethostname", lambda: host)
    assert distributed.workers_role() == role


def test_distributed_lineups_follow_layout_weights_and_rotate_agents():
    from colosseum.distributed import distributed_lineups, distributed_setup

    cfg = make_test_config("ffa", agents={"a": {}, "b": {}}, matchmaking={"layouts": {"2p": 0.25, "4p": 0.75}})
    setup = distributed_setup(cfg, ["a", "b"])
    lineups = distributed_lineups(setup, ["a", "b"], 4000, random.Random(0))
    counts = Counter(lu.layout for lu in lineups)
    assert abs(counts["2p"] / 4000 - 0.25) < 0.05
    for e, lineup in enumerate(lineups[:6]):
        expected = "ab"[e % 2]
        assert all(s.agent_id == expected and s.network_id == "latest" and s.collect for s in lineup.seats)
        assert len(lineup.seats) == len(setup.spec.layouts[lineup.layout])


def test_asymmetric_agents_are_refused_with_a_pointer_to_sp5(tmp_path, monkeypatch, restore_root_logging):
    import colosseum.distributed as distributed

    path = write_test_config(tmp_path / "asym.yaml", "asymmetric")
    overrides = {"run.dir": str(tmp_path / "runs")}
    with pytest.raises(ConfigError, match="SP5"):
        distributed.run_distributed_learner(str(path), "hunter", 0, "localhost:1", overrides=overrides)
    monkeypatch.setattr(distributed.mp, "Process", _ExitedProcess)
    with pytest.raises(ConfigError, match="SP5"):
        distributed.run_distributed_workers(str(path), "localhost:1", {"hunter": "localhost:2"}, overrides)
    assert not (tmp_path / "runs").exists()


def test_unknown_agents_are_refused_before_a_run_dir_exists(tmp_path, restore_root_logging):
    import colosseum.distributed as distributed

    path = write_test_config(tmp_path / "cfg.yaml", "turns")
    overrides = {"run.dir": str(tmp_path / "runs")}
    with pytest.raises(ConfigError, match=r"agents \['ghost'\] are not trainable agents.*agent_0"):
        distributed.run_distributed_learner(str(path), "ghost", 0, "localhost:1", overrides=overrides)
    assert not (tmp_path / "runs").exists()


def test_workers_entry_point_records_the_base_run_name(tmp_path, monkeypatch, restore_root_logging):
    import colosseum.distributed as distributed

    monkeypatch.setattr(mp, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(distributed.mp, "Process", _ExitedProcess)
    monkeypatch.setattr(distributed.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(distributed.socket, "gethostname", lambda: "node 7")
    path = write_test_config(tmp_path / "cfg.yaml", "turns", run={"dir": str(tmp_path / "runs")})
    assert distributed.run_distributed_workers(str(path), "localhost:1", {"agent_0": "localhost:2"}) == 0
    (root,) = (tmp_path / "runs").iterdir()
    resolved = load_config(root / "config.resolved.yaml")
    assert re.fullmatch(r"cfg-\d{8}-\d{6}", resolved.run.name)
    assert root.name == f"{resolved.run.name}-workers-node-7"


@pytest.mark.parametrize(("worker_exit", "expected"), [(0, 0), (3, 1)])
def test_run_workers_returns_an_exit_code(worker_exit, expected, tmp_path, monkeypatch, restore_root_logging,
                                          capsys):
    import colosseum.distributed as distributed

    class Exited(_ExitedProcess):
        exitcode = worker_exit

    monkeypatch.setattr(mp, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(distributed.mp, "Process", Exited)
    path = write_test_config(tmp_path / "cfg.yaml", "turns", run={"dir": str(tmp_path / "runs")})
    assert distributed.run_distributed_workers(str(path), "localhost:1", {"agent_0": "localhost:2"}) == expected
    assert ("worker-0 died (exit 3)" in capsys.readouterr().err) == (worker_exit == 3)


@pytest.fixture
def fake_learner_role(tmp_path, monkeypatch):
    """run_distributed_learner without gRPC servers: returns the config path."""
    pytest.importorskip("grpc")  # the patched modules import grpc (an optional extra)
    import colosseum.transport.grpc_transport as grpc_transport
    import colosseum.weight_store.grpc_store as grpc_store

    class FakeServer:
        def stop(self, grace):
            pass

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(grpc_transport, "serve_trajectory_receiver", lambda *a, **k: FakeServer())
    monkeypatch.setattr(grpc_store, "GRPCWeightStore", FakeStore)
    return write_test_config(tmp_path / "cfg.yaml", "turns", run={"dir": str(tmp_path / "runs")})


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_run_learner_returns_128_plus_signum(sig, fake_learner_role, monkeypatch, restore_root_logging):
    import colosseum.distributed as distributed
    import colosseum.learner.learner as learner_module

    def fake_learner_process(*, stop_event, **kwargs):
        os.kill(os.getpid(), sig)
        assert stop_event.wait(10)

    monkeypatch.setattr(learner_module, "learner_process", fake_learner_process)
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    assert distributed.run_distributed_learner(str(fake_learner_role), "agent_0", 0, "localhost:1") == 128 + sig
    assert {s: signal.getsignal(s) for s in before} == before


def test_run_learner_setup_failure_leaves_signal_handlers_untouched(fake_learner_role, monkeypatch,
                                                                     restore_root_logging, restore_global_rng):
    import colosseum.distributed as distributed

    def failing_seed(seed):
        raise RuntimeError("seeding failed")

    monkeypatch.setattr(distributed, "apply_global_seed", failing_seed)
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(RuntimeError, match="seeding failed"):
        distributed.run_distributed_learner(str(fake_learner_role), "agent_0", 0, "localhost:1")
    assert {s: signal.getsignal(s) for s in before} == before


def test_agent_rotation_continues_across_the_workers_of_a_machine(tmp_path, monkeypatch, restore_root_logging):
    """One env per worker and two agents: worker 0 plays agent a, worker 1 agent b."""
    import colosseum.distributed as distributed

    started: list[dict] = []

    class Recorded(_ExitedProcess):
        def __init__(self, *args, kwargs=None, **other):
            started.append(kwargs)

    monkeypatch.setattr(mp, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(distributed.mp, "Process", Recorded)
    path = write_test_config(tmp_path / "cfg.yaml", "turns", run={"dir": str(tmp_path / "runs")},
                             agents={"a": {}, "b": {}}, rollout={"num_workers": 2, "envs_per_worker": 1})
    assert distributed.run_distributed_workers(str(path), "localhost:1", {"a": "h:1", "b": "h:2"}) == 0
    owners = [{s.agent_id for s in kw["lineups"][0].seats} for kw in started]
    assert owners == [{"a"}, {"b"}]
