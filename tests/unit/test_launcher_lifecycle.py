"""Launcher lifecycle in-process (T6.5): validation once, partial-start teardown, exit codes
from child exits, drains that never block on a dead producer, CLI config errors."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import struct
import sys
import threading
import time

import pytest
import yaml
from click.testing import CliRunner
from fake_wandb import FakeWandb

import colosseum.launcher as launcher_module
from colosseum.cli import main
from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import ColosseumConfig, load_config
from colosseum.launcher import Launcher
from helpers import example_config, make_test_run_dir

TTT = "examples.tic_tac_toe"


def ttt_data(**training) -> dict:
    return {
        "env": {"env_class": f"{TTT}.env.TicTacToeEnv", "num_players": 2},
        "networks": {
            "encoder_class": f"{TTT}.networks.TicTacToeEncoder",
            "policy_class": f"{TTT}.networks.TicTacToePolicy",
            "value_class": f"{TTT}.networks.TicTacToeValue",
        },
        "learner": {"device": "cpu"},
        "rollout": {"num_workers": 1, "envs_per_worker": 2},
        "training": training,
    }


class _Stop(Exception):
    pass


def fake_process_class(started: list, fail_on_worker: bool = True):
    """mp.Process stand-in: learners 'run' until stop_event is set; starting a worker raises."""

    class FakeProcess:
        def __init__(self, target, kwargs, name=None, daemon=None):
            self.target, self.kwargs, self.name = target, kwargs, name
            self.joined = self.terminated = False

        def start(self):
            if fail_on_worker and self.target is launcher_module._worker_target:
                raise _Stop
            started.append(self)

        def is_alive(self):
            return not self.kwargs["stop_event"].is_set() and not self.terminated

        @property
        def exitcode(self):
            return None if self.is_alive() else 0

        def join(self, timeout=None):
            self.joined = True

        def terminate(self):
            self.terminated = True

        kill = terminate

    return FakeProcess


# ---------------------------------------------------------------------------
# Validation runs once on the CLI path (carried T6.2 item)
# ---------------------------------------------------------------------------


@pytest.fixture
def count_validations(monkeypatch):
    import colosseum.core.registry as registry

    calls: list[str] = []
    real = registry.validate_config

    def counting(config):
        calls.append(config.env.env_class)
        return real(config)

    monkeypatch.setattr(registry, "validate_config", counting)
    return calls


def test_run_training_validates_agent_configs_once(tmp_path, monkeypatch, count_validations,
                                                   restore_root_logging, restore_global_rng):
    monkeypatch.setattr(launcher_module.mp, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher_module.mp, "Process", fake_process_class([]))
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(ttt_data()))
    with pytest.raises(_Stop):
        launcher_module.run_training(str(path), {"run.dir": str(tmp_path / "runs"), "run.name": "once"})
    assert len(count_validations) == 1


def test_direct_launch_still_validates(tmp_path, monkeypatch, count_validations):
    monkeypatch.setattr(launcher_module.mp, "Process", fake_process_class([]))
    cfg = ColosseumConfig.model_validate(ttt_data())
    with pytest.raises(_Stop):
        Launcher(cfg, make_test_run_dir(cfg, tmp_path)).launch()
    assert len(count_validations) == 1


# ---------------------------------------------------------------------------
# A failed start tears down what was started (carried T6.3 item)
# ---------------------------------------------------------------------------


def test_failed_start_stops_started_children_and_closes_metrics(tmp_path, monkeypatch, caplog):
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    started: list = []
    monkeypatch.setattr(launcher_module.mp, "Process", fake_process_class(started))
    data = ttt_data()
    data["agents"] = {"alpha": None, "beta": None}
    data["metrics"] = {"use_wandb": True}
    cfg = ColosseumConfig.model_validate(data)
    run = make_test_run_dir(cfg, tmp_path)
    launcher = Launcher(cfg, run)
    sigint_before, sigterm_before = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)

    with caplog.at_level(logging.ERROR, logger="colosseum.launcher"), pytest.raises(_Stop):
        launcher.launch()

    assert [p.name for p in started] == ["learner-alpha", "learner-beta"]
    assert launcher._stop_event.is_set()  # the started learners were told to stop ...
    assert not any(p.is_alive() for p in started)
    assert launcher._supervisor.names == ["learner-alpha", "learner-beta"]  # ... and were supervised
    assert "Starting the child processes failed; stopping the 2 already started" in caplog.text
    assert launcher._metrics_writer.closed  # the hub was closed with its file
    assert '"kind": "system"' in run.metrics_path.read_text()
    assert fake_wandb.finished
    assert signal.getsignal(signal.SIGINT) == sigint_before
    assert signal.getsignal(signal.SIGTERM) == sigterm_before


def test_stop_requested_by_the_caller_is_a_clean_exit(tmp_path, monkeypatch, caplog):
    """scripts/bench_throughput.py stops a run by setting stop_event: children exiting then
    is not a failure."""
    monkeypatch.setattr(launcher_module.mp, "Process", fake_process_class([], fail_on_worker=False))
    cfg = ColosseumConfig.model_validate(ttt_data())
    cfg.training.total_timesteps = 10**9
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    timer = threading.Timer(0.5, launcher._stop_event.set)
    timer.start()
    try:
        with caplog.at_level(logging.INFO, logger="colosseum.launcher"):
            assert launcher.launch() == 0
    finally:
        timer.cancel()
    assert "Stop requested" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_signal_racing_with_the_stop_check_still_gives_its_exit_code(tmp_path, monkeypatch):
    """stop_event and received_signal set together (a signal right after the loop's signal
    check): the run reports 128 + signum, not a clean caller stop."""
    monkeypatch.setattr(launcher_module.mp, "Process", fake_process_class([], fail_on_worker=False))
    cfg = ColosseumConfig.model_validate(ttt_data())
    cfg.training.total_timesteps = 10**9
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))

    def signal_arrives():
        launcher._stop_event.set()
        launcher._supervisor.received_signal = signal.SIGTERM

    timer = threading.Timer(0.5, signal_arrives)
    timer.start()
    try:
        assert launcher.launch() == 128 + signal.SIGTERM
    finally:
        timer.cancel()


def test_failed_start_error_is_not_hidden_by_a_failing_teardown(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(launcher_module.mp, "Process", fake_process_class([]))
    cfg = ColosseumConfig.model_validate(ttt_data())
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))

    def failing_drain():
        raise OSError("disk full")

    monkeypatch.setattr(launcher, "_drain_all_checkpoints", failing_drain)
    with caplog.at_level(logging.ERROR, logger="colosseum.launcher"), pytest.raises(_Stop) as excinfo:
        launcher.launch()
    assert "Shutdown afterwards also failed: OSError('disk full')" in getattr(excinfo.value, "__notes__", [])
    assert "Shutdown after _Stop failed too" in caplog.text and "disk full" in caplog.text
    assert launcher._metrics_writer.closed


# ---------------------------------------------------------------------------
# Exit codes from child exits (carried T2.5 item a) and the single grace constant
# ---------------------------------------------------------------------------


def _exit_after_stop(stop_event, code: int, ready) -> None:
    ready.set()
    stop_event.wait(60)
    os._exit(code)


def _ignore_stop(ready) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # only kill() ends it
    ready.set()
    time.sleep(120)


def _launcher_with(tmp_path, children: dict) -> Launcher:
    """A launcher supervising real children; returns once every child reported it is ready."""
    cfg = load_config(example_config("tic_tac_toe.yaml"))
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    ctx = mp.get_context("spawn")
    ready_events = []
    for name, (target, args) in children.items():
        ready = ctx.Event()
        ready_events.append(ready)
        args = (*(launcher._stop_event if a == "stop" else a for a in args), ready)
        proc = ctx.Process(target=target, args=args, name=name, daemon=True)
        proc.start()
        launcher._supervisor.add(name, proc)
    assert all(ready.wait(60) for ready in ready_events), "children did not start"
    return launcher


def test_nonzero_child_exit_fails_a_run_that_reached_its_budget(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(launcher_module, "SHUTDOWN_GRACE_SEC", 1.5)
    launcher = _launcher_with(tmp_path, {
        "learner-a": (_exit_after_stop, ("stop", 0)),
        "worker-0": (_exit_after_stop, ("stop", 3)),
        "worker-1": (_ignore_stop, ()),
    })
    start = time.monotonic()
    with caplog.at_level(logging.INFO, logger="colosseum.launcher"):
        killed = launcher._shutdown()
        code = launcher._exit_code_after_shutdown(0, killed)
    # The main process waits SHUTDOWN_GRACE_SEC (patched here), then terminates and kills.
    assert time.monotonic() - start < 1.5 + 5.0
    assert killed == ["worker-1"]
    assert code == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors == [f"worker-0 died (exit 3), see {launcher._run_dir.logs / 'worker-0.log'}"]
    assert "worker-1" in caplog.text and "did not stop within" in caplog.text  # killed: a warning only


def test_signal_exit_code_stays_when_a_child_exits_nonzero(tmp_path, caplog):
    launcher = _launcher_with(tmp_path, {"worker-0": (_exit_after_stop, ("stop", 2))})
    killed = launcher._shutdown()
    with caplog.at_level(logging.WARNING, logger="colosseum.launcher"):
        assert launcher._exit_code_after_shutdown(130, killed) == 130
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "worker-0 died (exit 2)" in caplog.text


def test_clean_children_keep_exit_code_zero(tmp_path, caplog):
    launcher = _launcher_with(tmp_path, {"learner-a": (_exit_after_stop, ("stop", 0))})
    with caplog.at_level(logging.INFO, logger="colosseum.launcher"):
        assert launcher._exit_code_after_shutdown(0, launcher._shutdown()) == 0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# A learner killed mid-message never blocks the main process (carried T5.3 item)
# ---------------------------------------------------------------------------


def _die_mid_message(q) -> None:
    """Send one complete item, then write a message header announcing 1 MiB plus a few bytes
    into the queue's pipe and die (as a learner OOM-killed while its feeder thread flushes
    the final checkpoint)."""
    from multiprocessing.reduction import ForkingPickler

    q._sem.acquire()  # as put() does, then the feeder's write
    q._writer.send_bytes(ForkingPickler.dumps({"agent_id": "agent_0", "policy_version": 1}))
    os.write(q._writer.fileno(), struct.pack("!i", 1 << 20) + b"x" * 1000)
    os.kill(os.getpid(), signal.SIGKILL)


def test_drain_skips_checkpoint_queue_of_a_learner_killed_mid_message(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(launcher_module, "SHUTDOWN_GRACE_SEC", 1.0)
    cfg = load_config(example_config("tic_tac_toe.yaml"))
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    launcher._coordinator = Coordinator(cfg, checkpoint_dir=tmp_path / "ckpt")
    ctx = mp.get_context("spawn")
    cq = ctx.Queue()
    launcher._checkpoint_queues = {"agent_0": cq}
    launcher._all_queues = [cq]
    saved: list[dict] = []
    launcher._save_checkpoint = saved.append
    proc = ctx.Process(target=_die_mid_message, args=(cq,), daemon=True)
    proc.start()
    launcher._supervisor.add("learner-agent_0", proc)
    proc.join(60)
    assert proc.exitcode == -signal.SIGKILL

    done = threading.Event()

    def drain_and_shutdown():
        launcher._drain_all_checkpoints()
        launcher._shutdown()
        done.set()

    start = time.monotonic()
    with caplog.at_level(logging.ERROR, logger="colosseum.launcher"):
        threading.Thread(target=drain_and_shutdown, daemon=True).start()
        assert done.wait(20), "the main process blocked on a partial checkpoint payload"
    assert time.monotonic() - start < 1.0 + 3.0  # given up at the (patched) grace deadline
    assert ("An incomplete item on the checkpoint-agent_0 queue was never completed "
            "(learner-agent_0 died while sending)") in caplog.text
    assert saved == [{"agent_id": "agent_0", "policy_version": 1}]  # the complete item is kept


def _slow_to_unpickle() -> dict:
    time.sleep(4.0)  # far longer than the old fixed 1 s give-up
    return {"slow": True}


class SlowToUnpickle:
    def __reduce__(self):
        return (_slow_to_unpickle, ())


def _send_slow_payload_and_exit(q) -> None:
    q.put({"agent_id": "agent_0", "policy_version": 2, "payload": SlowToUnpickle()})
    q.close()
    q.join_thread()  # flushed into the pipe; exit 0 before the main process finished reading


def test_slow_read_from_an_exited_learner_is_kept_until_the_deadline(tmp_path, monkeypatch, caplog):
    """A complete final payload still being read/unpickled when its learner has exited is
    not dropped: the shutdown waits for it until its deadline."""
    monkeypatch.setattr(launcher_module, "SHUTDOWN_GRACE_SEC", 6.0)
    cfg = load_config(example_config("tic_tac_toe.yaml"))
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    ctx = mp.get_context("spawn")
    cq = ctx.Queue()
    launcher._checkpoint_queues = {"agent_0": cq}
    launcher._all_queues = [cq]
    saved: list[dict] = []
    launcher._save_checkpoint = saved.append
    proc = ctx.Process(target=_send_slow_payload_and_exit, args=(cq,), daemon=True)
    proc.start()
    launcher._supervisor.add("learner-agent_0", proc)
    proc.join(60)
    assert proc.exitcode == 0
    with caplog.at_level(logging.ERROR, logger="colosseum.launcher"):
        assert launcher._drain(cq, "checkpoint-agent_0", lambda: ["learner-agent_0"]) == []  # read in progress
        launcher._shutdown()
    assert saved == [{"agent_id": "agent_0", "policy_version": 2, "payload": {"slow": True}}]
    assert "never completed" not in caplog.text


# ---------------------------------------------------------------------------
# Every CLI command turns a config error into one line and exit code 1 (D10)
# ---------------------------------------------------------------------------


@pytest.fixture
def bad_key_config(tmp_path):
    data = yaml.safe_load(example_config("tic_tac_toe.yaml").read_text())
    data["rollout"]["num_worker"] = 3
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize("args", [
    ["train"],
    ["validate"],
    ["run-learner", "--weight-store", "localhost:1"],
    ["run-workers", "--weight-store", "localhost:1", "-l", "agent_0=localhost:2"],
    ["bc", "-d", "{data}", "-o", "{out}"],
    ["eval", "-a", "x:{out}", "-a", "y:{out}"],
], ids=lambda a: a[0])
def test_cli_config_error_is_one_line_and_exit_1(args, bad_key_config, tmp_path, monkeypatch, restore_root_logging):
    monkeypatch.setattr(launcher_module.mp, "set_start_method", lambda *a, **k: None)
    data = tmp_path / "data.pt"
    data.write_bytes(b"")
    args = [a.format(data=data, out=tmp_path / "out.pt") for a in args]
    result = CliRunner().invoke(main, [args[0], "-c", str(bad_key_config), *args[1:]])
    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("Config error:") and "num_worker" in result.stderr
    assert "Traceback" not in result.output


def test_serve_trajectory_command_is_gone():
    assert "serve-trajectory" not in main.commands
    assert {"train", "validate", "bc", "eval", "run-learner", "run-workers", "serve-weight-store"} <= set(main.commands)


def test_main_puts_cwd_first_on_sys_path(tmp_path, monkeypatch):
    """User code next to the config (cwd) is importable, also from the console script (R5-19)."""
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(tmp_path)])
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(ttt_data()))
    result = CliRunner().invoke(main, ["validate", "-c", str(cfg)])
    assert result.exit_code == 0, result.output
    assert sys.path[0] == os.getcwd() == str(tmp_path)  # cwd is tmp_path (conftest)


@pytest.mark.parametrize(("worker_exit", "expected"), [(0, 0), (3, 1)])
def test_run_workers_returns_an_exit_code(worker_exit, expected, tmp_path, monkeypatch, restore_root_logging,
                                          capsys):
    import colosseum.distributed as distributed

    class ExitedProcess:
        exitcode = worker_exit

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(mp, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(distributed.mp, "Process", ExitedProcess)
    data = ttt_data()
    data["run"] = {"dir": str(tmp_path / "runs")}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    assert distributed.run_distributed_workers(str(path), "localhost:1", {"agent_0": "localhost:2"}) == expected
    # The entry point logs to stderr (its own handlers replace pytest's capture handler).
    assert ("[ERROR] workers-main colosseum.distributed: worker-0 died (exit 3)" in capsys.readouterr().err) == (
        worker_exit == 3)


# ---------------------------------------------------------------------------
# Exit codes of run-learner / run-workers / train for signals and early Ctrl-C
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_run_learner_returns_128_plus_signum(sig, tmp_path, monkeypatch, restore_root_logging):
    import colosseum.distributed as distributed
    import colosseum.learner.learner as learner_module
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

    def fake_learner_process(*, stop_event, **kwargs):
        os.kill(os.getpid(), sig)
        assert stop_event.wait(10)

    monkeypatch.setattr(grpc_transport, "serve_trajectory_receiver", lambda *a, **k: FakeServer())
    monkeypatch.setattr(grpc_store, "GRPCWeightStore", FakeStore)
    monkeypatch.setattr(learner_module, "learner_process", fake_learner_process)
    data = ttt_data()
    data["run"] = {"dir": str(tmp_path / "runs")}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    assert distributed.run_distributed_learner(str(path), "agent_0", 0, "localhost:1") == 128 + sig
    assert {s: signal.getsignal(s) for s in before} == before  # handlers restored


def _raise_keyboard_interrupt(*args, **kwargs):
    raise KeyboardInterrupt


@pytest.mark.parametrize(("args", "target"), [
    (["train"], "colosseum.launcher.run_training"),
    (["run-learner", "--weight-store", "localhost:1"], "colosseum.distributed.run_distributed_learner"),
    (["run-workers", "--weight-store", "localhost:1", "-l", "agent_0=localhost:2"],
     "colosseum.distributed.run_distributed_workers"),
], ids=lambda a: a[0] if isinstance(a, list) else None)
def test_ctrl_c_before_the_run_handles_signals_exits_130(args, target, tmp_path, monkeypatch):
    monkeypatch.setattr(target, _raise_keyboard_interrupt)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(ttt_data()))
    result = CliRunner().invoke(main, [args[0], "-c", str(cfg), *args[1:]])
    assert result.exit_code == 130, result.output
    assert result.stderr == "Interrupted\n"
    assert "Aborted" not in result.output


@pytest.mark.parametrize(("args", "target"), [
    (["train"], "colosseum.launcher.run_training"),
    (["run-learner", "--weight-store", "localhost:1"], "colosseum.distributed.run_distributed_learner"),
    (["run-workers", "--weight-store", "localhost:1", "-l", "agent_0=localhost:2"],
     "colosseum.distributed.run_distributed_workers"),
], ids=lambda a: a[0] if isinstance(a, list) else None)
def test_cli_exits_with_the_run_exit_code(args, target, tmp_path, monkeypatch):
    monkeypatch.setattr(target, lambda *a, **k: 143)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(ttt_data()))
    assert CliRunner().invoke(main, [args[0], "-c", str(cfg), *args[1:]]).exit_code == 143
