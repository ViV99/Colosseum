"""SP2 launcher lifecycle in-process (SP1 guarantees, T7.1), on SP5's hub (T2.5 b; the hub is the evolved
launcher, R5): validation once, partial-start teardown, exit codes from child exits, CLI config errors, metrics
teardown. The hub's children are the local node and the SP4 pulse (R6); processes are faked through the hub's spawn
context (``Hub._mp``)."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner
from fake_wandb import FakeWandb

import colosseum.hub.hub as hub_module
from colosseum.arena.matchlog import Commit
from colosseum.cli import main
from colosseum.core.config import ColosseumConfig
from colosseum.core.types import MatchResult
from game_helpers import make_test_config, make_test_run_dir, scripted_agent
from hub_kit import make_hub

TTT = "examples.tic_tac_toe"


def ttt_data(**training) -> dict:
    """A small two-seat toy-game config as raw data (one worker, two envs, CPU learner)."""
    data = make_test_config("turns").model_dump(mode="json", by_alias=True)
    data["rollout"].update(num_workers=1, envs_per_worker=2)
    data["training"] = {**data["training"], "seed": None, **training}
    data["metrics"] = {**data["metrics"], "console_interval_sec": 10.0}
    return data


class _Stop(Exception):
    pass


class FakeContext:
    """The hub's spawn context with ``Process`` faked: a child 'runs' until the hub's stop event (the first ``Event``
    this context made) is set; starting a child named in ``fail`` raises ``_Stop``. Events and queues are real."""

    def __init__(self, started: list, fail: tuple[str, ...] = ("node",)) -> None:
        self._real = mp.get_context("spawn")
        self.started, self.fail = started, set(fail)
        self.stop_event = None

    def Event(self):
        event = self._real.Event()
        if self.stop_event is None:
            self.stop_event = event
        return event

    def Queue(self, *args, **kwargs):
        return self._real.Queue(*args, **kwargs)

    def Process(self, target, kwargs, name=None, daemon=None):
        return FakeProcess(self, target, kwargs, name)


class FakeProcess:
    def __init__(self, ctx: FakeContext, target, kwargs, name):
        self.ctx, self.target, self.kwargs, self.name = ctx, target, kwargs, name
        self.joined = self.terminated = False

    def start(self):
        if self.name in self.ctx.fail:
            raise _Stop
        self.ctx.started.append(self)

    def is_alive(self):
        return not self.ctx.stop_event.is_set() and not self.terminated

    @property
    def exitcode(self):
        return None if self.is_alive() else 0

    def join(self, timeout=None):
        self.joined = True

    def terminate(self):
        self.terminated = True

    kill = terminate


def fake_hub(config, run, started: list, fail: tuple[str, ...] = ("node",)):
    """A ``train`` hub whose children are ``FakeProcess``es."""
    hub = make_hub(config, run)
    ctx = FakeContext(started, fail)
    ctx.stop_event = hub._stop_event
    hub._mp = ctx
    return hub


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
    ctx = FakeContext([])
    monkeypatch.setattr(hub_module, "mp", SimpleNamespace(set_start_method=lambda *args, **kwargs: None,
                                                          get_context=lambda *args, **kwargs: ctx))
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(ttt_data()))
    with pytest.raises(_Stop):
        hub_module.run_training(str(path), {"run.dir": str(tmp_path / "runs"), "run.name": "once"})
    assert len(count_validations) == 1


def test_direct_launch_still_validates(tmp_path, count_validations):
    cfg = ColosseumConfig.model_validate(ttt_data())
    with pytest.raises(_Stop):
        fake_hub(cfg, make_test_run_dir(cfg, tmp_path), []).launch()
    assert len(count_validations) == 1


# ---------------------------------------------------------------------------
# A failed start tears down what was started (carried T6.3 item)
# ---------------------------------------------------------------------------


def test_failed_start_stops_started_children_and_closes_metrics(tmp_path, monkeypatch, caplog):
    """The local node starts, the pulse (R6) fails to start: the node is stopped and the sinks are closed."""
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    started: list = []
    data = ttt_data()
    data["agents"] = {"alpha": None, "beta": None, "rnd": scripted_agent()}
    data["metrics"] = {"use_wandb": True}
    data["evaluation"] = {"enabled": True}
    cfg = ColosseumConfig.model_validate(data)
    run = make_test_run_dir(cfg, tmp_path)
    hub = fake_hub(cfg, run, started, fail=("pulse",))
    sigint_before, sigterm_before = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)

    with caplog.at_level(logging.ERROR, logger="colosseum.hub.hub"), pytest.raises(_Stop):
        hub.launch()

    assert [p.name for p in started] == ["node"]
    assert hub._stop_event.is_set()  # the started node was told to stop ...
    assert not any(p.is_alive() for p in started)
    assert hub._supervisor.names == ["node"]  # ... and was supervised
    assert "Starting the local node or the pulse failed; stopping the 1 already started" in caplog.text
    assert hub._metrics_writer.closed  # the metrics hub was closed with its file
    assert '"kind": "system"' in run.metrics_path.read_text()
    assert fake_wandb.finished
    assert signal.getsignal(signal.SIGINT) == sigint_before
    assert signal.getsignal(signal.SIGTERM) == sigterm_before


def test_stop_requested_by_the_caller_is_a_clean_exit(tmp_path, caplog):
    """scripts/bench_throughput.py stops a run with ``request_stop``: children exiting then is not a failure."""
    cfg = ColosseumConfig.model_validate(ttt_data())
    cfg.training.total_timesteps = 10**9
    hub = fake_hub(cfg, make_test_run_dir(cfg, tmp_path), [], fail=())
    timer = threading.Timer(0.5, hub.request_stop)
    timer.start()
    try:
        with caplog.at_level(logging.INFO, logger="colosseum.hub.hub"):
            assert hub.launch() == 0
    finally:
        timer.cancel()
    assert "Stop requested" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_signal_racing_with_the_stop_check_still_gives_its_exit_code(tmp_path):
    """A stop request and a received signal together (a signal right after the loop's signal check): the run
    reports 128 + signum, not a clean caller stop."""
    cfg = ColosseumConfig.model_validate(ttt_data())
    cfg.training.total_timesteps = 10**9
    hub = fake_hub(cfg, make_test_run_dir(cfg, tmp_path), [], fail=())

    def signal_arrives():
        hub.request_stop()
        hub._supervisor.received_signal = signal.SIGTERM

    timer = threading.Timer(0.5, signal_arrives)
    timer.start()
    try:
        assert hub.launch() == 128 + signal.SIGTERM
    finally:
        timer.cancel()


def test_failed_start_error_is_not_hidden_by_a_failing_teardown(tmp_path, caplog):
    """The shutdown's drain (uploads of final snapshots, results) fails: the start's error stays the one raised."""
    cfg = ColosseumConfig.model_validate(ttt_data())
    hub = fake_hub(cfg, make_test_run_dir(cfg, tmp_path), [])
    real_drain = hub._drain_inbox
    failed: list[bool] = []

    def failing_drain(timeout):
        if hub._shutting_down and not failed:
            failed.append(True)
            raise OSError("disk full")
        return real_drain(timeout)

    hub._drain_inbox = failing_drain
    with caplog.at_level(logging.ERROR, logger="colosseum.hub.hub"), pytest.raises(_Stop) as excinfo:
        hub.launch()
    assert "Shutdown afterwards also failed: OSError('disk full')" in getattr(excinfo.value, "__notes__", [])
    assert "Shutdown after _Stop failed too" in caplog.text and "disk full" in caplog.text
    assert hub._metrics_writer.closed


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


def _hub_with(tmp_path, children: dict):
    """A hub supervising real children; returns once every child reported it is ready."""
    cfg = make_test_config("turns")
    hub = make_hub(cfg, make_test_run_dir(cfg, tmp_path))
    ctx = mp.get_context("spawn")
    ready_events = []
    for name, (target, args) in children.items():
        ready = ctx.Event()
        ready_events.append(ready)
        args = (*(hub._stop_event if a == "stop" else a for a in args), ready)
        proc = ctx.Process(target=target, args=args, name=name, daemon=True)
        proc.start()
        hub._supervisor.add(name, proc)
    assert all(ready.wait(60) for ready in ready_events), "children did not start"
    return hub


def test_nonzero_child_exit_fails_a_run_that_reached_its_budget(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(hub_module, "LOCAL_SHUTDOWN_GRACE_SEC", 1.5)
    hub = _hub_with(tmp_path, {
        "learner-a": (_exit_after_stop, ("stop", 0)),
        "worker-0": (_exit_after_stop, ("stop", 3)),
        "worker-1": (_ignore_stop, ()),
    })
    start = time.monotonic()
    with caplog.at_level(logging.INFO, logger="colosseum.hub.hub"):
        killed = hub._shutdown()
        code = hub._exit_code_after_shutdown(0, killed)
    # The hub waits its shutdown grace (patched here), then terminates and kills.
    assert time.monotonic() - start < 1.5 + 5.0
    assert killed == ["worker-1"]
    assert code == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors == [f"worker-0 died (exit 3), see {hub._run_dir.logs / 'worker-0.log'}"]
    assert "worker-1" in caplog.text and "did not stop within" in caplog.text  # killed: a warning only


def test_signal_exit_code_stays_when_a_child_exits_nonzero(tmp_path, caplog):
    hub = _hub_with(tmp_path, {"worker-0": (_exit_after_stop, ("stop", 2))})
    killed = hub._shutdown()
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.hub"):
        assert hub._exit_code_after_shutdown(130, killed) == 130
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "worker-0 died (exit 2)" in caplog.text


def test_clean_children_keep_exit_code_zero(tmp_path, caplog):
    hub = _hub_with(tmp_path, {"learner-a": (_exit_after_stop, ("stop", 0))})
    with caplog.at_level(logging.INFO, logger="colosseum.hub.hub"):
        assert hub._exit_code_after_shutdown(0, hub._shutdown()) == 0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# Every CLI command turns a config error into one line and exit code 1 (D10)
# ---------------------------------------------------------------------------


@pytest.fixture
def bad_key_config(tmp_path):
    data = ttt_data()
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
    monkeypatch.setattr(hub_module.mp, "set_start_method", lambda *a, **k: None)
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


# ---------------------------------------------------------------------------
# Exit codes of run-learner / run-workers / train for signals and early Ctrl-C
# ---------------------------------------------------------------------------


def _raise_keyboard_interrupt(*args, **kwargs):
    raise KeyboardInterrupt


def test_ctrl_c_stops_serve_weight_store_with_exit_130(monkeypatch, restore_root_logging):
    """The CLI-wide rule: Ctrl-C ends serve-weight-store with "Interrupted" and 130, after stopping
    the server (also when a Ctrl-C lost during startup is taken over afterwards)."""
    pytest.importorskip("grpc")  # grpc_store imports grpc (an optional extra)
    import colosseum.weight_store.grpc_store as grpc_store

    stopped = []

    class FakeServer:
        def wait_for_termination(self):
            raise KeyboardInterrupt

        def stop(self, grace):
            stopped.append(grace)

    monkeypatch.setattr(grpc_store, "serve_weight_store", lambda **kwargs: FakeServer())
    result = CliRunner().invoke(main, ["serve-weight-store", "--port", "1"])
    assert result.exit_code == 130, result.output
    assert result.stderr == "Interrupted\n" and stopped == [0]


@pytest.mark.parametrize(("args", "target"), [
    (["train"], "colosseum.hub.hub.run_training"),
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
    (["train"], "colosseum.hub.hub.run_training"),
    (["run-learner", "--weight-store", "localhost:1"], "colosseum.distributed.run_distributed_learner"),
    (["run-workers", "--weight-store", "localhost:1", "-l", "agent_0=localhost:2"],
     "colosseum.distributed.run_distributed_workers"),
], ids=lambda a: a[0] if isinstance(a, list) else None)
def test_cli_exits_with_the_run_exit_code(args, target, tmp_path, monkeypatch):
    monkeypatch.setattr(target, lambda *a, **k: 143)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(ttt_data()))
    assert CliRunner().invoke(main, [args[0], "-c", str(cfg), *args[1:]]).exit_code == 143


# ---------------------------------------------------------------------------
# Metrics teardown (from SP1's test_metrics_jsonl.py and test_wandb_logger.py)
# ---------------------------------------------------------------------------


class _CommittedWriter:
    """A journal writer stand-in with one committed node batch to account (``take_commits``, R8)."""

    def __init__(self, results: list) -> None:
        self._commits = [Commit(("node", "session", 1), list(results), 0, len(results))]
        self.dead = False
        self.stats = SimpleNamespace(as_dict=lambda: {})

    def take_commits(self) -> list:
        out, self._commits = self._commits, []
        return out

    def stop(self, timeout):
        return SimpleNamespace(written=1, duplicates=0, dropped=0)


def test_the_hub_closes_metrics_file_when_final_drain_fails(tmp_path):
    from colosseum.metrics.hub import MetricsHub
    from colosseum.metrics.jsonl import MetricsWriter

    config = make_test_config("turns")
    run = make_test_run_dir(config, tmp_path)
    hub = make_hub(config, run)
    writer = MetricsWriter(run.metrics_path)
    hub._metrics_writer = writer
    hub._hub = MetricsHub(writer=writer, ratings_path=run.ratings_path, agent_ids=["agent_0"],
                          total_timesteps=10, log_interval=1, console_interval_sec=10.0)

    class FailingCoordinator:
        def report_match_result(self, result):
            raise RuntimeError("boom")

        def ratings_snapshot(self):
            return {}

    # a node's batch, committed by the journal writer: accounted in the final drain (R8)
    hub._match_log_writer = _CommittedWriter([MatchResult("w0_e0_ep0", "2p", "wdl", [], [], 1)])
    hub._coordinator = FailingCoordinator()
    with pytest.raises(RuntimeError, match="boom"):
        hub._finish_metrics()
    assert writer.closed


def test_the_hub_finishes_wandb_when_final_metrics_fail(monkeypatch, tmp_path):
    from colosseum.core.config import MetricsConfig
    from colosseum.metrics.hub import MetricsHub
    from colosseum.metrics.jsonl import MetricsWriter
    from colosseum.metrics.wandb_logger import WandBLogger

    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    config = make_test_config("turns")
    run = make_test_run_dir(config, tmp_path)
    hub = make_hub(config, run)
    hub._wandb = WandBLogger(MetricsConfig(use_wandb=True))
    hub._metrics_writer = MetricsWriter(run.metrics_path)
    hub._hub = MetricsHub(writer=hub._metrics_writer, ratings_path=run.ratings_path,
                          agent_ids=["agent_0"], total_timesteps=10, log_interval=1,
                          console_interval_sec=10.0, wandb_logger=hub._wandb)

    class FailingCoordinator:
        def ratings_snapshot(self):
            raise RuntimeError("boom")

    hub._coordinator = FailingCoordinator()
    with pytest.raises(RuntimeError, match="boom"):
        hub._finish_metrics()
    assert hub._metrics_writer.closed and fake.finished
