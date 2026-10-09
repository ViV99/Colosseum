"""SP2 CLI: exit codes, signals, child death, quickstart (SP1 guarantees, T7.1). Linux (/proc) only."""
from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import pytest
import yaml

from cli_runner import TTT_CONFIG, run_in_session, run_train, training_process, wait_for

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")

TESTS_DIR = Path(__file__).resolve().parents[1]
FOREVER = {"training.total_timesteps": "100000000"}


def _proc_table() -> dict[int, tuple[int, str]]:
    table = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
        except OSError:
            continue
        rest = stat[stat.rfind(")") + 2:].split()
        table[int(d.name)] = (int(rest[1]), rest[0])  # (ppid, state)
    return table


def descendants(pid: int) -> set[int]:
    table = _proc_table()
    found, frontier = set(), [pid]
    while frontier:
        parent = frontier.pop()
        for child, (ppid, _state) in table.items():
            if ppid == parent and child not in found:
                found.add(child)
                frontier.append(child)
    return found


def pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat[stat.rfind(")") + 2:].split()[0] != "Z"


def logged_pid(root: Path, process_name: str) -> int | None:
    path = root / "logs" / f"{process_name}.log"
    if not path.exists():
        return None
    match = re.search(rf"{re.escape(process_name)} started \(pid (\d+)\)", path.read_text())
    return int(match.group(1)) if match else None


def training_started(root: Path) -> bool:
    path = root / "metrics.jsonl"
    return path.exists() and '"kind": "train"' in path.read_text()


def final_checkpoints(root: Path, agent: str = "agent_0") -> list[dict]:
    metas = [json.loads((c / "meta.json").read_text())
             for c in sorted((root / "checkpoints" / agent).glob("ckpt_v*"))]
    return [m for m in metas if m["final"]]


def test_normal_finish_exit_0_and_final_checkpoint(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="normal")
    assert run.returncode == 0, run.stderr[-3000:]
    main_log = run.log("main")
    assert "Training budget reached" in main_log
    assert "unexpected" not in main_log.lower() and "died" not in main_log
    assert "[ERROR]" not in main_log and "[ERROR]" not in run.stderr  # (T2.5 item c)
    ckpts = sorted((run.root / "checkpoints" / "agent_0").glob("ckpt_v*"))
    assert ckpts, "no checkpoint saved"
    metas = [json.loads((c / "meta.json").read_text()) for c in ckpts]
    assert any(m["final"] for m in metas)


def test_killed_worker_gives_exit_1_and_points_to_its_log(tmp_path):
    with training_process(TTT_CONFIG, tmp_path, name="killw", overrides=FOREVER) as (proc, root):
        assert wait_for(lambda: logged_pid(root, "worker-0") is not None and training_started(root), 120)
        kids = descendants(proc.pid)
        os.kill(logged_pid(root, "worker-0"), signal.SIGKILL)
        assert proc.wait(60) == 1
        assert wait_for(lambda: not any(pid_alive(k) for k in kids), 10)
    stderr = (tmp_path / "killw.stderr").read_text()
    assert f"worker-0 died (exit -9), see {root / 'logs' / 'worker-0.log'}" in stderr


def test_learner_crash_in_train_step_gives_exit_1(tmp_path):
    """train_step raising mid-training fails the run with an error naming the learner (T2.5 item b)."""
    run = run_train(TTT_CONFIG, tmp_path, name="crash", overrides={
        **FOREVER, "algorithm.algorithm_class": "game_helpers.CrashingAPPO"},
        env={"PYTHONPATH": os.pathsep.join(filter(None, [str(TESTS_DIR), os.environ.get("PYTHONPATH")]))})
    assert run.returncode == 1, run.stderr[-3000:]
    assert f"learner-agent_0 died (exit 1), see {run.root / 'logs' / 'learner-agent_0.log'}" in run.stderr
    learner_log = run.log("learner-agent_0")
    assert "learner-agent_0 crashed" in learner_log and "injected train_step failure" in learner_log


def test_sigterm_stops_all_descendants_within_10s(tmp_path):
    overrides = {**FOREVER, "rollout.vec_env": "subprocess", "rollout.subproc_workers": "2"}
    with training_process(TTT_CONFIG, tmp_path, name="term", overrides=overrides) as (proc, root):
        # learner + worker + 2 env processes (+ resource tracker)
        assert wait_for(lambda: len(descendants(proc.pid)) >= 4 and training_started(root), 120)
        kids = descendants(proc.pid)
        t0 = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(15) == 143
        assert wait_for(lambda: not any(pid_alive(k) for k in kids), max(0.0, 10 - (time.monotonic() - t0)))
    assert final_checkpoints(root), "final checkpoint not saved on SIGTERM"


def test_killed_main_process_takes_all_descendants_with_it(tmp_path):
    """PR_SET_PDEATHSIG: children and env grandchildren die with their parent (R3-17, R6-07)."""
    overrides = {**FOREVER, "rollout.vec_env": "subprocess", "rollout.subproc_workers": "2"}
    with training_process(TTT_CONFIG, tmp_path, name="kill9", overrides=overrides) as (proc, root):
        assert wait_for(lambda: len(descendants(proc.pid)) >= 4 and training_started(root), 120)
        kids = descendants(proc.pid)
        proc.kill()  # the main process only
        assert proc.wait(15) == -signal.SIGKILL
        assert wait_for(lambda: not any(pid_alive(k) for k in kids), 10)


def test_sigint_to_process_group_exits_130_with_final_checkpoint(tmp_path):
    """Ctrl-C reaches every process of the group; children ignore it, so the learner still
    sends its final checkpoint and the main process saves it (T5.3 carried item)."""
    with training_process(TTT_CONFIG, tmp_path, name="int", overrides=FOREVER) as (proc, root):
        assert wait_for(lambda: training_started(root), 120)
        kids = descendants(proc.pid)
        os.killpg(proc.pid, signal.SIGINT)  # like Ctrl-C in a terminal
        assert proc.wait(15) == 130
        assert wait_for(lambda: not any(pid_alive(k) for k in kids), 10)
    assert "KeyboardInterrupt" not in (tmp_path / "int.stderr").read_text()
    assert final_checkpoints(root), "final checkpoint not saved on Ctrl-C"


def test_sigint_during_startup_exits_130_not_aborted(tmp_path):
    """Ctrl-C while the CLI is still importing torch / loading the config (before the run's
    signal handling exists): "Interrupted" and 130, never click's "Aborted!" with 1."""
    marker = tmp_path / "entered"
    with training_process(TTT_CONFIG, tmp_path, name="early", overrides=FOREVER,
                          env={"COLOSSEUM_TEST_STARTUP_MARKER": str(marker)}) as (proc, _root):
        assert wait_for(marker.exists, 60, interval=0.01)
        os.killpg(proc.pid, signal.SIGINT)
        assert proc.wait(30) == 130
    stderr = (tmp_path / "early.stderr").read_text()
    assert "Aborted" not in stderr and "Traceback" not in stderr
    # Normally it lands in the imports ("Interrupted"); on a very fast start the run's own
    # handler may already have taken over - both are a clean 130.
    assert "Interrupted" in stderr or "Received SIGINT" in stderr


@pytest.mark.parametrize("builtin", ["exec", "eval"])
def test_ctrl_c_inside_string_exec_during_startup_still_exits_130(tmp_path, builtin):
    """A Ctrl-C that lands while a string ``exec``/``eval`` runs (``import torch`` builds
    hundreds of dataclass methods that way) makes CPython flag the KeyboardInterrupt as
    unhandled although ``_interrupts`` catches it; under ``python -m`` the process then
    killed itself with SIGINT (-2) instead of exiting 130 (FIX-2)."""
    # The Ctrl-C, deterministically: a KeyboardInterrupt raised inside the string's code (an
    # expression, so eval takes it too) of a command run through the real CLI group.
    (tmp_path / "ki_in_exec_cli.py").write_text(
        "from colosseum.cli import main\n\n"
        "@main.command('startup')\n"
        "def startup():\n"
        f"    {builtin}('(_ for _ in ()).throw(KeyboardInterrupt)', {{}})\n\n"
        "main()\n")
    env = {"PYTHONPATH": os.pathsep.join(filter(None, [str(tmp_path), os.environ.get("PYTHONPATH")]))}
    proc = run_in_session([sys.executable, "-m", "ki_in_exec_cli", "startup"], timeout=60, env=env)
    assert proc.returncode == 130, proc.stderr[-3000:]
    assert proc.stderr == "Interrupted\n"


def test_ctrl_c_lost_in_a_weakref_callback_during_startup_still_exits_130(tmp_path):
    """A Ctrl-C raised inside a weakref callback (importlib runs them during imports) cannot
    propagate; Python used to print "Exception ignored ... KeyboardInterrupt" and carry on.
    The CLI records it: a command that then finishes still ends with "Interrupted" and 130
    (``train`` hands it to the run's signal handling instead; unit-tested) (FIX-2)."""
    (tmp_path / "ki_in_callback_cli.py").write_text(
        "import weakref\n\n"
        "from colosseum.cli import main\n\n"
        "class Target:\n    pass\n\n"
        "def callback(_ref):\n    raise KeyboardInterrupt\n\n"
        "@main.command('startup')\n"
        "def startup():\n"
        "    target = Target()\n"
        "    ref = weakref.ref(target, callback)\n"
        "    del target\n"
        "    print('finished', ref() is None)\n\n"
        "main()\n")
    env = {"PYTHONPATH": os.pathsep.join(filter(None, [str(tmp_path), os.environ.get("PYTHONPATH")]))}
    proc = run_in_session([sys.executable, "-m", "ki_in_callback_cli", "startup"], timeout=60, env=env)
    assert proc.stdout == "finished True\n"
    assert proc.returncode == 130, proc.stderr[-3000:]
    assert proc.stderr == "Interrupted\n"


def test_config_error_exit_1_without_traceback(tmp_path):
    bad = tmp_path / "bad.yaml"
    data = yaml.safe_load(TTT_CONFIG.read_text())
    data["rollout"]["num_worker"] = 3
    bad.write_text(yaml.safe_dump(data))
    proc = run_in_session([sys.executable, "-m", "colosseum", "train", "-c", str(bad)], timeout=120)
    assert proc.returncode == 1
    assert "Config error" in proc.stderr and "num_worker" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_readme_quickstart_from_repo_root(tmp_path):
    """The README quickstart: the console script trains the example config from the repo root."""
    exe = Path(sys.executable).parent / "colosseum"
    assert exe.exists(), "console script missing; run scripts/setup-dev.sh"
    cmd = [str(exe), "train", "-c", "configs/examples/tic_tac_toe.yaml",
           "--set", "training.total_timesteps=2000",
           "--set", f"run.dir={tmp_path / 'runs'}", "--set", "run.name=quickstart"]
    proc = run_in_session(cmd, timeout=240)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert (tmp_path / "runs" / "quickstart" / "config.resolved.yaml").exists()
