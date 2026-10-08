"""Run ``colosseum train`` in a subprocess for integration tests."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TTT_CONFIG = REPO_ROOT / "configs" / "examples" / "tic_tac_toe.yaml"
TTT_MULTI_CONFIG = REPO_ROOT / "configs" / "examples" / "tic_tac_toe_multi.yaml"

# Small, fast settings for tic-tac-toe runs (about 10 s with 1 worker on CPU).
TINY: dict[str, str] = {
    "training.total_timesteps": "3000",
    "rollout.num_workers": "1",
    "rollout.envs_per_worker": "8",
    "rollout.chunk_length": "16",
    "rollout.weight_sync_interval_sec": "0.5",
    "rollout.match_refresh_interval_sec": "1.0",
    "learner.batch_chunks": "2",
    "learner.queue_size": "16",
    "self_play.checkpoint_interval": "20",
    "self_play.pool_size": "5",
    "metrics.log_interval": "1",
    "metrics.console_interval_sec": "1.0",
}


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({"WANDB_MODE": "disabled", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    return env


def train_cmd(config: Path, run_parent: Path, name: str, overrides: dict[str, str] | None = None) -> list[str]:
    sets = {**TINY, **(overrides or {}), "run.dir": str(run_parent), "run.name": name}
    cmd = [sys.executable, "-m", "colosseum", "train", "-c", str(config)]
    for key, value in sets.items():
        cmd += ["--set", f"{key}={value}"]
    return cmd


@dataclass
class TrainRun:
    returncode: int
    stdout: str
    stderr: str
    root: Path

    def records(self, kind: str | None = None) -> list[dict]:
        path = self.root / "metrics.jsonl"
        if not path.exists():
            return []
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [r for r in records if kind is None or r["kind"] == kind]

    def log(self, process_name: str) -> str:
        return (self.root / "logs" / f"{process_name}.log").read_text()

    def ratings(self) -> dict:
        return json.loads((self.root / "ratings.json").read_text())


def run_train(config: Path, tmp_path: Path, name: str = "run", overrides: dict[str, str] | None = None,
              timeout: float = 240.0, env: dict[str, str] | None = None) -> TrainRun:
    """``env`` entries are added to ``child_env()``."""
    run_parent = tmp_path / "runs"
    proc = run_in_session(train_cmd(config, run_parent, name, overrides), timeout, env)
    return TrainRun(proc.returncode, proc.stdout, proc.stderr, run_parent / name)


def run_in_session(cmd: list[str], timeout: float, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run ``cmd`` from the repo root in its own session (``env`` added to ``child_env()``);
    whatever is left of the session afterwards (also on a timeout) is SIGKILLed."""
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env={**child_env(), **(env or {})}, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    finally:
        _kill_group(proc)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the session started for ``proc`` (whatever is left of it) and reap ``proc``."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # start_new_session: pgid == pid
    except ProcessLookupError:
        pass
    proc.wait()


def start_train(config: Path, tmp_path: Path, name: str = "run",
                overrides: dict[str, str] | None = None) -> tuple[subprocess.Popen, Path]:
    """Start training in its own session; stdout/stderr go to files next to the run dir."""
    run_parent = tmp_path / "runs"
    # The child keeps its own copies of the descriptors; the parent's handles close here.
    with open(tmp_path / f"{name}.stdout", "w") as out, open(tmp_path / f"{name}.stderr", "w") as err:
        proc = subprocess.Popen(train_cmd(config, run_parent, name, overrides), cwd=REPO_ROOT, env=child_env(),
                                stdout=out, stderr=err, text=True, start_new_session=True)
    return proc, run_parent / name


@contextmanager
def training_process(config: Path, tmp_path: Path, name: str = "run",
                     overrides: dict[str, str] | None = None) -> Iterator[tuple[subprocess.Popen, Path]]:
    """``start_train`` whose whole process group (main, children, env grandchildren) is
    SIGKILLed and reaped on exit, so a failing test leaves no orphans behind."""
    proc, root = start_train(config, tmp_path, name, overrides)
    try:
        yield proc, root
    finally:
        _kill_group(proc)


def wait_for(predicate: Callable[[], bool], timeout: float, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
