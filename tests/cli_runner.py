"""Run ``colosseum train`` in a subprocess for integration and learning tests."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
TTT_CONFIG = REPO_ROOT / "configs" / "examples" / "tic_tac_toe.yaml"
TTT_MULTI_CONFIG = REPO_ROOT / "configs" / "examples" / "tic_tac_toe_multi.yaml"
_CKPT_RE = re.compile(r"ckpt_v(\d+)")

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
    "checkpoint.interval": "20",
    "checkpoint.keep_last": "5",
    "metrics.log_interval": "1",
    "metrics.console_interval_sec": "1.0",
}


def child_env() -> dict[str, str]:
    """The test process's environment for a CLI child; ``tests/`` goes on PYTHONPATH, so configs
    may name support modules (``game_helpers.*``)."""
    env = dict(os.environ)
    env.update({"WANDB_MODE": "disabled", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(TESTS_DIR), os.environ.get("PYTHONPATH")]))
    return env


def train_cmd(config: Path, run_parent: Path, name: str, overrides: Mapping[str, object] | None = None,
              module: str = "colosseum", tiny: bool = True) -> list[str]:
    """``python -m <module> train -c <config>`` with the ``TINY`` settings (unless ``tiny=False``:
    the config's own budget and sizes, as in the slow learning tests), then ``overrides``."""
    sets = {**(TINY if tiny else {}), **(overrides or {}), "run.dir": str(run_parent), "run.name": name}
    cmd = [sys.executable, "-m", module, "train", "-c", str(config)]
    for key, value in sets.items():
        cmd += ["--set", f"{key}={value}"]
    return cmd


def read_records(root: Path, kind: str | None = None) -> list[dict]:
    """The ``metrics.jsonl`` records of the run at ``root`` (of one ``kind``, or all)."""
    path = Path(root) / "metrics.jsonl"
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in records if kind is None or r["kind"] == kind]


def read_ratings(root: Path) -> dict:
    """``ratings.json`` of the run at ``root``: ``{"env_steps": N, "layouts": {<layout>: {...}}}``."""
    return json.loads((Path(root) / "ratings.json").read_text())


@dataclass
class TrainRun:
    returncode: int
    stdout: str
    stderr: str
    root: Path

    def records(self, kind: str | None = None) -> list[dict]:
        return read_records(self.root, kind)

    def log(self, process_name: str) -> str:
        return (self.root / "logs" / f"{process_name}.log").read_text()

    def ratings(self) -> dict:
        return read_ratings(self.root)


def newest_checkpoint(root: Path, agent_id: str) -> Path:
    """The ``ckpt_v<N>`` directory of ``agent_id`` with the largest N in the run at ``root``."""
    agent_dir = Path(root) / "checkpoints" / agent_id
    versions = {int(m.group(1)): d for d in agent_dir.iterdir()
                if d.is_dir() and (m := _CKPT_RE.fullmatch(d.name))} if agent_dir.is_dir() else {}
    assert versions, f"no checkpoints of {agent_id} in {root}"
    return versions[max(versions)]


def run_train(config: Path, tmp_path: Path, name: str = "run", overrides: Mapping[str, object] | None = None,
              timeout: float = 240.0, env: dict[str, str] | None = None, module: str = "colosseum",
              tiny: bool = True) -> TrainRun:
    """``env`` entries are added to ``child_env()``; ``tiny`` as in ``train_cmd``."""
    run_parent = tmp_path / "runs"
    proc = run_in_session(train_cmd(config, run_parent, name, overrides, module, tiny), timeout, env)
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


def start_train(config: Path, tmp_path: Path, name: str = "run", overrides: dict[str, str] | None = None,
                env: dict[str, str] | None = None, module: str = "colosseum") -> tuple[subprocess.Popen, Path]:
    """Start training in its own session; stdout/stderr go to files next to the run dir."""
    run_parent = tmp_path / "runs"
    # The child keeps its own copies of the descriptors; the parent's handles close here.
    with open(tmp_path / f"{name}.stdout", "w") as out, open(tmp_path / f"{name}.stderr", "w") as err:
        proc = subprocess.Popen(train_cmd(config, run_parent, name, overrides, module), cwd=REPO_ROOT,
                                env={**child_env(), **(env or {})},
                                stdout=out, stderr=err, text=True, start_new_session=True)
    return proc, run_parent / name


@contextmanager
def training_process(config: Path, tmp_path: Path, name: str = "run", overrides: dict[str, str] | None = None,
                     env: dict[str, str] | None = None,
                     module: str = "colosseum") -> Iterator[tuple[subprocess.Popen, Path]]:
    """``start_train`` whose whole process group (main, children, env grandchildren) is
    SIGKILLed and reaped on exit, so a failing test leaves no orphans behind."""
    proc, root = start_train(config, tmp_path, name, overrides, env, module)
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


# ---- colosseum hub (SP5 T2.5; amendment A4) ------------------------------------------------------------------------


def hub_cmd(config: Path, run_parent: Path, name: str, overrides: Mapping[str, object] | None = None, *,
            port: int | None = 0, insecure: bool = True, listen: str | None = "127.0.0.1",
            args: tuple[str, ...] | list[str] = ()) -> list[str]:
    """``python -m colosseum hub -c <config>`` with ``overrides`` (no ``TINY``) and the run dir; a ``None`` / ``False``
    keyword emits no flag; ``args`` are appended."""
    sets = {**(overrides or {}), "run.dir": str(run_parent), "run.name": name}
    cmd = [sys.executable, "-m", "colosseum", "hub", "-c", str(config)]
    if listen is not None:
        cmd += ["--listen", str(listen)]
    if port is not None:
        cmd += ["--port", str(port)]
    if insecure:
        cmd.append("--insecure")
    for key, value in sets.items():
        cmd += ["--set", f"{key}={value}"]
    return cmd + list(args)


def start_hub(config: Path, run_parent: Path, name: str, overrides: Mapping[str, object] | None = None, *,
              port: int | None = 0, insecure: bool = True, listen: str | None = "127.0.0.1",
              args: tuple[str, ...] | list[str] = (), cwd: Path | None = None,
              env: dict[str, str] | None = None) -> tuple[subprocess.Popen, Path]:
    """Start ``colosseum hub`` in its own session (stdout / stderr into ``<run_parent>/<name>.stdout|stderr``);
    returns the process and the run dir. The caller waits for it (``wait_for_hub_json``) and reaps it."""
    run_parent = Path(run_parent)
    run_parent.mkdir(parents=True, exist_ok=True)
    with open(run_parent / f"{name}.stdout", "w") as out, open(run_parent / f"{name}.stderr", "w") as err:
        proc = subprocess.Popen(hub_cmd(config, run_parent, name, overrides, port=port, insecure=insecure,
                                        listen=listen, args=args),
                                cwd=cwd or REPO_ROOT, env={**child_env(), **(env or {})}, stdout=out, stderr=err,
                                text=True, start_new_session=True)
    return proc, run_parent / name


def wait_for_hub_json(run_root: Path, timeout: float):
    """The ``HubInfo`` of ``run_root/hub.json`` once the hub wrote it (AssertionError after ``timeout``)."""
    from colosseum.hub.registry import read_hub_json

    deadline = time.monotonic() + timeout
    while True:
        info = read_hub_json(run_root)
        if info is not None:
            return info
        if time.monotonic() > deadline:
            raise AssertionError(f"no hub.json in {run_root} within {timeout} s")
        time.sleep(0.1)


def hub_address(run_root: Path, timeout: float = 30.0) -> str:
    """The address a client on this machine uses for the hub of ``run_root`` (its loopback, else its first
    listen address)."""
    info = wait_for_hub_json(run_root, timeout)
    return info.loopback or info.addresses[0]
