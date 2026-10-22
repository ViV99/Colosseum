"""Localhost clusters for SP5 tests (T3.1; later tasks extend, amendment A4): an in-process hub (``Hub.launch`` in a
thread of the test process: hub logic is inspectable, exit codes are returned) and ``NodeAgent``s in threads, each
with its own identity root, log dir and data port (port 0); their children (workers, learners) are real processes,
so a cluster test counts them against the "2 loaded CPU processes" rule. Waits are on counters and files, never on
wall-clock values. Every helper stops what it started in ``finally`` (``cluster``).

The hub listens on 127.0.0.1 (``HubOptions(mode="hub", listen="127.0.0.1", port=0)``) with its identity under
``tmp_path``; nodes find it through ``HubThread.address``; with ``insecure=False`` (default) the hub port is TLS and
data ports are mutual TLS, as in production. ``cluster_config`` is the tic-tac-toe example with the TINY sizes of
``cli_runner``, fast distributed timings, its run dir under ``tmp_path`` and CPU learners (pre-flight ruling
P-PF3-10; nodes get no GPUs unless a test passes them). ``hub_process`` runs ``colosseum hub`` as a CLI process.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cli_runner import TINY, TTT_CONFIG, child_env, hub_address, wait_for
from cli_runner import start_hub as start_hub_process
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader, MatchRow
from colosseum.core.config import ColosseumConfig, load_config, parse_override_value
from colosseum.core.run_dir import RunDir
from colosseum.hub.hub import Hub, HubOptions
from colosseum.hub.registry import read_hub_json
from colosseum.hub.state import HUB_STATE_FILE
from colosseum.net.security import HubIdentity, Token
from colosseum.node.agent import NodeAgent, NodeOptions

CLUSTER_TINY: dict[str, str] = {
    **TINY,
    "learner.device": "cpu",
    "distributed.heartbeat_sec": "0.25",
    "distributed.node_lost_after_sec": "5.0",
    "distributed.connect_timeout_sec": "1.0",
    "distributed.results_flush_sec": "0.1",
    "distributed.shutdown_grace_sec": "5.0",
}


def wait_until(predicate: Callable[[], bool], timeout: float, interval: float = 0.1) -> bool:
    return wait_for(predicate, timeout, interval)


def cluster_config(tmp_path: Path, overrides: Mapping[str, Any] | None = None, *, base: Path = TTT_CONFIG,
                   name: str = "cluster") -> ColosseumConfig:
    """``base`` with ``CLUSTER_TINY``, ``overrides`` (``--set`` style keys; values are YAML literals or Python
    values) and ``run.dir`` = ``tmp_path / "runs"``, ``run.name`` = ``name``."""
    sets: dict[str, Any] = {k: parse_override_value(v, k) for k, v in CLUSTER_TINY.items()}
    for key, value in (overrides or {}).items():
        sets[key] = parse_override_value(value, key) if isinstance(value, str) else value
    sets.update({"run.dir": str(Path(tmp_path) / "runs"), "run.name": name})
    return load_config(str(base), sets)


@dataclass
class HubThread:
    hub: Hub
    thread: threading.Thread
    root: Path                       # runs/<name>
    address: str                     # host:port of the hub port
    token: Token
    insecure: bool
    exit_codes: list[int] = field(default_factory=list)

    def stop(self, timeout: float = 60.0) -> int | None:
        if self.thread.is_alive():
            self.hub.request_stop()
            self.thread.join(timeout)
        assert not self.thread.is_alive(), "the hub did not stop"
        return self.exit_codes[0] if self.exit_codes else None


def start_hub(tmp_path: Path, config: ColosseumConfig, *, insecure: bool = False, name: str = "cluster") -> HubThread:
    """A ``colosseum hub`` (mode ``hub``) in a thread, its run dir ``tmp_path/runs/<name>``, its identity under
    ``tmp_path``."""
    config = config.model_copy(deep=True)
    config.run.dir, config.run.name = str(Path(tmp_path) / "runs"), name
    run_dir = RunDir.create(config)
    hub = Hub(config, run_dir, options=HubOptions(mode="hub", listen="127.0.0.1", port=0, insecure=insecure,
                                                  identity_dir=Path(tmp_path) / "hub-identity"))
    codes: list[int] = []
    thread = threading.Thread(target=lambda: codes.append(hub.launch()), name="hub", daemon=True)
    thread.start()
    assert wait_until(lambda: read_hub_json(run_dir.root) is not None or not thread.is_alive(), 120)
    info = read_hub_json(run_dir.root)
    assert info is not None, f"the hub exited with {codes}"
    token = HubIdentity.load_or_create(Path(info.identity_dir)).token
    return HubThread(hub, thread, run_dir.root, info.addresses[0], token, insecure, codes)


@dataclass
class NodeThread:
    agent: NodeAgent
    thread: threading.Thread
    name: str
    exit_codes: list[int] = field(default_factory=list)

    def stop(self, timeout: float = 60.0) -> int | None:
        self.agent.request_stop()
        self.thread.join(timeout)
        assert not self.thread.is_alive(), f"node {self.name} did not stop"
        return self.exit_codes[0] if self.exit_codes else None


def start_node(tmp_path: Path, hub: HubThread, name: str, *, workers: int = 0, cpu_learners: int = 0,
               evaluators: int = 0, no_learners: bool = False, gpus: Sequence[int] = (), data_port: int | None = 0,
               data_addresses: Sequence[str] | None = None, once: bool = True) -> NodeThread:
    """A registered node in a thread: its identity under ``tmp_path/nodes/<name>``, logs in ``tmp_path/logs/<name>``,
    a data port on an ephemeral port (``data_port=None``: none)."""
    options = NodeOptions(hub_address=hub.address, token=hub.token, insecure=hub.insecure, workers=workers,
                          evaluators=evaluators, cpu_learners=cpu_learners, no_learners=no_learners, gpus=tuple(gpus),
                          name=name, once=once, identity_root=Path(tmp_path) / "nodes" / name,
                          log_dir=Path(tmp_path) / "logs" / name, data_port=data_port,
                          data_addresses=None if data_addresses is None else tuple(data_addresses))
    agent = NodeAgent(options)
    codes: list[int] = []
    thread = threading.Thread(target=lambda: codes.append(agent.run()), name=f"node-{name}", daemon=True)
    thread.start()
    return NodeThread(agent, thread, name, codes)


@contextmanager
def cluster(tmp_path: Path, config: ColosseumConfig, nodes: Sequence[Mapping[str, Any]], *,
            insecure: bool = False, learner_first: str | None = None,
            timeout: float = 120.0) -> Iterator[tuple[HubThread, list[NodeThread]]]:
    """A hub and ``nodes`` (``start_node`` keywords + ``name``). With ``learner_first=<agent>``, every node after the
    first starts only once the hub accepted a version of that agent (its learner runs on the first node: until T4.1
    the hub places learners on the first node with a learner slot)."""
    hub = start_hub(tmp_path, config, insecure=insecure)
    started: list[NodeThread] = []
    try:
        for i, spec in enumerate(nodes):
            if i == 1 and learner_first is not None:
                assert wait_until(lambda: hub_version(hub, learner_first) is not None, timeout), \
                    f"no version of {learner_first} announced"
            started.append(start_node(tmp_path, hub, **dict(spec)))
        yield hub, started
    finally:
        try:
            hub.stop()
        finally:
            for node in started:
                node.stop()


def _root(hub_or_root: HubThread | Path | str) -> Path:
    return hub_or_root.root if isinstance(hub_or_root, HubThread) else Path(hub_or_root)


def read_hub_state(hub_or_root: HubThread | Path | str) -> dict:
    """``hub_state.json`` of a hub (thread) or run dir; ``{}`` before its first write."""
    import json

    path = _root(hub_or_root) / HUB_STATE_FILE
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):            # not written yet (atomic replace: never half a file)
        return {}


def hub_counter(hub_or_root: HubThread | Path | str, name: str) -> int:
    """A run total of the hub's counter book (``hub_state.json`` ``counters.book.totals``; 0 when absent)."""
    totals = ((read_hub_state(hub_or_root).get("counters") or {}).get("book") or {}).get("totals") or {}
    return int(totals.get(name, 0))


def hub_version(hub_or_root: HubThread | Path | str, agent_id: str) -> int | None:
    """The newest version of ``agent_id`` the hub accepted, from ``hub_state.json`` (written every metrics tick)."""
    latest = ((read_hub_state(hub_or_root).get("versions") or {}).get(agent_id) or {}).get("latest")
    return None if not latest else int(latest["version"])


def journal_rows(hub_or_root: HubThread | Path | str) -> list[MatchRow]:
    """Every training match of the run's journal (``matches.db``), newest first."""
    db = _root(hub_or_root) / MATCH_LOG_FILE
    if not db.exists():
        return []
    with MatchLogReader(db) as reader:
        return reader.matches(source="train", limit=None)


def node_of_uid(uid: str) -> int | None:
    """The node number of a training match uid (``<run_id>/train/n<node>/...``, spec 4.3)."""
    part = next((p for p in uid.split("/") if p.startswith("n") and p[1:].isdigit()), None)
    return None if part is None else int(part[1:])


def journal_results_by_node(hub_or_root: HubThread | Path | str) -> dict[int, int]:
    """Training results in ``matches.db`` per node number."""
    out: dict[int, int] = {}
    for row in journal_rows(hub_or_root):
        node = node_of_uid(row.uid)
        if node is not None:
            out[node] = out.get(node, 0) + 1
    return out


@dataclass
class HubProc:
    """A ``colosseum hub`` CLI process (``hub_process``)."""

    proc: subprocess.Popen
    root: Path
    address: str
    token_file: Path
    env: dict[str, str]

    def wait(self, timeout: float) -> int:
        return self.proc.wait(timeout)

    def interrupt(self) -> None:
        """SIGINT to the hub's process group (Ctrl+C)."""
        try:
            os.killpg(self.proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    def kill(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait()

    def tail(self, chars: int = 3000) -> str:
        path = self.root.parent / f"{self.root.name}.stderr"
        return path.read_text()[-chars:] if path.exists() else ""


def hub_process(config: Path, tmp_path: Path, name: str, *, overrides: Mapping[str, Any] | None = None,
                args: Sequence[str] = (), insecure: bool = False, timeout: float = 120.0) -> HubProc:
    """``colosseum hub -c <config>`` in its own session on ``127.0.0.1:<ephemeral>`` (``cli_runner.start_hub``),
    its run dir ``tmp_path/runs/<name>``; the token written to ``tmp_path/<name>.token`` (until T6.3: from the hub
    identity the process uses). The caller stops it (``interrupt`` + ``wait``, or ``kill``)."""
    env = child_env()
    proc, root = start_hub_process(Path(config), Path(tmp_path) / "runs", name, overrides, port=0,
                                   insecure=insecure, listen="127.0.0.1", args=tuple(args), env=env)
    try:
        address = hub_address(root, timeout)
        info = read_hub_json(root)
        token_file = Path(tmp_path) / f"{name}.token"
        token_file.write_text(str(HubIdentity.load_or_create(Path(info.identity_dir)).token) + "\n")
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise
    return HubProc(proc, root, address, token_file, env)
