"""The hub process (SP5 spec block 1; ruling R5): the SP4 ``Launcher`` with its inputs from node sessions.

The hub owns the run dir and its state (coordinator, snapshot storage and retention, the journal and its writer, live
ratings, metrics, ``league_state.json``, ``hub_state.json``, ``run_state.json``) and listens on one gRPC port
(``colosseum.hub.server``). Nodes run every worker and learner: ``train`` starts one local node as a child process on
a plain loopback port (R12, R18; ``train --listen`` also opens a TLS port for other machines), ``colosseum hub``
serves nodes that connect. The SP4 pulse child stays a child of the hub (R6).

One thread (the caller of ``launch``, the "main loop") owns every piece of state; the servicer threads only put
events on ``self._inbox`` (``hub.server``: sessions, node messages, uploads, registrations, admin calls) and get
replies through futures or a session's outbound queue. The main loop (``_main_loop``):

1. drains the inbox (``_drain_inbox``): every event goes to its handler (``_on_session_opened``,
   ``_on_session_closed``, ``_on_message`` -> ``_on_heartbeat`` / ``_on_lineup_request`` / ``_on_result_batch`` /
   ``_on_metrics_batch`` / ``_on_version`` / ``_on_event``, ``_on_upload``, ``_on_register``, ``_on_admin``); a
   handler's failure is one ERROR per handler (nothing a node sends stops the hub);
2. accounts the journal writer's commits (``_drain_commits``, R8: the coordinator and the metrics hub get a result
   once, after its commit; then the batch is acked);
3. reruns retention after a live refit, feeds the pulse, marks silent nodes lost, broadcasts the periodic
   ``Announce`` (env steps for the learners' LR progress, new versions, evicted snapshots), ticks the metrics (and
   writes ``league_state.json`` / ``hub_state.json``);
4. stops at a failure (``_fail_run``: a node's reported child error or a failed learner, also when reported during
   the shutdown; the supervisor: a dead local node or pulse child), a signal, ``request_stop`` or the budget (the
   hub is the only budget authority, spec 4.4).

Lineups (spec 4.2): a node asks for cells; every cell gets an epoch (``CellTable``, epochs reserved in blocks with an
fsync'd ``hub_state.json``, R7), an owner by the deficit rotation over the agents that can collect anywhere (an
accepted latest version and a learner), and a lineup from the coordinator's matchmaker (PFSP, ``shuffle_seats``,
``check_lineup``) whose seats of agents without a learner on that node do not collect (amendment A11, until T3.2).
No cell is requested while no agent can collect (P-PF2-3).

Results (spec 4.3, R8): each node batch is checked (``ResultIntake.check``: the node's own uids only), stamped with
the hub's env steps (P-PF2-13) and submitted to the journal writer with its token ``(node, session, seq)``; the
results are accounted in commit order and the batch acked only then. A batch the writer cannot take whole is neither
accounted nor acked (the node resends it); without the journal, or with a dead writer, results are accounted at
intake. Drops are counted in ``Hub._book`` (``dropped_match_results_by_reason.*``, amendment A2).

Until T4.1 every learner is generation 1 and is placed on the local node (``train``) or on the first node with
learner capacity (``hub``); until T4.2 a failed learner stops the run (A12).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
import queue
import shutil
import signal
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.ipc import QueueReader
from colosseum.core.run_dir import RunDir
from colosseum.hub.run_sinks import RunSinks
from colosseum.hub.server import (
    LOOPBACK_KEY,
    AdminIn,
    BlobStore,
    NodeMessageIn,
    RegisterIn,
    SessionClosed,
    SessionOpened,
    StopRequested,
    UploadIn,
    blob_ident,
    build_server,
    hello_from_proto,
)
from colosseum.hub.state import CellKey, restrict_collecting
from colosseum.launcher import (  # one definition until T2.5(b) moves them here
    RunSetup,
    _record_run_state,
    setup_run,
    validate_run_config,
    warn_static_ownership_skew,
)
from colosseum.metrics.counters import CounterBook
from colosseum.metrics.wandb_logger import WandBLogger
from colosseum.net.messages import lineup_to_proto, version_from_proto, version_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.players.registry import FixedPlayers
from colosseum.utils.logging import setup_process_logging
from colosseum.utils.process import SHUTDOWN_GRACE_SEC, ProcessSupervisor, start_process

if TYPE_CHECKING:
    from colosseum.learner.factory import InitState, TeacherSpec

logger = logging.getLogger(__name__)

__all__ = [
    "INBOX_POLL_SEC", "LOCAL_SHUTDOWN_GRACE_SEC", "Hub", "HubOptions", "LocalNodeOptions", "RunSetup",
    "run_hub", "run_training", "setup_run", "validate_run_config", "warn_static_ownership_skew",
]

INBOX_POLL_SEC = 0.05                     # the main loop waits this long for the first inbox event of a pass
INBOX_EVENTS_PER_PASS = 2_000             # a flood of events never starves the rest of a pass
LOCAL_SHUTDOWN_GRACE_SEC = SHUTDOWN_GRACE_SEC   # train: the local node's stop (its children get 2 s less)
# Pulse results (SP4 T6.1): one message per evaluated snapshot; the pulse process puts with a blocking put.
_PULSE_RESULTS_QUEUE_SIZE = 64
RESULT_DROP_GROUP = "dropped_match_results_by_reason"
_CHECKPOINT_KIND = {"checkpoint": "periodic", "final": "final"}   # UploadHeader.kind -> the frame's meta kind (A16)
# A local client of a hub listening on all interfaces (or on loopback) connects to the loopback address of the
# family the hub bound: "[::1]" for "[::]" (also under net.ipv6.bindv6only=1), "127.0.0.1" for IPv4.
_LOOPBACK_OF = {"[::]": "[::1]", "0.0.0.0": "127.0.0.1", "127.0.0.1": "127.0.0.1", "localhost": "localhost",
                "[::1]": "[::1]"}
PENDING_EVICTIONS_MAX = 10_000            # evictions kept per node while its stream is down (oldest dropped)


@dataclass(frozen=True)
class LocalNodeOptions:
    """``train``'s local node (R12): a child of the hub. ``evaluators`` is 0 until T5.1 (the SP4 pulse child does the
    pulse, R6; amendment A10); ``data_port`` only with ``train --listen`` (``NodeOptions.data_port = -1``)."""

    workers: int
    evaluators: int
    log_dir: Path
    data_port: bool = False

    @classmethod
    def from_config(cls, config: ColosseumConfig, run_dir: RunDir) -> LocalNodeOptions:
        return cls(workers=int(config.rollout.num_workers), evaluators=0, log_dir=Path(run_dir.logs),
                   data_port=False)


@dataclass(frozen=True)
class HubOptions:
    """How the hub runs: ``mode`` ``train`` (with ``local_node``) or ``hub`` (nodes connect); ``listen``: train: None
    = loopback only, else ``ADDR:distributed.port`` (TLS) too; hub: None = all interfaces; ``port``: hub: None =
    ``distributed.port``; ``insecure``: no TLS (also ``distributed.insecure``); ``identity_dir``: the hub identity
    (default ``security.hub_identity_dir()``)."""

    mode: Literal["train", "hub", "idle"] = "train"
    listen: str | None = None
    port: int | None = None
    insecure: bool = False
    local_node: LocalNodeOptions | None = None
    identity_dir: Path | None = None


class Hub(RunSinks):
    """The hub (module docstring). ``launch`` blocks until the run ends and returns the exit code (0 budget reached or
    ``request_stop``, 1 a failure, 130 / 143 after SIGINT / SIGTERM); it may run off the main thread (signal handlers
    are then not installed, A6). ``request_stop`` and ``env_steps_done`` are thread-safe."""

    def __init__(self, config: ColosseumConfig, run_dir: RunDir, *, options: HubOptions,
                 validated: bool = False) -> None:
        """``validated``: the config was already checked by ``validate_run_config`` (``run_training`` does it before
        the run dir exists), so ``launch`` does not repeat the env spec checks and the dummy forward passes."""
        from colosseum.hub.state import BudgetCounter

        self._config = config
        self._run_dir = run_dir
        self._options = options
        self._validated = validated
        self._dist = config.distributed
        ctx = mp.get_context("spawn")
        self._mp = ctx
        self._stop_event = ctx.Event()         # the pulse child's stop; set by the signal watcher too
        # Named children (the local node, the pulse), signals and teardown.
        self._supervisor = ProcessSupervisor(self._stop_event, log_dir=run_dir.logs)
        self._stop_requested = threading.Event()
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._budget = BudgetCounter()
        self._book = CounterBook()
        self._incarnation = uuid.uuid4().hex
        self._readers: dict[int, QueueReader] = {}
        self._reported_failures: set[str] = set()
        self._signal_logged = False
        self._failure: str | None = None
        # Set by _prepare(): the coordinator, the run setup, per-agent teachers / init / checkpoint meta.
        self._coordinator: Coordinator | None = None
        self._setup: RunSetup | None = None
        self._agent_ids: list[str] = []
        self._known_agents: set[str] = set()
        self._checkpoint_meta: dict[str, dict] = {}
        self._teachers: dict[str, TeacherSpec | None] = {}
        self._init_states: dict[str, InitState | None] = {}
        self._resume_states: dict[str, dict | None] = {}
        self._run_id: str | None = None
        self._identity = None
        self._blobs = None
        self._fixed_blobs: list = []              # pb.FixedBlob of Welcome.fixed
        self._start_blobs: dict[str, tuple[str, str]] = {}   # agent -> (recovery blob id, trainer_state blob id)
        # Pure hub state (built by _prepare).
        self._registry = None
        self._epochs = None
        self._cells = None
        self._rotation = None
        self._versions = None
        self._intake = None
        self._sessions: dict[str, Any] = {}       # node id -> its current SessionHandle
        self._learner_node: dict[str, int] = {}   # agent -> node number of its learner
        self._learner_alive: dict[str, bool] = {}
        self._journal_row: int | None = None
        self._admin_handlers: dict[str, Callable[[dict], dict]] = {"hub.status": self._admin_status}
        self._handlers: dict[type, Callable[[Any], None]] = {
            NodeMessageIn: self._on_message, SessionOpened: self._on_session_opened,
            SessionClosed: self._on_session_closed, UploadIn: self._on_upload, RegisterIn: self._on_register,
            AdminIn: self._on_admin, StopRequested: self._on_stop_requested}
        self._new_versions: dict[str, Any] = {}   # accepted since the last Announce, by agent
        self._pending_evictions: dict[str, list[tuple[str, str]]] = {}   # node id -> evictions not delivered yet
        self._announced_once: set[str] = set()    # agents with an accepted version (logged once)
        self._run_stopped: set[str] = set()      # node ids that finished stop_run
        self._nodes_lost = 0
        self._warned: set[tuple] = set()
        self._handler_errors: set[str] = set()
        self._shutting_down = False
        self._next_announce = 0.0
        self._next_lost_check = 0.0
        # Sinks (set by _open_sinks).
        self._hub = None
        self._metrics_writer = None
        self._wandb: WandBLogger | None = None
        self._match_log_writer = None
        self._comparison_table = None
        self._match_log_stopped = False
        self._results_received = 0
        self._pulse_results_received = 0
        self._live_ratings = None
        self._league_state_error_logged = False
        self._hub_state_error_logged = False
        self._retention_version: int | None = None
        # The pulse (R6; set by _prepare with evaluation.enabled).
        self._pulse_plan = None
        self._pulse_feeder = None
        self._pulse_requests = None
        self._pulse_results = None
        self._pulse_state_error_logged = False
        # The server (set by _start_server).
        self._server = None
        self._ports: dict[str, int] = {}
        self._hub_info = None

    # ---- public -----------------------------------------------------------------------------------------------

    @property
    def env_steps_done(self) -> int:
        """Env steps taken by every node so far (the global budget counter)."""
        return self._budget.value

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def ports(self) -> dict[str, int]:
        """The bound ports (``hub.server.build_server``'s map); empty before the server starts."""
        return dict(self._ports)

    @property
    def identity(self) -> Any:
        return self._identity

    def request_stop(self) -> None:
        """Stop the run (exit code 0); thread-safe."""

        self._stop_requested.set()
        self._inbox.put(StopRequested())

    def _build_coordinator(self, setup: RunSetup) -> Coordinator:
        """The run's coordinator; its matchmaker reads the global env-step counter (share schedules)."""
        return Coordinator(self._config, setup.spec, setup.player_roles, checkpoint_dir=self._run_dir.checkpoints,
                           env_steps=lambda: int(self._budget.value), run_id=self._run_id)

    def launch(self) -> int:
        """Run the hub until the run ends; returns the exit code (class docstring). Everything that can fail with a
        ConfigError runs before any port or child exists (``_prepare``)."""
        self._prepare()
        code = 1
        try:
            # SIGINT / SIGTERM only record the signal from here on: the shutdown saves the final checkpoints.
            self._supervisor.install_signal_handlers()
            self._open_sinks()
            self._start_server()
            started = False
            try:
                self._start_local_node()
                self._start_pulse()
                started = True
                logger.info("Training started. Press Ctrl+C to stop." if self._options.mode == "train"
                            else "Hub started; waiting for nodes. Press Ctrl+C to stop.")
                code = self._main_loop()
            except BaseException as error:
                # Also after a failed start or a main-loop error the started children are stopped (final
                # checkpoints saved); then the original error propagates.
                if not started:
                    logger.error(f"Starting the local node or the pulse failed; stopping the "
                                 f"{len(self._supervisor.names)} already started")
                self._shutdown_after_error(error)
                raise
            killed = self._shutdown()
            code = self._exit_code_after_shutdown(code, killed)
            if code == 0 and self._failure is not None:      # reported during the shutdown (SP4 R3-08)
                code = 1
        finally:
            try:
                self._finish_metrics()
            finally:
                try:
                    self._stop_server()
                finally:
                    try:
                        self._supervisor.restore_signal_handlers()
                    finally:
                        self._release_pulse_queues()
        return code

    # ---- start ------------------------------------------------------------------------------------------------

    def _prepare(self) -> None:
        """Everything before the first port or child: the setup, the coordinator, resume / init / teachers, the hub
        identity, the registered blobs, the pulse plan and the pure hub state. Raises ConfigError / EnvContractError
        for a bad config or resume source."""
        from colosseum.core.config import config_hash
        from colosseum.hub.state import (
            CellTable,
            EpochAllocator,
            NodeRegistry,
            OwnerRotation,
            ResultIntake,
            VersionRegistry,
        )
        from colosseum.learner.factory import resolve_teacher
        from colosseum.net.security import HubIdentity

        cfg = self._config
        trainable = cfg.get_trainable_agent_ids()
        self._run_id = self._run_dir.ensure_run_id()
        logger.info(f"Colosseum: Starting the hub ({self._options.mode}; run id {self._run_id})")
        logger.info(f"  Agents: {trainable}")
        logger.info(f"  Algorithm: {cfg.algorithm.name}")
        logger.info(f"  Workers: {cfg.rollout.num_workers}" + (" (local node)" if self._options.mode == "train"
                                                                else " (per node; nodes choose their own)"))
        logger.info(f"  Envs per worker: {cfg.rollout.envs_per_worker}")
        logger.info(f"  Chunk length: {cfg.rollout.chunk_length}")
        setup = setup_run(cfg, validate=not self._validated)
        self._setup = setup
        logger.info(f"  Roles: {setup.agent_roles}; layouts: {sorted(setup.spec.layouts)}")
        logger.info(f"  Fixed players: {cfg.fixed_agent_ids() or 'none'}")
        warn_static_ownership_skew(cfg)
        self._agent_ids = list(trainable)
        self._known_agents = set(cfg.agent_ids())

        coordinator = self._build_coordinator(setup)
        self._resume_states = self._resolve_resume(setup.agent_configs, setup.role_specs)
        self._load_league_state(coordinator)          # before the import: evictions there forget PFSP entries
        self._import_snapshot_pool(coordinator)
        # Kickstart teachers and init weights are resolved here (ConfigError before any process starts).
        self._teachers = {aid: resolve_teacher(cfg, aid, setup.spec) for aid in trainable}
        self._init_states = self._resolve_init(setup.spec, self._resume_states)
        cfg_hash = config_hash(cfg)
        self._checkpoint_meta = {
            aid: {"networks": setup.agent_configs[aid].networks.model_dump(mode="json", by_alias=True),
                  "config_hash": cfg_hash}
            for aid in trainable
        }
        self._coordinator = coordinator
        self._identity = HubIdentity.load_or_create(self._options.identity_dir)
        self._blobs = BlobStore(self._run_dir)
        shutil.rmtree(self._blobs.incoming, ignore_errors=True)    # upload spools of a crashed hub
        self._register_blobs(setup, self._resume_states)
        self._pulse_setup()
        self._registry = NodeRegistry()
        self._epochs = EpochAllocator(0, self._reserve_epochs)
        self._cells = CellTable(self._epochs)
        self._rotation = OwnerRotation(trainable)
        self._versions = VersionRegistry()
        self._intake = ResultIntake(journal=bool(cfg.match_log.enabled))

    def _register_blobs(self, setup: RunSetup, resume_states: Mapping[str, dict | None]) -> None:
        """What nodes download from the hub: frozen agents' weights, neural kickstart teachers and ``init`` weights
        (``Welcome.fixed``), and every resumed agent's start state (A28 until T4.1: a ``recovery`` weights frame and
        its ``trainer_state`` bytes, named by the agent's ``LearnerAssignment``)."""
        from colosseum.net.chunks import WeightsMeta, encode_weights

        def register(kind: str, agent_id: str, data: bytes) -> str:
            ident = blob_ident(kind, agent_id, data)
            self._blobs.register_fixed(ident, data, kind=kind, agent_id=agent_id)
            return ident

        for agent_id, frozen in setup.fixed.frozen.items():
            frame = encode_weights(frozen.model_state, WeightsMeta(
                agent_id, "fixed", 0, 0, architecture=dict(frozen.networks), roles=tuple(frozen.roles)))
            self._fixed_blobs.append(pb.FixedBlob(agent_id=agent_id, blob_id=register("fixed", agent_id, frame),
                                                  kind="fixed"))
        for agent_id, teacher in self._teachers.items():
            if teacher is None or teacher.kind != "neural" or teacher.frozen is None:
                continue
            frame = encode_weights(teacher.frozen.model_state, WeightsMeta(
                agent_id, "teacher", 0, 0, architecture=dict(teacher.frozen.networks),
                roles=tuple(teacher.frozen.roles)))
            self._fixed_blobs.append(pb.FixedBlob(agent_id=agent_id, blob_id=register("teacher", agent_id, frame),
                                                  kind="teacher"))
        for agent_id, init in self._init_states.items():
            if init is None:
                continue
            frame = encode_weights(init.model_state, WeightsMeta(agent_id, "init", 0, 0))
            self._fixed_blobs.append(pb.FixedBlob(agent_id=agent_id, blob_id=register("init", agent_id, frame),
                                                  kind="init"))
        for agent_id, state in resume_states.items():
            if state is None:
                continue
            frame = encode_weights(state["model_state"], WeightsMeta(
                agent_id, "recovery", 0, int(state["policy_version"])))
            trainer = state.get("trainer_state")
            self._start_blobs[agent_id] = (register("recovery", agent_id, frame),
                                           "" if trainer is None else register("trainer_state", agent_id,
                                                                               bytes(trainer)))

    def _pulse_setup(self) -> None:
        """R6: the pulse plan, its feeder and the two queues (with ``evaluation.enabled``)."""
        cfg = self._config
        if not cfg.evaluation.enabled:
            return
        from colosseum.arena.pulse import PulseFeeder, pulse_plan

        self._pulse_plan = pulse_plan(cfg, self._setup.spec, self._setup.player_roles)
        self._pulse_feeder = PulseFeeder(self._pulse_plan.agents, cfg.evaluation.every, cfg.evaluation.max_backlog)
        self._pulse_requests = self._mp.Queue(maxsize=1)
        self._pulse_results = self._mp.Queue(maxsize=_PULSE_RESULTS_QUEUE_SIZE)

    def _open_sinks(self) -> None:
        """The metrics hub, the journal (and its writer), the live ratings and the retention wiring, before any port
        or child exists: an open failure cannot orphan a node; ``_finish_metrics`` closes whatever was opened."""
        self._open_metrics(self._resume_states)
        self._open_match_log(self._coordinator)
        self._open_live_ratings()
        self._wire_retention()

    def _start_server(self) -> None:
        """The hub's gRPC server (``hub.server.build_server``), ``hub.json`` and the running-hubs registry entry.
        ``train`` without ``--listen``: one plain loopback port; ``train --listen ADDR``: TLS on
        ``ADDR:distributed.port`` too; ``hub``: TLS on all interfaces (``--listen`` narrows) at ``--port`` /
        ``distributed.port``; ``--insecure`` / ``distributed.insecure``: plain everywhere."""
        from colosseum.hub.registry import register_running, this_process_info, write_hub_json

        opts, dist = self._options, self._dist
        insecure = bool(opts.insecure or dist.insecure)
        if opts.mode == "train":
            loopback_plain = True
            hosts = [] if opts.listen is None else [opts.listen]
            port = int(dist.port)
        else:
            loopback_plain = False
            hosts = [opts.listen or "[::]"]
            port = int(opts.port if opts.port is not None else dist.port)

        def build(listen_hosts: list[str]):
            return build_server(inbox=self._inbox, identity=self._identity, insecure=insecure,
                                listen=[f"{h}:{port}" for h in listen_hosts], loopback_plain=loopback_plain,
                                blobs=self._blobs, max_message_mb=int(dist.max_message_mb))

        from colosseum.core.errors import ConfigError

        try:
            try:
                server, ports = build(hosts)
            except OSError:
                if hosts != ["[::]"]:
                    raise
                hosts = ["0.0.0.0"]                # no IPv6 on this machine: every IPv4 interface
                server, ports = build(hosts)
        except OSError as e:
            raise ConfigError(f"the hub cannot listen: {e}; choose another port (--port or distributed.port; 0 = any "
                              f"free port) or address (--listen)") from e
        server.start()
        self._server, self._ports = server, dict(ports)
        addresses = tuple(f"{h}:{ports[f'{h}:{port}']}" for h in hosts)
        if LOOPBACK_KEY in ports:
            loopback = f"127.0.0.1:{ports[LOOPBACK_KEY]}"
        elif hosts and hosts[0] in _LOOPBACK_OF:          # the same address family as the bound socket (M8)
            loopback = f"{_LOOPBACK_OF[hosts[0]]}:{ports[f'{hosts[0]}:{port}']}"
        else:
            loopback = None
        info = this_process_info(run_id=self._run_id, run_dir=str(self._run_dir.root), addresses=addresses,
                                 loopback=loopback, tls=bool(addresses) and not insecure,
                                 ca_fingerprint=self._identity.fingerprint, identity_dir=str(self._identity.path),
                                 mode=opts.mode)
        write_hub_json(self._run_dir, info)
        register_running(self._identity.path, info)
        self._hub_info = info
        tls = "plain" if insecure else "TLS"
        for address in addresses:
            logger.info(f"Hub listening on {address} ({tls})")
        if LOOPBACK_KEY in ports:
            logger.info(f"Hub listening on {loopback} (loopback, plain)")

    def _stop_server(self) -> None:
        """Stop the server at once (pending Register / Upload / Admin handlers end; carried item of T2.3) and remove
        the running-registry entry; ``hub.json`` stays."""
        from colosseum.hub.registry import unregister_running

        server, self._server = self._server, None
        try:
            if server is not None:
                server.stop(None).wait(5.0)
        finally:
            if self._hub_info is not None:
                try:
                    unregister_running(self._identity.path, self._hub_info)
                except OSError as e:
                    logger.warning(f"cannot remove the running-hub entry of this run: {e}")

    def _start_local_node(self) -> None:
        """R12: ``train``'s node as a spawned child (number 0, an in-memory node record: no ``Register``) on the
        plain loopback port; it starts the workers and the learners."""
        local = self._options.local_node
        if local is None:
            return
        from colosseum.node.agent import NodeOptions, local_node_target
        from colosseum.node.hardware import machine_id

        record, secret = self._identity.nodes.add_ephemeral("local", machine_id())
        options = NodeOptions(hub_address=f"127.0.0.1:{self._ports[LOOPBACK_KEY]}", token=self._identity.token,
                              loopback_plain=True, workers=int(local.workers), evaluators=int(local.evaluators),
                              local=True, credentials=(record.node_id, 0, secret), log_dir=Path(local.log_dir),
                              data_port=-1 if local.data_port else None)
        # Not daemonic: the node starts its own children (carried item of T2.4); it dies with the hub (PDEATHSIG)
        # and the shutdown waits for it (LOCAL_SHUTDOWN_GRACE_SEC) before it is killed.
        proc = self._mp.Process(target=local_node_target, name="node",
                                kwargs={"options": options, "log_dir": str(local.log_dir)}, daemon=False)
        start_process(proc)
        self._supervisor.add("node", proc)
        logger.info(f"Local node started with {local.workers} workers")

    def _start_pulse(self) -> None:
        """R6: the SP4 pulse child (with ``evaluation.enabled``)."""
        if self._pulse_feeder is None:
            return
        from colosseum.arena.pulse import pulse_target

        cfg, setup = self._config, self._setup
        fixed, names = setup.fixed, self._pulse_plan.field
        field_specs = FixedPlayers(bots={n: fixed.bots[n] for n in names if n in fixed.bots},
                                   frozen={n: fixed.frozen[n] for n in names if n in fixed.frozen},
                                   roles={n: fixed.roles[n] for n in names})
        pulse_proc = self._mp.Process(target=pulse_target, name="pulse", daemon=True, kwargs=dict(
            log_dir=str(self._run_dir.logs), config=cfg, requests=self._pulse_requests,
            results=self._pulse_results, stop_event=self._stop_event, field_specs=field_specs, field=list(names),
            agents=self._pulse_plan.agents, run_id=self._run_id))
        start_process(pulse_proc)
        self._supervisor.add("pulse", pulse_proc)
        logger.info(f"Pulse started for {list(self._pulse_plan.agents)} against {names}")

    # ---- the main loop ----------------------------------------------------------------------------------------

    def _main_loop(self) -> int:
        """Run until the budget is reached, a signal arrives, ``request_stop``, or a failure (module docstring).
        Returns the exit code: 0 budget reached (or ``request_stop``), 1 failure, 128+signum on SIGINT/SIGTERM."""
        total = self._config.training.total_timesteps
        while True:
            self._drain_inbox(INBOX_POLL_SEC)
            self._drain_commits()
            self._reapply_retention()   # after a new live refit; the refit thread never touches storage
            self._pump_pulse()
            now = time.monotonic()
            if now >= self._next_lost_check:
                self._next_lost_check = now + min(1.0, self._dist.heartbeat_sec)
                for entry in self._registry.lost(now, self._dist.node_lost_after_sec):
                    self._on_node_lost(entry)
            if now >= self._next_announce:
                self._broadcast_announce()
                self._next_announce = now + float(self._dist.heartbeat_sec)
            env_steps = self._budget.value
            # The method itself: the hub builds the ratings snapshot only when a tick is due (SP4 spec block 7.3).
            if self._hub.maybe_tick(env_steps=env_steps, ratings=self._coordinator.ratings_snapshot,
                                    queue_depths=self._hub.intake_queue_depths(), match_log=self._match_log_stats(),
                                    bt=self._live_snapshot()):
                self._save_league_state()
                self._save_hub_state(fsync=False)
            if self._failure is not None:
                return 1
            if self._supervisor.received_signal is not None:
                return self._signal_exit_code()
            if self._stop_requested.is_set():
                if self._supervisor.received_signal is not None:     # arrived since the check above
                    return self._signal_exit_code()
                logger.info(f"Stop requested after {env_steps} env steps (budget {total})")
                return 0
            if env_steps >= total:
                logger.info(f"Training budget reached: {env_steps} env steps (budget {total})")
                return 0
            failure = self._supervisor.first_failure()     # the local node or the pulse child ended
            if failure is not None:
                logger.error(failure.message())
                self._reported_failures.add(failure.name)
                return 1

    def _drain_inbox(self, timeout: float) -> int:
        """Handle the inbox's events: wait up to ``timeout`` for the first, then take what is there (at most
        ``INBOX_EVENTS_PER_PASS``). Returns how many were handled."""
        try:
            event = self._inbox.get(timeout=timeout) if timeout > 0 else self._inbox.get_nowait()
        except queue.Empty:
            return 0
        handled = 0
        while True:
            self._dispatch(event)
            handled += 1
            if handled >= INBOX_EVENTS_PER_PASS:
                return handled
            try:
                event = self._inbox.get_nowait()
            except queue.Empty:
                return handled

    def _dispatch(self, event: object) -> None:
        """The handler of ``event``'s type (one dict lookup per event: the result path's cost, criterion 2)."""
        handler = self._handlers.get(type(event))
        if handler is None:
            logger.warning(f"Hub: ignored an unknown inbox event {type(event).__name__}")
            return
        self._guarded(handler, event)

    def _on_stop_requested(self, _event: Any) -> None:
        self._stop_requested.set()

    def _guarded(self, handler: Callable[[Any], None], event: object) -> None:
        """Run one handler; its exception is one ERROR per handler (with the traceback the first time): nothing a
        node sends stops the hub."""
        try:
            handler(event)
        except Exception as e:  # noqa: BLE001 - one bad event never stops the hub
            name = getattr(handler, "__name__", "handler")
            first = name not in self._handler_errors
            self._handler_errors.add(name)
            logger.error(f"Hub: {name} failed on a {type(event).__name__} ({type(e).__name__}: {e})",
                         exc_info=first)

    # ---- sessions ---------------------------------------------------------------------------------------------

    def _on_session_opened(self, ev: Any) -> None:
        """A node's ``Hello`` (P-PF2-1 (c)): an older live stream of the node is closed as superseded; the node is
        connected (``new`` / ``resumed`` / ``returned``: a returned node's cells are dropped), its batch bookkeeping
        opened with ``Hello.last_acked_seq``, its learners placed, and a ``Welcome`` sent."""

        handle, hello = ev.handle, ev.hello
        record = self._identity.nodes.get(handle.node_id)
        if record is None:
            handle.close("unknown node")
            return
        old = self._sessions.get(handle.node_id)
        if old is not None and old is not handle:
            old.close("superseded")
        self._sessions[handle.node_id] = handle
        self._run_stopped.discard(handle.node_id)
        node_hello = hello_from_proto(hello)
        kind = self._registry.connect(handle.node_id, record.number, record.name, hello.session_id, node_hello,
                                      time.monotonic(), data_path="local" if record.number == 0 else "relay",
                                      serial=handle.serial)
        if kind == "returned":
            self._cells.drop_node(record.number)
        self._intake.open(handle.node_id, hello.session_id, int(hello.last_acked_seq))
        self._budget.report(handle.node_id, hello.session_id, int(hello.env_steps))
        placed = self._place_learners(record.number, node_hello)
        cfg = self._config
        agents = [pb.AgentState(agent_id=aid, kind=cfg.agent_kind(aid), status="active", start_env_steps=0)
                  for aid in cfg.agent_ids()]
        versions = [version_to_proto(v) for aid in self._agent_ids if (v := self._versions.latest(aid)) is not None]
        handle.send(pb.HubMessage(welcome=pb.Welcome(
            node_number=record.number, run_id=self._run_id, config_json=cfg.model_dump_json(by_alias=True),
            agents=agents, versions=versions, evictions=self._take_evictions(handle.node_id),
            learners=[self._assignment(a) for a in placed], evaluators=0, env_steps=self._budget.value,
            fixed=list(self._fixed_blobs), collect_allowed=True)))
        logger.info(f"node {record.number} ({record.name}) connected ({kind}); learners here: {placed or 'none'}")
        if self._shutting_down:
            self._send_stop_run(handle)

    def _place_learners(self, number: int, hello: Any) -> list[str]:
        """The trainable agents whose learner runs on node ``number``. Until T4.1 (A11): ``train`` places every
        learner on its local node (number 0); ``hub`` on the first node that connects with learner capacity (not
        ``--no-learners``, CPU learner slots or GPUs). Every placed learner is generation 1."""
        if not self._learner_node:
            slots = hello.slots or {}
            if self._options.mode == "train":
                capable = number == 0
            else:
                capable = (not slots.get("no_learners", False)
                           and (int(slots.get("cpu_learners", 0) or 0) != 0 or bool(slots.get("gpus"))))
            if capable:
                for agent in self._agent_ids:
                    self._versions.new_generation(agent)
                    self._learner_node[agent] = number
                    self._learner_alive[agent] = True
                logger.info(f"Learners of {self._agent_ids} placed on node {number}")
        return [a for a in self._agent_ids if self._learner_node.get(a) == number]

    def _assignment(self, agent: str) -> Any:
        """The ``LearnerAssignment`` of ``agent`` (A28: a resumed agent names its ``recovery`` / ``trainer_state``
        blobs)."""

        weights, trainer = self._start_blobs.get(agent, ("", ""))
        return pb.LearnerAssignment(agent_id=agent, generation=self._versions.generation(agent),
                                    version_floor=self._versions.version_floor(agent),
                                    device=str(self._setup.agent_configs[agent].learner.device),
                                    weights_blob=weights, trainer_state_blob=trainer, env_steps=self._budget.value)

    def _on_session_closed(self, ev: Any) -> None:
        """Only the node's current stream counts (P-PF2-1 (c)); the node stays online until ``node_lost_after_sec``
        (it may come back with its session and cells)."""
        if self._sessions.get(ev.node_id) is ev.handle:
            self._registry.disconnect(ev.node_id, ev.handle.serial)
            del self._sessions[ev.node_id]
            if not self._shutting_down:
                logger.info(f"node {self._node_label(ev.node_id)}: session closed ({ev.reason})")

    def _on_node_lost(self, entry: Any, *, reason: str = "lost", relocate_now: bool = False) -> None:
        """A node silent for ``node_lost_after_sec`` (or removed: ``reason``): its cells go (their unfinished
        matches are never reported), its stream is closed. T4.2 adds the relocation of its learners."""
        self._cells.drop_node(entry.number)
        handle = self._sessions.pop(entry.node_id, None)
        if handle is not None:
            handle.close(reason)
        self._nodes_lost += 1
        why = ""
        if reason == "lost":                          # it may come back and resend its unacked batches
            why = f": no heartbeat for more than {self._dist.node_lost_after_sec:g} s"
        else:                                         # gone for good (nodes remove, T6.3; ResultIntake contract)
            self._intake.forget_node(entry.node_id)
            self._pending_evictions.pop(entry.node_id, None)
        logger.warning(f"node {entry.number} ({entry.name}) {reason}{why}; its cells are dropped")

    def _node_label(self, node_id: str) -> str:
        entry = self._registry.by_id(node_id)
        return node_id[:8] if entry is None else f"{entry.number} ({entry.name})"

    # ---- node messages ----------------------------------------------------------------------------------------

    def _on_message(self, ev: Any) -> None:
        """One ``NodeMessage`` of a node's current stream (a superseded stream's messages are ignored, P-PF2-1),
        dispatched to the handler of its oneof."""
        handle = self._sessions.get(ev.node_id)
        if handle is None or handle.serial != ev.serial:
            return
        entry = self._registry.by_id(ev.node_id)
        if entry is None or entry.state != "online":
            return
        message = ev.message
        kind = message.WhichOneof("m")
        if kind == "heartbeat":
            self._on_heartbeat(ev.node_id, ev.session_id, message.heartbeat)
        elif kind == "lineup_request":
            self._on_lineup_request(ev.node_id, ev.session_id, message.lineup_request)
        elif kind == "results":
            self._on_result_batch(ev)
        elif kind == "metrics":
            self._on_metrics_batch(entry, message.metrics)
        elif kind == "version":
            self._on_version(entry, message.version.info)
        elif kind == "event":
            self._on_event(entry, message.event)
        elif kind == "hello":
            self._warn_once(("hello", entry.number), f"node {entry.number} ({entry.name}) sent a second Hello on its "
                                                     f"stream; ignored")
        else:
            logger.debug(f"node {entry.number}: ignored a {kind} message (not handled by this hub yet)")

    def _on_heartbeat(self, node_id: str, session_id: str, heartbeat: Any) -> None:
        """Registry, budget (cumulative env steps per (node, session), spec 4.4) and the node's counters
        (``Hub._book`` source ``<node_id>/<session_id>``, A2)."""
        if not self._registry.heartbeat(node_id, session_id, time.monotonic()):
            return
        self._budget.report(node_id, session_id, int(heartbeat.env_steps))
        self._book.report(f"{node_id}/{session_id}", {str(k): int(v) for k, v in heartbeat.counters.items()})

    def _collecting_any(self) -> list[str]:
        """Agents that can collect anywhere: an accepted latest version and a learner (spec 4.2, 4.7)."""
        return [a for a in self._agent_ids
                if self._learner_alive.get(a) and self._versions.latest(a) is not None]

    def _collecting_on(self, number: int) -> list[str]:
        """Agents that can collect on node ``number``: until T3.2 only those whose learner runs there (A11)."""
        return [a for a in self._collecting_any() if self._learner_node.get(a) == number]

    def _on_lineup_request(self, node_id: str, session_id: str, req: Any) -> None:
        """Answer a node's ``LineupRequest`` in this pass (spec 4.2; P-PF2-2, P-PF2-3, P-PF2-17; A11)."""

        entry = self._registry.by_id(node_id)
        handle = self._sessions.get(node_id)
        if entry is None or handle is None or entry.session_id != session_id:
            return
        collecting = self._collecting_any()
        if not collecting:                           # P-PF2-3: nothing allocated; the node asks again
            logger.debug(f"node {entry.number}: no agent can collect yet; {len(req.cells)} cells wait")
            return
        workers = int(entry.slots.get("workers", 0) or 0)
        envs = int(self._config.rollout.envs_per_worker)
        now = time.monotonic()
        requested, bad = [], 0
        seen = set()
        for c in req.cells:
            if 0 <= c.worker < workers and 0 <= c.env < envs and (c.worker, c.env) not in seen:
                seen.add((c.worker, c.env))
                requested.append((CellKey(entry.number, int(c.worker), int(c.env)), int(c.epoch),
                                  int(c.incarnation)))
            else:
                bad += 1
        if bad:                                      # P-PF2-17: outside the node's slots
            self._count_drop(entry.number, "node_messages", "invalid", "requested cells outside its slots", bad)
        if not requested:
            return
        cells = self._cells.request(entry.number, requested, now)
        keys = [c.key for c in cells]
        owners = self._rotation.assign(keys, collecting=collecting, owners=self._cells.owners(exclude=keys),
                                       previous={c.key: c.owner if c.owner is not None else c.previous
                                                 for c in cells})                     # P-PF2-2
        local = self._collecting_on(entry.number)
        ttl = float(self._config.rollout.match_refresh_interval_sec)
        grants = []
        for cell in cells:
            owner = owners[cell.key]
            lineup = restrict_collecting(self._coordinator.lineup_for(owner), local)
            lineup = dataclasses.replace(lineup, epoch=cell.epoch)
            self._cells.grant(cell.key, owner, ttl, now)
            grants.append(pb.Grant(worker=cell.key.worker, env=cell.key.env, epoch=cell.epoch, ttl_sec=ttl,
                                   lineup=lineup_to_proto(lineup)))
        handle.send(pb.HubMessage(grants=pb.LineupGrants(grants=grants)))

    def _on_metrics_batch(self, entry: Any, metrics: Any) -> None:
        """Learners' train records, workers' stats (keyed by (node, worker, incarnation)) and learners' intake
        reports (queue depth, chunk drops: ``MetricsHub.on_intake_stats``, A2)."""
        for train in metrics.train:
            if train.agent_id not in self._agent_ids:
                self._count_drop(entry.number, "dropped_metrics", "invalid", "train records of unknown agents")
                continue
            self._hub.on_train_metrics({"agent_id": train.agent_id, "train_step": int(train.train_step),
                                        **{str(k): float(v) for k, v in train.values.items()}})
        for stats in metrics.workers:
            self._hub.on_worker_stats({"node": entry.number, "worker_id": int(stats.worker),
                                       "incarnation": int(stats.incarnation),
                                       **{str(k): int(v) for k, v in stats.counters.items()}})
        for intake in metrics.intake:
            if intake.agent_id not in self._agent_ids:
                self._count_drop(entry.number, "dropped_metrics", "invalid", "intake reports of unknown agents")
                continue
            self._hub.on_intake_stats(intake.agent_id, int(intake.generation), int(intake.depth),
                                      {str(k): int(v) for k, v in intake.dropped.items()})

    def _on_version(self, entry: Any, info_proto: Any) -> None:
        """A learner's ``VersionAnnounce``: accepted only from the agent's learner node and generation (spec 4.5);
        an accepted version makes the agent collect and goes out with the next ``Announce``."""

        info = version_from_proto(info_proto)
        if self._learner_node.get(info.agent_id) != entry.number or info.node != entry.number:
            outcome = "invalid"
        else:
            outcome = self._versions.announce(info)
        if outcome == "accepted":
            self._new_versions[info.agent_id] = info
            if info.agent_id not in self._announced_once:
                self._announced_once.add(info.agent_id)
                logger.info(f"agent {info.agent_id}: first weights (v{info.version}) published on node "
                            f"{entry.number}; its seats collect from now on")
            return
        self._count_drop(entry.number, "dropped_announcements", outcome, f"version announcements of "
                         f"{info.agent_id!r} (v{info.version}, generation {info.generation})")

    def _on_event(self, entry: Any, event: Any) -> None:
        """A node's ``Event`` (A1 kinds). A reported child error and a failed learner stop the run (A12, until
        T4.2: ``_fail_run``, first wins); a death without a report is a WARNING (the node restarts the worker and
        counts ``worker_restarts`` itself, A2)."""
        where = f"node {entry.number} ({entry.name})"
        kind = event.kind
        if kind == "child_error":
            log = f"; see {event.log_path}" if event.log_path else ""
            self._fail_run(f"{event.proc} failed ({event.error_kind}): {event.message}{log}",
                           context=event.context or where)
        elif kind == "learner_failed":
            code = f" (exit {event.exit_code})" if event.exit_code else ""
            why = f": {event.message}" if event.message else ""
            log = f"; see {event.log_path}" if event.log_path else ""
            for agent, number in self._learner_node.items():
                if number == entry.number and event.proc == f"learner-{agent}":
                    self._learner_alive[agent] = False
            self._fail_run(f"{event.proc} on {where} ended{code}{why}{log}")
        elif kind == "proc_died":
            logger.warning(f"{where}: {event.proc} died (exit {event.exit_code}) without reporting an error; the node "
                           f"restarts it; see {event.log_path}")
        elif kind == "slot_disabled":
            logger.warning(f"{where}: {event.proc} died too often; its slot is disabled; see {event.log_path}")
        elif kind == "run_stopped":
            self._run_stopped.add(entry.node_id)
            logger.info(f"{where} stopped its processes")
        elif kind == "warning":
            logger.warning(f"{where}: {event.proc}: {event.message}")
        else:
            logger.debug(f"{where}: event {kind!r} ignored")

    def _fail_run(self, message: str, *, context: str = "") -> None:
        """Stop the run with exit code 1 (A6): one ERROR; the first failure wins (a crashed learner reports both
        ``child_error`` and ``learner_failed``: logged once)."""
        if self._failure is not None:
            return
        self._failure = f"{message} [{context}]" if context else message
        logger.error(f"Run failed: {self._failure}")

    # ---- results (R8) -----------------------------------------------------------------------------------------

    def _on_result_batch(self, ev: Any) -> None:
        """A node's ``ResultBatch`` (module docstring, R8)."""
        entry = self._registry.by_id(ev.node_id)
        if entry is None:
            return
        seq = int(ev.message.results.seq)
        if not self._intake.begin(ev.node_id, ev.session_id, seq):
            if self._intake.is_acked(ev.node_id, ev.session_id, seq):
                self._ack(ev.node_id, seq)                      # the node missed the ack
            return
        if ev.results is None:                     # undecodable: dropped whole and acked (no poison resend)
            self._count_drop(entry.number, RESULT_DROP_GROUP, "invalid", f"match results ({ev.error})",
                             max(1, len(ev.message.results.results)))
            self._settled(ev.node_id, ev.session_id, seq)
            return
        prefix = f"{self._run_id}/train/n{entry.number}/"
        valid = []
        for result in ev.results:
            reason = self._intake.check(result, spec=self._setup.spec, agents=self._known_agents,
                                        versions=self._versions, node=entry.number, uid_prefix=prefix)
            if reason is not None:
                self._count_drop(entry.number, RESULT_DROP_GROUP, reason, "match results")
                continue
            result.env_steps = self._budget.value               # stamped at intake (spec 4.3; P-PF2-13)
            valid.append(result)
        if not valid:
            self._settled(ev.node_id, ev.session_id, seq)
            return
        writer = self._match_log_writer
        if writer is None or writer.dead:
            if writer is not None:                              # the journal is unusable: accounted, not journaled
                self._drop_writer(len(valid))
            self._results_received += len(valid)
            self._account(valid)
            self._settled(ev.node_id, ev.session_id, seq)
            return
        if writer.submit(valid, token=(ev.node_id, ev.session_id, seq)):
            self._intake.failed(ev.node_id, ev.session_id, seq)  # no room: the node resends it
            return
        self._results_received += len(valid)

    def _drain_commits(self) -> None:
        """Account the writer's commits in commit order, then ack their batches (R8)."""
        if self._match_log_writer is None:
            return
        for commit in self._match_log_writer.take_commits():
            node_id, session_id, seq = commit.token
            if commit.failed:
                self._intake.failed(node_id, session_id, seq)    # resent; a dead writer then takes the intake path
                continue
            self._account(commit.accepted)                       # duplicates (uid already journaled) are not
            if commit.last_row is not None:
                self._journal_row = max(self._journal_row or 0, int(commit.last_row))
            self._settled(node_id, session_id, seq)

    def _account(self, results: list) -> None:
        for result in results:
            self._coordinator.report_match_result(result)
            self._hub.on_match_result(result)

    def _settled(self, node_id: str, session_id: str, seq: int) -> None:
        self._intake.committed(node_id, session_id, seq)
        self._ack(node_id, seq)

    def _ack(self, node_id: str, seq: int) -> None:

        handle = self._sessions.get(node_id)
        if handle is not None:
            handle.send(pb.HubMessage(ack=pb.Ack(seqs=[int(seq)])))

    def _count_drop(self, node: int, group: str, reason: str, what: str, n: int = 1) -> None:
        """Count ``n`` dropped items in ``Hub._book`` as ``<group>.<reason>`` (A2, A6); one WARNING per (node,
        group, reason)."""
        self._book.add(f"{group}.{reason}", int(n))
        if (node, group, reason) not in self._warned:
            self._warned.add((node, group, reason))
            logger.warning(f"node {node}: dropped {n} {what} ({group}.{reason}); further drops of this kind are only "
                           f"counted")

    def _drop_writer(self, n: int) -> None:
        """Results accounted without the journal because its writer is dead (``..._by_reason.writer``); one ERROR."""
        self._book.add(f"{RESULT_DROP_GROUP}.writer", int(n))
        if ("writer",) not in self._warned:
            self._warned.add(("writer",))
            logger.error("Match log writer is not running (the journal is unusable): match results are accounted "
                         "without being journaled (counted in dropped_match_results_by_reason.writer)")

    def _warn_once(self, key: tuple, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(text)

    # ---- uploads, registrations, admin ------------------------------------------------------------------------

    def _on_upload(self, ev: Any) -> None:
        """A learner's snapshot (``checkpoint`` / ``final``): checked (P-PF2-9 (b): a trainable agent of the run whose
        learner runs on the uploading node, its current generation, a matching frame), saved with the hub's env
        steps (R9) and answered. The spool file may have vanished (the servicer's timeout or a cancel deleted it
        and told the node so): then nothing is saved (carried item of T2.3)."""
        from colosseum.net.chunks import decode_checkpoint
        from colosseum.net.frame import payload_byte_cap

        header = ev.header
        record = self._identity.nodes.get(ev.node_id)
        number = -1 if record is None else record.number
        claimed = ev.path.with_name(ev.path.name + ".claimed")

        def reply(accepted: bool, reason: str = "", checkpoint_id: str = "") -> None:
            if not ev.reply.done():
                ev.reply.set_result(pb.UploadReply(accepted=accepted, reason=reason, checkpoint_id=checkpoint_id))

        try:
            try:
                ev.path.rename(claimed)               # the servicer's later unlink cannot take it any more
            except FileNotFoundError:
                logger.info(f"node {number}: an upload of {header.agent_id!r} v{header.version} was given up "
                            f"before the hub read it")
                return
            if header.kind not in _CHECKPOINT_KIND:
                self._count_drop(number, "dropped_uploads", "invalid", f"{header.kind!r} uploads")
                reply(False, "kind")
                return
            agent, generation = header.agent_id, int(header.generation)
            if agent not in self._agent_ids or self._learner_node.get(agent) != number or generation < 1:
                self._count_drop(number, "dropped_uploads", "invalid", f"uploads of {agent!r}")
                reply(False, "invalid")
                return
            if generation != self._versions.generation(agent):
                self._count_drop(number, "dropped_uploads", "stale_generation", f"uploads of {agent!r}")
                reply(False, "stale_generation")
                return
            with open(claimed, "rb") as f:
                data = bytearray(f.read())            # writable: the saved arrays are views of it
            try:
                payload, meta = decode_checkpoint(data, cap=payload_byte_cap(int(self._dist.max_message_mb)))
            except ValueError as e:                   # FrameError: a malformed frame
                self._count_drop(number, "dropped_uploads", "invalid", f"undecodable uploads ({e})")
                reply(False, "invalid")
                return
            mismatch = [what for what, ok in (
                ("agent", payload["agent_id"] == agent), ("generation", int(meta["generation"]) == generation),
                ("version", int(payload["policy_version"]) == int(header.version)),
                ("kind", meta["kind"] == _CHECKPOINT_KIND[header.kind])) if not ok]
            if mismatch:                              # the frame says something else than its header (M6)
                self._count_drop(number, "dropped_uploads", "invalid", f"uploads of {agent!r} whose frame does not "
                                 f"match the header ({', '.join(mismatch)})")
                reply(False, f"invalid: {', '.join(mismatch)}")
                return
            checkpoint_id = self._save_checkpoint(payload)
            reply(True, checkpoint_id=checkpoint_id)
        except Exception as e:  # noqa: BLE001 - one bad upload never stops the hub
            logger.error(f"Saving the {header.kind} snapshot v{header.version} of {header.agent_id!r} from node "
                         f"{number} failed ({type(e).__name__}: {e})", exc_info=True)
            reply(False, f"error: {type(e).__name__}")
        finally:
            claimed.unlink(missing_ok=True)
            ev.path.unlink(missing_ok=True)

    def _on_register(self, ev: Any) -> None:
        """``Register`` with the token (T6.3 adds the version / platform checks before it)."""

        req = ev.request
        try:
            record, secret = self._identity.nodes.register(machine_id=req.machine_id, name=req.name,
                                                           tags=list(req.tags), node_id=req.node_id or None,
                                                           run_id=self._run_id or "")
            cert = self._identity.issue_node_certificate(record.node_id, req.public_key_pem.encode("ascii"))
        except Exception as e:  # noqa: BLE001 - the servicer maps it to a status
            ev.reply.set_exception(e)
            return
        logger.info(f"node {record.number} ({record.name}) registered from {ev.peer}")
        ev.reply.set_result(pb.RegisterReply(node_id=record.node_id, number=record.number, name=record.name,
                                             certificate_pem=cert.decode("ascii"), node_secret=secret,
                                             ca_pem=self._identity.ca_pem.decode("ascii")))

    def _on_admin(self, ev: Any) -> None:
        """An admin command (``hub.admin.ADMIN_COMMANDS``): code 0 with the handler's data, 2 for an unknown command,
        1 when the handler fails."""
        from concurrent.futures import Future


        handler = self._admin_handlers.get(ev.command)
        if handler is None:
            reply = pb.AdminReply(code=2, text=f"unknown command {ev.command[:64]!r}; this hub knows "
                                               f"{sorted(self._admin_handlers)}")
        else:
            try:
                reply = pb.AdminReply(code=0, text="ok", data_json=json.dumps(handler(dict(ev.args))))
            except Exception as e:  # noqa: BLE001 - the caller gets the error
                reply = pb.AdminReply(code=1, text=f"{ev.command}: {type(e).__name__}: {e}")
        reply.done = True
        if isinstance(ev.reply, Future):
            if not ev.reply.done():
                ev.reply.set_result(reply)
        else:
            ev.reply.put(reply)

    def _admin_status(self, args: dict) -> dict:
        return {"run_id": self._run_id, "mode": self._options.mode, "env_steps": self._budget.value,
                "nodes": [{"number": e.number, "name": e.name, "state": e.state} for e in self._registry.all()]}

    # ---- periodic ---------------------------------------------------------------------------------------------

    def _broadcast_announce(self) -> None:
        """The periodic ``Announce`` to every live session: the hub's env steps (the learners' LR progress), the
        versions accepted since the last one and the snapshots evicted since the node's last ``Announce`` or
        ``Welcome`` (kept per online node while its stream is down, at most ``PENDING_EVICTIONS_MAX``)."""
        new = [(a, c) for a, ids in sorted(self._coordinator.take_evictions().items()) for c in ids]
        if new:
            for entry in self._registry.all():
                if entry.state == "online":
                    pending = self._pending_evictions.setdefault(entry.node_id, [])
                    pending.extend(new)
                    del pending[:-PENDING_EVICTIONS_MAX]
        versions = [version_to_proto(v) for v in self._new_versions.values()]
        self._new_versions = {}
        for node_id, handle in list(self._sessions.items()):
            handle.send(pb.HubMessage(announce=pb.Announce(
                versions=versions, evictions=self._take_evictions(node_id), env_steps=self._budget.value,
                collect_allowed=True)))

    def _take_evictions(self, node_id: str) -> list:
        """The evictions not yet delivered to ``node_id`` (``pb.Eviction``), now handed out."""
        return [pb.Eviction(agent_id=a, checkpoint_id=c) for a, c in self._pending_evictions.pop(node_id, [])]

    def _reserve_epochs(self, until: int) -> None:
        """R7: ``hub_state.json`` with fsync before an epoch above ``until``'s block start is handed out."""
        self._save_hub_state(fsync=True, epoch_reserved_until=until)

    def _save_hub_state(self, *, fsync: bool, epoch_reserved_until: int | None = None) -> None:
        """``hub_state.json`` (A3), atomically. A failing reservation (``fsync``) propagates: no epoch is granted
        unreserved; a failing tick write is one ERROR."""
        from colosseum.hub.state import HUB_STATE_FILE, HUB_STATE_VERSION, HubStateData, save_hub_state

        reserved = max(int(self._epochs.reserved_until) if self._epochs is not None else 0,
                       int(epoch_reserved_until or 0))
        state = self._budget.state()
        data = HubStateData(version=HUB_STATE_VERSION, run_id=self._run_id or "", env_steps=int(state["env_steps"]),
                            sessions=dict(state["sessions"]), epoch_reserved_until=reserved,
                            versions=self._versions.state() if self._versions is not None else {},
                            counters={"book": self._book.state(),
                                      "intake_book": self._hub.intake_book.state() if self._hub is not None else {}},
                            nodes_lost=int(self._nodes_lost))
        path = self._run_dir.root / HUB_STATE_FILE
        if fsync:
            save_hub_state(path, data, fsync=True)
            return
        try:
            save_hub_state(path, data, fsync=False)
        except Exception as e:  # noqa: BLE001 - bookkeeping must never stop training
            logger.error(f"Writing {path} failed ({type(e).__name__}: {e})", exc_info=not self._hub_state_error_logged)
            self._hub_state_error_logged = True

    # ---- stop -------------------------------------------------------------------------------------------------

    def _send_stop_run(self, handle: Any) -> None:

        handle.send(pb.HubMessage(command=pb.Command(kind="stop_run", reason=f"grace={self._grace():g}")))

    def _grace(self) -> float:
        return float(LOCAL_SHUTDOWN_GRACE_SEC if self._options.mode == "train" else self._dist.shutdown_grace_sec)

    def _shutdown(self) -> list[str]:
        """Stop every node and child within the grace (``LOCAL_SHUTDOWN_GRACE_SEC`` for ``train``,
        ``distributed.shutdown_grace_sec`` otherwise); returns the children that had to be terminated.

        ``stop_run`` goes to every live session and the pulse is told to stop; meanwhile the inbox, the commits and
        the pulse are drained (final snapshots arrive as uploads, the last results as batches), until every node
        that has a session finished its stop (``run_stopped``) and every child exited, or the grace passed. Then the
        rest is terminated and killed, and the pulse queues' reads get until the deadline (at least 1 s after the
        kill)."""
        grace = self._grace()
        deadline = time.monotonic() + grace
        self._shutting_down = True
        self._stop_event.set()
        for handle in list(self._sessions.values()):
            self._send_stop_run(handle)
        first_error: Exception | None = None

        def poll() -> None:
            nonlocal first_error
            for step in (lambda: self._drain_inbox(INBOX_POLL_SEC), self._drain_commits, self._pump_pulse):
                try:
                    step()
                except Exception as e:  # noqa: BLE001 - re-raised after the teardown
                    if first_error is None:
                        logger.exception("Draining during the shutdown failed")
                    first_error = first_error or e

        killed: list[str] = []
        try:
            while time.monotonic() < deadline:
                poll()
                nodes_done = all(node_id in self._run_stopped for node_id in self._sessions)
                if nodes_done and not self._supervisor.alive():
                    break
            poll()
        finally:
            killed = self._supervisor.kill_remaining()
            if killed:
                logger.warning(f"Terminated processes that did not stop within {grace:.0f}s: {', '.join(killed)}")
            self._finish_reads(max(deadline, time.monotonic() + 1.0), poll)
            for q in (self._pulse_results,):
                if q is not None:
                    try:
                        q.cancel_join_thread()
                    except (AttributeError, OSError):
                        pass
            logger.info("All processes stopped")
        if first_error is not None:
            raise first_error
        return killed

    def _shutdown_after_error(self, error: BaseException) -> None:
        """Tear down after ``error``. A teardown failure is logged and noted on ``error``,
        which stays the exception the caller raises."""
        try:
            self._shutdown()
        except Exception as teardown_error:  # noqa: BLE001 - must not replace ``error``
            logger.exception(f"Shutdown after {type(error).__name__} failed too")
            error.add_note(f"Shutdown afterwards also failed: {teardown_error!r}")

    def _exit_code_after_shutdown(self, code: int, killed: list[str]) -> int:
        """A child that exited non-zero on its own (not terminated by the shutdown) fails the
        run, even if the budget was reached (R3-08). After a signal the signal's code stays
        and such exits are only warned about. A signal that arrived while the run was
        finishing (e.g. right after the budget was reached) still gives 128 + signum."""
        if code == 0 and self._supervisor.received_signal is not None:
            code = self._signal_exit_code()
        for failure in self._supervisor.failures(exclude=set(killed) | self._reported_failures):
            self._reported_failures.add(failure.name)
            if code in (0, 1):
                logger.error(failure.message())
                code = 1
            else:
                logger.warning(failure.message())
        return code

    def _signal_exit_code(self) -> int:
        """128 + signum of the first SIGINT/SIGTERM; logged once (the handler itself only records it)."""
        signum = int(self._supervisor.received_signal)
        if not self._signal_logged:
            self._signal_logged = True
            logger.warning(f"Received {signal.Signals(signum).name}, shutting down")
        return 128 + signum

# ---------------------------------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------------------------------


def _run(config_path: str | None, overrides: dict | None, make_options: Callable[[ColosseumConfig, RunDir],
                                                                                 HubOptions], *, what: str) -> int:
    """SP4's ``run_training`` body around a ``Hub``: spawn, logging, the config validated before the run dir exists,
    the resume source checked, the global seed, the run dir and its resolved config, ``run_state.json``."""
    from colosseum.utils.seeding import apply_global_seed

    mp.set_start_method("spawn", force=True)
    setup_process_logging(None, "main", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    # Before the run dir exists: an invalid config leaves nothing behind, so a corrected retry
    # with the same run.name works. Hub.launch does not validate again.
    validate_run_config(config)
    if config.training.resume_from:
        from colosseum.coordinator.checkpoint_manager import classify_resume_source
        classify_resume_source(config.training.resume_from)  # exists and has a known layout (nothing loaded)
    apply_global_seed(config.training.seed)  # after overrides: --set training.seed works
    run_dir = RunDir.create(config, config_path)
    config = run_dir.with_run_name(config)  # the resolved config and checkpoint hashes agree
    setup_process_logging(run_dir.logs, "main", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
    _record_run_state(run_dir, "running")
    try:
        code = Hub(config, run_dir, options=make_options(config, run_dir), validated=True).launch()
    except KeyboardInterrupt:
        # Ctrl-C before launch installed its signal handlers (the CLI exits 130): nothing was saved
        _record_run_state(run_dir, "stopped", exit_code=130)
        raise
    except BaseException:
        _record_run_state(run_dir, "failed", exit_code=1)
        raise
    # 130 / 143: stopped by a signal after saving the final checkpoints (not a failure); 1: a failure.
    _record_run_state(run_dir, "finished" if code in (0, 130, 143) else "failed", exit_code=code)
    if code == 0:
        logger.info(f"{what} finished; outputs in {run_dir.root}")
    return code


def run_training(config_path: str | None, overrides: dict | None = None, *, listen: str | None = None) -> int:
    """Entry point of ``colosseum train``: a hub and its local node on this machine (``listen``: also accept other
    machines' nodes on ``listen:distributed.port``, TLS). Returns the process exit code."""
    def options(config: ColosseumConfig, run_dir: RunDir) -> HubOptions:
        local = replace(LocalNodeOptions.from_config(config, run_dir), data_port=listen is not None)
        return HubOptions(mode="train", listen=listen, local_node=local)

    return _run(config_path, overrides, options, what="Training")


def run_hub(config_path: str | None, overrides: dict | None = None, *, listen: str | None = None,
            port: int | None = None, insecure: bool = False) -> int:
    """Entry point of ``colosseum hub``: a hub that nodes join (no local node). Returns the process exit code."""
    def options(config: ColosseumConfig, run_dir: RunDir) -> HubOptions:
        return HubOptions(mode="hub", listen=listen, port=port, insecure=insecure, local_node=None)

    return _run(config_path, overrides, options, what="Training")
