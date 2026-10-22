"""The SP4 run sinks of the hub (SP5 T2.5; rulings R5, R6; controller ruling R27): the metrics hub, the journal and
its writer, the live ratings, snapshot retention, resume / init / league state, checkpoint saving and the SP4 pulse
child's queues. The code moved here unchanged from the SP4 ``Launcher`` (``launcher.py``, through ``hub.py``); ``Hub``
inherits ``RunSinks``, so every method keeps its name on ``Hub`` (A6, R5) and reads the hub's attributes (built in
``Hub.__init__`` / ``Hub._prepare``)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from colosseum.coordinator.coordinator import LEAGUE_STATE_FILE, Coordinator
from colosseum.core.config import ColosseumConfig
from colosseum.core.ipc import QueueReader, release_command_queues
from colosseum.envs.game import GameSpec, RoleSpec
from colosseum.metrics.wandb_logger import WandBLogger
from colosseum.utils.process import SHUTDOWN_GRACE_SEC

if TYPE_CHECKING:
    from colosseum.learner.factory import InitState

# The hub's logger: these lines read as before the move ("colosseum.hub.hub" in main.log).
logger = logging.getLogger("colosseum.hub.hub")

__all__ = ["RunSinks"]

# Bound on each step of releasing the pulse queues after the teardown.
_QUEUE_RELEASE_SEC = 1.0


class RunSinks:
    """Mixin of ``colosseum.hub.hub.Hub`` (module docstring)."""

    # ---- moved from the SP4 Launcher (R5) ---------------------------------------------------------------------

    def _open_metrics(self, resume_states: dict[str, dict | None]) -> None:
        """metrics.jsonl writer, optional WandB viewer and the hub that feeds both (T6.3/T6.4).

        Resumed runs start the rate baselines at the resumed counters, so the first system
        record has no resume spike.
        """
        from colosseum.metrics.hub import MetricsHub
        from colosseum.metrics.jsonl import MetricsWriter

        cfg = self._config
        self._metrics_writer = MetricsWriter(self._run_dir.metrics_path)
        # Optional viewer of the same records (metrics.jsonl stays the source of truth, T6.4).
        self._wandb = WandBLogger(
            cfg.metrics,
            run_name=self._run_dir.run_name or self._run_dir.root.name,
            run_config=cfg.model_dump(mode="json", by_alias=True),
        )
        self._hub = MetricsHub(
            writer=self._metrics_writer,
            ratings_path=self._run_dir.ratings_path,
            agent_ids=self._agent_ids,
            total_timesteps=cfg.training.total_timesteps,
            log_interval=cfg.metrics.log_interval,
            console_interval_sec=cfg.metrics.console_interval_sec,
            initial_env_steps=int(self._budget.value),
            initial_train_steps={aid: int(s["policy_version"]) if s else 0 for aid, s in resume_states.items()},
            wandb_logger=self._wandb,
            stored_snapshots=self._stored_snapshot_versions,
        )

    def _open_match_log(self, coordinator: Coordinator) -> None:
        """The match log (spec block 1). The journal is opened here first, so a schema or file problem is a
        ConfigError before any child starts (a file that is not an SQLite database is moved aside by
        ``MatchLog``); a run-dir resume imports the resumed run's journal (an SP3 run dir has none: INFO, the
        journal starts empty); the in-memory comparison table of the live ratings is read from it once; then the
        writer thread starts with its own connection (opened in the thread). Evicted snapshots reach the writer
        through the coordinator's evict listener. Node batches are submitted with their token (R8)."""
        import sqlite3
        from functools import partial

        from colosseum.arena.matchlog import (
            MATCH_LOG_FILE,
            RUN_SOURCES,
            MatchLog,
            MatchLogError,
            MatchLogWriter,
            RunInfo,
        )
        from colosseum.coordinator.checkpoint_manager import RESUME_RUN_DIR, classify_resume_source
        from colosseum.core.config import config_hash
        from colosseum.core.errors import ConfigError
        from colosseum.core.run_dir import read_run_id

        cfg = self._config
        if not cfg.match_log.enabled:
            logger.info("Match log: off (match_log.enabled is false)")
            return
        resume_from = cfg.training.resume_from
        parent = (Path(resume_from) if resume_from and classify_resume_source(resume_from) == RESUME_RUN_DIR
                  else None)
        run = RunInfo(run_id=self._run_id, name=self._run_dir.run_name or self._run_dir.root.name,
                      parent_run_id=read_run_id(parent) if parent is not None else None, game=cfg.env.env_class,
                      config_hash=config_hash(cfg), checkpoint_interval=cfg.checkpoint.interval)
        path = self._run_dir.matches_path
        factory = partial(MatchLog, path, run, min_free_disk_mb=cfg.match_log.min_free_disk_mb)
        hint = "move it away or set match_log.enabled: false"
        try:
            log = factory()
        except MatchLogError as e:
            raise ConfigError(f"Match log: {e}; {hint}") from e
        except (sqlite3.Error, OSError) as e:
            raise ConfigError(f"Match log {path} cannot be opened ({type(e).__name__}: {e}; locked by another "
                              f"process or unreadable); {hint}") from e
        try:
            if parent is not None:
                parent_log = parent / MATCH_LOG_FILE
                if not parent_log.exists():
                    logger.info(f"Match log: the resumed run dir {parent} has no {MATCH_LOG_FILE} (an SP3 run "
                                f"dir); the journal starts empty")
                else:
                    try:
                        stats = log.import_parent(parent_log)
                    except (MatchLogError, sqlite3.Error, OSError) as e:
                        raise ConfigError(f"Match log: importing the resumed run's journal {parent_log} failed "
                                          f"({type(e).__name__}: {e}); this run's journal is unchanged; move "
                                          f"{parent_log} away to resume without it, or set match_log.enabled: "
                                          f"false") from e
                    if not stats.skipped:
                        logger.info(f"Match log: imported {parent_log}: {stats.counts_rows} counts rows, "
                                    f"{stats.folded} detailed training matches folded into counts, "
                                    f"{stats.copied} tournament / pulse matches copied")
            try:
                table = log.comparison_table(RUN_SOURCES)
            except (MatchLogError, sqlite3.Error) as e:
                raise ConfigError(f"Match log {path} cannot be read ({type(e).__name__}: {e}); {hint}") from e
        finally:
            log.close()
        self._comparison_table = table
        mlc = cfg.match_log
        self._match_log_writer = MatchLogWriter(
            factory, compact_interval_sec=mlc.compact_interval_sec,
            window_bytes=int(mlc.detail_window_mb * 1024 * 1024), table=table)
        self._match_log_writer.start()
        coordinator.add_evict_listener(self._on_snapshot_evicted)
        logger.info(f"Match log: {path} (detail window {mlc.detail_window_mb:g} MB, min free disk "
                    f"{mlc.min_free_disk_mb:g} MB, compaction every {mlc.compact_interval_sec:g} s)")

    def _open_live_ratings(self) -> None:
        """The live Bradley-Terry refit thread over the writer's in-memory table (spec block 2); none without the
        journal. Lower bounds at ``checkpoint.top_k_quantile``, ``P(top-k)`` with ``checkpoint.keep_top_k``."""
        if self._comparison_table is None:
            return
        from colosseum.arena.live import LiveRatings, rating_inputs

        cfg = self._config
        options, fixed_agents, random_bot = rating_inputs(cfg)
        self._live_ratings = LiveRatings(
            self._comparison_table, interval_sec=cfg.ratings.refit_interval_sec,
            options=options, fixed_agents=fixed_agents,
            random_bot=random_bot, agent_order=cfg.agent_ids(), top_k=cfg.checkpoint.keep_top_k,
            quantile=cfg.checkpoint.top_k_quantile,
            env_steps_fn=lambda: int(self._budget.value), seed=cfg.training.seed or 0)
        self._live_ratings.start()
        logger.info(f"Live ratings: Bradley-Terry refit every {cfg.ratings.refit_interval_sec:g} s")

    def _wire_retention(self) -> None:
        """T4.1: top-k retention reads the live ratings (spec block 4). Evictions already reach the journal through
        T1.3's evict listener (``_on_snapshot_evicted``), whose next compaction pass folds the evicted entity's
        detailed matches into counts."""
        if self._live_ratings is not None:
            self._coordinator.set_ratings_source(self._live_ratings.latest)

    def _reapply_retention(self) -> None:
        """After every live refit (a new ``RatingsSnapshot.version``), rerun retention for every trainable agent;
        only with ``keep_top_k > 0``, where scores decide (spec block 4). Main loop only."""
        if self._live_ratings is None or self._config.checkpoint.keep_top_k == 0 or self._coordinator is None:
            return
        snapshot = self._live_ratings.latest()
        if snapshot is None or snapshot.version == self._retention_version:
            return
        self._retention_version = snapshot.version
        for aid in self._agent_ids:
            self._coordinator.checkpoint_manager.reapply_retention(aid)

    def _live_snapshot(self):
        """The newest live ratings snapshot (None without live ratings or before the first refit)."""
        return None if self._live_ratings is None else self._live_ratings.latest()

    def _stored_snapshot_versions(self) -> dict[str, set[int]] | None:
        """Per trainable agent the versions of its stored snapshots, from the checkpoint manager's index (the
        compact ratings record's ``best``, ruling A-M4); None before the coordinator exists."""
        from colosseum.arena.entities import snapshot_version

        if self._coordinator is None:
            return None
        manager = self._coordinator.checkpoint_manager
        return {aid: {v for c in manager.list_checkpoints(aid) if (v := snapshot_version(c.checkpoint_id)) is not None}
                for aid in self._coordinator.trainable_agents}

    def _finish_live_ratings(self):
        """Stop the refit thread and refit once more on the final data (the journal writer has flushed), unless
        that would be slow (``LiveRatings.finish``). Returns the newest snapshot (None without live ratings)."""
        if self._live_ratings is None:
            return None
        return self._live_ratings.finish(SHUTDOWN_GRACE_SEC)

    def _on_snapshot_evicted(self, agent_id: str, checkpoint_id: str) -> None:
        """Coordinator evict listener (main loop): the journal folds the evicted snapshot's training matches."""
        from colosseum.arena.entities import entity_key, snapshot_version

        version = snapshot_version(checkpoint_id)
        if self._match_log_writer is not None and version is not None:
            self._match_log_writer.note_evicted([entity_key(agent_id, version)])

    def _match_log_stats(self) -> dict | None:
        """The journal writer's stats for the ``system`` record (None without a journal)."""
        return None if self._match_log_writer is None else self._match_log_writer.stats.as_dict()

    def _stop_match_log(self) -> None:
        """Flush and stop the journal writer (within SHUTDOWN_GRACE_SEC, after the nodes stopped and the final
        drains are done); account the commits made while it stopped (P-PF2-14); log the run's result accounting.
        Runs once."""
        if self._match_log_writer is None or self._match_log_stopped:
            return
        self._match_log_stopped = True
        stats = self._match_log_writer.stop(SHUTDOWN_GRACE_SEC)
        self._drain_commits()
        worker_drops = self._hub.system_cumulative("dropped_results") if self._hub is not None else 0
        workers, pulse = self._results_received, self._pulse_results_received
        received = (f"{workers + pulse} received ({workers} from workers, {pulse} from the pulse)"
                    if self._pulse_feeder is not None else f"{workers} received")
        logger.info(f"Match results: {received}, {stats.written} journaled, "
                    f"{stats.duplicates} duplicates, {stats.dropped} dropped by the journal writer, {worker_drops} "
                    f"dropped by workers")

    def _finish_metrics(self) -> None:
        """Record what arrived during shutdown (final train metrics, last results), flush and stop
        the journal writer, write the final records and ratings.json; the writer is stopped, the
        metrics file closed and the WandB run finished even if any of that fails (or if the hub
        was never built)."""
        try:
            if self._hub is not None:
                for _ in range(16):              # what the nodes sent last (bounded: a node may still be talking)
                    if not self._drain_inbox(0.0):
                        break
                self._drain_commits()
                self._pump_pulse()                # before the writer stops: the last pulse results (A17, A20)
                self._stop_match_log()
                final_bt = self._finish_live_ratings()
                self._finish_pulse()
                self._hub.close(env_steps=int(self._budget.value),
                                ratings=self._coordinator.ratings_snapshot,
                                queue_depths=self._hub.intake_queue_depths(),
                                match_log=self._match_log_stats(), bt=final_bt)
                self._save_league_state()
                if self._versions is not None:
                    self._save_hub_state(fsync=False)
        finally:
            try:
                self._stop_match_log()            # also when the final drains failed
            except Exception:  # noqa: BLE001 - must not replace the original error
                logger.exception("Stopping the match log writer failed")
            try:
                if self._live_ratings is not None:
                    self._live_ratings.stop(SHUTDOWN_GRACE_SEC)
            except Exception:  # noqa: BLE001 - must not replace the original error
                logger.exception("Stopping the live ratings failed")
            try:
                if self._metrics_writer is not None:
                    self._metrics_writer.close()
            finally:
                if self._wandb is not None:
                    self._wandb.finish()

    def _resolve_resume(self, agent_configs: Mapping[str, ColosseumConfig],
                        role_specs: Mapping[str, RoleSpec]) -> dict[str, dict | None]:
        """Resolve ``training.resume_from`` for every trainable agent (see ``resolve_resume``).

        A checkpoint source must carry the role signature of the agent's roles (ConfigError naming
        the checkpoint otherwise); every resumed agent's weights are checked against its
        architecture (ConfigError before any process starts); the global env-step counter
        continues from the largest resumed ``env_steps`` (``BudgetCounter.add``), so the budget and LR progress
        continue. A checkpoint without ``trainer_state.pt`` (e.g. an imported or ``keep_every`` snapshot) while
        ``checkpoint.save_optimizer`` is on is announced by a WARNING: the optimizer / LR state starts fresh.
        """
        from colosseum.coordinator.checkpoint_manager import (
            RESUME_PT_FILE,
            check_model_state,
            classify_resume_source,
            resolve_resume,
        )
        from colosseum.core.registry import build_model
        from colosseum.core.roles import role_signature

        resume_from = self._config.training.resume_from
        resume_states: dict[str, dict | None] = {aid: None for aid in agent_configs}
        if not resume_from:
            return resume_states
        for aid, acfg in agent_configs.items():
            state = resolve_resume(resume_from, aid, expected_signature=role_signature(role_specs[aid]))
            if state is not None:
                check_model_state(build_model(acfg, role_specs[aid]), state["model_state"], state["source"])
                logger.info(f"Resume [{aid}]: {state['source']} (policy_version {state['policy_version']})")
                if (state.get("trainer_state") is None and self._config.checkpoint.save_optimizer
                        and classify_resume_source(resume_from) != RESUME_PT_FILE):
                    logger.warning(
                        f"Resume [{aid}]: checkpoint {state['source']} has no trainer_state.pt; the weights and "
                        f"policy_version are restored, the optimizer / LR state and trainer counters start fresh"
                    )
            resume_states[aid] = state
        start_env_steps = max((s["env_steps"] for s in resume_states.values() if s), default=0)
        if start_env_steps > 0:
            self._budget.add(int(start_env_steps))
            logger.info(f"Resume: env-step counter continues from {start_env_steps}")
        return resume_states

    def _resolve_init(self, spec: GameSpec, resume_states: Mapping[str, dict | None]) -> dict[str, InitState | None]:
        """``init`` weights per trainable agent (spec block 6). An agent restored by ``training.resume_from``
        ignores ``init`` (logged), so a run continues with the same config."""
        from colosseum.learner.factory import resolve_init

        states: dict[str, InitState | None] = {}
        for aid in self._config.get_trainable_agent_ids():
            resumed = resume_states.get(aid)
            if resumed is not None:
                if self._config.get_agent_config(aid).init.from_ is not None:
                    logger.info(f"Agent '{aid}' resumes from {resumed['source']}; its init.from is ignored")
                states[aid] = None
                continue
            state = resolve_init(self._config, aid, spec)
            if state is not None:
                for line in state.report:
                    logger.info(line)
            states[aid] = state
        return states

    def _load_league_state(self, coordinator: Coordinator) -> None:
        """A run-dir resume restores the resumed run's ratings and PFSP statistics (spec block 7.4); a run dir
        without league_state.json (an SP3 run dir, or an SP4 run that never saved one, e.g. killed before its first
        metrics tick) gives an INFO line and empty tables. A resume from a checkpoint dir or a .pt starts empty, as
        in SP3."""
        from colosseum.coordinator.checkpoint_manager import RESUME_RUN_DIR, classify_resume_source

        resume_from = self._config.training.resume_from
        if not resume_from or classify_resume_source(resume_from) != RESUME_RUN_DIR:
            return
        path = Path(resume_from) / LEAGUE_STATE_FILE
        if coordinator.load_league_state(path):
            logger.info(f"Resume: league state (ratings, PFSP statistics) restored from {path}")
        else:
            logger.info(f"Resume: {resume_from} has no {LEAGUE_STATE_FILE} (an SP3 run dir, or a run that never "
                        f"saved it, e.g. killed before its first metrics tick); ratings and PFSP statistics start "
                        f"empty")

    def _save_league_state(self) -> None:
        """Write league_state.json (each metrics tick and at the end) with the journal watermark (the highest
        journal row accounted; None without the journal); a failure is an ERROR, never a crash."""
        if self._coordinator is None:
            return
        try:
            self._coordinator.save_league_state(self._run_dir.league_state_path, env_steps=int(self._budget.value),
                                                journal_row=self._journal_row if self._intake is not None
                                                and self._intake.journal else None)
        except Exception as e:  # noqa: BLE001 - the league state must never stop training
            logger.error(f"Writing {self._run_dir.league_state_path} failed ({type(e).__name__}: {e})",
                         exc_info=not self._league_state_error_logged)
            self._league_state_error_logged = True

    def _import_snapshot_pool(self, coordinator: Coordinator) -> None:
        """A resume from a run dir carries its snapshot pool into this run (spec block 4), before the first
        lineups are drawn; a resume from a checkpoint dir or a .pt starts with an empty pool."""
        from colosseum.coordinator.checkpoint_manager import RESUME_RUN_DIR, classify_resume_source

        resume_from = self._config.training.resume_from
        if resume_from and classify_resume_source(resume_from) == RESUME_RUN_DIR:
            coordinator.import_snapshots(resume_from)

    def _save_checkpoint(self, payload: dict) -> str:
        """Save a learner snapshot with the hub's env steps (R9); the pulse gets it. Returns its checkpoint id."""
        aid = payload["agent_id"]
        env_steps = int(self._budget.value)
        meta = {**self._checkpoint_meta.get(aid, {}), "env_steps": env_steps}
        checkpoint_id = self._coordinator.save_checkpoint_payload(payload, meta_extra=meta)
        if self._pulse_feeder is not None:          # the pulse evaluates it even if retention evicts it (spec block 6)
            self._pulse_feeder.offer(aid, checkpoint_id, int(payload["policy_version"]), env_steps,
                                     payload["model_state"], final=bool(payload.get("final", False)))
        return checkpoint_id

    # ---- the pulse (R6) ---------------------------------------------------------------------------------------

    def _dead_producers(self, names: list[str]) -> list[str]:
        dead = []
        for name in names:
            proc = self._supervisor.get(name)
            if proc is not None and proc.exitcode is not None:
                dead.append(name)
        return dead

    def _drain(self, q: Any, label: str, producers: Callable[[], list[str]]) -> list:
        """Everything currently in ``q``, read without ever blocking on a dead producer."""
        reader = self._readers.get(id(q))
        if reader is None:
            reader = self._readers[id(q)] = QueueReader(q, label, lambda: self._dead_producers(producers()))
        return reader.drain()

    def _pump_pulse(self) -> None:
        """Hand waiting snapshots to the pulse process; give its results to the journal writer only (contract
        amendment A20: not to the coordinator, the online ELO, PFSP or the episodes aggregator); keep the
        ``system`` counters and ``pulse_state.json`` current. Main loop only."""
        feeder = self._pulse_feeder
        if feeder is None or self._pulse_requests is None:
            return
        handing_out = not self._stop_event.is_set()     # after the stop the pulse takes no new request
        if handing_out:
            feeder.pump(self._pulse_requests)
        if self._pulse_results is not None:
            for message in self._drain(self._pulse_results, "pulse-results", lambda: ["pulse"]):
                self._pulse_results_received += len(message.results)
                if self._match_log_writer is not None:
                    self._match_log_writer.submit(message.results)   # a drop is counted by the writer (T1.3)
                feeder.done(message)
            if handing_out:
                feeder.pump(self._pulse_requests)   # the next one, if the pulse took the previous request
        if self._hub is not None:
            self._hub.set_system_value("pulse_skipped_snapshots", feeder.skipped)
            self._hub.set_system_value("pulse_evaluated_snapshots", feeder.evaluated)
            self._hub.set_system_value("pulse_pending_snapshots", feeder.pending)
        if feeder.changed:
            self._write_pulse_state()

    def _write_pulse_state(self) -> None:
        """``pulse_state.json`` (bookkeeping; a failing write is logged and retried at the next change, never fatal)."""
        from colosseum.metrics.jsonl import write_json_atomic

        self._pulse_feeder.changed = False
        try:
            write_json_atomic(self._run_dir.pulse_state_path, self._pulse_feeder.state())
        except Exception as e:  # noqa: BLE001 - the pulse bookkeeping must never stop training
            logger.error(f"Writing {self._run_dir.pulse_state_path} failed ({type(e).__name__}: {e})",
                         exc_info=not self._pulse_state_error_logged)
            self._pulse_state_error_logged = True

    def _finish_pulse(self) -> None:
        """At the end: the final pulse state, and an INFO per final snapshot the run gave no time to evaluate."""
        if self._pulse_feeder is None:
            return
        for agent_id, checkpoint_id in self._pulse_feeder.unevaluated_final():
            why = ("the backlog skipped it" if self._pulse_feeder.skipped_by_backlog(agent_id, checkpoint_id)
                   else "the run stopped first")
            logger.info(f"pulse: the final snapshot {checkpoint_id} of {agent_id} was not evaluated ({why}); "
                        f"evaluate it with colosseum tournament")
        self._write_pulse_state()

    def _finish_reads(self, deadline: float, poll: Callable[[], None]) -> None:
        """Every child is gone: let in-progress pulse reads complete until ``deadline``, then abandon the stuck ones
        (a producer died mid-message) and take what they read before."""
        while any(r.busy for r in self._readers.values()) and time.monotonic() < deadline:
            poll()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        stuck = [r for r in self._readers.values() if r.busy and not r.abandoned]
        for reader in stuck:
            reader.abandon()
        poll()

    def _release_pulse_queues(self) -> None:
        """After the teardown: end abandoned reads, read back and close the request queue this process produces
        (``release_command_queues``), close the results queue unless a read of it is stuck; every step bounded by
        ``_QUEUE_RELEASE_SEC``."""
        for reader in self._readers.values():
            reader.release(_QUEUE_RELEASE_SEC)
        busy = {key: reader for key, reader in self._readers.items() if reader.busy}
        if self._pulse_requests is not None:
            release_command_queues([self._pulse_requests], _QUEUE_RELEASE_SEC)
        if self._pulse_results is not None:
            try:
                self._pulse_results.cancel_join_thread()
                if id(self._pulse_results) not in busy:
                    self._pulse_results.close()
            except (AttributeError, OSError):
                pass
        self._readers = busy
        self._pulse_requests = self._pulse_results = None
