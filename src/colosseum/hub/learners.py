"""The learners of the hub (SP5 T4.1; spec block 1, 2.6, 7.2; controller ruling R39): the learner table's placement
passes, learner starts and stops with generations, version floors, reconciliation of a reconnecting node, start points
(A28) and parked (hub-held) weights (A27). The code moved here unchanged from ``hub.py`` (the R27 pattern of
``run_sinks``); ``Hub`` inherits ``HubLearners``, so every method keeps its name on ``Hub`` and reads the hub's
attributes (built in ``Hub.__init__`` / ``Hub._prepare``)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from colosseum.core.types import LATEST_NETWORK_ID
from colosseum.hub.placement import LearnerSlot, capacity_from_hello, place
from colosseum.net.proto import colosseum_v1_pb2 as pb

# The hub's logger: these lines read as before the move ("colosseum.hub.hub" in main.log).
logger = logging.getLogger("colosseum.hub.hub")

__all__ = ["JOURNAL_FLOOR_MATCHES", "HubLearners"]

JOURNAL_FLOOR_MATCHES = 1000              # newest journaled matches per agent read for the version floor (P-PF3-3)


class HubLearners:
    """Mixin of ``colosseum.hub.hub.Hub`` (module docstring)."""

    def _train_only(self) -> bool:
        """``train`` without ``--listen``: its local node is the only node that can ever exist (W04-26)."""
        return self._options.mode == "train" and self._options.listen is None

    # ---- learners (T4.1) --------------------------------------------------------------------------------------

    def _reconcile_learners(self, number: int, running: Any) -> list[tuple[str, int]]:
        """``Hello.learners`` of node ``number`` against the learner table (``LearnerTable.reconcile``): adopted
        learners stop being parked; returns the ``(agent, generation)`` learners the node must stop. Every reported
        version raises that agent's version floor (P-PF3-3)."""
        reported: dict[str, int] = {}
        for learner in running:
            agent = str(learner.agent_id)
            reported[agent] = int(learner.generation)
            if agent in self._learners:
                self._hello_floors[agent] = max(self._hello_floors.get(agent, 0), int(learner.version) + 1)
        announced = {a: info.generation for a in self._agent_ids if (info := self._versions.accepted(a)) is not None}
        out = self._learners.reconcile(number, reported, announced=announced)
        for agent in out.adopted:
            entry = self._learners[agent]
            self._drop_parked(agent)
            logger.info(f"Learner of {agent}: generation {entry.generation} on node {number} adopted ({entry.status})")
        for agent, generation in out.stop:
            logger.info(f"Learner of {agent}: generation {generation} on node {number} is not the current one; "
                        f"it is told to stop")
        for agent in out.gone:
            logger.info(f"Learner of {agent}: node {number} came back without it; it is placed again")
        return list(out.stop)

    def _learners_on(self, number: int) -> list[str]:
        """Agents whose learner starts or runs on node ``number`` (the node's desired learner set)."""
        return [a for a in self._learners.on_node(number) if self._learners[a].status in ("starting", "running")]

    def _learner_node_of(self, agent: str) -> int | None:
        """The node of ``agent``'s learner while it starts or runs (the collect rule's ``learner_node``), else None."""
        if self._learners is None or agent not in self._learners or not self._learners.can_collect(agent):
            return None
        slot = self._learners[agent].slot
        return None if slot is None else slot.node

    def _node_capacities(self) -> list:
        """The learner slots of every online node (``hub.placement.capacity_from_hello``); train's local node
        (``data_path == "local"``) is unlimited (R18)."""
        out = []
        for entry in self._registry.online():
            kinds = {int(g.get("index", 0)): str(g.get("kind", "")) for g in (entry.hardware or {}).get("gpus", ())}
            out.append(capacity_from_hello(entry.number, entry.name, entry.tags, entry.slots or {}, kinds,
                                           unlimited=entry.data_path == "local"))
        return out

    def _place_learners(self) -> None:
        """One placement pass (main loop only; cheap, idempotent; module docstring): every agent without a learner
        gets a free slot (``_start_learner``), the rest are ``pending`` (an INFO when the reason changes; parked
        weights when they have any). No pass runs while no node is online. ``train`` without ``--listen`` fails the
        run when an agent stays pending with its local node online (W04-26)."""
        if self._learners is None or not self._agent_ids or not self._registry.online():
            return                                   # before any node: nothing to place, nobody to play parked weights
        configs = self._setup.agent_configs
        agents = {a: configs[a].learner for a in self._agent_ids if self._learners[a].status != "failed"}
        current = self._learners.current()
        plan = place(agents, self._node_capacities(), current=current)
        for agent, slot in plan.new(current).items():
            self._start_learner(agent, slot)
        for agent in plan.pending:
            reason = plan.reasons[agent]
            self._learners.set_pending(agent, reason)
            if self._pending_reasons.get(agent) != reason:
                self._pending_reasons[agent] = reason
                logger.info(f"Learner of {agent}: pending ({reason})")
            self._park(agent)
        for agent in plan.assigned:
            self._pending_reasons.pop(agent, None)
        self._update_pending_value()
        local_online = any(e.data_path == "local" for e in self._registry.online())
        if plan.pending and local_online and self._train_only():
            agent = plan.pending[0]
            self._fail_run(f"Learner of {agent} cannot run on this machine: {plan.reasons[agent]}")

    def _start_learner(self, agent: str, slot: LearnerSlot) -> None:
        """A new assignment of ``agent``'s learner to ``slot``: a new generation (``hub_state.json`` written with
        fsync before the command leaves, P-PF3-2), the version floor, the start point (A28), the ``start_learner``
        command (deferred until after the ``Welcome`` of a node being welcomed) and an immediate ``Announce`` of the
        agents' states."""
        generation = self._versions.new_generation(agent)
        self._learners.assign(agent, slot, generation)
        try:
            self._save_hub_state(fsync=True)
        except Exception:
            self._learners.set_pending(agent, "hub_state.json could not be written")
            raise
        blob, version, what = self._learner_start_source(agent)
        floor = self._version_floor(agent, start=(blob, version))
        assignment = pb.LearnerAssignment(agent_id=agent, generation=generation, version_floor=floor,
                                          device=slot.device, weights_blob=blob, trainer_state_blob="",
                                          env_steps=self._budget.value)
        self._assignments[agent] = assignment
        command = pb.HubMessage(command=pb.Command(kind="start_learner", learner=assignment))
        entry = self._registry.get(slot.node)
        if slot.node == self._welcoming:
            self._deferred.append(command)
        elif entry is not None and (handle := self._sessions.get(entry.node_id)) is not None:
            handle.send(command)
        name = "?" if entry is None else entry.name
        logger.info(f"Learner of {agent}: generation {generation} on node {slot.node} ({name}), device {slot.device}, "
                    f"from {what}, versions from {floor}")
        self._announce_agents()

    def _learner_start_source(self, agent: str) -> tuple[str, int, str]:
        """``(weights_blob, version, description)`` of ``agent``'s next learner start (A28): the hub-start source
        (``_resume_start_sources``; T4.2 puts the newest recovery point in front)."""
        return self._start_sources.get(agent, ("", 0, "factory weights"))

    def _version_floor(self, agent: str, *, start: tuple[str, int] | None = None) -> int:
        """The first version a new learner of ``agent`` may publish (R9, W04-7, P-PF3-3): above every version the
        registry accepted, the newest stored snapshot, the journal's newest seats of the agent (read once per hub
        start) and every learner version a node reported in a ``Hello``. T4.2 adds the recovery point.

        ``start`` = the start point ``(weights_blob, version)``: a start point that IS the newest stored snapshot (a
        run-dir ``training.resume_from`` resumes the parent's newest snapshot, which the pool import stored here; an
        in-place resume's ``start:ckpt_v<N>``, T4.3) may publish its own version again (the same weights), so the
        stored snapshot then bounds the floor at its version, not above it (SP2's "versions continue from the
        resumed ``ckpt_v<N>``")."""
        floors = [self._versions.version_floor(agent), self._hello_floors.get(agent, 0),
                  self._journal_version_floors().get(agent, 0)]
        latest = self._coordinator.checkpoint_manager.latest(agent) if self._coordinator is not None else None
        if latest is not None:
            stored = int(latest.policy_version)
            same = start is not None and int(start[1]) == stored and start[0] in ("start:resume",
                                                                                  f"start:ckpt_v{stored}")
            floors.append(stored if same else stored + 1)
        return max(floors)

    def _journal_version_floors(self) -> dict[str, int]:
        """P-PF3-3: per trainable agent, 1 + the highest ``version_start`` / ``version_end`` of its latest seats in
        the newest ``JOURNAL_FLOOR_MATCHES`` journaled training matches (versions that reached a result survive a
        crash of the hub even when ``hub_state.json`` is older); read once per hub start, {} without a journal."""
        if self._journal_floors is not None:
            return self._journal_floors
        from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader

        floors: dict[str, int] = {}
        path = self._run_dir.root / MATCH_LOG_FILE
        if self._config.match_log.enabled and path.exists():
            try:
                with MatchLogReader(path) as reader:
                    for agent in self._agent_ids:
                        for match in reader.matches(agent=agent, source="train", limit=JOURNAL_FLOOR_MATCHES):
                            for seat in match.seats:
                                if seat.agent_id == agent and seat.network_id == LATEST_NETWORK_ID:
                                    top = max(v for v in (seat.version_start, seat.version_end, -1) if v is not None)
                                    floors[agent] = max(floors.get(agent, 0), int(top) + 1)
            except Exception as e:  # noqa: BLE001 - the other floors still apply
                logger.warning(f"the journal's versions could not be read for the version floor "
                               f"({type(e).__name__}: {e})")
        self._journal_floors = floors
        return floors

    def _stop_learner(self, node: int, agent: str, generation: int, reason: str) -> None:
        """``stop_learner`` of ``agent``'s learner ``generation`` on ``node`` (reason ``superseded``: no final upload;
        T4.4 adds ``stop_run``), at most once per stream of that node; nothing while the node has no stream (its next
        ``Hello`` is reconciled)."""
        entry = self._registry.get(node)
        handle = None if entry is None else self._sessions.get(entry.node_id)
        if handle is None:
            return
        key = (int(handle.serial), agent, int(generation))
        if key in self._stops_sent:
            return
        self._stops_sent.add(key)
        handle.send(pb.HubMessage(command=pb.Command(
            kind="stop_learner", learner=pb.LearnerAssignment(agent_id=agent, generation=int(generation)),
            reason=reason)))
        logger.info(f"Learner of {agent}: generation {generation} on node {node} told to stop ({reason})")

    def _assignment_for(self, agent: str) -> Any:
        """The ``LearnerAssignment`` of ``agent``'s current learner (the one it was started with; an adopted learner
        of an earlier hub process gets one built from the table)."""
        entry = self._learners[agent]
        known = self._assignments.get(agent)
        if known is not None and int(known.generation) == entry.generation:
            return known
        blob, _version, _what = self._learner_start_source(agent)
        return pb.LearnerAssignment(agent_id=agent, generation=entry.generation,
                                    version_floor=self._version_floor(agent),
                                    device="" if entry.slot is None else entry.slot.device, weights_blob=blob,
                                    trainer_state_blob="", env_steps=self._budget.value)

    def _agent_states(self) -> list:
        """``AgentState`` of every agent: a trainable one ``active`` while its learner starts or runs, ``pending``
        while it waits for a slot or its node, ``failed``; fixed agents ``active``."""
        cfg = self._config
        out = []
        for aid in cfg.agent_ids():
            status = "active"
            if self._learners is not None and aid in self._learners:
                learner = self._learners[aid].status
                status = {"pending": "pending", "relocating": "pending", "failed": "failed"}.get(learner, "active")
            out.append(pb.AgentState(agent_id=aid, kind=cfg.agent_kind(aid), status=status, start_env_steps=0))
        return out

    def _announce_agents(self) -> None:
        """An immediate ``Announce`` (the periodic one's content plus every agent's state, A6)."""
        if self._sessions and self._coordinator is not None:
            self._broadcast_announce(agents=True)

    def _update_pending_value(self) -> None:
        """``system`` value ``pending_agents``: trainable agents waiting for a slot or for their node (A26)."""
        if self._hub is None or self._learners is None:
            return
        self._hub.set_system_value("pending_agents", [e.agent_id for e in self._learners.entries()
                                                      if e.status in ("pending", "relocating")])

    # ---- parked weights (T4.1; A27) ---------------------------------------------------------------------------

    def _playable(self, agent: str) -> bool:
        """A fixed agent, or a trainable one whose learner starts or runs or that has parked weights."""
        if self._learners is None or agent not in self._learners:
            return True
        return self._learners.can_collect(agent) or self._versions.parked(agent) is not None

    def _park_source(self, agent: str) -> tuple[int, dict, str] | None:
        """The newest hub-held weights of ``agent`` (``(version, numpy state_dict, description)``): its newest stored
        snapshot, its ``training.resume_from`` state, its ``init`` weights (version 0; a partial ``init`` completed on
        a freshly built model); None without any. T4.2 adds the recovery point."""
        candidates: list[tuple[int, int, Callable[[], dict], str]] = []
        latest = self._coordinator.checkpoint_manager.latest(agent)
        if latest is not None:
            candidates.append((int(latest.policy_version), 2, lambda: self._coordinator.checkpoint_manager.load_model(
                agent, latest.checkpoint_id), f"snapshot {latest.checkpoint_id}"))
        state = self._resume_states.get(agent)
        if state is not None:
            candidates.append((int(state["policy_version"]), 1, lambda: dict(state["model_state"]),
                               f"{state['source']}"))
        init = self._init_states.get(agent)
        if init is not None:
            candidates.append((0, 0, lambda: self._full_init_state(agent, init), f"init {init.source}"))
        if not candidates:
            return None
        version, _rank, load, what = max(candidates, key=lambda c: (c[0], c[1]))
        return version, load(), what

    def _full_init_state(self, agent: str, init: Any) -> dict:
        """``init`` weights as a full state_dict (a partial ``init`` is loaded onto a freshly built model)."""
        import numpy as np

        if init.strict:
            return dict(init.model_state)
        import torch

        from colosseum.core.registry import build_model

        model = build_model(self._setup.agent_configs[agent], self._setup.role_specs[agent])
        model.load_state_dict({k: torch.as_tensor(np.asarray(v)) for k, v in init.model_state.items()}, strict=False)
        return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def _park(self, agent: str) -> None:
        """Serve ``agent``'s newest hub-held weights as its latest while it has no learner (``_park_source``): the
        frame ``BlobStore.parked_path`` (written atomically), a ``VersionInfo`` with node -1, generation 0,
        ``relay_only`` (A27) in the registry and the next ``Announce``. No-op while it can collect or is parked; an
        agent without any weights is not drawn into lineups (one INFO)."""
        from colosseum.net.chunks import WeightsMeta, encode_weights
        from colosseum.net.grpc_util import blob_digest
        from colosseum.net.messages import VersionInfo

        if self._learners.can_collect(agent) or agent in self._parked or agent in self._unplayable:
            return
        try:
            source = self._park_source(agent)
            if source is not None:
                version, state_dict, what = source
                frame = encode_weights(state_dict, WeightsMeta(agent, "latest", generation=0,
                                                               policy_version=int(version)))
                self._blobs.write_parked(agent, frame)
        except Exception as e:  # noqa: BLE001 - a session open or a placement pass must go on
            logger.error(f"{agent}: its hub-held weights could not be prepared ({type(e).__name__}: {e}); it is not "
                         f"drawn into lineups until its learner runs", exc_info=True)
            self._unplayable.add(agent)
            return
        if source is None:
            self._unplayable.add(agent)
            logger.info(f"{agent} has no weights yet: it is not drawn into lineups until its learner runs")
            return
        info = VersionInfo(agent_id=agent, generation=0, version=int(version), size=len(frame),
                           digest=blob_digest(frame), node=-1, relay_only=True)
        self._versions.park(info)
        self._parked[agent] = int(version)
        self._new_versions[agent] = info
        logger.info(f"{agent}: plays with hub-held weights v{version} ({what}) while it has no learner")
        self._announce_agents()

    def _drop_parked(self, agent: str) -> None:
        """Stop serving ``agent``'s parked weights (its learner was adopted or published a version): the registry
        entry and the file go (a servicer that has it open keeps reading, POSIX)."""
        self._versions.unpark(agent)
        self._unplayable.discard(agent)
        if self._parked.pop(agent, None) is not None:
            self._blobs.drop_parked(agent)


    # ---- start points (T4.1; A28) ------------------------------------------------------------------------------

    def _resume_start_sources(self) -> dict[str, tuple[str, int, str]]:
        """The hub-start source of every trainable agent's learner (A28; T4.3 replaces it for an in-place resume):
        ``(weights_blob, version, description)``. A ``training.resume_from`` state becomes the start point
        ``start:resume`` (one checkpoint frame: weights plus trainer-state bytes, none = weights only, as SP3's
        ``.pt`` resume) written once to ``BlobStore.start_path(agent, "resume")``; else ``init`` (the ``Welcome.fixed``
        blob); else factory weights."""
        from colosseum.net.chunks import encode_checkpoint

        out: dict[str, tuple[str, int, str]] = {}
        for agent in self._agent_ids:
            state = self._resume_states.get(agent)
            init = self._init_states.get(agent)
            if state is not None:
                version = int(state["policy_version"])
                trainer = state.get("trainer_state")
                frame = encode_checkpoint({"agent_id": agent, "policy_version": version,
                                           "model_state": state["model_state"],
                                           "trainer_state_bytes": None if trainer is None else bytes(trainer),
                                           "final": False}, kind="recovery", generation=0)
                self._blobs.write_start(agent, "resume", frame)
                out[agent] = ("start:resume", version, f"{state['source']} (policy_version {version})")
            elif init is not None:
                out[agent] = ("init", 0, f"init {init.source}")
            else:
                out[agent] = ("", 0, "factory weights")
        return out
