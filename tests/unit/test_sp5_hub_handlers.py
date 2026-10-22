"""The real hub's upload, session, event, announce and node-loss handlers (SP5 T2.5 review round 1; pre-flight
rulings P-PF2-1 (c), P-PF2-9 (b); amendments A6, A12; SP4 ruling R3-08): driven in-process through ``hub_kit``."""
from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import Future

import numpy as np
import pytest

from colosseum.hub.hub import Hub, HubOptions
from colosseum.hub.server import SessionClosed, UploadIn
from colosseum.net.chunks import encode_checkpoint
from colosseum.net.proto import colosseum_v1_pb2 as pb
from game_helpers import frozen_agent, make_test_config, make_test_run_dir, scripted_agent, write_ckpt_dir
from hub_kit import accept_version, lineup_hub, node_message, open_node


def _config(tmp_path, **sections):
    base = make_test_config("turns", agents={"a": {}})
    frozen = write_ckpt_dir(tmp_path / "frozen-src", base, agent_id="a")
    return make_test_config("turns", agents={"a": {}, "random": scripted_agent(), "frz": frozen_agent(frozen)},
                            **sections)


def _frame(agent="a", version=20, *, kind="periodic", generation=1, final=False, w: float = 0.0,
           trainer: bytes | None = None) -> bytes:
    payload = {"agent_id": agent, "policy_version": version,
               "model_state": {"w": np.arange(4, dtype=np.float32) + np.float32(w)},
               "trainer_state_bytes": trainer, "final": final}
    return encode_checkpoint(payload, kind=kind, generation=generation)


def _upload(hub, handle, data: bytes | None, *, kind="checkpoint", agent="a", generation=1, version=20) -> Future:
    """An ``UploadIn`` with a spool file (``data`` None: the spool vanished) through the real ``Hub._on_upload``."""
    hub._blobs.incoming.mkdir(parents=True, exist_ok=True)
    path = hub._blobs.incoming / f"{uuid.uuid4().hex}.part"
    if data is not None:
        path.write_bytes(data)
    reply: Future = Future()
    header = pb.UploadHeader(kind=kind, agent_id=agent, generation=generation, version=version,
                             total_size=0 if data is None else len(data))
    hub._on_upload(UploadIn(header, path, handle.node_id, reply))
    assert not path.exists() and not list(hub._blobs.incoming.iterdir())     # the spool is always removed
    return reply


def _stored(hub, agent="a") -> list[str]:
    return [c.checkpoint_id for c in hub._coordinator.checkpoint_manager.list_checkpoints(agent)]


def test_accepted_periodic_and_final_uploads_are_saved_with_the_hubs_env_steps(tmp_path):
    hub = lineup_hub(_config(tmp_path), tmp_path)
    node = open_node(hub)
    hub._budget.report(node.node_id, node.session_id, 777)
    periodic = _upload(hub, node, _frame(version=20)).result(1)
    final = _upload(hub, node, _frame(version=25, kind="final", final=True), kind="final", version=25).result(1)
    assert (periodic.accepted, periodic.checkpoint_id) == (True, "ckpt_v20")
    assert (final.accepted, final.checkpoint_id) == (True, "ckpt_v25")
    assert _stored(hub) == ["ckpt_v20", "ckpt_v25"]
    metas = {c.checkpoint_id: c.meta for c in hub._coordinator.checkpoint_manager.list_checkpoints("a")}
    assert metas["ckpt_v20"]["env_steps"] == 777 and metas["ckpt_v25"]["final"] is True


def _files(hub, checkpoint_id: str, agent="a") -> tuple[dict, bytes, bytes]:
    """``(meta, model.pt bytes, trainer_state.pt bytes)`` of a stored snapshot."""
    info = next(c for c in hub._coordinator.checkpoint_manager.list_checkpoints(agent)
                if c.checkpoint_id == checkpoint_id)
    meta = json.loads((info.path / "meta.json").read_text())
    return meta, (info.path / "model.pt").read_bytes(), (info.path / "trainer_state.pt").read_bytes()


def test_a_late_periodic_of_a_stored_final_version_is_ignored_and_counted(tmp_path, caplog):
    """Ruling R35 (FX4): the node's transfer pool uploads the periodic and the final snapshot of the same version
    concurrently; when the periodic arrives second, the stored final of the current run stays (meta, model,
    trainer state) and the periodic is counted as ``dropped_uploads.not_newer``."""
    hub = lineup_hub(_config(tmp_path), tmp_path)
    node = open_node(hub)
    final = _upload(hub, node, _frame(version=310, kind="final", final=True, w=1.0, trainer=b"final-state"),
                    kind="final", version=310).result(1)
    assert (final.accepted, final.checkpoint_id) == (True, "ckpt_v310")
    before = _files(hub, "ckpt_v310")
    with caplog.at_level(logging.INFO, logger="colosseum.hub.hub"):
        late = _upload(hub, node, _frame(version=310, w=2.0, trainer=b"periodic-state"), version=310).result(1)
    assert (late.accepted, late.checkpoint_id, late.reason) == (True, "ckpt_v310", "not_newer")
    after = _files(hub, "ckpt_v310")
    assert after == before and after[0]["final"] is True
    assert _stored(hub) == ["ckpt_v310"]
    assert hub._book.get("dropped_uploads.not_newer") == 1
    assert "already stores the final snapshot v310" in caplog.text


def test_a_final_replaces_a_periodic_of_the_same_version(tmp_path):
    hub = lineup_hub(_config(tmp_path), tmp_path)
    node = open_node(hub)
    assert _upload(hub, node, _frame(version=310, trainer=b"periodic-state"), version=310).result(1).accepted
    assert _files(hub, "ckpt_v310")[0]["final"] is False
    final = _upload(hub, node, _frame(version=310, kind="final", final=True, w=1.0, trainer=b"final-state"),
                    kind="final", version=310).result(1)
    assert (final.accepted, final.checkpoint_id) == (True, "ckpt_v310")
    meta, _, trainer = _files(hub, "ckpt_v310")
    assert meta["final"] is True and trainer == b"final-state"
    assert hub._book.get("dropped_uploads.not_newer") == 0


def test_a_periodic_replaces_an_earlier_runs_final_of_the_same_version(tmp_path):
    """A final imported from an earlier run (another ``run_id``) is not the current run's: R35 does not apply."""
    hub = lineup_hub(_config(tmp_path), tmp_path)
    node = open_node(hub)
    hub._coordinator.checkpoint_manager.save("a", 310, {"w": np.zeros(4, dtype=np.float32)},
                                             meta_extra={"final": True, "run_id": "an-earlier-run"})
    reply = _upload(hub, node, _frame(version=310, trainer=b"periodic-state"), version=310).result(1)
    assert (reply.accepted, reply.reason) == (True, "")
    meta, _, trainer = _files(hub, "ckpt_v310")
    assert meta["final"] is False and meta["run_id"] == hub._coordinator.checkpoint_manager.run_id
    assert trainer == b"periodic-state" and hub._book.get("dropped_uploads.not_newer") == 0


@pytest.mark.parametrize("case, expected", [
    ("unknown agent", "invalid"), ("frozen agent", "invalid"), ("scripted agent", "invalid"),
    ("generation 0", "invalid"), ("learner elsewhere", "stale_generation"), ("frame agent", "invalid: agent"),
    ("frame generation", "invalid: generation"), ("frame version", "invalid: version"),
    ("frame kind", "invalid: kind"), ("stale generation", "stale_generation"), ("upload kind", "kind"),
    ("garbage", "invalid")])
def test_every_p_pf2_9_refusal_is_answered_counted_and_nothing_is_saved(tmp_path, case, expected):
    hub = lineup_hub(_config(tmp_path), tmp_path)
    node = open_node(hub)
    other = open_node(hub, name="other", cpu_learners=0, no_learners=True)
    upload = dict(handle=node, data=_frame())
    if case in ("unknown agent", "frozen agent", "scripted agent"):
        agent = {"unknown agent": "nobody", "frozen agent": "frz", "scripted agent": "random"}[case]
        upload.update(agent=agent, data=_frame(agent=agent))
    elif case == "generation 0":
        upload.update(generation=0, data=_frame(generation=0))
    elif case == "learner elsewhere":
        upload.update(handle=other)
    elif case == "frame agent":
        upload.update(data=_frame(agent="b"))
    elif case == "frame generation":
        upload.update(data=_frame(generation=2))
    elif case == "frame version":
        upload.update(version=21)
    elif case == "frame kind":
        upload.update(data=_frame(kind="final"))
    elif case == "stale generation":
        upload.update(generation=2, data=_frame(generation=2))
    elif case == "upload kind":
        upload.update(kind="agent_file")                               # T8.2: not handled by the main loop yet
    elif case == "garbage":
        upload.update(data=b"\0" * 64)
    handle, data = upload.pop("handle"), upload.pop("data")
    reply = _upload(hub, handle, data, **upload).result(1)
    assert not reply.accepted and reply.reason == expected
    assert _stored(hub) == []
    reason = "stale_generation" if expected == "stale_generation" else "invalid"
    assert hub._book.get(f"dropped_uploads.{reason}") == 1


def test_a_spool_that_vanished_before_the_hub_claimed_it_saves_nothing(tmp_path, caplog):
    hub = lineup_hub(_config(tmp_path), tmp_path)
    node = open_node(hub)
    with caplog.at_level(logging.INFO, logger="colosseum.hub.hub"):
        reply = _upload(hub, node, None)
    assert not reply.done() and _stored(hub) == []                     # the servicer already answered the node
    assert "was given up before the hub read it" in caplog.text


def test_a_newer_stream_supersedes_the_old_one_whose_late_close_and_messages_change_nothing(tmp_path):
    hub = lineup_hub(_config(tmp_path), tmp_path)
    old = open_node(hub, session_id="s1")
    new = open_node(hub, node_id=old.node_id, session_id="s1")              # reconnected before the hub saw the cut
    assert old.closed_reason == "superseded" and hub._sessions[old.node_id] is new
    hub._on_session_closed(SessionClosed(old.node_id, "s1", "stream cut", old))   # the old stream's late close
    entry = hub._registry.by_id(old.node_id)
    assert hub._sessions[old.node_id] is new and entry.connected and entry.serial == new.serial
    hub._on_message(node_message(old, heartbeat=pb.Heartbeat(env_steps=500)))     # a superseded serial's message
    assert hub.env_steps_done == 0
    hub._on_message(node_message(new, heartbeat=pb.Heartbeat(env_steps=300)))
    assert hub.env_steps_done == 300
    hub._on_session_closed(SessionClosed(new.node_id, "s1", "end of stream", new))
    assert old.node_id not in hub._sessions and not hub._registry.by_id(old.node_id).connected


def test_a_learners_error_is_a_warning_and_its_failure_carries_the_reason(tmp_path, caplog):
    """T4.2 (A12): a learner's ``child_error`` no longer stops the run (its death report follows: the node restarts
    it through the hub); its ``learner_failed`` names the reason the node kept (the ``ChildError`` text) and fails the
    agent; with every trainable agent failed the run stops."""
    hub = lineup_hub(_config(tmp_path), tmp_path)
    node = open_node(hub)
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.hub"):
        hub._on_message(node_message(node, event=pb.Event(
            kind="child_error", proc="learner-a", error_kind="crash", message="RuntimeError: boom",
            context="node 1 (n), learner a (generation 1)", log_path="/logs/learner-a.log")))
        assert hub._failure is None and "learner-a reported an error (crash): RuntimeError: boom" in caplog.text
        hub._on_message(node_message(node, event=pb.Event(
            kind="learner_failed", proc="learner-a", exit_code=1, log_path="/logs/learner-a.log", generation=1,
            message="3 restarts within 1800 s on this node; last: crash: RuntimeError: boom")))
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any(e.startswith("Learner of a failed (3 restarts") and "RuntimeError: boom" in e for e in errors)
    assert hub._learners["a"].status == "failed" and hub._failure == "every trainable agent failed; stopping"


@pytest.mark.parametrize("kind, proc", [("child_error", "worker-0"), ("learner_failed", "learner-a")])
def test_a_failure_reported_during_the_shutdown_fails_the_run(tmp_path, kind, proc):
    """SP4 R3-08: a child that fails while the run winds down fails it (exit 1) even after the budget (T4.2: a
    worker's reported error; the only trainable agent's failed learner)."""
    config = make_test_config("turns", agents={"a": {}}, training={"total_timesteps": 10**12})
    hub = Hub(config, make_test_run_dir(config, tmp_path), options=HubOptions(
        mode="hub", listen="127.0.0.1", port=0, insecure=True, identity_dir=tmp_path / "hub-identity"))

    def main_loop() -> int:                             # the budget is reached; the node reports while stopping
        node = open_node(hub)
        hub._inbox.put(node_message(node, event=pb.Event(kind=kind, proc=proc, error_kind="crash",
                                                         message="late", exit_code=1, generation=1)))
        hub._inbox.put(node_message(node, event=pb.Event(kind="run_stopped")))
        return 0

    hub._main_loop = main_loop
    assert hub.launch() == 1
    assert hub._failure == ("every trainable agent failed; stopping" if kind == "learner_failed"
                            else "worker-0 failed (crash): late [node 1 (n)]")


def test_evictions_wait_for_a_node_whose_stream_is_down_and_reach_it_with_its_welcome(tmp_path):
    hub = lineup_hub(_config(tmp_path), tmp_path)
    up, down = open_node(hub, name="up"), open_node(hub, name="down", cpu_learners=0)
    hub._on_session_closed(SessionClosed(down.node_id, down.session_id, "stream cut", down))
    evictions = iter([{"a": ["ckpt_v5"]}, {}])
    hub._coordinator.take_evictions = lambda: next(evictions)
    hub._broadcast_announce()
    assert [(e.agent_id, e.checkpoint_id) for e in up.of("announce")[-1].evictions] == [("a", "ckpt_v5")]
    back = open_node(hub, node_id=down.node_id, session_id=down.session_id)
    assert [(e.agent_id, e.checkpoint_id) for e in back.of("welcome")[0].evictions] == [("a", "ckpt_v5")]
    hub._broadcast_announce()
    assert not up.of("announce")[-1].evictions and not back.of("announce")[-1].evictions


def test_a_removed_node_is_forgotten_by_the_intake_a_lost_one_is_not(tmp_path, caplog):
    hub = lineup_hub(_config(tmp_path), tmp_path)
    lost, removed = open_node(hub, name="lost"), open_node(hub, name="removed", cpu_learners=0)
    for handle in (lost, removed):
        hub._intake.begin(handle.node_id, handle.session_id, 1)
        hub._intake.committed(handle.node_id, handle.session_id, 1)
    accept_version(hub, "a")
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.hub"):
        hub._on_node_lost(hub._registry.by_id(lost.node_id))
        hub._on_node_lost(hub._registry.by_id(removed.node_id), reason="removed")
    assert hub._intake.is_acked(lost.node_id, lost.session_id, 1)
    assert not hub._intake.is_acked(removed.node_id, removed.session_id, 1)
    assert "(lost) lost: no heartbeat for more than 10 s" in caplog.text
    assert "(removed) removed; its cells are dropped" in caplog.text and "removed: no heartbeat" not in caplog.text
    assert lost.closed_reason == "lost" and removed.closed_reason == "removed" and hub._nodes_lost == 2


def test_a_hub_on_all_interfaces_records_a_loopback_address_of_the_family_it_bound(tmp_path):
    """M8: ``hub.json``'s loopback reaches the bound socket ("[::1]" for "[::]", "127.0.0.1" for the IPv4 fallback)."""
    import threading
    import time

    from colosseum.hub.admin import HubTarget, call
    from colosseum.hub.registry import read_hub_json

    config = make_test_config("turns", agents={"a": {}}, training={"total_timesteps": 10**12})
    hub = Hub(config, make_test_run_dir(config, tmp_path), options=HubOptions(
        mode="hub", port=0, insecure=True, identity_dir=tmp_path / "hub-identity"))
    thread = threading.Thread(target=hub.launch, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 60
        while not hub.ports and time.monotonic() < deadline:
            time.sleep(0.05)
        info = read_hub_json(hub._run_dir)
        bound = info.addresses[0].rsplit(":", 1)[0]
        assert info.loopback.rsplit(":", 1)[0] == {"[::]": "[::1]", "0.0.0.0": "127.0.0.1"}[bound]
        assert call(HubTarget(info.loopback, False, None, hub.identity.token), "hub.status")["code"] == 0
    finally:
        hub.request_stop()
        thread.join(60)
