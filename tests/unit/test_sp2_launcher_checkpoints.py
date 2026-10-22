"""SP2 launcher checkpoints on SP5's hub (T2.5 b; R5): resume (with role signatures) and the budget's start point,
snapshot saves, the node worker's arguments, final snapshots that arrive as uploads during the shutdown (T5.4).

Ported from SP1's tests/unit/test_checkpoint_store.py (launcher part) onto lineups and roles. SP2's lineup resolution
and per-worker refresh deltas are gone (spec 4.2: nodes pull lineups after their weights are cached; no latest
fallback for a missing snapshot).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.config import config_hash
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model
from colosseum.core.roles import role_signature
from colosseum.hub.hub import setup_run
from colosseum.learner.learner import make_checkpoint_payload
from colosseum.net.proto import colosseum_v1_pb2 as pb
from game_helpers import FakeAlgorithm, agent_role_of, make_coordinator, make_test_config, make_test_run_dir
from hub_kit import lineup_hub, make_hub, node_message, open_node, upload_event


def sd(value: float = 0.0) -> dict[str, np.ndarray]:
    return {"w": np.full((2, 3), value, np.float32), "b": np.zeros(3, np.float32)}


def model_state(config, agent_id: str = "agent_0") -> dict[str, np.ndarray]:
    _roles, role = agent_role_of(config, agent_id)
    model = build_model(config.get_agent_config(agent_id), role)
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def signed_meta(config, agent_id: str = "agent_0", **extra) -> dict:
    roles, role = agent_role_of(config, agent_id)
    return {"roles": roles, "role_signature": role_signature(role), **extra}


def _old_run(tmp_path: Path, cfg, version: int, env_steps: int, **meta) -> Path:
    run = tmp_path / "old_run"
    CheckpointManager(run / "checkpoints").save(
        "agent_0", version, model_state(cfg), meta_extra={"env_steps": env_steps, **signed_meta(cfg), **meta},
    )
    return run


def _resume(cfg, tmp_path, name="resumed"):
    hub = make_hub(cfg, make_test_run_dir(cfg, tmp_path, name=name))
    setup = setup_run(cfg, validate=False)
    return hub, hub._resolve_resume(setup.agent_configs, setup.role_specs)


def test_resume_seeds_global_env_step_counter(tmp_path):
    run = _old_run(tmp_path, make_test_config("solo"), version=12, env_steps=1200)
    hub, states = _resume(make_test_config("solo", training={"resume_from": str(run)}), tmp_path)
    assert states["agent_0"]["policy_version"] == 12
    assert hub.env_steps_done == 1200                       # the budget counter's start point (BudgetCounter.add)


def test_resume_from_a_run_dir_written_by_the_hub(tmp_path):
    cfg = make_test_config("solo")
    first = make_test_run_dir(cfg, tmp_path, name="first")
    hub = make_hub(cfg, first)
    hub._coordinator = make_coordinator(cfg, first.checkpoints)  # as in launch()
    hub._budget.add(321)
    assert hub._save_checkpoint({"agent_id": "agent_0", "policy_version": 7, "final": True,
                                 "model_state": model_state(cfg), "trainer_state_bytes": None}) == "ckpt_v7"
    assert (first.checkpoints / "agent_0" / "ckpt_v7" / "model.pt").is_file()
    resumed, states = _resume(make_test_config("solo", training={"resume_from": str(first.root)}), tmp_path)
    assert states["agent_0"]["policy_version"] == 7 and resumed.env_steps_done == 321


def test_resume_rejects_architecture_mismatch_before_spawning(tmp_path):
    run = tmp_path / "old_run"
    cfg = make_test_config("solo")
    CheckpointManager(run / "checkpoints").save("agent_0", 3, sd(3), meta_extra=signed_meta(cfg))
    with pytest.raises(ConfigError, match="do not match"):
        _resume(make_test_config("solo", training={"resume_from": str(run)}), tmp_path)


def test_resume_rejects_another_role_signature_naming_the_checkpoint(tmp_path):
    run = _old_run(tmp_path, make_test_config("solo"), version=5, env_steps=50, role_signature="other-game")
    with pytest.raises(ConfigError, match="role signature") as exc:
        _resume(make_test_config("solo", training={"resume_from": str(run)}), tmp_path)
    assert "ckpt_v5" in str(exc.value)
    assert "same roles" in str(exc.value) and "drop training.resume_from" in str(exc.value)  # fix hint


def test_resume_rejects_an_unsigned_sp1_checkpoint_naming_it(tmp_path):
    run = tmp_path / "old_run"
    CheckpointManager(run / "checkpoints").save("agent_0", 4, model_state(make_test_config("solo")),
                                                meta_extra={"env_steps": 40})
    with pytest.raises(ConfigError, match="no role_signature") as exc:
        _resume(make_test_config("solo", training={"resume_from": str(run)}), tmp_path)
    assert str(run / "checkpoints" / "agent_0" / "ckpt_v4") in str(exc.value)
    assert "drop training.resume_from" in str(exc.value)  # fix hint


def test_resume_rejects_a_checkpoint_of_another_role(tmp_path):
    """Asymmetric game: prey's checkpoint offered to hunter (different spaces) fails on the
    signature, before any architecture check, naming the checkpoint."""
    cfg = make_test_config("asymmetric")
    prey_roles, prey_role = agent_role_of(cfg, "prey")
    prey_ckpt = tmp_path / "old_run" / "checkpoints" / "prey"
    CheckpointManager(prey_ckpt.parent).save(
        "prey", 2, model_state(cfg, "prey"),
        meta_extra={"env_steps": 20, "roles": prey_roles, "role_signature": role_signature(prey_role)})
    with pytest.raises(ConfigError, match="role signature") as exc:
        _resume(make_test_config("asymmetric", training={"resume_from": str(prey_ckpt / "ckpt_v2")}), tmp_path)
    assert "[hunter]" in str(exc.value) and str(prey_ckpt / "ckpt_v2") in str(exc.value)


def test_the_node_worker_passes_env_max_idle_steps_and_lineups_to_the_rollout_loop(tmp_path, monkeypatch):
    """R14: ``env.max_idle_steps`` reaches the node worker's ``RolloutLoop`` (with roles and its first lineups)."""
    from node_kit import run_worker_play

    cfg = make_test_config("turns", env={"max_idle_steps": 123})
    setup = setup_run(cfg, validate=False)
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    lineups = coord.generate_lineups(cfg.rollout.envs_per_worker, env_offset=0)
    recorded, _io = run_worker_play(monkeypatch, cfg, lineups)
    assert recorded["max_idle_steps"] == 123
    assert recorded["lineups"] == lineups and recorded["agent_roles"] == setup.agent_roles
    assert recorded["num_envs"] == cfg.rollout.envs_per_worker
    model = recorded["model_factories"]["agent_0"]()  # built for the agent's role spec
    assert model.state_dict().keys() == build_model(cfg, setup.role_specs["agent_0"]).state_dict().keys()


def _exit_after_stop(stop_event, code: int) -> None:
    stop_event.wait(60)
    os._exit(code)


def _hub_with_node(tmp_path, cfg):
    """A hub (coordinator, budget, identity, pure state: ``lineup_hub``) with one node session that runs every
    learner (generation 1), and a spawned child supervised as the node: it exits 0 once the hub stops."""
    hub = lineup_hub(cfg, tmp_path)
    node = open_node(hub)
    proc = mp.get_context("spawn").Process(target=_exit_after_stop, args=(hub._stop_event, 0), daemon=True)
    proc.start()
    hub._supervisor.add("node", proc)
    return hub, node, proc


def _node_stops(hub, node, uploads) -> threading.Thread:
    """The node's stop path, standing in for its learners: once the hub stopped (``stop_run``), each ``(delay, upload)``
    arrives ``delay`` seconds after the stop (a learner's final snapshot), then the node's ``run_stopped``."""
    def run() -> None:
        hub._stop_event.wait(60)
        start = time.monotonic()
        for delay, event in uploads:
            time.sleep(max(0.0, start + delay - time.monotonic()))
            hub._inbox.put(event)
        hub._inbox.put(node_message(node, event=pb.Event(kind="run_stopped")))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _payloads(agent_id: str = "agent_0") -> tuple[dict, dict]:
    algo = FakeAlgorithm()
    algo.train_once()
    periodic = make_checkpoint_payload(agent_id, algo)
    algo.train_once()
    return periodic, make_checkpoint_payload(agent_id, algo, final=True)


def test_shutdown_saves_the_snapshots_the_node_uploads(tmp_path):
    """A periodic snapshot still in the inbox and the final one uploaded during the shutdown are both saved, with the
    hub's env steps (R9) and the run's checkpoint meta."""
    cfg = make_test_config("solo")
    hub, node, proc = _hub_with_node(tmp_path, cfg)
    hub._budget.add(777)
    periodic, final = _payloads()
    queued = upload_event(hub, periodic)
    hub._inbox.put(queued)                                           # arrived before the stop, not handled yet
    last = upload_event(hub, final, "final")
    _node_stops(hub, node, [(0.0, last)])

    hub._shutdown()

    assert proc.exitcode == 0
    assert queued.reply.result(1).accepted and last.reply.result(1).accepted
    infos = hub._coordinator.checkpoint_manager.list_checkpoints("agent_0")
    assert [c.checkpoint_id for c in infos] == ["ckpt_v1", "ckpt_v2"]
    assert infos[-1].meta["final"] is True and infos[0].meta["final"] is False
    assert infos[-1].meta["env_steps"] == 777 and infos[-1].meta["config_hash"] == config_hash(cfg)
    assert infos[-1].meta["role_signature"] == hub._coordinator.role_signature("agent_0")


def test_an_upload_that_fails_to_save_is_refused_and_the_next_one_is_saved(tmp_path, caplog):
    cfg = make_test_config("solo")
    hub = lineup_hub(cfg, tmp_path)
    open_node(hub)
    events = [upload_event(hub, payload) for payload in _payloads()]
    real_save = hub._save_checkpoint

    def save(payload):
        if payload["policy_version"] == 1:
            raise OSError("disk full")
        return real_save(payload)

    hub._save_checkpoint = save
    for event in events:
        hub._inbox.put(event)
    hub._drain_inbox(0.0)
    assert [c.checkpoint_id for c in hub._coordinator.checkpoint_manager.list_checkpoints("agent_0")] == [
        "ckpt_v2"]
    first, second = (event.reply.result(1) for event in events)
    assert not first.accepted and "OSError" in first.reason and second.accepted   # the node is told
    assert "Saving the checkpoint snapshot v1 of 'agent_0'" in caplog.text and "disk full" in caplog.text
    assert "Traceback" in caplog.text


def test_shutdown_tears_children_down_even_if_a_save_fails(tmp_path):
    """Every save fails: each upload is refused (``accepted`` false, the hub goes on: R8 / P-PF2-9) and the shutdown
    still stops the node."""
    cfg = make_test_config("solo")
    hub, node, proc = _hub_with_node(tmp_path, cfg)
    calls: list[int] = []

    def failing_save(payload):
        calls.append(payload["policy_version"])
        raise OSError("disk full")

    hub._save_checkpoint = failing_save
    periodic, final = _payloads()
    queued = upload_event(hub, periodic)
    hub._inbox.put(queued)
    last = upload_event(hub, final, "final")
    _node_stops(hub, node, [(0.0, last)])
    hub._shutdown()
    assert calls == [1, 2] and not proc.is_alive() and proc.exitcode is not None
    assert not queued.reply.result(1).accepted and not last.reply.result(1).accepted


def test_shutdown_keeps_draining_after_a_failed_save(tmp_path):
    """A save failure does not end the grace window: a second learner's final snapshot, delivered later, is still
    saved."""
    cfg = make_test_config("solo", agents={"a0": {}, "a1": {}})
    hub, node, proc = _hub_with_node(tmp_path, cfg)
    real_save = hub._save_checkpoint

    def save(payload):
        if payload["agent_id"] == "a0" and not payload["final"]:
            raise OSError("disk full")
        return real_save(payload)

    hub._save_checkpoint = save
    a0_periodic, a0_final = _payloads("a0")
    _a1_periodic, a1_final = _payloads("a1")
    failing = upload_event(hub, a0_periodic)
    hub._inbox.put(failing)
    _node_stops(hub, node, [(0.0, upload_event(hub, a0_final, "final")),
                            (1.5, upload_event(hub, a1_final, "final"))])
    start = time.monotonic()
    hub._shutdown()
    assert time.monotonic() - start < 10.0
    assert not proc.is_alive() and proc.exitcode == 0
    assert not failing.reply.result(1).accepted
    mgr = hub._coordinator.checkpoint_manager
    assert [c.meta["final"] for c in mgr.list_checkpoints("a1")] == [True]
    assert [c.meta["final"] for c in mgr.list_checkpoints("a0")] == [True]
