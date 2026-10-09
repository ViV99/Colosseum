"""SP2 launcher: lineup resolution, refresh deltas, resume (with role signatures), shutdown saves (T5.4).

Ported from SP1's tests/unit/test_checkpoint_store.py (launcher part) onto lineups and roles.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
from pathlib import Path

import numpy as np
import pytest

from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.config import config_hash
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model
from colosseum.core.roles import role_signature
from colosseum.core.types import LATEST_NETWORK_ID, Lineup, SeatAssignment
from colosseum.launcher import Launcher, _resolve_lineups, setup_run
from colosseum.learner.learner import make_checkpoint_payload, send_checkpoint
from game_helpers import FakeAlgorithm, agent_role_of, make_coordinator, make_test_config, make_test_run_dir


def sd(value: float = 0.0) -> dict[str, np.ndarray]:
    return {"w": np.full((2, 3), value, np.float32), "b": np.zeros(3, np.float32)}


def model_state(config, agent_id: str = "agent_0") -> dict[str, np.ndarray]:
    _roles, role = agent_role_of(config, agent_id)
    model = build_model(config.get_agent_config(agent_id), role)
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def signed_meta(config, agent_id: str = "agent_0", **extra) -> dict:
    roles, role = agent_role_of(config, agent_id)
    return {"roles": roles, "role_signature": role_signature(role), **extra}


def test_missing_checkpoint_falls_back_to_latest_and_collects(tmp_path, caplog):
    cfg = make_test_config("turns")
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.checkpoint_manager.save("agent_0", 10, sd(10))
    layout = next(iter(coord.spec.layouts))
    lineup = Lineup(layout, [SeatAssignment("agent_0", "ckpt_v10", False),
                             SeatAssignment("agent_0", "ckpt_v999", False)])
    new_ckpts, (resolved,) = _resolve_lineups([lineup], coord, ["agent_0"])
    assert [(s.network_id, s.collect) for s in resolved.seats] == [("ckpt_v10", False), (LATEST_NETWORK_ID, True)]
    assert list(new_ckpts["agent_0"]) == ["ckpt_v10"] and "ckpt_v999" in caplog.text
    again, (resolved2,) = _resolve_lineups([lineup], coord, ["agent_0"], already_sent={"agent_0": {"ckpt_v10"}})
    assert again["agent_0"] == {}  # already on the worker: not reloaded or resent
    assert resolved2.seats[0].network_id == "ckpt_v10"


def test_resolve_lineups_two_agents(tmp_path):
    cfg = make_test_config("turns", agents={"alpha": {}, "beta": {}})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.checkpoint_manager.save("beta", 4, sd(4))
    layout = next(iter(coord.spec.layouts))
    lineups = [
        Lineup(layout, [SeatAssignment("alpha"), SeatAssignment("beta")]),
        Lineup(layout, [SeatAssignment("alpha"), SeatAssignment("beta", "ckpt_v4", False)]),
    ]
    new_ckpts, resolved = _resolve_lineups(lineups, coord, ["alpha", "beta"])
    assert [[(s.agent_id, s.network_id, s.collect) for s in lu.seats] for lu in resolved] == [
        [("alpha", "latest", True), ("beta", "latest", True)],
        [("alpha", "latest", True), ("beta", "ckpt_v4", False)],
    ]
    assert new_ckpts["alpha"] == {} and float(new_ckpts["beta"]["ckpt_v4"]["w"][0, 0]) == 4.0


def test_refresh_marks_checkpoints_sent_only_after_successful_put(tmp_path):
    cfg = make_test_config("turns", matchmaking={"mode": "self_play", "latest_prob": 0.0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.checkpoint_manager.save("agent_0", 10, sd(10))
    launcher = Launcher.__new__(Launcher)  # only _config is needed by _refresh_worker_matches
    launcher._config = cfg
    q = mp.get_context("spawn").Queue(maxsize=1)
    q.put("occupied")
    sent = [{"agent_0": set()}]
    launcher._refresh_worker_matches(coord, ["agent_0"], [q], sent)
    assert sent[0]["agent_0"] == set()  # put failed (queue full): nothing marked
    assert q.get(timeout=5) == "occupied"
    launcher._refresh_worker_matches(coord, ["agent_0"], [q], sent)
    cmd = q.get(timeout=5)
    assert "ckpt_v10" in cmd.new_checkpoints["agent_0"]
    assert len(cmd.lineups) == cfg.rollout.envs_per_worker
    assert sent[0]["agent_0"] == {"ckpt_v10"}
    launcher._refresh_worker_matches(coord, ["agent_0"], [q], sent)
    assert q.get(timeout=5).new_checkpoints == {}  # delta only


def _old_run(tmp_path: Path, cfg, version: int, env_steps: int, **meta) -> Path:
    run = tmp_path / "old_run"
    CheckpointManager(run / "checkpoints").save(
        "agent_0", version, model_state(cfg), meta_extra={"env_steps": env_steps, **signed_meta(cfg), **meta},
    )
    return run


def _resume(cfg, tmp_path, name="resumed"):
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path, name=name))
    setup = setup_run(cfg, validate=False)
    return launcher, launcher._resolve_resume(setup.agent_configs, setup.role_specs)


def test_resume_seeds_global_env_step_counter(tmp_path):
    run = _old_run(tmp_path, make_test_config("solo"), version=12, env_steps=1200)
    launcher, states = _resume(make_test_config("solo", training={"resume_from": str(run)}), tmp_path)
    assert states["agent_0"]["policy_version"] == 12
    assert launcher.env_steps_done == 1200


def test_resume_from_a_run_dir_written_by_the_launcher(tmp_path):
    cfg = make_test_config("solo")
    first = make_test_run_dir(cfg, tmp_path, name="first")
    launcher = Launcher(cfg, first)
    launcher._coordinator = make_coordinator(cfg, first.checkpoints)  # as in launch()
    launcher._env_step_counter.add(321)
    launcher._save_checkpoint({"agent_id": "agent_0", "policy_version": 7, "final": True,
                               "model_state": model_state(cfg), "trainer_state_bytes": None})
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


def test_worker_main_passes_env_max_idle_steps_and_lineups_to_the_rollout_worker(tmp_path, monkeypatch):
    """R14: ``env.max_idle_steps`` reaches ``rollout_worker_process`` (with roles and lineups)."""
    import colosseum.worker.rollout_worker as rollout_worker
    from colosseum.launcher import _worker_main

    recorded: dict = {}
    monkeypatch.setattr(rollout_worker, "rollout_worker_process", lambda **kwargs: recorded.update(kwargs))
    cfg = make_test_config("turns", env={"max_idle_steps": 123})
    setup = setup_run(cfg, validate=False)
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    lineups = coord.generate_lineups(cfg.rollout.envs_per_worker, env_offset=0)
    _worker_main(worker_id=0, config=cfg, agent_ids=["agent_0"], agent_roles=setup.agent_roles,
                 agent_configs=setup.agent_configs, role_specs=setup.role_specs, trajectory_queues={},
                 weight_queues={}, stop_event=None, lineups=lineups)
    assert recorded["max_idle_steps"] == 123
    assert recorded["lineups"] == lineups and recorded["agent_roles"] == setup.agent_roles
    assert recorded["num_envs"] == cfg.rollout.envs_per_worker
    model = recorded["model_factories"]["agent_0"]()  # built for the agent's role spec
    assert model.state_dict().keys() == build_model(cfg, setup.role_specs["agent_0"]).state_dict().keys()


def _learner_like_checkpoint_sender(cq, stop_event, periodic: dict | None, final: dict,
                                    final_delay: float = 0.0) -> None:
    """Child process standing in for a learner: an optional periodic snapshot, then the final
    one ``final_delay`` seconds after stop."""
    if periodic is not None:
        send_checkpoint(cq, periodic, block=False)
    stop_event.wait(timeout=60)
    time.sleep(final_delay)
    send_checkpoint(cq, final, block=True)
    cq.close()
    cq.join_thread()


def _launcher_with_queue(tmp_path, cfg, agent_ids):
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    launcher._coordinator = make_coordinator(cfg, tmp_path / "ckpt")
    launcher._agent_ids = list(agent_ids)
    ctx = mp.get_context("spawn")
    queues = {aid: ctx.Queue(maxsize=4) for aid in agent_ids}
    launcher._checkpoint_queues = queues
    launcher._all_queues = list(queues.values())
    return launcher, queues


def _start_sender(launcher, name, *args):
    proc = mp.get_context("spawn").Process(target=_learner_like_checkpoint_sender, args=args, daemon=True)
    proc.start()
    launcher._supervisor.add(name, proc)
    return proc


def _wait_not_empty(q) -> None:
    deadline = time.monotonic() + 60
    while q.empty() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not q.empty()


def test_shutdown_saves_checkpoints_still_queued(tmp_path):
    cfg = make_test_config("solo")
    launcher, queues = _launcher_with_queue(tmp_path, cfg, ["agent_0"])
    launcher._checkpoint_meta = {"agent_0": {"config_hash": config_hash(cfg)}}
    launcher._env_step_counter.add(777)
    algo = FakeAlgorithm()
    algo.train_once()
    periodic = make_checkpoint_payload("agent_0", algo)
    algo.train_once()
    final = make_checkpoint_payload("agent_0", algo, final=True)
    proc = _start_sender(launcher, "learner-agent_0", queues["agent_0"], launcher._stop_event, periodic, final)
    _wait_not_empty(queues["agent_0"])

    launcher._shutdown()

    assert proc.exitcode == 0
    infos = launcher._coordinator.checkpoint_manager.list_checkpoints("agent_0")
    assert [c.checkpoint_id for c in infos] == ["ckpt_v1", "ckpt_v2"]
    assert infos[-1].meta["final"] is True and infos[0].meta["final"] is False
    assert infos[-1].meta["env_steps"] == 777 and infos[-1].meta["config_hash"] == config_hash(cfg)
    assert infos[-1].meta["role_signature"] == launcher._coordinator.role_signature("agent_0")


def test_drain_saves_remaining_payloads_after_a_failed_save(tmp_path, caplog):
    cfg = make_test_config("solo")
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    launcher._coordinator = make_coordinator(cfg, tmp_path / "ckpt")
    q = queue.Queue()
    launcher._checkpoint_queues = {"agent_0": q}
    algo = FakeAlgorithm()
    for _ in range(2):
        algo.train_once()
        q.put(make_checkpoint_payload("agent_0", algo))
    real_save = launcher._save_checkpoint

    def save(payload):
        if payload["policy_version"] == 1:
            raise OSError("disk full")
        real_save(payload)

    launcher._save_checkpoint = save
    with pytest.raises(OSError, match="disk full"):
        launcher._drain_all_checkpoints()
    assert [c.checkpoint_id for c in launcher._coordinator.checkpoint_manager.list_checkpoints("agent_0")] == [
        "ckpt_v2"]
    assert "Failed to save checkpoint v1" in caplog.text and "Traceback" in caplog.text


def test_shutdown_tears_children_down_even_if_a_save_fails(tmp_path):
    cfg = make_test_config("solo")
    launcher, queues = _launcher_with_queue(tmp_path, cfg, ["agent_0"])
    calls: list[int] = []

    def failing_save(payload):
        calls.append(payload["policy_version"])
        raise OSError("disk full")

    launcher._save_checkpoint = failing_save
    algo = FakeAlgorithm()
    algo.train_once()
    proc = _start_sender(launcher, "learner-agent_0", queues["agent_0"], launcher._stop_event,
                         make_checkpoint_payload("agent_0", algo), make_checkpoint_payload("agent_0", algo, final=True))
    _wait_not_empty(queues["agent_0"])
    with pytest.raises(OSError, match="disk full"):
        launcher._shutdown()
    assert calls and not proc.is_alive() and proc.exitcode is not None


def test_shutdown_keeps_draining_after_a_failed_save(tmp_path):
    """A save failure does not end the grace window: a second learner's final snapshot,
    delivered later, is still saved; the error propagates after teardown."""
    cfg = make_test_config("solo", agents={"a0": {}, "a1": {}})
    launcher, queues = _launcher_with_queue(tmp_path, cfg, ["a0", "a1"])
    real_save = launcher._save_checkpoint

    def save(payload):
        if payload["agent_id"] == "a0" and not payload["final"]:
            raise OSError("disk full")
        real_save(payload)

    launcher._save_checkpoint = save
    algo = FakeAlgorithm()
    algo.train_once()
    procs = [
        _start_sender(launcher, "learner-a0", queues["a0"], launcher._stop_event,
                      make_checkpoint_payload("a0", algo), make_checkpoint_payload("a0", algo, final=True)),
        _start_sender(launcher, "learner-a1", queues["a1"], launcher._stop_event,
                      None, make_checkpoint_payload("a1", algo, final=True), 1.5),
    ]
    _wait_not_empty(queues["a0"])
    start = time.monotonic()
    with pytest.raises(OSError, match="disk full"):
        launcher._shutdown()
    assert time.monotonic() - start < 10.0
    assert all(not p.is_alive() and p.exitcode == 0 for p in procs)
    mgr = launcher._coordinator.checkpoint_manager
    assert [c.meta["final"] for c in mgr.list_checkpoints("a1")] == [True]
    assert [c.meta["final"] for c in mgr.list_checkpoints("a0")] == [True]
