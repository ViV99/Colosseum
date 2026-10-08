"""Checkpoint store: confinement, atomic writes, resume, final snapshot, missing-ckpt fallback (T5.3)."""
from __future__ import annotations

import io
import json
import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from colosseum.coordinator import checkpoint_manager as cm_module
from colosseum.coordinator.checkpoint_manager import (
    CheckpointManager,
    check_model_state,
    resolve_resume,
)
from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import ColosseumConfig, config_hash
from colosseum.core.errors import ConfigError
from colosseum.core.types import MatchConfig, PlayerSlot
from colosseum.learner.learner import apply_resume_state, make_checkpoint_payload, send_checkpoint

TTT = "examples.tic_tac_toe"


def make_config(**training) -> ColosseumConfig:
    return ColosseumConfig.model_validate({
        "env": {"env_class": f"{TTT}.env.TicTacToeEnv", "num_players": 2},
        "networks": {
            "encoder_class": f"{TTT}.networks.TicTacToeEncoder",
            "policy_class": f"{TTT}.networks.TicTacToePolicy",
            "value_class": f"{TTT}.networks.TicTacToeValue",
        },
        "training": {"phase": "self_play", **training},
        "self_play": {"pool_size": 2, "latest_prob": 0.0, "shuffle_seats": False},
    })


def sd(value: float = 0.0) -> dict[str, np.ndarray]:
    return {"w": np.full((2, 3), value, np.float32), "b": np.zeros(3, np.float32)}


class FakeAlgorithm:
    """Implements the BaseAlgorithm members used by checkpointing (contract T4.6)."""

    def __init__(self, version: int = 0):
        self._model = nn.Linear(3, 2)
        self._opt = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        self._version = version

    @property
    def model(self):
        return self._model

    @property
    def policy_version(self) -> int:
        return self._version

    def train_once(self):
        self._opt.zero_grad()
        self._model(torch.ones(4, 3)).sum().backward()
        self._opt.step()
        self._version += 1

    def state_dict(self):
        return {"optimizer": self._opt.state_dict(), "policy_version": self._version, "consumed_samples": 0}

    def load_state_dict(self, state):
        self._opt.load_state_dict(state["optimizer"])
        self._version = int(state["policy_version"])


def contains_tensor(obj) -> bool:
    if isinstance(obj, torch.Tensor):
        return True
    if isinstance(obj, dict):
        return any(contains_tensor(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(contains_tensor(v) for v in obj)
    return False


def test_eviction_never_touches_dirs_outside_base_dir(tmp_path):
    """Regression for R6-06: meta.json 'path' pointing at another run must be ignored."""
    outside = tmp_path / "other_run" / "agent_0" / "ckpt_v1"
    outside.mkdir(parents=True)
    (outside / "model.pt").write_bytes(b"precious")
    base = tmp_path / "run" / "checkpoints"
    copied = base / "agent_0" / "ckpt_v1"
    copied.mkdir(parents=True)
    torch.save({k: torch.tensor(v) for k, v in sd().items()}, copied / "model.pt")
    (copied / "meta.json").write_text(json.dumps({"path": str(outside), "policy_version": 1}))

    mgr = CheckpointManager(base, pool_size=2)
    mgr.save("agent_0", 2, sd(2))
    mgr.save("agent_0", 3, sd(3))

    assert (outside / "model.pt").read_bytes() == b"precious"
    assert sorted(p.name for p in (base / "agent_0").iterdir()) == ["ckpt_v2", "ckpt_v3"]
    assert [c.checkpoint_id for c in mgr.list_checkpoints("agent_0")] == ["ckpt_v2", "ckpt_v3"]


def test_scan_reads_only_own_dir_and_derives_path(tmp_path):
    base = tmp_path / "ckpts"
    CheckpointManager(base, pool_size=5).save("a", 7, sd(7))
    mgr = CheckpointManager(base, pool_size=5)
    info = mgr.latest("a")
    assert info.checkpoint_id == "ckpt_v7"
    assert info.path == base / "a" / "ckpt_v7"
    assert mgr.agents == ["a"]


def test_save_is_atomic_on_failure(tmp_path, monkeypatch):
    mgr = CheckpointManager(tmp_path, pool_size=5)
    mgr.save("a", 1, sd(1))

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(cm_module.json, "dumps", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        mgr.save("a", 2, sd(2))
    monkeypatch.undo()
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == ["ckpt_v1"]
    assert [c.checkpoint_id for c in mgr.list_checkpoints("a")] == ["ckpt_v1"]


def test_stale_tmp_dirs_are_cleaned_on_scan(tmp_path):
    (tmp_path / "a" / ".tmp-ckpt_v9-dead").mkdir(parents=True)
    CheckpointManager(tmp_path)
    assert not list((tmp_path / "a").glob(".tmp-*"))


def test_duplicate_id_is_replaced_not_duplicated(tmp_path):
    mgr = CheckpointManager(tmp_path, pool_size=5)
    mgr.save("a", 3, sd(1.0))
    mgr.save("a", 3, sd(30.0), trainer_state={"policy_version": 3})
    assert [c.checkpoint_id for c in mgr.list_checkpoints("a")] == ["ckpt_v3"]
    assert float(mgr.load_model("a", "ckpt_v3")["w"][0, 0]) == 30.0
    assert mgr.load_trainer_state("a", "ckpt_v3") == {"policy_version": 3}


def test_trainer_state_bytes_written_verbatim(tmp_path):
    buf = io.BytesIO()
    torch.save({"policy_version": 4, "x": torch.ones(2)}, buf)
    mgr = CheckpointManager(tmp_path)
    mgr.save("a", 4, sd(), trainer_state=buf.getvalue(), meta_extra={"env_steps": 99})
    assert mgr.load_trainer_state("a", "ckpt_v4")["policy_version"] == 4
    meta = json.loads((tmp_path / "a" / "ckpt_v4" / "meta.json").read_text())
    assert meta["env_steps"] == 99 and meta["agent_id"] == "a" and meta["policy_version"] == 4


def test_resolve_resume_checkpoint_dir_run_dir_and_pt(tmp_path):
    base = tmp_path / "old_run" / "checkpoints"
    mgr = CheckpointManager(base, pool_size=5)
    buf = io.BytesIO()
    torch.save({"policy_version": 12}, buf)
    mgr.save("agent_0", 5, sd(5), meta_extra={"env_steps": 500})
    mgr.save("agent_0", 12, sd(12), trainer_state=buf.getvalue(), meta_extra={"env_steps": 1200})

    from_run = resolve_resume(str(tmp_path / "old_run"), "agent_0")
    assert from_run["policy_version"] == 12 and from_run["env_steps"] == 1200
    assert isinstance(from_run["trainer_state"], bytes)
    assert float(from_run["model_state"]["w"][0, 0]) == 12.0

    from_dir = resolve_resume(str(base / "agent_0" / "ckpt_v5"), "agent_0")
    assert from_dir["policy_version"] == 5 and from_dir["trainer_state"] is None

    pt = tmp_path / "bc.pt"
    torch.save({k: torch.tensor(v) for k, v in sd(7).items()}, pt)
    from_pt = resolve_resume(str(pt), "agent_0")
    assert from_pt["policy_version"] == 0 and from_pt["trainer_state"] is None
    assert float(from_pt["model_state"]["w"][0, 0]) == 7.0

    assert resolve_resume(str(tmp_path / "old_run"), "unknown_agent") is None
    with pytest.raises(ConfigError, match="resume_from"):
        resolve_resume(str(tmp_path / "missing"), "agent_0")
    for result in (from_run, from_dir, from_pt):
        assert not contains_tensor(result)


def test_check_model_state_reports_architecture_mismatch():
    model = nn.Linear(2, 3)
    good = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    check_model_state(model, good, "ok.pt")
    with pytest.raises(ConfigError, match="shape mismatch weight"):
        check_model_state(model, {"weight": np.zeros((3, 4), np.float32), "bias": good["bias"]}, "bc.pt")
    with pytest.raises(ConfigError, match="missing keys"):
        check_model_state(model, {"weight": good["weight"]}, "bc.pt")


def test_final_checkpoint_roundtrip_and_version_continuation(tmp_path):
    algo = FakeAlgorithm()
    for _ in range(3):
        algo.train_once()
    payload = make_checkpoint_payload("agent_0", algo, final=True)
    assert not contains_tensor(payload)

    q = mp.get_context("spawn").Queue(maxsize=2)
    assert send_checkpoint(q, payload, block=True, timeout=5.0)
    received = q.get(timeout=5.0)

    cfg = make_config()
    coord = Coordinator(cfg, checkpoint_dir=tmp_path / "ckpt")
    meta_extra = {"networks": cfg.networks.model_dump(mode="json", by_alias=True),
                  "config_hash": config_hash(cfg), "env_steps": 321}
    ckpt_id = coord.save_checkpoint_payload(received, meta_extra=meta_extra)
    assert ckpt_id == "ckpt_v3"
    meta = json.loads((tmp_path / "ckpt" / "agent_0" / "ckpt_v3" / "meta.json").read_text())
    assert meta["final"] is True and meta["env_steps"] == 321
    assert meta["networks"]["encoder_class"].endswith("TicTacToeEncoder")
    assert len(meta["config_hash"]) == 16

    resumed = FakeAlgorithm()
    apply_resume_state(resumed, resolve_resume(str(tmp_path / "ckpt" / "agent_0" / "ckpt_v3"), "agent_0"))
    assert resumed.policy_version == 3
    for k, v in algo.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[k], v)
    resumed.train_once()
    assert resumed.policy_version == 4  # versions continue after resume


def test_apply_resume_without_trainer_state_keeps_version(tmp_path):
    mgr = CheckpointManager(tmp_path)
    model = FakeAlgorithm().model
    mgr.save("a", 40, {k: v.detach().numpy() for k, v in model.state_dict().items()})
    algo = FakeAlgorithm()
    apply_resume_state(algo, resolve_resume(str(tmp_path / "a" / "ckpt_v40"), "a"))
    assert algo.policy_version == 40


def test_send_checkpoint_nonblocking_reports_full_queue():
    q = mp.get_context("spawn").Queue(maxsize=1)
    assert send_checkpoint(q, {"policy_version": 1}, block=False)
    assert not send_checkpoint(q, {"policy_version": 2}, block=False)


def test_missing_checkpoint_falls_back_to_latest_and_collects(tmp_path, caplog):
    from colosseum.launcher import _derive_worker_configs

    coord = Coordinator(make_config(), checkpoint_dir=tmp_path / "ckpt")
    coord.checkpoint_manager.save("agent_0", 10, sd(10))
    match = MatchConfig(match_id="m", player_slots=[
        PlayerSlot("agent_0", None, True),
        PlayerSlot("agent_0", "ckpt_v10", False),
        PlayerSlot("agent_0", "ckpt_v999", False),
    ])
    new_ckpts, nets, collect, agents = _derive_worker_configs([match], coord, ["agent_0"])
    assert nets == [["latest", "ckpt_v10", "latest"]]
    assert collect == [[True, False, True]]
    assert list(new_ckpts["agent_0"]) == ["ckpt_v10"]
    assert "ckpt_v999" in caplog.text

    again, nets2, _, _ = _derive_worker_configs([match], coord, ["agent_0"],
                                               already_sent={"agent_0": {"ckpt_v10"}})
    assert again["agent_0"] == {}  # already on the worker: not reloaded or resent
    assert nets2 == [["latest", "ckpt_v10", "latest"]]


def test_refresh_marks_checkpoints_sent_only_after_successful_put(tmp_path):
    from colosseum.launcher import Launcher

    cfg = make_config()
    coord = Coordinator(cfg, checkpoint_dir=tmp_path / "ckpt")
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
    assert sent[0]["agent_0"] == {"ckpt_v10"}
    launcher._refresh_worker_matches(coord, ["agent_0"], [q], sent)
    assert q.get(timeout=5).new_checkpoints == {}  # delta only


# ---------------------------------------------------------------------------
# Controller rulings (T5.3): resume counters, global env-step counter, shutdown drain,
# LATEST_NETWORK_ID location.
# ---------------------------------------------------------------------------


def _model_numpy_state(model: nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def test_learner_resume_continues_train_step_and_consumed_samples():
    """train_step continues from the restored policy_version, consumed_samples from the trainer state."""
    from colosseum.algorithms.appo import APPO
    from colosseum.core.config import AlgorithmConfig, LearnerConfig
    from colosseum.core.types import TrajectoryChunk
    from colosseum.learner.learner import learner_process
    from dataflow_helpers import CheckedQueue, TinyModel, chunk_payload

    source = APPO(TinyModel(), AlgorithmConfig(), device="cpu")
    for step in range(2):
        source.train_step([TrajectoryChunk.from_payload(chunk_payload(T=4, version=step + v)) for v in range(2)])
    assert source.policy_version == 2 and source.consumed_samples == 16
    payload = make_checkpoint_payload("a", source)
    resume_state = {"model_state": payload["model_state"], "trainer_state": payload["trainer_state_bytes"],
                    "policy_version": payload["policy_version"], "env_steps": 0, "source": "test"}

    traj, wq, mq = CheckedQueue(), CheckedQueue(maxsize=1), CheckedQueue()
    for version in range(2):
        traj.put(chunk_payload(T=4, version=2 + version))
    stop = threading.Event()
    built: list[APPO] = []

    def factory() -> APPO:
        built.append(APPO(TinyModel(), AlgorithmConfig(), device="cpu"))
        return built[-1]

    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=factory, trajectory_queue=traj, weight_queues=[wq],
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"),
        stop_event=stop, metrics_queue=mq, resume_state=resume_state,
    ))
    thread.start()
    deadline = time.monotonic() + 60
    while mq.qsize() < 1 and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    metrics = mq.get_nowait()
    assert metrics["train_step"] == 3  # 2 restored + 1 new, not 1
    assert metrics["consumed_samples"] == 24  # 16 restored + 8 new, not 8
    assert built[0].policy_version == 3 and built[0].consumed_samples == 24


def _old_run_with_checkpoint(tmp_path: Path, cfg: ColosseumConfig, version: int, env_steps: int) -> Path:
    from colosseum.core.registry import build_model

    run = tmp_path / "old_run"
    CheckpointManager(run / "checkpoints").save(
        "agent_0", version, _model_numpy_state(build_model(cfg)), meta_extra={"env_steps": env_steps},
    )
    return run


def test_resume_seeds_global_env_step_counter(tmp_path):
    """The main process continues the global SharedCounter from the resumed checkpoint's env_steps."""
    from colosseum.launcher import Launcher

    run = _old_run_with_checkpoint(tmp_path, make_config(), version=12, env_steps=1200)
    cfg = make_config(resume_from=str(run))
    launcher = Launcher(cfg)
    states = launcher._resolve_resume({"agent_0": cfg.get_agent_config("agent_0")})
    assert states["agent_0"]["policy_version"] == 12
    assert launcher.env_steps_done == 1200
    assert launcher._env_step_counter.value == 1200


def test_resume_rejects_architecture_mismatch_before_spawning(tmp_path):
    from colosseum.launcher import Launcher

    run = tmp_path / "old_run"
    CheckpointManager(run / "checkpoints").save("agent_0", 3, sd(3))
    cfg = make_config(resume_from=str(run))
    launcher = Launcher(cfg)
    with pytest.raises(ConfigError, match="do not match"):
        launcher._resolve_resume({"agent_0": cfg.get_agent_config("agent_0")})
    assert launcher.env_steps_done == 0


def _learner_like_checkpoint_sender(cq, stop_event, periodic: dict, final: dict) -> None:
    """Child process standing in for a learner: one periodic snapshot, the final one after stop."""
    send_checkpoint(cq, periodic, block=False)
    stop_event.wait(timeout=60)
    send_checkpoint(cq, final, block=True)
    cq.close()
    cq.join_thread()


def test_shutdown_saves_checkpoints_still_queued(tmp_path):
    """Payloads still in a real checkpoint queue at shutdown, incl. the final snapshot sent
    after stop by a live child process, are saved before the children are torn down."""
    from colosseum.launcher import Launcher

    cfg = make_config()
    launcher = Launcher(cfg)
    launcher._coordinator = Coordinator(cfg, checkpoint_dir=tmp_path / "ckpt")
    launcher._agent_ids = ["agent_0"]
    launcher._checkpoint_meta = {"agent_0": {"config_hash": config_hash(cfg)}}
    cq = mp.get_context("spawn").Queue(maxsize=4)
    launcher._checkpoint_queues = {"agent_0": cq}
    launcher._all_queues = [cq]
    launcher._env_step_counter.add(777)

    algo = FakeAlgorithm()
    algo.train_once()
    periodic = make_checkpoint_payload("agent_0", algo)
    algo.train_once()
    final = make_checkpoint_payload("agent_0", algo, final=True)
    proc = mp.get_context("spawn").Process(
        target=_learner_like_checkpoint_sender, args=(cq, launcher._stop_event, periodic, final), daemon=True,
    )
    proc.start()
    launcher._processes = [proc]
    deadline = time.monotonic() + 60
    while cq.empty() and time.monotonic() < deadline:  # the periodic snapshot is queued
        time.sleep(0.05)
    assert not cq.empty()

    launcher._shutdown()

    assert proc.exitcode == 0
    infos = launcher._coordinator.checkpoint_manager.list_checkpoints("agent_0")
    assert [c.checkpoint_id for c in infos] == ["ckpt_v1", "ckpt_v2"]
    assert infos[-1].meta["final"] is True and infos[0].meta["final"] is False
    assert infos[-1].meta["env_steps"] == 777
    assert infos[-1].meta["config_hash"] == config_hash(cfg)


def test_learner_sends_final_checkpoint_on_stop():
    """A stopped learner puts a final snapshot on its checkpoint queue (R3-07)."""
    from colosseum.algorithms.appo import APPO
    from colosseum.core.config import AlgorithmConfig, LearnerConfig
    from colosseum.learner.learner import learner_process
    from dataflow_helpers import CheckedQueue, TinyModel

    stop = threading.Event()
    stop.set()
    ckq = CheckedQueue()
    learner_process(
        agent_id="a", algorithm_factory=lambda: APPO(TinyModel(), AlgorithmConfig(), device="cpu"),
        trajectory_queue=CheckedQueue(), weight_queues=[CheckedQueue(maxsize=1)],
        config=LearnerConfig(batch_chunks=2, device="cpu"), stop_event=stop,
        checkpoint_queue=ckq, checkpoint_interval=100,
    )
    final = ckq.get_nowait()
    assert final["final"] is True and final["agent_id"] == "a" and final["policy_version"] == 0
    assert isinstance(final["trainer_state_bytes"], bytes)
    assert ckq.empty()


def test_latest_network_id_lives_in_core_types():
    import inspect

    import colosseum.coordinator.coordinator as coordinator_module
    from colosseum.core.types import LATEST_NETWORK_ID
    from colosseum.worker.rollout_loop import LATEST_NETWORK_ID as loop_id
    from colosseum.worker.rollout_worker import LATEST_NETWORK_ID as worker_id

    assert LATEST_NETWORK_ID == loop_id == worker_id == "latest"
    assert "colosseum.worker" not in inspect.getsource(coordinator_module)
