"""Checkpoint store: confinement, atomic writes, resume, final snapshot, missing-ckpt fallback (T5.3)."""
from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import queue
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
from helpers import make_test_run_dir

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
    """Fix round 1: only write dirs older than STALE_TMP_AGE_SEC are leftovers; a young one
    may be another process's save in progress (shared checkpoint dir) and is kept."""
    stale = tmp_path / "a" / ".tmp-ckpt_v9-dead"
    stale.mkdir(parents=True)
    old = time.time() - cm_module.STALE_TMP_AGE_SEC - 60
    os.utime(stale, (old, old))
    live = tmp_path / "a" / ".tmp-ckpt_v10-live"
    live.mkdir()
    CheckpointManager(tmp_path)
    assert [p.name for p in (tmp_path / "a").glob(".tmp-*")] == [".tmp-ckpt_v10-live"]


def test_interrupted_replace_is_restored_on_scan(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.save("a", 3, sd(3))
    # Crash between the two os.replace calls of a same-id save: the old copy was moved aside.
    os.replace(tmp_path / "a" / "ckpt_v3", tmp_path / "a" / ".tmp-old-ckpt_v3-deadbeef")
    restored = CheckpointManager(tmp_path)
    assert [c.checkpoint_id for c in restored.list_checkpoints("a")] == ["ckpt_v3"]
    assert float(restored.load_model("a", "ckpt_v3")["w"][0, 0]) == 3.0
    # A moved-aside copy next to a complete checkpoint is garbage.
    (tmp_path / "a" / ".tmp-old-ckpt_v3-0123abcd").mkdir()
    CheckpointManager(tmp_path)
    assert not list((tmp_path / "a").glob(".tmp-*"))


@pytest.mark.parametrize("bad", ["../x", "/x", "a/b", "..", "", ".hidden"])
def test_unsafe_ids_are_rejected(tmp_path, bad):
    import pydantic

    base = tmp_path / "base"
    mgr = CheckpointManager(base)
    mgr.save("a", 1, sd())
    with pytest.raises(ConfigError, match="agent id"):
        mgr.save(bad, 1, sd())
    with pytest.raises(ConfigError, match="agent id"):
        mgr.load_model(bad, "ckpt_v1")
    with pytest.raises(ConfigError, match="checkpoint id"):
        mgr.load_model("a", bad)
    with pytest.raises(ConfigError, match="checkpoint id"):
        mgr.load_trainer_state("a", bad)
    with pytest.raises(ConfigError, match="agent id"):
        resolve_resume(str(base / "a" / "ckpt_v1"), bad)
    with pytest.raises(pydantic.ValidationError, match="Invalid agent id"):
        ColosseumConfig.model_validate({**make_config().model_dump(mode="json"), "agents": {bad: {}}})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["base"]  # nothing written outside
    assert sorted(p.name for p in base.iterdir()) == ["a"]


def test_reserved_agent_id_is_rejected(tmp_path):
    """T6.4 fix round 1: the checkpoint store validates agent ids with check_agent_id, so an
    agent id reserved for global metrics is rejected there too."""
    base = tmp_path / "base"
    mgr = CheckpointManager(base)
    mgr.save("a", 1, sd())
    msg = r"Invalid agent id 'ratings': reserved for global metrics"
    with pytest.raises(ConfigError, match=msg):
        mgr.save("ratings", 1, sd())
    with pytest.raises(ConfigError, match=msg):
        mgr.load_model("ratings", "ckpt_v1")
    with pytest.raises(ConfigError, match=msg):
        resolve_resume(str(base / "a" / "ckpt_v1"), "ratings")
    assert sorted(p.name for p in base.iterdir()) == ["a"]


def test_dotted_agent_id_is_rejected(tmp_path):
    """T6.1 fix round 1: agent ids may not contain '.' (it separates --set path parts).
    Checkpoint ids keep allowing it."""
    import pydantic

    base = tmp_path / "base"
    mgr = CheckpointManager(base)
    mgr.save("a", 1, sd())
    msg = r"Invalid agent id 'a\.b': '\.' is not allowed"
    with pytest.raises(ConfigError, match=msg):
        mgr.save("a.b", 1, sd())
    with pytest.raises(ConfigError, match=msg):
        mgr.load_model("a.b", "ckpt_v1")
    with pytest.raises(ConfigError, match=msg):
        resolve_resume(str(base / "a" / "ckpt_v1"), "a.b")
    with pytest.raises(pydantic.ValidationError, match=msg):
        ColosseumConfig.model_validate({**make_config().model_dump(mode="json"), "agents": {"a.b": {}}})
    assert sorted(p.name for p in base.iterdir()) == ["a"]


def test_normal_ids_still_work(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.save("team-a_v2-0", 5, sd(5))
    assert float(mgr.load_model("team-a_v2-0", "ckpt_v5")["w"][0, 0]) == 5.0
    cfg = ColosseumConfig.model_validate({**make_config().model_dump(mode="json"), "agents": {"team-a_v2-0": {}}})
    assert cfg.get_trainable_agent_ids() == ["team-a_v2-0"]


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


def test_classify_resume_source_reads_only_the_layout(tmp_path):
    from colosseum.coordinator.checkpoint_manager import (
        RESUME_CHECKPOINT_DIR,
        RESUME_PT_FILE,
        RESUME_RUN_DIR,
        classify_resume_source,
    )

    CheckpointManager(tmp_path / "run" / "checkpoints").save("a", 1, sd(1))
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a torch file")  # classified by name only, never loaded
    assert classify_resume_source(tmp_path / "run") == RESUME_RUN_DIR
    assert classify_resume_source(str(tmp_path / "run" / "checkpoints" / "a" / "ckpt_v1")) == RESUME_CHECKPOINT_DIR
    assert classify_resume_source(garbage) == RESUME_PT_FILE
    for bad in (tmp_path / "missing", tmp_path / "run" / "checkpoints", tmp_path / "x.txt"):
        with pytest.raises(ConfigError, match="resume_from"):
            classify_resume_source(bad)


def test_resolve_resume_wraps_unreadable_checkpoints_in_config_error(tmp_path):
    base = tmp_path / "run" / "checkpoints"
    CheckpointManager(base).save("a", 2, sd(2), meta_extra={"env_steps": None})
    ckpt = base / "a" / "ckpt_v2"
    assert resolve_resume(str(ckpt), "a")["env_steps"] == 0  # null env_steps (distributed) -> 0

    good_meta = (ckpt / "meta.json").read_text()
    (ckpt / "meta.json").write_text("{not json")
    with pytest.raises(ConfigError, match="cannot read checkpoint"):
        resolve_resume(str(ckpt), "a")

    (ckpt / "meta.json").write_text(good_meta)
    (ckpt / "model.pt").write_bytes(b"garbage")
    with pytest.raises(ConfigError, match="cannot read checkpoint"):
        resolve_resume(str(ckpt), "a")
    with pytest.raises(ConfigError, match="cannot read checkpoint"):
        resolve_resume(str(tmp_path / "run"), "a")


def _two_checkpoint_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    mgr = CheckpointManager(run / "checkpoints")
    mgr.save("a", 1, sd(1))
    mgr.save("a", 2, sd(2))
    return run, run / "checkpoints" / "a"


@pytest.mark.parametrize("bad_meta", [
    "{not json",
    "[]",
    json.dumps({"policy_version": 2, "timestamp": "yesterday"}),
    json.dumps({"timestamp": 1.0}),
    json.dumps({"policy_version": "2"}),
], ids=["corrupt", "non_object", "bad_timestamp", "no_version", "str_version"])
def test_run_dir_resume_fails_loudly_on_malformed_latest_meta(tmp_path, bad_meta):
    """Explicit resume never falls back to an older checkpoint (fix round 2, M6a)."""
    run, agent_dir = _two_checkpoint_run(tmp_path)
    (agent_dir / "ckpt_v2" / "meta.json").write_text(bad_meta)
    with pytest.raises(ConfigError, match="ckpt_v2") as exc:
        resolve_resume(str(run), "a")
    assert "resume_from" in str(exc.value)


def test_run_dir_resume_with_all_metas_corrupt_is_an_error_not_a_fresh_start(tmp_path):
    run, agent_dir = _two_checkpoint_run(tmp_path)
    for d in ("ckpt_v1", "ckpt_v2"):
        (agent_dir / d / "meta.json").write_text("{not json")
    with pytest.raises(ConfigError, match="Malformed checkpoint"):
        resolve_resume(str(run), "a")


@pytest.mark.parametrize("bad_meta", ["[]", json.dumps({"policy_version": 2, "timestamp": "yesterday"})],
                         ids=["non_object", "bad_timestamp"])
def test_scan_skips_malformed_meta_with_a_warning(tmp_path, caplog, bad_meta):
    """A background index degrades gracefully (fix round 2, M6b)."""
    _, agent_dir = _two_checkpoint_run(tmp_path)
    (agent_dir / "ckpt_v2" / "meta.json").write_text(bad_meta)
    mgr = CheckpointManager(agent_dir.parent)
    assert [c.checkpoint_id for c in mgr.list_checkpoints("a")] == ["ckpt_v1"]
    assert "Skipping malformed checkpoint" in caplog.text and "ckpt_v2" in caplog.text


def test_config_hash_is_stable_and_sensitive():
    a, b = make_config(), make_config()
    assert config_hash(a) == config_hash(b) == config_hash(a)
    assert config_hash(make_config(seed=1)) != config_hash(a)
    assert config_hash(make_config(seed=1)) != config_hash(make_config(seed=2))


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
    assert agents == [["agent_0"] * 3]
    assert nets == [["latest", "ckpt_v10", "latest"]]
    assert collect == [[True, False, True]]
    assert list(new_ckpts["agent_0"]) == ["ckpt_v10"]
    assert "ckpt_v999" in caplog.text

    again, nets2, _, _ = _derive_worker_configs([match], coord, ["agent_0"],
                                               already_sent={"agent_0": {"ckpt_v10"}})
    assert again["agent_0"] == {}  # already on the worker: not reloaded or resent
    assert nets2 == [["latest", "ckpt_v10", "latest"]]


def test_derive_worker_configs_two_agents(tmp_path):
    """Per-env slot maps for two agents (replaces the deleted test_derive_worker_configs)."""
    from colosseum.launcher import _derive_worker_configs

    cfg = ColosseumConfig.model_validate({**make_config().model_dump(mode="json"), "agents": {"alpha": {}, "beta": {}}})
    coord = Coordinator(cfg, checkpoint_dir=tmp_path / "ckpt")
    coord.checkpoint_manager.save("beta", 4, sd(4))
    slots = [
        [PlayerSlot("alpha", None, True), PlayerSlot("beta", None, True)],
        [PlayerSlot("beta", None, True), PlayerSlot("alpha", None, False)],
        [PlayerSlot("alpha", None, True), PlayerSlot("beta", "ckpt_v4", False)],
    ]
    matches = [MatchConfig(match_id=f"m{i}", player_slots=row) for i, row in enumerate(slots)]
    new_ckpts, nets, collect, agents = _derive_worker_configs(matches, coord, ["alpha", "beta"])
    assert agents == [["alpha", "beta"], ["beta", "alpha"], ["alpha", "beta"]]
    assert nets == [["latest", "latest"], ["latest", "latest"], ["latest", "ckpt_v4"]]
    assert collect == [[True, True], [True, False], [True, False]]
    assert new_ckpts["alpha"] == {} and list(new_ckpts["beta"]) == ["ckpt_v4"]
    assert float(new_ckpts["beta"]["ckpt_v4"]["w"][0, 0]) == 4.0


def test_worker_seats_latest_for_unloaded_network_and_reports_it():
    """A slot naming a network the worker does not have plays (and reports) latest and collects."""
    from colosseum.core.types import LATEST_NETWORK_ID, WorkerCommand
    from dataflow_helpers import EnvFactory, GridStepEnv, make_loop, make_tiny_model

    loop, col = make_loop(
        EnvFactory(GridStepEnv, lengths=(3,)), make_tiny_model, agent_ids=("a",), num_envs=1,
        slot_agent_map=[["a", "a"]], slot_network_map=[["latest", "ckpt_v9"]], collect_mask=[[True, False]],
    )
    assert loop._slot_network_map == [[LATEST_NETWORK_ID, LATEST_NETWORK_ID]]
    assert loop._collect_mask == [[True, True]]
    for _ in range(3):
        loop.step()
    assert col.results
    assert all(s.network_id == LATEST_NETWORK_ID for r in col.results for s in r.seats)

    col.commands.append(WorkerCommand(
        slot_agent_map=[["a", "a"]], slot_network_map=[["ckpt_v7", "latest"]],
        collect_mask=[[False, True]], new_checkpoints={},
    ))
    for _ in range(6):
        loop.step()
    assert loop._slot_network_map == [[LATEST_NETWORK_ID, LATEST_NETWORK_ID]]
    assert loop._collect_mask == [[True, True]]
    assert len(col.results) >= 3
    assert all(s.network_id == LATEST_NETWORK_ID for r in col.results for s in r.seats)
    # Both seats collected the whole time: 9 steps x 2 seats, chunk_length 4.
    assert len(col.chunks) == 4
    loop.close()


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
    from colosseum.core.ipc import SharedCounter
    from colosseum.core.types import TrajectoryChunk
    from colosseum.learner.learner import learner_process
    from dataflow_helpers import CheckedQueue, TinyModel, chunk_payload

    algo_cfg = AlgorithmConfig(lr_schedule="linear")
    source = APPO(TinyModel(), algo_cfg, device="cpu")
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
        built.append(APPO(TinyModel(), algo_cfg, device="cpu"))
        return built[-1]

    # The main process seeds the global env-step counter from the resumed checkpoint (D3).
    counter = SharedCounter()
    counter.add(400)
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=factory, trajectory_queue=traj, weight_queues=[wq],
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"),
        stop_event=stop, metrics_queue=mq, resume_state=resume_state,
        progress_counter=counter, total_timesteps=1000,
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
    # LR progress continues from the restored budget, not from the schedule start.
    assert metrics["progress"] == pytest.approx(0.4)
    assert metrics["lr"] == pytest.approx(algo_cfg.learning_rate * 0.6)


def test_resumed_learner_lr_follows_restored_progress():
    """Before any new step, the resumed algorithm's LR is the schedule at the restored progress."""
    from colosseum.algorithms.appo import APPO
    from colosseum.core.config import AlgorithmConfig, LearnerConfig
    from colosseum.core.types import TrajectoryChunk
    from colosseum.learner.learner import learner_process
    from dataflow_helpers import CheckedQueue, TinyModel, chunk_payload

    algo_cfg = AlgorithmConfig(lr_schedule="linear")
    source = APPO(TinyModel(), algo_cfg, device="cpu")
    source.set_progress(0.4)
    source.train_step([TrajectoryChunk.from_payload(chunk_payload(T=4, version=v)) for v in range(2)])
    payload = make_checkpoint_payload("a", source)
    built: list[APPO] = []

    def factory() -> APPO:
        built.append(APPO(TinyModel(), algo_cfg, device="cpu"))
        return built[-1]

    stop = threading.Event()
    stop.set()
    learner_process(
        agent_id="a", algorithm_factory=factory, trajectory_queue=CheckedQueue(),
        weight_queues=[CheckedQueue(maxsize=1)], config=LearnerConfig(batch_chunks=2, device="cpu"),
        stop_event=stop,
        resume_state={"model_state": payload["model_state"], "trainer_state": payload["trainer_state_bytes"],
                      "policy_version": 1, "env_steps": 400, "source": "test"},
    )
    state = built[0].state_dict()
    assert state["progress"] == pytest.approx(0.4)
    assert state["optimizer"]["param_groups"][0]["lr"] == pytest.approx(algo_cfg.learning_rate * 0.6)


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
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    states = launcher._resolve_resume({"agent_0": cfg.get_agent_config("agent_0")})
    assert states["agent_0"]["policy_version"] == 12
    assert launcher.env_steps_done == 1200
    assert launcher._env_step_counter.value == 1200


def test_resume_from_a_run_dir_created_by_run_dir_create(tmp_path):
    """Checkpoints the launcher saves into ``RunDir.checkpoints`` resume from the run root (T6.2)."""
    from colosseum.core.registry import build_model
    from colosseum.launcher import Launcher

    cfg = make_config()
    first = make_test_run_dir(cfg, tmp_path, name="first")
    launcher = Launcher(cfg, first)
    launcher._coordinator = Coordinator(cfg, checkpoint_dir=first.checkpoints)  # as in launch()
    launcher._env_step_counter.add(321)
    launcher._save_checkpoint({"agent_id": "agent_0", "policy_version": 7, "final": True,
                               "model_state": _model_numpy_state(build_model(cfg)), "trainer_state_bytes": None})
    assert (first.checkpoints / "agent_0" / "ckpt_v7" / "model.pt").is_file()

    resumed_cfg = make_config(resume_from=str(first.root))
    resumed = Launcher(resumed_cfg, make_test_run_dir(resumed_cfg, tmp_path, name="second"))
    states = resumed._resolve_resume({"agent_0": resumed_cfg.get_agent_config("agent_0")})
    assert states["agent_0"]["policy_version"] == 7
    assert resumed.env_steps_done == 321


def test_resume_rejects_architecture_mismatch_before_spawning(tmp_path):
    from colosseum.launcher import Launcher

    run = tmp_path / "old_run"
    CheckpointManager(run / "checkpoints").save("agent_0", 3, sd(3))
    cfg = make_config(resume_from=str(run))
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    with pytest.raises(ConfigError, match="do not match"):
        launcher._resolve_resume({"agent_0": cfg.get_agent_config("agent_0")})
    assert launcher.env_steps_done == 0


def _learner_like_checkpoint_sender(cq, stop_event, periodic: dict | None, final: dict,
                                    final_delay: float = 0.0) -> None:
    """Child process standing in for a learner: an optional periodic snapshot, then the
    final one ``final_delay`` seconds after stop."""
    if periodic is not None:
        send_checkpoint(cq, periodic, block=False)
    stop_event.wait(timeout=60)
    time.sleep(final_delay)
    send_checkpoint(cq, final, block=True)
    cq.close()
    cq.join_thread()


def test_shutdown_saves_checkpoints_still_queued(tmp_path):
    """Payloads still in a real checkpoint queue at shutdown, incl. the final snapshot sent
    after stop by a live child process, are saved before the children are torn down."""
    from colosseum.launcher import Launcher

    cfg = make_config()
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
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


def test_drain_saves_remaining_payloads_after_a_failed_save(tmp_path, caplog):
    from colosseum.launcher import Launcher

    cfg = make_config()
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    launcher._coordinator = Coordinator(cfg, checkpoint_dir=tmp_path / "ckpt")
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
    from colosseum.launcher import Launcher

    cfg = make_config()
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    launcher._coordinator = Coordinator(cfg, checkpoint_dir=tmp_path / "ckpt")
    launcher._agent_ids = ["agent_0"]
    cq = mp.get_context("spawn").Queue(maxsize=4)
    launcher._checkpoint_queues = {"agent_0": cq}
    launcher._all_queues = [cq]
    calls: list[int] = []

    def failing_save(payload):
        calls.append(payload["policy_version"])
        raise OSError("disk full")

    launcher._save_checkpoint = failing_save
    algo = FakeAlgorithm()
    algo.train_once()
    periodic = make_checkpoint_payload("agent_0", algo)
    final = make_checkpoint_payload("agent_0", algo, final=True)
    proc = mp.get_context("spawn").Process(
        target=_learner_like_checkpoint_sender, args=(cq, launcher._stop_event, periodic, final), daemon=True,
    )
    proc.start()
    launcher._processes = [proc]
    deadline = time.monotonic() + 60
    while cq.empty() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not cq.empty()

    with pytest.raises(OSError, match="disk full"):
        launcher._shutdown()
    assert calls  # the save was attempted ...
    assert not proc.is_alive() and proc.exitcode is not None  # ... and the child was still torn down


def test_shutdown_keeps_draining_after_a_failed_save(tmp_path):
    """A save failure does not end the grace window: a second learner's final snapshot,
    delivered later, is still saved; the error propagates after teardown."""
    from colosseum.launcher import Launcher

    cfg = ColosseumConfig.model_validate({**make_config().model_dump(mode="json"), "agents": {"a0": {}, "a1": {}}})
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    launcher._coordinator = Coordinator(cfg, checkpoint_dir=tmp_path / "ckpt")
    launcher._agent_ids = ["a0", "a1"]
    ctx = mp.get_context("spawn")
    queues = {aid: ctx.Queue(maxsize=4) for aid in launcher._agent_ids}
    launcher._checkpoint_queues = queues
    launcher._all_queues = list(queues.values())
    real_save = launcher._save_checkpoint

    def save(payload):
        if payload["agent_id"] == "a0" and not payload["final"]:
            raise OSError("disk full")
        real_save(payload)

    launcher._save_checkpoint = save
    algo = FakeAlgorithm()
    algo.train_once()
    procs = [
        ctx.Process(target=_learner_like_checkpoint_sender, daemon=True, args=(
            queues["a0"], launcher._stop_event, make_checkpoint_payload("a0", algo),
            make_checkpoint_payload("a0", algo, final=True))),
        ctx.Process(target=_learner_like_checkpoint_sender, daemon=True, args=(
            queues["a1"], launcher._stop_event, None, make_checkpoint_payload("a1", algo, final=True), 1.5)),
    ]
    for proc in procs:
        proc.start()
    launcher._processes = procs
    deadline = time.monotonic() + 60
    while queues["a0"].empty() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not queues["a0"].empty()

    with pytest.raises(OSError, match="disk full"):
        launcher._shutdown()
    assert all(not p.is_alive() and p.exitcode == 0 for p in procs)
    mgr = launcher._coordinator.checkpoint_manager
    assert [c.meta["final"] for c in mgr.list_checkpoints("a1")] == [True]  # arrived 1.5 s after the failure
    assert [c.meta["final"] for c in mgr.list_checkpoints("a0")] == [True]


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
