"""Checkpoint store (SP1 guarantees) plus roles and role signatures in meta.json (T5.3)."""
from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
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
    check_role_signature,
    load_checkpoint_dir,
    resolve_resume,
)
from colosseum.core.config import ColosseumConfig, config_hash
from colosseum.core.errors import ConfigError
from colosseum.learner.learner import apply_resume_state, make_checkpoint_payload, send_checkpoint
from game_helpers import FakeAlgorithm, make_coordinator, make_test_config


def make_config(**training) -> ColosseumConfig:
    return make_test_config("solo", training=training, checkpoint={"pool_size": 2},
                            matchmaking={"latest_prob": 0.0, "shuffle_seats": False})


def sd(value: float = 0.0) -> dict[str, np.ndarray]:
    return {"w": np.full((2, 3), value, np.float32), "b": np.zeros(3, np.float32)}


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


def test_interrupted_eviction_never_leaves_a_half_deleted_checkpoint(tmp_path, monkeypatch):
    """Eviction renames the victim to ``.tmp-evict-*`` before deleting it, so a crash or a
    partial delete never leaves a ``ckpt_v*`` dir that breaks a strict run-dir resume (T5.4)."""
    run = tmp_path / "run"
    mgr = CheckpointManager(run / "checkpoints", pool_size=2)
    mgr.save("a", 1, sd(1))
    mgr.save("a", 2, sd(2))
    deleted: list[Path] = []

    def partial_rmtree(path, ignore_errors=False, **kwargs):  # dies after removing meta.json
        deleted.append(Path(path))
        (Path(path) / "meta.json").unlink()

    monkeypatch.setattr(cm_module.shutil, "rmtree", partial_rmtree)
    mgr.save("a", 3, sd(3))
    monkeypatch.undo()

    agent_dir = run / "checkpoints" / "a"
    assert sorted(p.name for p in agent_dir.glob("ckpt_v*")) == ["ckpt_v2", "ckpt_v3"]
    assert [p.name.startswith(".tmp-evict-ckpt_v1-") for p in deleted] == [True]
    assert resolve_resume(str(run), "a")["policy_version"] == 3
    assert [c.checkpoint_id for c in CheckpointManager(run / "checkpoints").list_checkpoints("a")] == [
        "ckpt_v2", "ckpt_v3"]
    # The leftover is an ordinary stale tmp dir for the scan.
    old = time.time() - cm_module.STALE_TMP_AGE_SEC - 60
    os.utime(deleted[0], (old, old))
    CheckpointManager(run / "checkpoints")
    assert not list(agent_dir.glob(".tmp-*"))


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


def test_checkpoint_dir_resume_requires_meta_json(tmp_path):
    """The version comes from meta.json only, never from the dir name (T5.4)."""
    base = tmp_path / "run" / "checkpoints"
    CheckpointManager(base).save("a", 5, sd(5))
    ckpt = base / "a" / "ckpt_v5"
    (ckpt / "meta.json").unlink()
    with pytest.raises(ConfigError, match="cannot read checkpoint.*meta.json"):
        resolve_resume(str(ckpt), "a")


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
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    meta_extra = {"networks": cfg.networks.model_dump(mode="json", by_alias=True),
                  "config_hash": config_hash(cfg), "env_steps": 321}
    ckpt_id = coord.save_checkpoint_payload(received, meta_extra=meta_extra)
    assert ckpt_id == "ckpt_v3"
    meta = json.loads((tmp_path / "ckpt" / "agent_0" / "ckpt_v3" / "meta.json").read_text())
    assert meta["final"] is True and meta["env_steps"] == 321
    assert meta["networks"]["model_class"] == "game_helpers.GameTestModel"
    assert len(meta["config_hash"]) == 16
    assert meta["roles"] == coord.agent_roles["agent_0"]
    assert meta["role_signature"] == coord.role_signature("agent_0")

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


# ---------------------------------------------------------------------------
# SP2: roles and role signatures (spec block 9)
# ---------------------------------------------------------------------------


def _signed(mgr: CheckpointManager, agent: str, version: int, signature: str = "sig-A") -> None:
    mgr.save(agent, version, sd(version), meta_extra={"roles": ["player"], "role_signature": signature})


def test_meta_roles_and_signature_are_read_back(tmp_path):
    mgr = CheckpointManager(tmp_path / "run" / "checkpoints")
    _signed(mgr, "a", 3)
    loaded = load_checkpoint_dir(tmp_path / "run" / "checkpoints" / "a" / "ckpt_v3")
    assert loaded["roles"] == ["player"] and loaded["role_signature"] == "sig-A"
    state = resolve_resume(str(tmp_path / "run"), "a", expected_signature="sig-A")
    assert state["roles"] == ["player"] and state["role_signature"] == "sig-A" and state["policy_version"] == 3


@pytest.mark.parametrize("source", ["run", "ckpt"])
def test_resume_rejects_a_different_role_signature_naming_the_path(tmp_path, source):
    mgr = CheckpointManager(tmp_path / "run" / "checkpoints")
    _signed(mgr, "a", 3, signature="sig-A")
    path = tmp_path / "run" if source == "run" else tmp_path / "run" / "checkpoints" / "a" / "ckpt_v3"
    with pytest.raises(ConfigError, match="role signature") as exc:
        resolve_resume(str(path), "a", expected_signature="sig-B")
    assert "ckpt_v3" in str(exc.value) and "resume_from" in str(exc.value)


def test_resume_rejects_a_checkpoint_without_signature(tmp_path):
    CheckpointManager(tmp_path / "run" / "checkpoints").save("a", 1, sd(1))
    with pytest.raises(ConfigError, match="no role_signature"):
        resolve_resume(str(tmp_path / "run"), "a", expected_signature="sig-A")
    assert resolve_resume(str(tmp_path / "run"), "a")["role_signature"] is None  # unchecked without expectation


def test_pt_resume_has_no_signature_to_check(tmp_path):
    pt = tmp_path / "bc.pt"
    torch.save({k: torch.tensor(v) for k, v in sd(7).items()}, pt)
    state = resolve_resume(str(pt), "a", expected_signature="sig-A")
    assert state["roles"] is None and state["role_signature"] is None


@pytest.mark.parametrize("meta", [{"roles": "player"}, {"roles": []}, {"roles": [1]}, {"role_signature": 5}],
                         ids=["str_roles", "empty_roles", "int_role", "int_signature"])
def test_malformed_roles_or_signature_make_the_meta_invalid(tmp_path, meta):
    mgr = CheckpointManager(tmp_path)
    mgr.save("a", 2, sd(2), meta_extra=meta)
    with pytest.raises(ConfigError, match="Malformed checkpoint"):
        load_checkpoint_dir(tmp_path / "a" / "ckpt_v2")


def test_check_role_signature_messages():
    check_role_signature({"source": "x", "role_signature": "s"}, "s", "ctx")
    with pytest.raises(ConfigError, match="ctx: checkpoint x has role signature 's'"):
        check_role_signature({"source": "x", "role_signature": "s"}, "t", "ctx")
