"""Snapshot storage (SP3 T2.1, spec block 4): keep_last / keep_every / final, trainer_state only in the
keep_last window, eviction callbacks, the pool_size translation and the run-dir resume import."""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import pytest
import yaml

from colosseum.coordinator import checkpoint_manager as cm_module
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.config import CheckpointConfig, ColosseumConfig, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model
from colosseum.core.roles import role_signature
from colosseum.launcher import Launcher, setup_run
from game_helpers import agent_role_of, make_coordinator, make_test_config, make_test_run_dir


def sd(value: float) -> dict[str, np.ndarray]:
    return {"w": np.full((2,), value, np.float32)}


def ids(manager, agent_id="a") -> list[str]:
    return [c.checkpoint_id for c in manager.list_checkpoints(agent_id)]


def model_state(config, agent_id="agent_0") -> dict[str, np.ndarray]:
    _roles, role = agent_role_of(config, agent_id)
    return {k: v.detach().numpy().copy() for k, v in build_model(config.get_agent_config(agent_id), role)
            .state_dict().items()}


def _data(**sections) -> dict:
    return {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}, **sections}


def _write(tmp_path, data) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(path)


def test_keep_last_keep_every_and_the_trainer_state_window(tmp_path):
    evicted = []
    mgr = CheckpointManager(tmp_path, keep_last=3, keep_every=2, interval=10,
                            on_evict=lambda agent, ckpt: evicted.append((agent, ckpt)))
    for version in range(10, 101, 10):
        mgr.save("a", version, sd(version), trainer_state=b"opt")
    assert ids(mgr) == ["ckpt_v20", "ckpt_v40", "ckpt_v60", "ckpt_v80", "ckpt_v90", "ckpt_v100"]
    assert evicted == [("a", "ckpt_v10"), ("a", "ckpt_v30"), ("a", "ckpt_v50"), ("a", "ckpt_v70")]
    agent_dir = tmp_path / "a"
    assert sorted(p.parent.name for p in agent_dir.glob("ckpt_v*/trainer_state.pt")) == [
        "ckpt_v100", "ckpt_v80", "ckpt_v90"]
    assert sorted(p.name for p in agent_dir.iterdir()) == sorted(ids(mgr))   # no leftovers
    assert ids(CheckpointManager(tmp_path, keep_last=3, keep_every=2, interval=10)) == ids(mgr)   # a scan agrees


def test_a_final_snapshot_is_never_evicted_and_keep_every_0_is_fifo(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last=2, keep_every=0, interval=10)
    mgr.save("a", 7, sd(7), trainer_state=b"opt", meta_extra={"final": True})
    for version in (10, 20, 30):
        mgr.save("a", version, sd(version), trainer_state=b"opt")
    assert ids(mgr) == ["ckpt_v7", "ckpt_v20", "ckpt_v30"]
    assert not (tmp_path / "a" / "ckpt_v7" / "trainer_state.pt").exists()
    assert json.loads((tmp_path / "a" / "ckpt_v7" / "meta.json").read_text())["final"] is True
    assert float(mgr.load_model("a", "ckpt_v7")["w"][0]) == 7.0


def test_the_snapshot_just_saved_is_kept_even_when_it_is_older(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last=1)
    mgr.save("a", 20, sd(20))
    mgr.save("a", 5, sd(5))
    assert ids(mgr) == ["ckpt_v5", "ckpt_v20"]


def test_import_snapshots_links_model_and_meta_reads_the_source_only_and_applies_retention(tmp_path):
    old_dir = tmp_path / "old" / "checkpoints"
    old = CheckpointManager(old_dir, keep_last=10)
    for version in (10, 20, 30, 40):
        old.save("a", version, sd(version), trainer_state=b"opt", meta_extra={"role_signature": "sig"})
    leftover = old_dir / "a" / ".tmp-ckpt_v50-0123abcd"            # a stale write dir of the old run
    leftover.mkdir()
    stale = time.time() - cm_module.STALE_TMP_AGE_SEC - 60
    os.utime(leftover, (stale, stale))
    before = sorted(str(p.relative_to(tmp_path / "old")) for p in (tmp_path / "old").rglob("*"))
    evicted = []
    new = CheckpointManager(tmp_path / "new" / "checkpoints", keep_last=2, keep_every=2, interval=10,
                            on_evict=lambda agent, ckpt: evicted.append((agent, ckpt)))
    assert new.import_snapshots(old_dir, "a", expected_signature="sig") == ["ckpt_v20", "ckpt_v30", "ckpt_v40"]
    assert evicted == [("a", "ckpt_v10")]
    for info in new.list_checkpoints("a"):
        assert sorted(p.name for p in info.path.iterdir()) == ["meta.json", "model.pt"]
        src = old_dir / "a" / info.checkpoint_id
        assert (info.path / "model.pt").read_bytes() == (src / "model.pt").read_bytes()
    assert sorted(str(p.relative_to(tmp_path / "old")) for p in (tmp_path / "old").rglob("*")) == before
    assert float(new.load_model("a", "ckpt_v40")["w"][0]) == 40.0
    new.save("a", 50, sd(50), trainer_state=b"opt")                    # retention goes on across saves
    assert ids(new) == ["ckpt_v20", "ckpt_v40", "ckpt_v50"]
    assert new.import_snapshots(tmp_path / "old" / "checkpoints", "nobody") == []


def test_import_snapshots_checks_the_role_signature(tmp_path):
    old = CheckpointManager(tmp_path / "old")
    old.save("a", 10, sd(10), meta_extra={"role_signature": "other-game"})
    new = CheckpointManager(tmp_path / "new")
    with pytest.raises(ConfigError, match="role signature") as info:
        new.import_snapshots(tmp_path / "old", "a", expected_signature="sig")
    assert "ckpt_v10" in str(info.value) and ids(new) == []


def test_checkpoint_config_defaults_and_the_pool_size_translation(tmp_path, caplog):
    cfg = ColosseumConfig.model_validate(_data())
    assert (cfg.checkpoint.keep_last, cfg.checkpoint.keep_every) == (20, 10)
    with caplog.at_level(logging.WARNING, logger="colosseum.core.config"):
        old = load_config(_write(tmp_path, _data(checkpoint={"interval": 50, "pool_size": 5})))
    assert (old.checkpoint.keep_last, old.checkpoint.keep_every, old.checkpoint.interval) == (5, 10, 50)
    assert len([r for r in caplog.records if "pool_size" in r.getMessage()]) == 1
    assert "pool_size" not in old.model_dump(mode="json", by_alias=True)["checkpoint"]
    assert load_config(_write(tmp_path, _data()), {"checkpoint.pool_size": 7}).checkpoint.keep_last == 7
    with pytest.raises(ConfigError, match="both"):
        load_config(_write(tmp_path, _data(checkpoint={"pool_size": 5, "keep_last": 3})))
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, _data(checkpoint={"keep_every": -1})))
    with pytest.raises(ValueError):
        CheckpointConfig(pool_size=0)


def test_the_coordinator_stores_snapshots_by_the_configured_rules(tmp_path):
    cfg = make_test_config("solo", checkpoint={"interval": 10, "keep_last": 2, "keep_every": 3})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    for version in range(10, 71, 10):
        coord.save_checkpoint_payload({"agent_id": "agent_0", "policy_version": version, "model_state": sd(version),
                                       "trainer_state_bytes": b"opt"})
    assert ids(coord.checkpoint_manager, "agent_0") == ["ckpt_v30", "ckpt_v60", "ckpt_v70"]


def test_a_run_dir_resume_imports_the_pool_and_a_checkpoint_dir_resume_does_not(tmp_path):
    cfg = make_test_config("turns", agents={"a": {}, "b": {}}, checkpoint={"keep_last": 5})
    old = make_coordinator(cfg, tmp_path / "old" / "checkpoints")
    for aid in ("a", "b"):
        for version in (20, 40):
            old.save_checkpoint_payload({"agent_id": aid, "policy_version": version,
                                         "model_state": model_state(cfg, aid), "trainer_state_bytes": b"opt"},
                                        meta_extra={"env_steps": version})
    resumed = make_test_config("turns", agents={"a": {}, "b": {}}, checkpoint={"keep_last": 5},
                               training={"resume_from": str(tmp_path / "old")})
    launcher = Launcher(resumed, make_test_run_dir(resumed, tmp_path, name="new"))
    coord = make_coordinator(resumed, launcher._run_dir.checkpoints)
    launcher._import_snapshot_pool(coord)
    assert ids(coord.checkpoint_manager, "a") == ids(coord.checkpoint_manager, "b") == ["ckpt_v20", "ckpt_v40"]
    from_ckpt = make_test_config("turns", agents={"a": {}, "b": {}},
                                 training={"resume_from": str(tmp_path / "old" / "checkpoints" / "a" / "ckpt_v40")})
    launcher2 = Launcher(from_ckpt, make_test_run_dir(from_ckpt, tmp_path, name="new2"))
    coord2 = make_coordinator(from_ckpt, launcher2._run_dir.checkpoints)
    launcher2._import_snapshot_pool(coord2)
    assert ids(coord2.checkpoint_manager, "a") == []


def test_eviction_callback_errors_after_the_first_are_logged(tmp_path, caplog):
    def on_evict(agent, ckpt):
        raise RuntimeError(f"callback failed for {ckpt}")

    old = CheckpointManager(tmp_path / "old", keep_last=10)
    for version in (10, 20, 30):
        old.save("a", version, sd(version))
    mgr = CheckpointManager(tmp_path / "store", keep_last=1, keep_every=0, on_evict=on_evict)
    with caplog.at_level(logging.ERROR, logger="colosseum.coordinator.checkpoint_manager"):
        with pytest.raises(RuntimeError, match="ckpt_v10"):
            mgr.import_snapshots(tmp_path / "old", "a")
    logged = [r for r in caplog.records if r.exc_info is not None]
    assert [str(r.exc_info[1]) for r in logged] == ["callback failed for ckpt_v20"]


def test_a_raising_eviction_callback_never_leaves_deleted_snapshots_in_the_index(tmp_path):
    seen = []

    def on_evict(agent, ckpt):
        seen.append(ckpt)
        raise RuntimeError(f"callback failed for {ckpt}")

    mgr = CheckpointManager(tmp_path / "store", keep_last=1, keep_every=0, on_evict=on_evict)
    mgr.save("a", 10, sd(10))
    with pytest.raises(RuntimeError, match="ckpt_v10"):
        mgr.save("a", 20, sd(20))
    assert ids(mgr) == ["ckpt_v20"] and seen == ["ckpt_v10"]
    assert sorted(p.name for p in (tmp_path / "store" / "a").iterdir()) == ["ckpt_v20"]
    old = CheckpointManager(tmp_path / "old", keep_last=10)
    for version in (30, 40, 50):
        old.save("a", version, sd(version))
    with pytest.raises(RuntimeError, match="ckpt_v20"):             # every callback runs, the first error is raised
        mgr.import_snapshots(tmp_path / "old", "a")
    assert ids(mgr) == ["ckpt_v50"] and seen == ["ckpt_v10", "ckpt_v20", "ckpt_v30", "ckpt_v40"]
    assert all(info.path.is_dir() for info in mgr.list_checkpoints("a"))


def test_a_resume_from_a_snapshot_without_trainer_state_warns_that_the_optimizer_starts_fresh(tmp_path, caplog):
    cfg = make_test_config("solo")
    old = CheckpointManager(tmp_path / "old" / "checkpoints")
    old.save("agent_0", 10, model_state(cfg), meta_extra={"env_steps": 100, **_signed(cfg)})   # no trainer_state.pt

    def resume(source, name, **checkpoint):
        resumed = make_test_config("solo", training={"resume_from": str(source)}, checkpoint=checkpoint)
        launcher = Launcher(resumed, make_test_run_dir(resumed, tmp_path, name=name))
        setup = setup_run(resumed, validate=False)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="colosseum.launcher"):
            launcher._resolve_resume(setup.agent_configs, setup.role_specs)
        return [r.getMessage() for r in caplog.records if "trainer_state.pt" in r.getMessage()]

    for source in (tmp_path / "old", tmp_path / "old" / "checkpoints" / "agent_0" / "ckpt_v10"):
        (msg,) = resume(source, f"r-{source.name}")
        assert "ckpt_v10" in msg and "optimizer" in msg and "start fresh" in msg
    assert resume(tmp_path / "old", "no-opt", save_optimizer=False) == []
    old.save("agent_0", 20, model_state(cfg), trainer_state=b"opt", meta_extra={"env_steps": 200, **_signed(cfg)})
    assert resume(tmp_path / "old", "with-state") == []


def _signed(config) -> dict:
    roles, role = agent_role_of(config, "agent_0")
    return {"roles": roles, "role_signature": role_signature(role)}
