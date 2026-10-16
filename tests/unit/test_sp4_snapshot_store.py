"""Snapshot storage with SP4 retention (T4.1, spec block 4): run_id / pinned in meta.json, the current run's
final, pins written on disk while a run trains, top-k through the scores provider, reapply after a refit,
the cap, config checks, the validate printout, the coordinator wiring, the distributed FIFO cap."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pytest
import yaml

from colosseum.arena.retention import SnapshotScore
from colosseum.coordinator.checkpoint_manager import CheckpointManager, read_snapshots, set_pinned
from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import validate_config
from colosseum.distributed import check_distributed_scope, distributed_checkpoint_manager
from game_helpers import make_coordinator, make_test_config


def sd(value: float) -> dict[str, np.ndarray]:
    return {"w": np.full((2,), value, np.float32)}


def ids(manager, agent_id="a") -> list[str]:
    return [c.checkpoint_id for c in manager.list_checkpoints(agent_id)]


def meta(manager, checkpoint_id, agent_id="a") -> dict:
    return json.loads((manager.base_dir / agent_id / checkpoint_id / "meta.json").read_text())


def test_new_snapshots_carry_run_id_and_pinned_and_imported_finals_are_ordinary(tmp_path):
    old = CheckpointManager(tmp_path / "old", keep_last=5, run_id="r1")
    old.save("a", 7, sd(7), meta_extra={"final": True})
    old.save("a", 10, sd(10))
    assert meta(old, "ckpt_v7")["run_id"] == "r1" and meta(old, "ckpt_v7")["pinned"] is False
    new = CheckpointManager(tmp_path / "new", keep_last=1, run_id="r2")
    assert new.import_snapshots(tmp_path / "old", "a") == ["ckpt_v10"]       # r1's final is not protected in r2
    new.save("a", 20, sd(20), meta_extra={"final": True})
    new.save("a", 30, sd(30))
    assert ids(new) == ["ckpt_v20", "ckpt_v30"]                              # r2's own final is
    assert meta(new, "ckpt_v20")["run_id"] == "r2"
    legacy = CheckpointManager(tmp_path / "legacy", keep_last=1)             # no run id: the SP3 rule
    legacy.save("a", 7, sd(7), meta_extra={"final": True})
    legacy.save("a", 8, sd(8))
    assert ids(legacy) == ["ckpt_v7", "ckpt_v8"] and "run_id" not in meta(legacy, "ckpt_v7")


def test_a_pin_written_on_disk_by_another_process_is_honoured(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last=1, run_id="r")
    mgr.save("a", 10, sd(10))
    set_pinned(tmp_path / "a" / "ckpt_v10", True)                            # what `colosseum pin` does
    mgr.save("a", 20, sd(20))
    mgr.save("a", 30, sd(30))
    assert ids(mgr) == ["ckpt_v10", "ckpt_v30"] and mgr.pinned("a") == ["ckpt_v10"]
    mgr.pin("a", "ckpt_v10", pinned=False)
    assert meta(mgr, "ckpt_v10")["pinned"] is False
    mgr.save("a", 40, sd(40))
    assert ids(mgr) == ["ckpt_v40"]
    with pytest.raises(ConfigError, match="ckpt_v99"):
        mgr.pin("a", "ckpt_v99")


def test_set_pinned_replaces_the_file_so_a_hard_linked_source_is_untouched(tmp_path):
    old = CheckpointManager(tmp_path / "old", keep_last=5, run_id="r1")
    old.save("a", 10, sd(10))
    new = CheckpointManager(tmp_path / "new", keep_last=5, run_id="r2")
    new.import_snapshots(tmp_path / "old", "a")
    new.pin("a", "ckpt_v10")
    assert meta(new, "ckpt_v10")["pinned"] is True and meta(old, "ckpt_v10")["pinned"] is False
    again = CheckpointManager(tmp_path / "third", keep_last=1, run_id="r3")
    again.import_snapshots(tmp_path / "new", "a")                            # the pin survives a resume
    again.save("a", 20, sd(20))
    again.save("a", 30, sd(30))
    assert ids(again) == ["ckpt_v10", "ckpt_v30"]


def test_top_k_uses_the_scores_provider_and_reapply_evicts_after_a_refit(tmp_path):
    scores = {"ckpt_v10": SnapshotScore(1.0, 500), "ckpt_v20": SnapshotScore(5.0, 500),
              "ckpt_v30": SnapshotScore(3.0, 500)}
    evicted = []
    mgr = CheckpointManager(tmp_path, keep_last=1, keep_top_k=1, top_k_min_games=100, run_id="r",
                            scores_fn=lambda agent: dict(scores),
                            on_evict=lambda a, c: evicted.append((a, c, ids(mgr))))
    for version in (10, 20, 30, 40):
        mgr.save("a", version, sd(version))
    assert ids(mgr) == ["ckpt_v20", "ckpt_v40"]
    scores.update({"ckpt_v20": SnapshotScore(0.0, 500), "ckpt_v40": SnapshotScore(9.0, 500)})
    evicted.clear()
    assert mgr.reapply_retention("a") == ["ckpt_v20"]
    assert ids(mgr) == ["ckpt_v40"]
    assert evicted == [("a", "ckpt_v20", ["ckpt_v40"])]                      # the index is updated first (SP3 #24)
    assert mgr.reapply_retention("a") == [] and mgr.reapply_retention("nobody") == []


def test_max_snapshots_caps_keep_every_snapshots_oldest_first_and_warns_once(tmp_path, caplog):
    mgr = CheckpointManager(tmp_path, keep_last=2, keep_every=1, interval=10, max_snapshots=3, run_id="r")
    for version in range(10, 61, 10):
        mgr.save("a", version, sd(version))
    assert ids(mgr) == ["ckpt_v40", "ckpt_v50", "ckpt_v60"]
    pinned = CheckpointManager(tmp_path / "p", keep_last=2, max_snapshots=2, run_id="r")
    with caplog.at_level(logging.WARNING, logger="colosseum.coordinator.checkpoint_manager"):
        for version in (10, 20, 30, 40):
            pinned.save("a", version, sd(version))
            pinned.pin("a", f"ckpt_v{version}")
    assert ids(pinned) == ["ckpt_v10", "ckpt_v20", "ckpt_v30", "ckpt_v40"]
    assert len([r for r in caplog.records if "exceed checkpoint.max_snapshots" in r.getMessage()]) == 1


def test_read_snapshots_lists_without_cleaning(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last=5)
    mgr.save("a", 10, sd(10))
    leftover = tmp_path / "a" / ".tmp-ckpt_v20-0123abcd"
    leftover.mkdir()
    assert [c.checkpoint_id for c in read_snapshots(tmp_path / "a")] == ["ckpt_v10"]
    assert leftover.is_dir() and read_snapshots(tmp_path / "missing") == []


def _write(tmp_path, **sections) -> str:
    data = {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}, **sections}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(path)


def test_checkpoint_config_defaults_and_checks(tmp_path):
    cfg = load_config(_write(tmp_path))
    c = cfg.checkpoint
    assert (c.keep_top_k, c.top_k_quantile, c.top_k_min_games, c.max_snapshots) == (0, 0.1, 200, 100)
    assert load_config(_write(tmp_path, checkpoint={"max_snapshots": None})).checkpoint.max_snapshots is None
    with pytest.raises(ConfigError, match="max_snapshots"):
        load_config(_write(tmp_path, checkpoint={"keep_last": 20, "max_snapshots": 5}))
    with pytest.raises(ConfigError, match="match_log"):
        load_config(_write(tmp_path, checkpoint={"keep_top_k": 3}, match_log={"enabled": False}))
    for bad in ({"top_k_quantile": 0.5}, {"top_k_quantile": 0.0}, {"keep_top_k": -1}, {"top_k_min_games": -1},
                {"max_snapshots": 0}):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, checkpoint=bad))
    assert load_config(_write(tmp_path, checkpoint={"keep_top_k": 3})).checkpoint.keep_top_k == 3   # log on
    default = ColosseumConfig.model_validate(make_test_config("turns").model_dump(by_alias=True))
    assert default.checkpoint.max_snapshots == 100


def test_validate_prints_the_retention_rule_and_warns_about_pins_over_the_cap(tmp_path, caplog):
    cfg = make_test_config("turns", agents={"a": {}, "b": {}}, checkpoint={"keep_last": 2, "max_snapshots": 3})
    lines = validate_config(cfg).lines
    assert sum(1 for line in lines if line.startswith("agent 'a': snapshots") or
               line.startswith("agent 'b': snapshots")) == 2
    old = tmp_path / "old"
    mgr = CheckpointManager(old / "checkpoints", keep_last=5, run_id="r1")
    for version in (10, 20):
        mgr.save("a", version, sd(version))
        mgr.pin("a", f"ckpt_v{version}")
    resumed = make_test_config("turns", agents={"a": {}, "b": {}}, checkpoint={"keep_last": 2, "max_snapshots": 3},
                               training={"resume_from": str(old)})
    with caplog.at_level(logging.WARNING, logger="colosseum.core.validation"):
        warnings = [line for line in validate_config(resumed).lines if line.startswith("WARNING")]
    assert len(warnings) == 1 and "agent 'a'" in warnings[0] and "2 pinned" in warnings[0]
    assert any("2 pinned" in r.getMessage() for r in caplog.records)


@dataclass
class _Fit:
    rating: dict
    games: dict


@dataclass
class _Unc:
    lcb: dict
    contrast_samples: dict = field(default_factory=dict)


@dataclass
class _Snapshot:       # the RatingsSnapshot fields snapshot_scores reads
    version: int
    fits: dict
    uncertainty: dict
    scores: dict = field(default_factory=dict)
    env_steps: int = 0


def test_the_coordinator_feeds_top_k_from_the_live_ratings_and_tells_listeners(tmp_path):
    cfg = make_test_config("turns", checkpoint={"interval": 10, "keep_last": 1, "keep_top_k": 1,
                                                "top_k_min_games": 0, "keep_every": 0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    seen = []
    coord.add_evict_listener(lambda agent, ckpt: seen.append((agent, ckpt)))
    snap = _Snapshot(1, {"2p": _Fit({"agent_0@v10": 5.0, "agent_0@v20": 1.0}, {"agent_0@v10": 9, "agent_0@v20": 9})},
                     {"2p": _Unc({"agent_0@v10": 4.0, "agent_0@v20": 0.0})})
    coord.set_ratings_source(lambda: snap)
    assert coord.snapshot_scores("agent_0") == {"ckpt_v10": SnapshotScore(4.0, 9), "ckpt_v20": SnapshotScore(0.0, 9)}
    for version in (10, 20, 30):
        coord.save_checkpoint_payload({"agent_id": "agent_0", "policy_version": version, "model_state": sd(version),
                                       "trainer_state_bytes": b"opt"})
    assert ids(coord.checkpoint_manager, "agent_0") == ["ckpt_v10", "ckpt_v30"]
    assert seen == [("agent_0", "ckpt_v20")]
    coord.set_ratings_source(lambda: None)
    assert coord.snapshot_scores("agent_0") == {}


def test_the_distributed_learner_caps_its_pool_fifo_and_refuses_top_k(tmp_path):
    config = make_test_config("turns", checkpoint={"interval": 10, "keep_last": 1, "keep_every": 1,
                                                   "max_snapshots": 3})
    manager = distributed_checkpoint_manager(config, tmp_path / "checkpoints", run_id="r")
    for version in range(10, 71, 10):
        manager.save("agent_0", version, sd(version))
    assert ids(manager, "agent_0") == ["ckpt_v50", "ckpt_v60", "ckpt_v70"]
    with pytest.raises(ConfigError, match="keep_top_k"):
        check_distributed_scope(make_test_config("turns", checkpoint={"keep_top_k": 2}))


def test_imported_sp3_snapshots_are_ordinary_not_final(tmp_path):
    """T0.1 baseline: SP3 snapshots carry no run_id, so their final one is not protected in an SP4 run."""
    from game_helpers import copy_sp3_run, sp3_manifest

    manifest = sp3_manifest()
    agent, final = manifest["agent"], manifest["final_version"]
    source = copy_sp3_run(tmp_path)
    assert all("run_id" not in c.meta for c in read_snapshots(source / "checkpoints" / agent))
    mgr = CheckpointManager(tmp_path / "new", keep_last=1, run_id="r")
    assert mgr.import_snapshots(source / "checkpoints", agent) == [f"ckpt_v{final}"]   # keep_last 1: the newest
    mgr.save(agent, final + 20, sd(1))
    assert ids(mgr, agent) == [f"ckpt_v{final + 20}"]
    legacy = CheckpointManager(tmp_path / "legacy", keep_last=1)      # no run id: the SP3 rule keeps it
    legacy.import_snapshots(source / "checkpoints", agent)
    legacy.save(agent, final + 20, sd(1))
    assert ids(legacy, agent) == [f"ckpt_v{final}", f"ckpt_v{final + 20}"]


def test_the_launcher_reapplies_retention_once_per_new_live_refit(tmp_path):
    from colosseum.launcher import Launcher

    calls = []

    class _Manager:
        def reapply_retention(self, agent_id):
            calls.append(agent_id)
            return []

    class _Coord:
        checkpoint_manager = _Manager()

        def set_ratings_source(self, fn):
            calls.append(("source", fn()))

    class _Live:
        snapshot = None

        def latest(self):
            return self.snapshot

    launcher = object.__new__(Launcher)          # only the attributes the two methods read
    launcher._config = make_test_config("turns", checkpoint={"keep_top_k": 2})
    launcher._coordinator, launcher._live_ratings, launcher._agent_ids = _Coord(), _Live(), ["a", "b"]
    launcher._retention_version = None
    launcher._wire_retention()
    assert calls == [("source", None)]
    calls.clear()
    launcher._reapply_retention()                # no refit yet
    launcher._live_ratings.snapshot = _Snapshot(1, {}, {})
    launcher._reapply_retention()
    launcher._reapply_retention()                # the same refit: nothing new
    launcher._live_ratings.snapshot = _Snapshot(2, {}, {})
    launcher._reapply_retention()
    assert calls == ["a", "b", "a", "b"]
    calls.clear()
    launcher._config = make_test_config("turns")  # keep_top_k 0: scores never decide, no rerun
    launcher._live_ratings.snapshot = _Snapshot(3, {}, {})
    launcher._reapply_retention()
    assert calls == []
