"""``colosseum pin`` (SP4 T4.2, spec blocks 3 and 4): pin / unpin / list, errors with hints, and a pin that a
running training honours."""
from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from cli_runner import TTT_CONFIG, training_process, wait_for
from colosseum.cli import main
from colosseum.coordinator.checkpoint_manager import CheckpointManager, parse_snapshot_ref, pin_snapshot
from colosseum.core.errors import ConfigError


def invoke(*args):
    return CliRunner().invoke(main, ["pin", *map(str, args)])


def make_run(tmp_path):
    root = tmp_path / "run"
    mgr = CheckpointManager(root / "checkpoints", keep_last=10, run_id="r")
    for aid in ("main", "rival"):
        for version in (10, 20, 30):
            mgr.save(aid, version, {"w": np.zeros(2, np.float32)})
    return root


def pinned_flag(root, aid, version) -> bool:
    return json.loads((root / "checkpoints" / aid / f"ckpt_v{version}" / "meta.json").read_text())["pinned"]


def test_pin_unpin_and_list(tmp_path):
    root = make_run(tmp_path)
    result = invoke(root, "main@v20")
    assert result.exit_code == 0, result.output
    assert "Pinned main@v20" in result.output and pinned_flag(root, "main", 20) is True
    assert invoke(root, "main@v20").exit_code == 0                      # idempotent
    listing = invoke(root, "--list")
    assert listing.exit_code == 0
    assert "main: ckpt_v10, ckpt_v20 (pinned), ckpt_v30" in listing.output and "rival: ckpt_v10" in listing.output
    result = invoke(root, "main@v20", "--remove")
    assert result.exit_code == 0 and "Unpinned main@v20" in result.output and pinned_flag(root, "main", 20) is False


def test_pin_of_a_hard_linked_snapshot_leaves_the_other_run_unchanged(tmp_path):
    root = make_run(tmp_path)
    other = tmp_path / "other" / "checkpoints" / "main" / "ckpt_v20"
    other.mkdir(parents=True)
    for f in (root / "checkpoints" / "main" / "ckpt_v20").iterdir():
        (other / f.name).hardlink_to(f)
    assert invoke(root, "main@v20").exit_code == 0
    assert pinned_flag(root, "main", 20) is True
    assert json.loads((other / "meta.json").read_text())["pinned"] is False


def test_errors_name_the_problem_and_a_hint(tmp_path):
    root = make_run(tmp_path)
    result = invoke(root, "main@v25")
    assert result.exit_code == 1 and "ckpt_v25" in result.output and "ckpt_v30" in result.output
    assert "--list" in result.output
    result = invoke(root, "nobody@v10")
    assert result.exit_code == 1 and "main" in result.output and "rival" in result.output
    result = invoke(tmp_path / "missing", "main@v10")
    assert result.exit_code != 0
    result = invoke(tmp_path, "main@v10")
    assert result.exit_code == 1 and "not a run dir" in result.output
    assert invoke(root, "main-v10").exit_code == 2
    assert invoke(root).exit_code == 2                                   # neither a snapshot nor --list
    assert invoke(root, "main@v10", "--list").exit_code == 2
    assert invoke(root, "--list", "--remove").exit_code == 2
    assert parse_snapshot_ref("b_1@v7") == ("b_1", 7)
    with pytest.raises(ConfigError, match="agent@v"):
        parse_snapshot_ref("main@7")
    with pytest.raises(ConfigError):
        pin_snapshot(root, "main", 99)


def _evict_like_the_trainer(ckpt_dir):
    ckpt_dir.rename(ckpt_dir.parent / f".tmp-evict-{ckpt_dir.name}-test")    # CheckpointManager._evict's first step


@pytest.mark.parametrize("when", ["before the write", "after the write"])
def test_a_snapshot_evicted_while_it_is_pinned_is_a_config_error_with_a_hint(tmp_path, monkeypatch, when):
    """The race with a training run's eviction (final review A, M8): no traceback, no "Pinned" for a snapshot that is
    gone, and a pointer to the tournaments' linked copies."""
    import colosseum.coordinator.checkpoint_manager as checkpoint_manager

    root = make_run(tmp_path)
    original = checkpoint_manager.write_text_atomic

    def racing_write(path, text):
        if when == "before the write":
            _evict_like_the_trainer(path.parent)
        original(path, text)
        if when == "after the write":
            _evict_like_the_trainer(path.parent)

    monkeypatch.setattr(checkpoint_manager, "write_text_atomic", racing_write)
    result = invoke(root, "main@v20")
    assert isinstance(result.exception, SystemExit)                     # a ConfigError exit, not a raw OSError
    assert result.exit_code == 1 and "Traceback" not in result.output and "Pinned" not in result.output
    assert "main@v20 was evicted" in result.output and "tournaments" in result.output and "--list" in result.output


def test_an_unpin_followed_by_an_eviction_is_no_error(tmp_path, monkeypatch):
    """Unpinning makes the snapshot evictable: an eviction right after the write is the expected outcome."""
    import colosseum.coordinator.checkpoint_manager as checkpoint_manager

    root = make_run(tmp_path)
    assert invoke(root, "main@v20").exit_code == 0
    original = checkpoint_manager.write_text_atomic

    def write_then_evict(path, text):
        original(path, text)
        _evict_like_the_trainer(path.parent)

    monkeypatch.setattr(checkpoint_manager, "write_text_atomic", write_then_evict)
    result = invoke(root, "main@v20", "--remove")
    assert result.exit_code == 0 and "Unpinned main@v20" in result.output


def test_a_running_training_keeps_a_snapshot_pinned_from_outside(tmp_path):
    overrides = {"training.total_timesteps": "40000", "checkpoint.interval": "5", "checkpoint.keep_last": "2",
                 "checkpoint.keep_every": "0"}
    with training_process(TTT_CONFIG, tmp_path, name="live", overrides=overrides) as (proc, root):
        agent_dir = root / "checkpoints" / "agent_0"
        assert wait_for(lambda: len(list(agent_dir.glob("ckpt_v*"))) >= 1 or proc.poll() is not None, 120)
        first = min(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))
        result = invoke(root, f"agent_0@v{first}")
        if result.exit_code != 0:          # evicted between the glob and the pin: take the newest
            first = max(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))
            result = invoke(root, f"agent_0@v{first}")
        assert result.exit_code == 0, result.output
        assert wait_for(lambda: max(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))
                        >= first + 20 or proc.poll() is not None, 120)
        assert (agent_dir / f"ckpt_v{first}").is_dir() and pinned_flag(root, "agent_0", first) is True
        assert not (agent_dir / f"ckpt_v{first + 5}").exists()     # retention did run past the pin (keep_last 2)
