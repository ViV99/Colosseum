"""In-place resume preparation (SP5 T4.3, spec 5.4): refusal while the run's process lives, the saved config or a
new one with a diff WARNING, ``--set`` in both cases, the run dir pinned, ``training.resume_from`` not applied
again, the start env-step count of an SP4 run dir, the at-budget WARNING, the numbered backup of the previous
resolved config."""
from __future__ import annotations

import json
import logging
import os

import pytest
import yaml

from colosseum.core.errors import ConfigError
from colosseum.core.run_state import write_run_state
from colosseum.hub.resume import (
    check_not_running,
    commit_resume,
    config_diff,
    flatten,
    next_backup_path,
    prepare_resume,
    resume_port,
    start_env_steps,
)
from colosseum.hub.state import HUB_STATE_FILE, HubStateData, save_hub_state
from game_helpers import copy_sp4_run


def _saved(root) -> dict:
    return yaml.safe_load((root / "config.resolved.yaml").read_text())


def test_refused_while_the_run_process_lives_and_allowed_after(tmp_path):
    root = copy_sp4_run(tmp_path)
    write_run_state(root, "running", os.getpid())                  # this test process: alive
    with pytest.raises(ConfigError, match=f"is running \\(pid {os.getpid()}\\)"):
        check_not_running(root)
    with pytest.raises(ConfigError, match="two hubs cannot share one run dir"):
        prepare_resume(root, None)
    write_run_state(root, "running", 2**22 + 12345)                # a dead pid: a killed run
    check_not_running(root)
    write_run_state(root, "finished", os.getpid(), exit_code=0)
    check_not_running(root)


def test_without_c_the_saved_config_with_set_and_the_dir_pinned(tmp_path, caplog):
    root = copy_sp4_run(tmp_path)
    moved = tmp_path / "elsewhere"
    root.rename(moved)                                              # a run dir copied to another machine / path
    with caplog.at_level(logging.WARNING):
        plan = prepare_resume(moved, None, {"training.total_timesteps": 10**7})
    assert plan.config.run.name == "elsewhere" and plan.config.run.dir == str(tmp_path)
    assert plan.config.training.total_timesteps == 10**7
    assert plan.diff == [] and not plan.from_file and not plan.sp5
    assert plan.run_id == json.loads((moved / "run.json").read_text())["run_id"]
    assert "differs" not in caplog.text and "names run dir" not in caplog.text


def test_with_c_the_new_config_and_a_diff_warning(tmp_path, caplog):
    root = copy_sp4_run(tmp_path)
    data = _saved(root)
    old_chunks = data["learner"]["batch_chunks"]
    data["learner"]["batch_chunks"] = old_chunks + 5
    data["run"]["name"] = "other"
    path = tmp_path / "new.yaml"
    path.write_text(yaml.safe_dump(data))
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.resume"):
        plan = prepare_resume(root, path, {"training.total_timesteps": 10**7})
    assert plan.from_file and plan.config.learner.batch_chunks == old_chunks + 5
    assert plan.config.run.name == root.name
    assert plan.diff == [f"learner.batch_chunks: {old_chunks!r} -> {old_chunks + 5!r}",
                         "training.total_timesteps: 6000 -> 10000000"]          # --set is listed too (round 1)
    assert "differs from the run's saved config in 2 settings" in caplog.text
    assert "the resumed run stays in" in caplog.text


def test_with_c_set_applies_to_the_new_config_only(tmp_path, caplog):
    """Review round 1: with -c, --set never reaches the saved config: an override of an agent only the new file has
    is valid, and the diff lists the --set changes too."""
    root = copy_sp4_run(tmp_path)
    data = _saved(root)
    data["agents"]["greedy2"] = {"kind": "scripted", "class": "colosseum.players.RandomBot", "kwargs": {}}
    path = tmp_path / "new.yaml"
    path.write_text(yaml.safe_dump(data))
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.resume"):
        plan = prepare_resume(root, path, {"training.total_timesteps": 10**7, "agents.greedy2.kwargs": {"x": 1}})
    assert plan.config.agents["greedy2"].kwargs == {"x": 1}
    assert "training.total_timesteps: 6000 -> 10000000" in plan.diff
    assert any(line.startswith("agents.greedy2.") for line in plan.diff)
    assert "differs from the run's saved config" in caplog.text


def test_with_c_an_agent_created_by_set_shows_in_the_diff(tmp_path):
    root = copy_sp4_run(tmp_path)
    data = _saved(root)
    data["agents"]["b"] = {"kind": "trainable"}
    path = tmp_path / "new.yaml"
    path.write_text(yaml.safe_dump(data))
    plan = prepare_resume(root, path, {"training.total_timesteps": 10**7, "agents.b.learner.device": "cpu"})
    assert plan.diff and any(line.startswith("agents.b.") for line in plan.diff)
    assert "training.total_timesteps: 6000 -> 10000000" in plan.diff


def test_resume_from_is_not_applied_again(tmp_path, caplog):
    root = copy_sp4_run(tmp_path)
    data = _saved(root)
    data["training"]["resume_from"] = str(tmp_path / "parent")
    (root / "config.resolved.yaml").write_text(yaml.safe_dump(data))
    with caplog.at_level(logging.INFO, logger="colosseum.hub.resume"):
        prepare_resume(root, None, {"training.total_timesteps": 10**7})
    assert "was imported when the run started" in caplog.text
    data["training"]["resume_from"] = str(tmp_path / "another")
    path = tmp_path / "new.yaml"
    path.write_text(yaml.safe_dump(data))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.resume"):
        prepare_resume(root, path, {"training.total_timesteps": 10**7})
    assert "is ignored: an in-place resume continues the run's own state" in caplog.text


def test_sp4_start_env_steps_and_the_budget_warning(tmp_path, caplog):
    root = copy_sp4_run(tmp_path)
    steps = start_env_steps(root)
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert steps == max(r["env_steps"] for r in records if "env_steps" in r) > 0
    with open(root / "metrics.jsonl", "a") as fh:
        fh.write('{"kind": "system", "env_st')                        # a torn last line of a killed run
    assert start_env_steps(root) == steps
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.resume"):
        plan = prepare_resume(root, None)
    assert plan.start_env_steps == steps
    assert "at or past training.total_timesteps" in caplog.text


def test_the_saved_transport_section_is_dropped_before_set(tmp_path, caplog):
    """A29: an SP4 config.resolved.yaml still has ``transport``; an in-place resume drops it before ``--set`` (no
    translation WARNING, no conflict with ``--set distributed.max_message_mb``)."""
    root = copy_sp4_run(tmp_path)
    assert "transport" in _saved(root)
    with caplog.at_level(logging.WARNING):
        plan = prepare_resume(root, None, {"training.total_timesteps": 10**7, "distributed.max_message_mb": 32})
    assert plan.config.distributed.max_message_mb == 32
    assert "transport" not in caplog.text
    commit_resume(plan)
    assert "transport" not in _saved(root)


def test_an_sp3_run_dir_has_no_run_id_yet(tmp_path):
    root = copy_sp4_run(tmp_path)
    (root / "run.json").unlink()
    assert prepare_resume(root, None, {"training.total_timesteps": 10**7}).run_id is None


def test_hub_state_counts_win_and_a_malformed_one_is_refused(tmp_path):
    root = copy_sp4_run(tmp_path)
    run_id = json.loads((root / "run.json").read_text())["run_id"]
    save_hub_state(root / HUB_STATE_FILE, HubStateData(version=1, run_id=run_id, env_steps=123, sessions={},
                                                       epoch_reserved_until=0, versions={}, counters={},
                                                       nodes_lost=0), fsync=False)
    plan = prepare_resume(root, None, {"training.total_timesteps": 10**7})
    assert plan.sp5 and plan.start_env_steps == 123
    (root / "hub_state.json").write_text("{")
    with pytest.raises(ConfigError, match="hub_state.json"):
        prepare_resume(root, None)


def test_a_hub_state_of_another_run_is_refused_before_anything_is_written(tmp_path):
    """Review round 1 (T4.3(b)): the run-id check runs in prepare_resume, before commit_resume writes anything."""
    root = copy_sp4_run(tmp_path)
    save_hub_state(root / HUB_STATE_FILE, HubStateData(version=1, run_id="other", env_steps=5, sessions={},
                                                       epoch_reserved_until=0, versions={}, counters={},
                                                       nodes_lost=0), fsync=False)
    before = (root / "config.resolved.yaml").read_text()
    with pytest.raises(ConfigError, match="belongs to run other"):
        prepare_resume(root, None, {"training.total_timesteps": 10**7})
    assert (root / "config.resolved.yaml").read_text() == before and not (root / "config.resolved.1.yaml").exists()


def test_not_a_run_dir(tmp_path):
    with pytest.raises(ConfigError, match="not a run dir"):
        prepare_resume(tmp_path / "missing", None)
    (tmp_path / "x" / "checkpoints").mkdir(parents=True)
    with pytest.raises(ConfigError, match="has no config.resolved.yaml"):
        prepare_resume(tmp_path / "x", None)


def test_commit_keeps_numbered_backups(tmp_path):
    root = copy_sp4_run(tmp_path)
    original = (root / "config.resolved.yaml").read_text()
    plan = prepare_resume(root, None, {"training.total_timesteps": 10**7})
    assert next_backup_path(root).name == "config.resolved.1.yaml"
    assert commit_resume(plan) == root / "config.resolved.1.yaml"
    assert (root / "config.resolved.1.yaml").read_text() == original
    assert _saved(root)["training"]["total_timesteps"] == 10**7
    plan = prepare_resume(root, None, {"training.total_timesteps": 2 * 10**7})
    assert commit_resume(plan).name == "config.resolved.2.yaml"


def test_flatten_and_diff_helpers(tmp_path):
    assert flatten({"a": {"b": 1, "c": {"d": [1]}}, "e": {}}) == {"a.b": 1, "a.c.d": [1], "e": {}}
    root = copy_sp4_run(tmp_path)
    a = prepare_resume(root, None, {"training.total_timesteps": 10**7}).config
    b = a.model_copy(update={"run": a.run.model_copy(update={"name": "zzz"})})
    assert config_diff(a, b) == []                                   # run.dir / run.name are pinned
    kwargs = {**a.networks.kwargs, "extra": 7}
    c = a.model_copy(update={"networks": a.networks.model_copy(update={"kwargs": kwargs})})
    assert config_diff(a, c) == ["networks.kwargs.extra: added (7)"]
    d = a.model_copy(update={"networks": a.networks.model_copy(update={"kwargs": {}})})
    assert config_diff(a, d) == [f"networks.kwargs.{k}: removed (was {v!r})"
                                 for k, v in sorted(a.networks.kwargs.items())]


def test_a_resumed_hub_listens_on_its_previous_port():
    assert resume_port(["10.0.0.5:47123", "192.168.1.2:47123"], None, 47100) == 47123
    assert resume_port(["10.0.0.5:47123"], 5000, 47100) == 5000                # --port wins
    assert resume_port([], None, 47100) == 47100                               # no hub.json: the config's
    assert resume_port(["a:1", "b:2"], None, 47100) == 47100                   # ambiguous: the config's


def test_strict_checks_accept_the_saved_config_and_refuse_another_architecture(tmp_path):
    from colosseum.hub.hub import setup_run
    from colosseum.hub.resume import check_resumed_agents

    root = copy_sp4_run(tmp_path)
    plan = prepare_resume(root, None, {"training.total_timesteps": 10**7})
    setup = setup_run(plan.config, validate=False)
    checked = check_resumed_agents(root, setup.agent_configs, setup.role_specs)
    assert set(checked) == set(setup.agent_configs) and all("ckpt_v" in s for s in checked.values())
    data = _saved(root)
    data["networks"]["kwargs"] = {**(data["networks"].get("kwargs") or {}), "hidden": 7}
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(data))
    plan = prepare_resume(root, path, {"training.total_timesteps": 10**7})
    setup = setup_run(plan.config, validate=False)
    with pytest.raises(ConfigError, match="cannot continue with this config"):
        check_resumed_agents(root, setup.agent_configs, setup.role_specs)
