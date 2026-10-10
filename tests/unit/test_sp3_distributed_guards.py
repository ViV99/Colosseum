"""Distributed roles keep the SP2 scope (spec block 8, T6.1): SP3 settings are ConfigErrors naming
SP5, any opponent mix is reduced to latest only with one warning, decisions by values, retention."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from colosseum.core.config import load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model
from colosseum.distributed import check_distributed_scope, distributed_checkpoint_manager, distributed_setup
from game_helpers import agent_role_of, make_test_config, write_test_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SP2_TTT = REPO_ROOT / "tests" / "fixtures" / "sp2" / "configs" / "tic_tac_toe.yaml"
RANDOM_BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}
LATEST_ONLY = {"latest": 1.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.0}


class _ExitedProcess:
    """mp.Process stand-in (never reached when the scope check refuses the config)."""

    exitcode = 0

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


def _reductions(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == "colosseum.distributed" and r.levelno == logging.WARNING
            and "reduced to latest only" in r.getMessage()]


def _teacher_pt(tmp_path) -> Path:
    cfg = make_test_config("turns")
    _roles, role = agent_role_of(cfg, "agent_0")
    path = tmp_path / "teacher.pt"
    torch.save(build_model(cfg.get_agent_config("agent_0"), role).state_dict(), path)
    return path


@pytest.mark.parametrize(("sections", "needle"), [
    ({"agents": {"bot": RANDOM_BOT}}, "bot"),
    ({"agents": {"old": {"kind": "frozen", "path": "missing.pt"}}}, "old"),
    ({"init": {"from": "bc.pt"}}, "init.from"),
    ({"init": {"critic_warmup_steps": 5}}, "critic_warmup_steps"),
    ({"matchmaking": {"matchmaker_class": "my_game.league.Mine"}}, "matchmaker_class"),
    ({"agents": {"agent_0": {"matchmaking": {"teammates": "mixed"}}}}, "agents.agent_0.matchmaking"),
    ({"agents": {"agent_0": {"kickstart": {"lambda": 0.5}}}}, "agents.agent_0.kickstart"),
    ({"kickstart": {"teacher": "runs/x/checkpoints/agent_0/ckpt_v10"}}, "kickstart.teacher"),
], ids=["scripted", "frozen", "init-from", "critic-warmup", "matchmaker-class", "agent-matchmaking",
        "agent-kickstart", "teacher-dir"])
def test_sp3_settings_are_refused_with_a_pointer_to_sp5(sections, needle):
    config = make_test_config("turns", **sections)
    with pytest.raises(ConfigError, match="SP5") as info:
        check_distributed_scope(config)
    assert needle in str(info.value)


@pytest.mark.parametrize(("sections", "needles"), [
    # run-workers passes no teachers and the gRPC weight store carries no teacher_active: labels would be missing
    ({"agents": {"agent_0": {}, "bot": RANDOM_BOT}, "kickstart": {"teacher": "bot"}},
     ["kickstart.teacher 'bot'", "scripted teacher"]),
    ({"agents": {"agent_0": {"kickstart": {"teacher": "bot"}}, "bot": RANDOM_BOT}},
     ["agents.agent_0.kickstart"]),
    # run-learner applies no init: a per-agent init.from / critic warm-up is refused, not ignored
    ({"agents": {"agent_0": {"init": {"from": "bc.pt"}}}}, ["init.from 'bc.pt' of agent 'agent_0'"]),
    ({"agents": {"agent_0": {"init": {"critic_warmup_steps": 1}}}},
     ["init.critic_warmup_steps=1 of agent 'agent_0'"]),
], ids=["scripted-teacher", "agent-scripted-teacher", "agent-init-from", "agent-critic-warmup"])
def test_scripted_teachers_and_any_init_are_refused(sections, needles):
    config = make_test_config("turns", **sections)
    with pytest.raises(ConfigError, match="SP5") as info:
        check_distributed_scope(config)
    for needle in needles:
        assert needle in str(info.value)


def test_a_directory_named_like_a_pt_file_is_not_a_pt_teacher(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.mkdir()                  # a checkpoint dir: the teacher would get its own architecture from meta.json
    with pytest.raises(ConfigError, match="SP5") as info:
        check_distributed_scope(make_test_config("turns", kickstart={"teacher": str(ckpt)}))
    assert "kickstart.teacher" in str(info.value)


def test_values_decide_not_the_presence_of_a_key():
    config = make_test_config(
        "turns", init={"strict": False},
        agents={"agent_0": {"matchmaking": {"teammates": "self"}, "kickstart": {"lambda": 1.0},
                            "init": {"strict": False, "critic_warmup_steps": 0}}})
    check_distributed_scope(config)          # every value equals the global one: nothing to refuse


@pytest.mark.parametrize(("opponents", "warned"), [
    (LATEST_ONLY, False),
    ({**LATEST_ONLY, "snapshots": {0: 0.0, 100000: 0.0}}, False),       # a schedule of zeros is latest only
    ({**LATEST_ONLY, "snapshots": {0: 0.0, 100000: 0.2}}, True),
    ({"latest": 0.7, "snapshots": 0.2, "rivals": 0.0, "anchors": 0.1}, True),   # the SP3 default
], ids=["latest-only", "zero-schedule", "schedule", "default"])
def test_any_mix_but_latest_only_is_reduced_with_one_warning(opponents, warned, caplog):
    config = make_test_config("turns", matchmaking={"opponents": opponents})
    with caplog.at_level(logging.WARNING, logger="colosseum.distributed"):
        distributed_setup(config, ["agent_0"])
    assert len(_reductions(caplog)) == (1 if warned else 0)


@pytest.mark.parametrize("opponents", [LATEST_ONLY, {"latest": 0.7, "snapshots": 0.2, "rivals": 0.0, "anchors": 0.1}],
                         ids=["latest-only", "default"])
def test_the_one_warning_also_says_that_mixed_teammates_are_ignored(opponents, caplog):
    config = make_test_config("turns", matchmaking={"opponents": opponents, "teammates": "mixed"})
    with caplog.at_level(logging.WARNING, logger="colosseum.distributed"):
        distributed_setup(config, ["agent_0"])
    warnings = [r.getMessage() for r in caplog.records
                if r.name == "colosseum.distributed" and r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "teammates: mixed is ignored" in warnings[0], warnings
    assert ("reduced to latest only" in warnings[0]) == (opponents is not LATEST_ONLY)


def test_sp2_configs_resolved_configs_and_the_sp2_kickstart_form_start(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="colosseum.distributed"):
        distributed_setup(load_config(SP2_TTT), ["agent_0"])         # mode self_play, latest_prob 0.8
        assert len(_reductions(caplog)) == 1
        caplog.clear()
        resolved = write_test_config(tmp_path / "resolved.yaml", "turns")   # every field written explicitly
        distributed_setup(load_config(resolved), ["agent_0"])
        assert len(_reductions(caplog)) == 1
    raw = yaml.safe_load(resolved.read_text())
    raw.pop("kickstart", None)
    raw["training"]["kickstart_teacher"] = str(_teacher_pt(tmp_path))     # SP2 form, translated by T4.1
    sp2_form = tmp_path / "sp2_kickstart.yaml"
    sp2_form.write_text(yaml.safe_dump(raw, sort_keys=False))
    distributed_setup(load_config(sp2_form), ["agent_0"])
    top_level = make_test_config("turns", kickstart={"teacher": str(_teacher_pt(tmp_path))})
    distributed_setup(top_level, ["agent_0"])


def test_the_distributed_learner_keeps_snapshots_like_the_local_one(tmp_path):
    config = make_test_config("turns", checkpoint={"interval": 10, "keep_last": 2, "keep_every": 3})
    manager = distributed_checkpoint_manager(config, tmp_path / "checkpoints")
    state = {"w": np.zeros(2, np.float32)}
    for version in range(10, 101, 10):
        manager.save("agent_0", version, state, trainer_state={"step": version})
    manager.save("agent_0", 105, state, trainer_state={"step": 105}, meta_extra={"final": True})
    agent_dir = tmp_path / "checkpoints" / "agent_0"
    kept = sorted(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))
    assert kept == [30, 60, 90, 100, 105]          # every 3rd interval, the newest 2, the final one
    with_state = sorted(int(p.parent.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*/trainer_state.pt"))
    assert with_state == [100, 105]                # trainer_state.pt only inside the keep_last window


def test_entry_points_refuse_before_a_run_dir_exists(tmp_path, monkeypatch, restore_root_logging):
    import colosseum.distributed as distributed

    path = write_test_config(tmp_path / "cfg.yaml", "turns", agents={"bot": RANDOM_BOT})
    overrides = {"run.dir": str(tmp_path / "runs")}
    with pytest.raises(ConfigError, match="SP5"):
        distributed.run_distributed_learner(str(path), "agent_0", 0, "localhost:1", overrides=overrides)
    monkeypatch.setattr(distributed.mp, "Process", _ExitedProcess)
    with pytest.raises(ConfigError, match="SP5"):
        distributed.run_distributed_workers(str(path), "localhost:1", {"agent_0": "localhost:2"}, overrides)
    assert not (tmp_path / "runs").exists()
