"""validate of fixed agents (SP3 T1.5, spec block 1): bots are played under the legality gate, frozen agents
are loaded and checked; `colosseum validate` lists every agent with its kind."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.config import load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model, validate_config
from colosseum.core.roles import role_signature
from colosseum.core.validation import ValidationReport
from colosseum.players import ScriptedBot
from game_helpers import (
    SP2_CHECKPOINT,
    SP2_TTT_TINY,
    agent_role_of,
    frozen_agent,
    make_test_config,
    scripted_agent,
    write_test_config,
)


class IllegalTurnBot(ScriptedBot):
    def act(self, obs, mask, info):
        return np.int64(2)                     # TurnTakingGame always masks action 2


class CrashingTurnBot(ScriptedBot):
    def act(self, obs, mask, info):
        raise RuntimeError("bot bug")


def _model_state(config):
    _roles, role = agent_role_of(config, "agent_0")
    model = build_model(config.get_agent_config("agent_0"), role)
    return {k: v.detach().numpy() for k, v in model.state_dict().items()}


def _pt(tmp_path, hidden=16):
    cfg = make_test_config("turns", networks={"kwargs": {"core": "none", "hidden": hidden}})
    path = tmp_path / f"net{hidden}.pt"
    torch.save({k: torch.from_numpy(v) for k, v in _model_state(cfg).items()}, path)
    return path


def _turns(**agents):
    return make_test_config("turns", agents={"agent_0": {}, **agents})


def test_validate_plays_scripted_agents_and_loads_frozen_ones(tmp_path):
    report = validate_config(_turns(rnd=scripted_agent(), old=frozen_agent(_pt(tmp_path))))
    assert isinstance(report, ValidationReport)
    assert any(line.startswith("agent 'rnd' (scripted colosseum.players.RandomBot): roles ['player']; played ")
               for line in report.lines), report.lines
    assert any(line.startswith("agent 'old' (frozen): ") for line in report.lines), report.lines


@pytest.mark.parametrize("class_path, message", [
    ("test_sp3_validate_players.IllegalTurnBot",
     r"validate, seat 0, episode step 0, layout 2p: agent 'bad': illegal action: action 2"),
    ("test_sp3_validate_players.CrashingTurnBot", r"agent 'bad': act raised RuntimeError: bot bug"),
    ("game_helpers.TurnTakingGame", "must subclass colosseum.players.ScriptedBot"),
    ("no_such_module.Bot", "cannot be imported"),
])
def test_a_broken_scripted_agent_fails_validate(class_path, message):
    with pytest.raises(ConfigError, match=message):
        validate_config(_turns(bad=scripted_agent(class_path)))


def test_broken_frozen_agents_fail_validate(tmp_path):
    cfg = make_test_config("turns")
    roles, role = agent_role_of(cfg, "agent_0")
    other = tmp_path / "other"
    CheckpointManager(other).save("x", 1, _model_state(cfg), meta_extra={"roles": roles, "role_signature": "nope"})
    with pytest.raises(ConfigError, match="role signature"):
        validate_config(_turns(old=frozen_agent(other / "x" / "ckpt_v1")))
    signed = tmp_path / "signed"
    CheckpointManager(signed).save("x", 1, _model_state(cfg),
                                   meta_extra={"roles": roles, "role_signature": role_signature(role)})
    with pytest.raises(ConfigError, match="meta.json gives"):
        validate_config(_turns(old=frozen_agent(signed / "x" / "ckpt_v1", roles=["player"])))
    with pytest.raises(ConfigError, match="do not match"):
        validate_config(_turns(old=frozen_agent(_pt(tmp_path, hidden=32))))     # wide weights, default networks
    meta = json.loads((signed / "x" / "ckpt_v1" / "meta.json").read_text())
    assert "networks" not in meta                                               # the config's networks are used
    validate_config(_turns(old=frozen_agent(signed / "x" / "ckpt_v1")))


def test_the_sp2_checkpoint_validates_as_a_frozen_agent():
    cfg = load_config(SP2_TTT_TINY, {"agents.old.kind": "frozen", "agents.old.path": str(SP2_CHECKPOINT)})
    report = validate_config(cfg)
    assert any(line.startswith(f"agent 'old' (frozen): {SP2_CHECKPOINT}") for line in report.lines), report.lines


def test_cli_validate_lists_every_agent_with_its_kind(tmp_path):
    path = write_test_config(tmp_path / "cfg.yaml", "turns",
                             agents={"agent_0": {}, "rnd": scripted_agent(), "old": frozen_agent(_pt(tmp_path))})
    result = CliRunner().invoke(main, ["validate", "-c", str(path)])
    assert result.exit_code == 0, result.output
    for line in ("OK: agent 'agent_0' (trainable)", "OK: agent 'rnd' (scripted)", "OK: agent 'old' (frozen)",
                 "agent 'rnd' (scripted colosseum.players.RandomBot)", "Config is valid."):
        assert line in result.output, result.output
