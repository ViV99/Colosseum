"""eval with players (SP3 T1.5, spec blocks 3 and 7): play_lineups with bots, -a <name> of fixed agents."""
from __future__ import annotations

import functools
import json

import numpy as np
import pytest
import torch
import yaml
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.config import load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.eval import evaluate, play_lineups, schedule_lineups
from colosseum.players import ScriptedBot
from colosseum.players.registry import BotSpec, make_bot
from colosseum.worker.match_runner import ScriptedPlayer
from game_helpers import (
    SP2_CHECKPOINT,
    SP2_TTT_TINY,
    TurnTakingGame,
    agent_role_of,
    frozen_agent,
    make_test_config,
    make_test_model,
    scripted_agent,
    write_test_config,
)


class FirstLegalBot(ScriptedBot):
    def __init__(self) -> None:
        self.calls = 0

    def act(self, obs, mask, info):
        self.calls += 1
        return np.int64(np.flatnonzero(mask)[0] if mask is not None else 0)


def _pt(tmp_path):
    cfg = make_test_config("turns")
    _roles, role = agent_role_of(cfg, "agent_0")
    path = tmp_path / "net.pt"
    torch.save(build_model(cfg.get_agent_config("agent_0"), role).state_dict(), path)
    return path


def invoke(*args):
    return CliRunner().invoke(main, ["eval", *map(str, args)])


def test_play_lineups_takes_bot_prototypes_and_scripted_players():
    spec = TurnTakingGame().spec
    model = make_test_model(spec.roles["player"])
    model.train()
    proto = FirstLegalBot()
    lineups = schedule_lineups(spec, "2p", {"net": ["player"], "bot": ["player"]}, 4)
    results = play_lineups(env_fn=TurnTakingGame, models={"net": model, "bot": proto}, lineups=lineups, num_envs=2,
                           seed=0)
    assert len(results) == 4 and proto.calls == 0          # every seat plays a deep copy of the prototype
    assert {s.agent_id for r in results for s in r.seats} == {"net", "bot"}
    assert model.training                                   # train/eval flags are restored
    rnd = ScriptedPlayer(functools.partial(make_bot, BotSpec("colosseum.players.RandomBot", {}), spec))
    assert len(play_lineups(env_fn=TurnTakingGame, models={"net": model, "bot": rnd}, lineups=lineups,
                            num_envs=2, seed=0)) == 4


def test_play_lineups_never_collects_so_default_seat_assignments_work_for_bots():
    spec = TurnTakingGame().spec
    model = make_test_model(spec.roles["player"])
    lineups = [Lineup("2p", [SeatAssignment("net"), SeatAssignment("bot")]),        # default collect=True
               Lineup("2p", [SeatAssignment("bot"), SeatAssignment("net")])]
    results = play_lineups(env_fn=TurnTakingGame, models={"net": model, "bot": FirstLegalBot()}, lineups=lineups,
                           num_envs=2, seed=0)
    assert len(results) == 2 and {s.agent_id for r in results for s in r.seats} == {"net", "bot"}
    assert all(seat.collect for lineup in lineups for seat in lineup.seats)    # the caller's lineups are untouched


def test_evaluate_takes_scripted_and_frozen_agents_by_name(tmp_path):
    pt = _pt(tmp_path)
    cfg = make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent(), "old": frozen_agent(pt)})
    report = evaluate(cfg, {"rnd": None, "old": None, "net": str(pt)}, layouts=None, num_matches=2, seed=0,
                      num_envs=2)
    pairs = {(r["agent_a"], r["agent_b"]) for r in report.layouts["2p"]["pairs"]}
    assert {("rnd", "old"), ("rnd", "net"), ("old", "net")} <= pairs
    with pytest.raises(ConfigError, match=r"trainable agent.*-a agent_0=<checkpoint dir or \.pt>"):
        evaluate(cfg, {"agent_0": None}, layouts=None, num_matches=1)
    with pytest.raises(ConfigError, match="Unknown agent 'ghost'"):
        evaluate(cfg, {"ghost": None}, layouts=None, num_matches=1)


def test_eval_cli_resolves_names_of_scripted_and_frozen_agents(tmp_path):
    pt = _pt(tmp_path)
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns",
                                 agents={"agent_0": {}, "rnd": scripted_agent(), "old": frozen_agent(pt)})
    out = tmp_path / "r.json"
    result = invoke("-c", cfg_path, "-a", "rnd", "-a", "old", "-a", f"net={pt}", "-n", 2, "--num-envs", 2,
                    "--seed", 0, "-o", out)
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["agents"] == ["rnd", "old", "net"]
    result = invoke("-c", cfg_path, "-a", "agent_0", "-a", "rnd")
    assert result.exit_code == 1 and "-a agent_0=" in result.stderr, result.output
    result = invoke("-c", cfg_path, "-a", "nobody")
    assert result.exit_code == 2 and "nobody" in result.output
    assert invoke("-c", cfg_path, "-a", "rnd=").exit_code == 2


def test_eval_cli_plays_the_sp2_checkpoint_as_a_frozen_agent_by_name(tmp_path):
    cfg = load_config(SP2_TTT_TINY, {"agents.old.kind": "frozen", "agents.old.path": str(SP2_CHECKPOINT),
                                     "agents.rnd.kind": "scripted", "agents.rnd.class": "colosseum.players.RandomBot"})
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json", by_alias=True), sort_keys=False))
    out = tmp_path / "r.json"
    result = invoke("-c", cfg_path, "-a", "old", "-a", "rnd", "-n", 2, "--num-envs", 2, "--seed", 0, "-o", out)
    assert result.exit_code == 0, result.output
    rows = json.loads(out.read_text())["layouts"]["2p"]["pairs"]
    assert {(r["agent_a"], r["agent_b"]) for r in rows} == {("old", "rnd"), ("rnd", "old")}


def test_a_player_error_is_one_line_on_stderr_with_exit_code_1(capsys):
    """Pre-flight ruling P17: ``cli._config_errors`` reports a PlayerError as "Player error: ..." (exit 1)."""
    from colosseum.cli import _config_errors
    from colosseum.core.errors import PlayerError

    with pytest.raises(SystemExit) as exc, _config_errors():
        raise PlayerError("eval, env 0, seat 1, episode step 3, layout 2p: agent 'bot': illegal action: x")
    assert exc.value.code == 1
    assert capsys.readouterr().err == ("Player error: eval, env 0, seat 1, episode step 3, layout 2p: agent 'bot': "
                                       "illegal action: x\n")
