"""`validate` prints the effective opponent mix and checks a custom matchmaker class (spec block 5, T3.3)."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.errors import ConfigError
from colosseum.core.registry import validate_config
from colosseum.core.types import LATEST_NETWORK_ID, Lineup, SeatAssignment
from colosseum.league.base import BaseMatchmaker
from game_helpers import make_test_config, write_test_config

BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}


class SelfOnly(BaseMatchmaker):
    def lineup_for(self, owner):
        return Lineup("2p", [SeatAssignment(owner, LATEST_NETWORK_ID, True)] * 2)


class WrongLayout(BaseMatchmaker):
    def lineup_for(self, owner):
        return Lineup("9p", [SeatAssignment(owner)] * 9)


class Exploding(BaseMatchmaker):
    def lineup_for(self, owner):
        raise RuntimeError("boom")


def test_validate_report_holds_every_agents_mix():
    report = validate_config(make_test_config("turns", agents={"agent_0": {}, "bot": BOT}))
    assert any(line.startswith("agent 'agent_0': opponents by pfsp hard") for line in report.lines)
    assert "  anchors: bot 1" in report.lines


def test_cli_validate_prints_the_mix(tmp_path):
    path = write_test_config(tmp_path / "cfg.yaml", "turns", agents={"agent_0": {}, "bot": BOT})
    result = CliRunner().invoke(main, ["validate", "-c", str(path)])
    assert result.exit_code == 0, result.output
    assert "opponents by pfsp hard" in result.output and "anchors: bot 1" in result.output


def test_a_valid_custom_matchmaker_passes():
    report = validate_config(make_test_config("turns",
                                              matchmaking={"matchmaker_class": "test_sp3_validate_mix.SelfOnly"}))
    assert report.lines[0].startswith("matchmaker: test_sp3_validate_mix.SelfOnly (custom")


@pytest.mark.parametrize("name, message", [
    ("WrongLayout", "WrongLayout.*unknown layout '9p'"),
    ("Exploding", "lineup_for\\('agent_0'\\) raised RuntimeError: boom"),
])
def test_a_broken_custom_matchmaker_fails_validate(name, message):
    with pytest.raises(ConfigError, match=message):
        validate_config(make_test_config("turns", matchmaking={"matchmaker_class": f"test_sp3_validate_mix.{name}"}))
