"""Docs and example config of SP4 (criterion 3.10, T8.4): the league example shows evaluation and keep_top_k and
still validates without pulse running by default; README and ARENA_GUIDE name every new command; every flag
the guide uses exists in the command's --help."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.config import load_config

REPO = Path(__file__).resolve().parents[2]
COMMANDS = ("tournament", "report", "dashboard", "pin")


def test_league_example_shows_evaluation_and_top_k_switched_off():
    raw = yaml.safe_load((REPO / "configs/examples/unit_harvest_league.yaml").read_text())
    assert raw["evaluation"]["enabled"] is False and "every" in raw["evaluation"]
    assert raw["checkpoint"]["keep_top_k"] == 0 and "max_snapshots" in raw["checkpoint"]
    cfg = load_config(REPO / "configs/examples/unit_harvest_league.yaml")
    assert not cfg.evaluation.enabled and cfg.match_log.enabled


@pytest.mark.parametrize("doc", ["README.md", "docs/ARENA_GUIDE.md"])
def test_docs_name_every_new_command(doc):
    text = (REPO / doc).read_text()
    for command in COMMANDS:
        assert f"colosseum {command}" in text, (doc, command)


@pytest.mark.parametrize("command", COMMANDS)
def test_every_flag_in_the_guide_exists(command):
    help_text = CliRunner().invoke(main, [command, "--help"]).output
    guide = (REPO / "docs/ARENA_GUIDE.md").read_text()
    used = set()
    for line in re.findall(rf"colosseum {command}\b(?:[^\n`\\]|\\\n)*", guide):   # with \-continued lines
        used |= set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", line))
    missing = sorted(flag for flag in used if flag not in help_text)
    assert not missing, (command, missing)
