"""Docs and example config of SP4 (criterion 3.10, T8.4): the league example shows evaluation and keep_top_k (both
off) and loads with pulse off and the journal on; README and ARENA_GUIDE name every new command; every flag README
and the guide give a new command exists, as a whole word, in that command's --help."""
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
DOCS = ("README.md", "docs/ARENA_GUIDE.md")


def test_league_example_shows_evaluation_and_top_k_switched_off():
    raw = yaml.safe_load((REPO / "configs/examples/unit_harvest_league.yaml").read_text())
    assert raw["evaluation"]["enabled"] is False and "every" in raw["evaluation"]
    assert raw["checkpoint"]["keep_top_k"] == 0 and "max_snapshots" in raw["checkpoint"]
    cfg = load_config(REPO / "configs/examples/unit_harvest_league.yaml")
    assert not cfg.evaluation.enabled and cfg.match_log.enabled


@pytest.mark.parametrize("doc", DOCS)
def test_docs_name_every_new_command(doc):
    text = (REPO / doc).read_text()
    for command in COMMANDS:
        assert f"colosseum {command}" in text, (doc, command)


def _has_flag(help_text: str, flag: str) -> bool:
    """``flag`` as a whole word: ``--block`` is not found in a help that has only ``--blocks``."""
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text) is not None


def test_a_truncated_flag_is_not_found():
    assert _has_flag("  --blocks INTEGER", "--blocks") and not _has_flag("  --blocks INTEGER", "--block")


@pytest.mark.parametrize("doc", DOCS)
@pytest.mark.parametrize("command", COMMANDS)
def test_every_flag_in_the_docs_exists(doc, command):
    help_text = CliRunner().invoke(main, [command, "--help"]).output
    text = (REPO / doc).read_text()
    used = set()
    for line in re.findall(rf"colosseum {command}\b(?:[^\n`\\]|\\\n)*", text):   # with \-continued lines
        used |= set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", line))
    missing = sorted(flag for flag in used if not _has_flag(help_text, flag))
    assert not missing, (doc, command, missing)
