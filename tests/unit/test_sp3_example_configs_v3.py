"""Example configs in the SP3 form (T6.6): no SP2 knob, no translation warning, and the matchmaking
and checkpoint sections equal the translation of their SP2 copies (tests/fixtures/sp2/configs)."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from colosseum.core.config import SP2_MATCHMAKING_KNOBS, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "configs" / "examples"
SP2_COPIES = REPO_ROOT / "tests" / "fixtures" / "sp2" / "configs"
CHANGED_ON_PURPOSE = {"team_tag.yaml"}          # T6.3: RandomBot anchor (spec block 9)


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")), ids=lambda p: p.name)
def test_example_configs_use_no_sp2_knob(path, caplog):
    raw = yaml.safe_load(path.read_text())
    assert not set(raw.get("matchmaking") or {}) & set(SP2_MATCHMAKING_KNOBS)
    assert "pool_size" not in (raw.get("checkpoint") or {})
    assert not [key for key in (raw.get("training") or {}) if key.startswith("kickstart_")]
    with caplog.at_level(logging.WARNING):
        load_config(path)
    assert [r.getMessage() for r in caplog.records if r.name.startswith("colosseum")] == []


@pytest.mark.parametrize("name", sorted(p.name for p in SP2_COPIES.glob("*.yaml") if p.name not in CHANGED_ON_PURPOSE))
def test_example_configs_behave_like_the_translation_of_their_sp2_copies(name):
    live, old = load_config(EXAMPLES / name), load_config(SP2_COPIES / name)
    assert live.matchmaking == old.matchmaking
    assert live.checkpoint == old.checkpoint
