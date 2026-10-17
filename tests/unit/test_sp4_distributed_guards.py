"""Distributed roles and SP4 (spec block 8, criterion 3.9; T8.1): pulse evaluation and top-k are refused
with a ConfigError naming SP5; the journal and ratings sections are accepted (distributed roles write no
match log: one INFO when it is enabled); ``max_snapshots`` caps the distributed learner's storage FIFO."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from colosseum.core.errors import ConfigError
from colosseum.distributed import check_distributed_scope, distributed_checkpoint_manager
from game_helpers import make_test_config, scripted_agent


@pytest.mark.parametrize(("sections", "needle"), [
    ({"evaluation": {"enabled": True},
      "agents": {"agent_0": {}, "bot": scripted_agent()}}, "evaluation.enabled"),
    ({"checkpoint": {"keep_top_k": 3}}, "keep_top_k"),
], ids=["pulse", "top-k"])
def test_sp4_settings_are_refused_with_a_pointer_to_sp5(sections, needle):
    config = make_test_config("turns", **sections)
    with pytest.raises(ConfigError, match="SP5") as info:
        check_distributed_scope(config)
    assert needle in str(info.value)


def test_journal_and_ratings_sections_are_accepted_and_the_missing_journal_is_noted(caplog):
    config = make_test_config("turns", match_log={"detail_window_mb": 64.0}, ratings={"refit_interval_sec": 5.0},
                              checkpoint={"max_snapshots": 7})
    with caplog.at_level(logging.INFO, logger="colosseum.distributed"):
        check_distributed_scope(config)
    notes = [r.getMessage() for r in caplog.records if "match log" in r.getMessage()]
    assert len(notes) == 1 and "SP5" in notes[0]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="colosseum.distributed"):
        check_distributed_scope(make_test_config("turns", match_log={"enabled": False}))
    assert not [r for r in caplog.records if "match log" in r.getMessage()]


def test_max_snapshots_caps_the_distributed_learner_oldest_keep_every_first(tmp_path):
    config = make_test_config("turns", checkpoint={"interval": 10, "keep_last": 2, "keep_every": 3,
                                                   "max_snapshots": 3})
    manager = distributed_checkpoint_manager(config, tmp_path / "checkpoints")
    state = {"w": np.zeros(2, np.float32)}
    for version in range(10, 101, 10):
        manager.save("agent_0", version, state, trainer_state={"step": version})
    kept = sorted(int(p.name.removeprefix("ckpt_v")) for p in (tmp_path / "checkpoints" / "agent_0").glob("ckpt_v*"))
    # keep_every: 30 / 60 / 90, newest: 90 / 100 -> 4 > 3, the oldest keep_every-only snapshot goes (FIFO)
    assert kept == [60, 90, 100]
