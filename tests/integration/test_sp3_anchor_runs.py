"""`colosseum train` with a scripted anchor: anchors are drawn by their share, never collect, are
rating entities (they appear in the ``games`` matrix), and the played shares reach metrics.jsonl
(spec block 5, T3.3)."""
from __future__ import annotations

from pathlib import Path

from cli_runner import run_train
from game_helpers import write_test_config


def test_a_scripted_anchor_plays_its_share_and_shows_in_metrics_and_ratings(tmp_path: Path):
    config = write_test_config(
        tmp_path / "anchor.yaml", "turns",
        agents={"agent_0": {}, "bot": {"kind": "scripted", "class": "colosseum.players.RandomBot"}},
        matchmaking={"opponents": {"latest": 0.5, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.5}},
    )
    run = run_train(config, tmp_path, name="anchor")
    assert run.returncode == 0, run.stderr[-3000:]
    teams = anchors = 0.0
    for record in run.records("episodes"):
        assert record["agent"] == "agent_0"                    # the bot never owns data
        cell = record["opponents"].get("2p")
        if cell:
            teams += cell["teams"]
            anchors += cell["teams"] * cell["categories"]["anchors"]
            assert set(cell["anchors"]) <= {"bot"}
    assert teams >= 100, teams
    assert 0.3 <= anchors / teams <= 0.7, anchors / teams
    assert run.ratings()["layouts"]["2p"]["games"]["agent_0"]["bot"] > 0   # P6: participation, not ELO keys
    assert not (run.root / "checkpoints" / "bot").exists()
