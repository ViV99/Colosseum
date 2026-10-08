"""Tic-tac-toe with mask and `active` beats a random legal-move player >= 80% of games (spec §3.3)."""
from __future__ import annotations

import sys
import time

import pytest

from cli_runner import TTT_CONFIG, run_in_session
from ttt_eval import load_latest_model, play_vs_random


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_tic_tac_toe_beats_random_80_percent(tmp_path, restore_global_rng):
    cmd = [sys.executable, "-m", "colosseum", "train", "-c", str(TTT_CONFIG),
           "--set", f"run.dir={tmp_path / 'runs'}", "--set", "run.name=ttt", "--set", "training.seed=0"]
    start = time.monotonic()
    proc = run_in_session(cmd, timeout=1100)
    elapsed = time.monotonic() - start
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = play_vs_random(load_latest_model(tmp_path / "runs" / "ttt"), num_games=400, seed=0)
    print(f"training took {elapsed:.0f}s; vs random: {result}")
    assert result["win_rate"] >= 0.80, result
