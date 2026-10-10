"""Fast smoke of the SP3 competition pipeline on tic-tac-toe through the CLI (spec section 3,
criterion 3; section 6): record the game's bot -> bc -> train with init from the frozen BC net
(critic warm-up), a neural kickstart teacher and scripted / frozen anchors -> eval against the bot,
RandomBot and the frozen BC net. Short budgets, no thresholds."""
from __future__ import annotations

import pytest

import pipeline_kit as kit

pytestmark = pytest.mark.usefixtures("restore_root_logging")


def test_tic_tac_toe_pipeline_smoke(tmp_path):
    game = kit.TIC_TAC_TOE
    result = kit.run_pipeline(game, tmp_path, seed=0, settings=kit.TIC_TAC_TOE_SMOKE, in_process=True)
    assert result.record["roles"]["player"]["decisions"] > 0
    assert (tmp_path / "data" / "player" / "part-00000.pt").is_file() and result.bc_path.is_file()
    assert result.run.records("train"), "the learner never trained"
    games = result.run.ratings()["layouts"]["2p"]["games"][kit.MAIN]      # P6: the games matrix, not ELO keys
    assert all(games.get(anchor, 0) > 0 for anchor in (game.bot, kit.RANDOM, kit.BC_NET)), \
        f"an anchor never played main in training: {games}"
    assert set(result.pairs) == {game.bot, kit.RANDOM, kit.BC_NET}
    assert all(row["n"] == 4 for row in result.pairs.values())
