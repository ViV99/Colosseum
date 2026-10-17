"""Slow ladder test (SP4 criterion 3.3): ``colosseum tournament --mode select`` with the default sizes ranks
the ladder bots (one bot with a share epsilon of random moves) in their true order on tic_tac_toe (1v1)
and tron (FFA, 3p), selects the strongest, confirms it on fresh seeds, and the BT intervals of neighbours
do not contradict the order. Sizes and the time bound: T8.2 rulings; changing one needs a ruling with
measurements."""
from __future__ import annotations

import json

import pytest

import arena_kit as kit

pytestmark = [pytest.mark.slow, pytest.mark.timeout(900)]

LADDER_SEED = 100            # not among the measurement seeds 0-9 of T8.2
# T8.2 ruling (docs/benchmarks/sp4-tournament-defaults.json): the default sizes (--halving-blocks 8,
# --confirm-blocks 100, --finalists 3) are the plan's floors: already the smallest grid point 8 / 32 passed the
# rule on both ladders at seeds 0-9 (winner 10/10, order 9/10, confirmed 10/10, BT 10/10).
# T8.3 ruling: ladder tic_tac_toe 5 s / tron 6 s at seed 100 (pytest calls 4.6 s / 6.1 s) -> max(60, 2 x 6.1),
# rounded up to 30 s = 60 s.
MAX_SECONDS = 60.0


@pytest.mark.parametrize("game", kit.LADDER_GAMES, ids=lambda g: g.name)
def test_select_ranks_the_ladder_and_confirms_the_strongest(tmp_path, game):
    run = kit.run_select(game, tmp_path, seed=LADDER_SEED)
    rows = {name: {k: run.rows[name][k] for k in ("score", "score_ci", "bt", "bt_ci", "games")}
            for name in kit.TRUE_ORDER}
    print(f"[ladder {game.name}] {run.seconds:.0f}s ranking {run.ranking} winner {run.winner} "
          f"confirmed {run.confirmed}\n{json.dumps(rows)}")
    assert kit.ladder_problems(run) == []
    assert run.seconds <= MAX_SECONDS
