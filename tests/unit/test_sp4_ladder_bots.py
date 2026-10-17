"""Ladder bots of known strength (SP4 criterion 3.3, T8.2): epsilon is validated, epsilon 0 is the base bot,
the tron bot avoids walls, and against a fixed field (RandomBot + the bot at epsilon 0.1) the mean pairwise
score falls with epsilon on the real MatchRunner. Bands: at the plan's measurement (tic_tac_toe 0.742 /
0.532 / 0.315, tron 3p 0.872 / 0.435 / 0.326 for epsilon 0 / 0.5 / 1 over 400 / 300 games) adjacent gaps
are >= 0.1 with a per-game sd <= 0.5, so 200 / 150 games give an SE of the gap <= 0.05: an inversion is
a > 2 SE event (and the observed gaps are 2-4 times larger)."""
from __future__ import annotations

import functools

import numpy as np
import pytest

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.eval import play_lineups
from colosseum.players.registry import BotSpec, make_bot
from colosseum.worker.match_runner import ScriptedPlayer
from examples.tic_tac_toe.bots import TicTacToeBot
from examples.tic_tac_toe.game import TicTacToeGame
from examples.tic_tac_toe.ladder import EpsilonTicTacToeBot
from examples.tron.bots import FloodFillTronBot, free_area
from examples.tron.game import TronGame

TTT = "examples.tic_tac_toe.ladder.EpsilonTicTacToeBot"
TRON = "examples.tron.bots.FloodFillTronBot"


def _player(spec, cls: str, **kwargs) -> ScriptedPlayer:
    return ScriptedPlayer(functools.partial(make_bot, BotSpec(cls, kwargs), spec))


@pytest.mark.parametrize("cls", [EpsilonTicTacToeBot, FloodFillTronBot])
@pytest.mark.parametrize("epsilon", [-0.1, 1.5])
def test_epsilon_outside_the_unit_interval_is_refused(cls, epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        cls(epsilon=epsilon)


def test_epsilon_zero_plays_exactly_the_base_bot():
    rng = np.random.default_rng(0)
    for _ in range(50):
        board = rng.choice([0, 1, 2], size=9, p=[0.5, 0.25, 0.25])
        if not (board == 0).any():
            continue
        obs = np.stack([board == 1, board == 2, board == 0]).reshape(3, 3, 3).astype(np.float32)
        base, ladder = TicTacToeBot(), EpsilonTicTacToeBot(epsilon=0.0)
        for bot in (base, ladder):
            bot.game_spec = TicTacToeGame.spec
            bot.reset(role="player", seat=0, layout="2p", rng=np.random.default_rng(7))
        assert ladder.act(obs, board == 0, None) == base.act(obs, board == 0, None)


def test_tron_bot_never_takes_a_blocked_move_when_a_free_one_exists():
    window = np.zeros((2, 7, 7), np.uint8)
    window[0, 2, 3] = 1                      # wall straight ahead (heading up)
    window[0, 3, 2] = 1                      # wall on the left
    bot = FloodFillTronBot()
    bot.reset(role="player", seat=0, layout="2p", rng=np.random.default_rng(0))
    assert all(bot.act(window, None, None) == 2 for _ in range(10))          # turn right
    assert free_area(window[0] > 0, (2, 3)) == -1


def _mean_score(results, agent: str) -> float:
    total = 0.0
    for r in results:
        ranks = {t.team: t.rank for t in r.teams}
        (mine,) = {s.team for s in r.seats if s.agent_id == agent}
        others = [rank for team, rank in ranks.items() if team != mine]
        total += np.mean([1.0 if ranks[mine] < o else 0.5 if ranks[mine] == o else 0.0 for o in others])
    return total / len(results)


@pytest.mark.parametrize(("env_cls", "cls", "layout", "games"), [(TicTacToeGame, TTT, "2p", 200),
                                                                  (TronGame, TRON, "3p", 150)], ids=["ttt", "tron"])
def test_strength_falls_with_epsilon_against_a_fixed_field(env_cls, cls, layout, games):
    spec = env_cls().spec
    scores = []
    for eps in (0.0, 0.5, 1.0):
        models = {"c": _player(spec, cls, epsilon=eps), "random": _player(spec, "colosseum.players.RandomBot"),
                  "ref": _player(spec, cls, epsilon=0.1)}
        seats = [SeatAssignment(a, "fixed", False) for a in ("c", "random", "ref")]
        if layout == "2p":
            lineups = [Lineup(layout, [seats[0], seats[1 + (m // 2) % 2]][:: 1 if m % 2 == 0 else -1])
                       for m in range(games)]
        else:
            lineups = [Lineup(layout, seats[k % 3:] + seats[:k % 3]) for k in range(games)]
        results = play_lineups(env_fn=env_cls, models=models, lineups=lineups, num_envs=8, seed=3)
        scores.append(_mean_score(results, "c"))
    assert scores[0] > scores[1] > scores[2], scores
