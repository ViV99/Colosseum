"""Demo-game bots (spec block 2, T6.2): tic-tac-toe wins, blocks, takes the center, else plays a
random legal cell; the unit_harvest bot is scripted_action with legal builds; both beat RandomBot
on the real MatchRunner (legality checked by the framework on every decision)."""
from __future__ import annotations

import functools

import numpy as np
import pytest

from colosseum.core.types import Lineup, MatchResult, SeatAssignment
from colosseum.eval import play_lineups
from colosseum.players.registry import BotSpec, make_bot
from colosseum.worker.match_runner import ScriptedPlayer
from examples.tic_tac_toe.bots import TicTacToeBot
from examples.tic_tac_toe.game import TicTacToeGame
from examples.unit_harvest.bots import HarvestBot
from examples.unit_harvest.game import UnitHarvestGame, scripted_action


def _board(own: list[int], opp: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """``(obs, mask)`` of a tic-tac-toe position from the mover's side."""
    board = np.zeros(9, np.int8)
    board[own] = 1
    board[opp] = 2
    obs = np.stack([(board == 1), (board == 2), (board == 0)]).reshape(3, 3, 3).astype(np.float32)
    return obs, board == 0


def _ttt_bot(seed: int = 0) -> TicTacToeBot:
    bot = TicTacToeBot()
    bot.game_spec = TicTacToeGame.spec
    bot.reset(role="player", seat=0, layout="2p", rng=np.random.default_rng(seed))
    return bot


def test_tic_tac_toe_bot_wins_then_blocks_then_takes_the_center():
    assert _ttt_bot().act(*_board([0, 1], [3, 4]), None) == 2        # its own line beats blocking 3-4-5
    assert _ttt_bot().act(*_board([0, 8], [3, 4]), None) == 5        # block 3-4-5 (0-4-8 is taken)
    assert _ttt_bot().act(*_board([], []), None) == 4                # the center
    obs, mask = _board([4], [0])
    picks = {_ttt_bot(seed).act(obs, mask, None) for seed in range(20)}
    assert picks <= set(np.flatnonzero(mask).tolist()) and len(picks) > 1   # else a random legal cell


def test_harvest_bot_builds_only_when_the_mask_allows_it():
    env = UnitHarvestGame()
    bot = HarvestBot()
    bot.game_spec = env.spec
    bot.reset(role="player", seat=0, layout="2p", rng=np.random.default_rng(0))
    result = env.reset(0, "2p")
    obs = {key: np.array(value, copy=True) for key, value in result.obs[0].items()}
    obs["base"][0] = 0.5                       # the observation claims a stock of 5 ...
    mask = result.action_masks[0]
    assert not mask["base"][1]                 # ... but the env forbids building (stock 0)
    assert scripted_action(obs)["base"] == 1
    action = bot.act(obs, mask, None)
    assert action["base"] == 0
    np.testing.assert_array_equal(action["workers"], scripted_action(obs)["workers"])


def _wdl(results: list[MatchResult], agent: str) -> tuple[float, float, float]:
    w = d = losses = 0
    for r in results:
        ranks = {t.team: t.rank for t in r.teams}
        (mine,) = {s.team for s in r.seats if s.agent_id == agent}
        (other,) = set(ranks) - {mine}
        w += ranks[mine] < ranks[other]
        d += ranks[mine] == ranks[other]
        losses += ranks[mine] > ranks[other]
    n = len(results)
    return w / n, d / n, losses / n


@pytest.mark.parametrize(("env_cls", "bot_class", "matches", "min_win", "max_loss"), [
    (TicTacToeGame, "examples.tic_tac_toe.bots.TicTacToeBot", 200, 0.80, 0.05),
    (UnitHarvestGame, "examples.unit_harvest.bots.HarvestBot", 40, 0.95, 0.0),
], ids=["tic_tac_toe", "unit_harvest"])
def test_demo_bots_beat_random_bot_on_the_match_runner(env_cls, bot_class, matches, min_win, max_loss):
    spec = env_cls().spec
    players = {"bot": ScriptedPlayer(functools.partial(make_bot, BotSpec(bot_class, {}), spec)),
               "random": ScriptedPlayer(functools.partial(make_bot, BotSpec("colosseum.players.RandomBot", {}), spec))}
    seats = [SeatAssignment("bot", collect=False), SeatAssignment("random", collect=False)]   # bots never collect
    lineups = [Lineup("2p", seats if m % 2 == 0 else seats[::-1]) for m in range(matches)]
    win, _draw, loss = _wdl(play_lineups(env_fn=env_cls, models=players, lineups=lineups, num_envs=8, seed=0), "bot")
    assert win >= min_win and loss <= max_loss, (win, loss)
