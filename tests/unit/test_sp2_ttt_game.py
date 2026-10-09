"""Tic-tac-toe on the SP2 contract: turns, masks, outcomes, contract checks (T7.2)."""
from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from cli_runner import REPO_ROOT
from colosseum.core.config import load_config
from colosseum.core.registry import build_model, validate_config
from colosseum.core.specs import ActionSpec
from colosseum.envs.contract import EpisodeTracker
from examples.tic_tac_toe.game import TicTacToeGame
from examples.tic_tac_toe.models import TicTacToePolicy

TTT_SP2 = REPO_ROOT / "configs" / "examples"


def play(game: TicTacToeGame, cells: list[int]):
    """Play ``cells`` in turn under the contract tracker; returns the last StepResult."""
    tracker = EpisodeTracker(game.spec, context="test")
    result = game.reset(0, "2p")
    tracker.on_reset("2p", result)
    for cell in cells:
        (seat,) = result.acting
        actions = {seat: np.int64(cell)}
        result = game.step(actions)
        tracker.on_step(actions, result)
    return result


def test_spec_turns_and_masks():
    game = TicTacToeGame()
    assert list(game.spec.layouts) == ["2p"] and game.spec.outcome_kind("2p") == "wdl"
    result = game.reset(0, "2p")
    assert result.acting == {0} and result.action_masks[0].all() and result.obs[0][2].all()
    result = game.step({0: 4})
    assert result.acting == {1} and not result.action_masks[1][4]
    assert result.obs[1][1].reshape(-1)[4] == 1.0  # the opponent's mark from the mover's side
    with pytest.raises(ValueError, match="layout"):
        game.reset(0, "4p")


def test_win_gives_both_seats_their_reward_and_ranks():
    result = play(TicTacToeGame(), [0, 3, 1, 4, 2])
    assert result.episode_over and not result.acting
    assert result.rewards == {0: 1.0, 1: -1.0} and result.outcome.team_rank == {0: 1.0, 1: 2.0}


def test_draw():
    result = play(TicTacToeGame(), [0, 1, 2, 4, 3, 5, 7, 6, 8])
    assert result.episode_over and result.rewards == {0: 0.0, 1: 0.0}
    assert result.outcome.team_rank == {0: 1.0, 1: 1.0}


def test_illegal_move_loses():
    game = TicTacToeGame()
    game.reset(0, "2p")
    game.step({0: 4})
    result = game.step({1: 4})
    assert result.episode_over and result.rewards == {1: -1.0, 0: 1.0}


def test_policy_requires_the_action_spec_and_masks_the_board():
    """No silent fallback: the policy needs the role's action spec, which build_model injects."""
    with pytest.raises(TypeError, match="action_spec"):
        TicTacToePolicy(in_dim=64)
    with pytest.raises(ValueError, match="Discrete"):
        TicTacToePolicy(in_dim=64, action_spec=ActionSpec.from_space(gymnasium.spaces.MultiDiscrete([3, 3])))
    model = build_model(load_config(TTT_SP2 / "tic_tac_toe.yaml"), TicTacToeGame.spec.roles["player"])
    game = TicTacToeGame()
    game.reset(0, "2p")
    result = game.step({0: 4})
    obs = torch.as_tensor(result.obs[1])[None]
    mask = torch.as_tensor(result.action_masks[1])[None]
    with torch.no_grad():
        dist = model.step(obs, model.initial_state(1), mask).dist
        probs = torch.cat([dist.log_prob(torch.tensor([cell])).exp() for cell in range(9)])
    assert probs[4].item() == 0.0 and probs.sum().item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("name", ["tic_tac_toe.yaml", "tic_tac_toe_multi.yaml", "tic_tac_toe_attention.yaml"])
def test_example_configs_are_valid(name):
    validate_config(load_config(TTT_SP2 / name))
