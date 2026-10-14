"""Slow SP3 pipeline test on unit_harvest (spec section 3, criterion 3): record the game's bot ->
bc -> train from the BC weights (critic warm-up) with the bot as a DAgger kickstart teacher and
scripted anchors -> eval against the bot, RandomBot and the frozen BC net, all through the CLI.
Thresholds: T6.4 rule (max(floor, min over three calibration runs - 0.05, rounded down to 0.05));
changing one needs a ruling with measurements."""
from __future__ import annotations

import json

import pytest

import pipeline_kit as kit
from demo_learning import LEARNING_SEED

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1500), pytest.mark.usefixtures("restore_global_rng")]

# T6.4 ruling (calibration seeds 0, 1, 2): vs random win 1.000 / 1.000 / 1.000, vs greedy score
# 0.483 / 0.500 / 0.475, vs bc_net score 0.483 / 0.500 / 0.483 (floors 0.80 / 0.35 / 0.45).
VS_RANDOM_MIN_WIN = 0.95      # rule: 1.000 - 0.05 (above the floor 0.80, SP2's bar for unit_harvest)
VS_BOT_MIN_SCORE = 0.40       # rule: 0.475 - 0.05 -> 0.40 (score = wins + draws / 2; trained policies tie the bot)
VS_BC_MIN_SCORE = 0.45        # floor (rule gives 0.40): RL must not end clearly below its own warm start


def _cell(row: dict) -> str:
    return f"W/D/L {row['wins']}/{row['draws']}/{row['losses']} score {row['score']:.3f} win {row['win_rate']:.3f}"


def test_unit_harvest_pipeline_beats_random_and_holds_the_bot_and_bc(tmp_path):
    game, settings = kit.UNIT_HARVEST, kit.UNIT_HARVEST_SETTINGS
    result = kit.run_pipeline(game, tmp_path, seed=LEARNING_SEED, settings=settings)
    vs_random, vs_bot, vs_bc = (result.pairs[name] for name in (kit.RANDOM, game.bot, kit.BC_NET))
    seconds = json.dumps({k: round(v) for k, v in result.seconds.items()})
    print(f"[unit_harvest pipeline] seed {LEARNING_SEED}; seconds {seconds}; recorded {result.record['decisions']} "
          f"decisions; env steps/s {_env_steps_per_sec(result):.0f}")
    print(f"[unit_harvest pipeline] vs random: {_cell(vs_random)}; vs {game.bot}: {_cell(vs_bot)}; "
          f"vs {kit.BC_NET}: {_cell(vs_bc)}")
    assert vs_random["win_rate"] >= VS_RANDOM_MIN_WIN
    assert vs_bot["score"] >= VS_BOT_MIN_SCORE
    assert vs_bc["score"] >= VS_BC_MIN_SCORE


def _env_steps_per_sec(result: kit.PipelineRun) -> float:
    system = result.run.records("system")
    if len(system) < 2 or system[-1]["ts"] <= system[0]["ts"]:
        return float("nan")
    return (system[-1]["env_steps"] - system[0]["env_steps"]) / (system[-1]["ts"] - system[0]["ts"])
