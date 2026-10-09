"""Fast SP2 learning checks (spec section 3, criterion 3, and section 6): a Units bandit (a ``Units``
action learns end to end with ``ratio_mode: per_unit``; not a credit-assignment test, the joint
ratio solves it too) and a cooperative bandit that needs both seats right at once."""
from __future__ import annotations

import time

import pytest

from colosseum.core.config import AlgorithmConfig
from colosseum.core.specs import ActionSpec
from colosseum.core.types import Lineup, SeatAssignment
from demo_learning import AgentSetup, GreedyPolicy, mean_team_score, play, train_in_process
from game_helpers import make_test_model
from sp2_bandits import CoopBandit, UnitsBandit, make_units_bandit_model

pytestmark = pytest.mark.usefixtures("restore_global_rng")


def _solo(agent: str, n: int) -> list[Lineup]:
    return [Lineup("solo", [SeatAssignment(agent)]) for _ in range(n)]


def test_units_bandit_is_solved_with_per_unit_ratios():
    role = UnitsBandit().spec.roles["player"]
    action_spec = ActionSpec.from_space(role.action_space)
    config = AlgorithmConfig(learning_rate=3e-3, lr_schedule="constant", entropy_coeff=0.003, ratio_mode="per_unit")

    def solved(models) -> bool:
        greedy = GreedyPolicy(models["agent_0"], action_spec)
        return mean_team_score(play(UnitsBandit, {"g": greedy}, _solo("g", 64)), "g") >= 0.95

    start = time.monotonic()
    metrics: dict[str, dict[str, float]] = {}
    updates = train_in_process(
        env_fn=UnitsBandit,
        agents={"agent_0": AgentSetup(["player"], make_units_bandit_model, config, action_spec)},
        lineups=_solo("agent_0", 16), max_updates=300, solved=solved, last_metrics=metrics)
    assert updates != -1, "units bandit not solved within 300 updates (random play scores 0.25)"
    # The per-unit decider path ran: an act slot has 1..8 live units, about 4.5 on average.
    assert metrics["agent_0"]["deciders_valid_mean"] > 1.0, metrics["agent_0"]
    assert time.monotonic() - start < 60


def test_coop_bandit_two_agents_learn_their_joint_answer():
    role = CoopBandit().spec.roles["player"]
    action_spec = ActionSpec.from_space(role.action_space)
    config = AlgorithmConfig(learning_rate=3e-3, lr_schedule="constant", entropy_coeff=0.003)
    setup = AgentSetup(["player"], lambda: make_test_model(role, "none", hidden=32), config, action_spec)
    lineups = [Lineup("coop2", [SeatAssignment("a"), SeatAssignment("b")]) for _ in range(16)]

    def solved(models) -> bool:
        greedy = {name: GreedyPolicy(models[name], action_spec) for name in ("a", "b")}
        results = play(CoopBandit, greedy, [Lineup("coop2", [SeatAssignment("a"), SeatAssignment("b")])
                                            for _ in range(64)])
        return mean_team_score(results, "a") >= 0.95

    start = time.monotonic()
    updates = train_in_process(env_fn=CoopBandit, agents={"a": setup, "b": setup}, lineups=lineups,
                               max_updates=300, solved=solved)
    assert updates != -1, "coop bandit not solved within 300 updates (random play scores 1/16)"
    assert time.monotonic() - start < 60
