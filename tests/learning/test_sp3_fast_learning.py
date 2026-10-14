"""Fast SP3 learning checks (spec section 6, «Быстрые учится»): a scripted kickstart teacher (DAgger
labels in the chunk) moves the student to its action on a bandit whose reward is always 0, and a
RandomBot anchor is beaten while only the latest seats collect data."""
from __future__ import annotations

import functools
import time

import pytest
import torch

from colosseum.core.config import AlgorithmConfig, ColosseumConfig
from colosseum.core.registry import build_model, env_spec
from colosseum.core.specs import ActionSpec
from colosseum.core.types import FIXED_NETWORK_ID, Lineup, SeatAssignment
from colosseum.learner.factory import build_algorithm, resolve_teacher
from colosseum.players.registry import BotSpec, FixedPlayers, make_bot
from colosseum.worker.match_runner import ScriptedPlayer
from demo_learning import AgentSetup, GreedyPolicy, play, train_in_process, win_rate
from game_learning_envs import make_mlp_model
from sp3_bandits import OFFSET_TARGETS, DuelBandit, SilentBandit

pytestmark = pytest.mark.usefixtures("restore_global_rng")
RANDOM_BOT = BotSpec("colosseum.players.RandomBot", {})


def test_scripted_kickstart_teacher_moves_the_student_to_its_action():
    config = ColosseumConfig.model_validate({
        "env": {"env_class": "sp3_bandits.SilentBandit", "kwargs": {}},
        "networks": {"model_class": "game_helpers.GameTestModel", "kwargs": {"core": "none", "hidden": 32}},
        "algorithm": {"learning_rate": 3.0e-3, "lr_schedule": "constant", "entropy_coeff": 0.0},
        "kickstart": {"teacher": "teacher", "lambda": 1.0, "decay_steps": 1_000_000},
        "matchmaking": {"anchors": []},
        "agents": {"agent_0": {}, "teacher": {"kind": "scripted", "class": "sp3_bandits.OffsetTeacher"}},
    })
    spec = env_spec(config)
    role = spec.roles["player"]
    agent_config = config.get_agent_config("agent_0")
    teacher = resolve_teacher(config, "agent_0", spec)
    assert teacher is not None and teacher.kind == "scripted" and teacher.bot is not None
    setup = AgentSetup(
        roles=["player"], model_fn=functools.partial(build_model, agent_config, role),
        config=agent_config.algorithm, action_spec=ActionSpec.from_space(role.action_space),
        algorithm_fn=functools.partial(build_algorithm, agent_config, role, spec, device="cpu", teacher=teacher))
    labelled: list[bool] = []

    def solved(models) -> bool:
        model = models["agent_0"]
        with torch.no_grad():
            probs = model.step(torch.eye(4), model.initial_state(4)).dist.log_prob(OFFSET_TARGETS).exp()
        return bool((probs >= 0.9).all())

    start = time.monotonic()
    updates = train_in_process(
        env_fn=SilentBandit, agents={"agent_0": setup},
        lineups=[Lineup("solo", [SeatAssignment("agent_0")]) for _ in range(8)], max_updates=300, solved=solved,
        teachers={"agent_0": teacher.bot},
        on_chunk=lambda c: labelled.append(c.has_teacher is not None and bool(c.has_teacher.any())))
    assert any(labelled), "no chunk carried teacher labels"
    assert updates != -1, "the student did not follow the scripted teacher within 300 updates (the reward is 0)"
    assert time.monotonic() - start < 60


def test_random_bot_anchor_is_beaten_while_only_latest_seats_collect():
    spec = DuelBandit.spec
    action_spec = ActionSpec.from_space(spec.roles["player"].action_space)
    config = AlgorithmConfig(learning_rate=3e-3, lr_schedule="constant", entropy_coeff=0.003)
    setup = AgentSetup(["player"], lambda: make_mlp_model(obs_dim=4, num_actions=4, hidden=32), config, action_spec)
    fixed = FixedPlayers(bots={"random": RANDOM_BOT}, frozen={}, roles={"random": ("player",)})
    seats = [SeatAssignment("agent_0"), SeatAssignment("random", FIXED_NETWORK_ID, collect=False)]
    lineups = [Lineup("2p", seats if e % 2 == 0 else seats[::-1]) for e in range(8)]
    random_player = ScriptedPlayer(functools.partial(make_bot, RANDOM_BOT, spec))
    collected: set[str] = set()

    def solved(models) -> bool:
        greedy = GreedyPolicy(models["agent_0"], action_spec)
        eval_seats = [SeatAssignment("g"), SeatAssignment("random")]          # play_lineups never collects
        results = play(DuelBandit, {"g": greedy, "random": random_player},
                       [Lineup("2p", eval_seats if m % 2 == 0 else eval_seats[::-1]) for m in range(64)])
        return win_rate(results, "g") >= 0.9

    start = time.monotonic()
    updates = train_in_process(env_fn=DuelBandit, agents={"agent_0": setup}, lineups=lineups, max_updates=300,
                               solved=solved, fixed_players=fixed, on_chunk=lambda c: collected.add(c.agent_id))
    assert collected == {"agent_0"}, collected          # the anchor's seats never produce chunks
    assert updates != -1, "RandomBot not beaten in 90% of duels within 300 updates"
    assert time.monotonic() - start < 60
