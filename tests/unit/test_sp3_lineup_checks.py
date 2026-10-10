"""check_lineup, validate_matchmaking, effective_mix and describe_mix (spec block 5, T3.2)."""
from __future__ import annotations

import re

import gymnasium
import numpy as np
import pytest

from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.registry import env_spec
from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID, Lineup, SeatAssignment
from colosseum.envs.game import GameSpec, RoleSpec, SeatSpec
from colosseum.league.lineups import check_lineup
from colosseum.league.mixture import describe_mix, effective_mix, validate_matchmaking
from game_helpers import make_matchmaker_context, make_test_config

OBS = gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)
ACT = gymnasium.spaces.Discrete(3)
HUNTER = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (4,), np.float32), gymnasium.spaces.Discrete(5))
PREY = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (3,), np.float32), gymnasium.spaces.Discrete(4))
BASE = {
    "env": {"env_class": "my_game.game.MyGame"},
    "networks": {"encoder_class": "my_game.models.Encoder", "policy_class": "my_game.models.Policy",
                 "value_class": "my_game.models.Value"},
}
BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}


def hunt_spec() -> GameSpec:
    return GameSpec(
        roles={"hunter": HUNTER, "prey": PREY},
        layouts={"1v3": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1), SeatSpec("prey", 1))},
    )


def cfg(**sections) -> ColosseumConfig:
    return ColosseumConfig.model_validate({**BASE, **sections})


def duel_context():
    return make_matchmaker_context(GameSpec.symmetric(2, OBS, ACT), {"a": ["player"], "b": ["player"],
                                                                     "bot": ["player"]},
                                   kinds={"bot": "scripted"}, snapshots={"a": ["ckpt_v1"]})


def test_valid_lineups_pass():
    ctx = duel_context()
    check_lineup(ctx, Lineup("2p", [SeatAssignment("a", source="owner"),
                                    SeatAssignment("a", "ckpt_v1", False, source="snapshots")]), "M")
    check_lineup(ctx, Lineup("2p", [SeatAssignment("a"), SeatAssignment("bot", FIXED_NETWORK_ID, False)]), "M")
    check_lineup(ctx, Lineup("2p", [SeatAssignment("a"), SeatAssignment("b", LATEST_NETWORK_ID, False)]), "M")


@pytest.mark.parametrize("seats, problem", [
    ([SeatAssignment("a")], "1 seats"),
    ([SeatAssignment("a"), SeatAssignment("ghost")], "unknown agent 'ghost'"),
    ([SeatAssignment("a"), SeatAssignment("a", "ckpt_v7", False)], "no snapshot 'ckpt_v7'"),
    ([SeatAssignment("a"), SeatAssignment("a", "ckpt_v1", True)], "collect"),
    ([SeatAssignment("a"), SeatAssignment("bot", LATEST_NETWORK_ID, False)], "must play network 'fixed'"),
    ([SeatAssignment("a"), SeatAssignment("bot", FIXED_NETWORK_ID, True)], "collect"),
    ([SeatAssignment("a"), SeatAssignment("a", source="bogus")], "unknown source 'bogus'"),
])
def test_invalid_lineups_name_the_matchmaker_and_the_problem(seats, problem):
    with pytest.raises(ValueError, match=re.escape(problem)) as info:
        check_lineup(duel_context(), Lineup("2p", seats), "MyMatchmaker")
    assert "MyMatchmaker" in str(info.value) and "2p" in str(info.value)


def test_unknown_layouts_roles_and_non_lineups_are_rejected():
    ctx = duel_context()
    with pytest.raises(ValueError, match="unknown layout '3p'"):
        check_lineup(ctx, Lineup("3p", [SeatAssignment("a")] * 3), "M")
    hunt = make_matchmaker_context(hunt_spec(), {"h": ["hunter"], "p": ["prey"]})
    with pytest.raises(ValueError, match="does not play role 'hunter'"):
        check_lineup(hunt, Lineup("1v3", [SeatAssignment("p")] * 4), "M")
    with pytest.raises(ValueError, match="must return a Lineup"):
        check_lineup(ctx, ["not", "a", "lineup"], "M")


def test_validate_matchmaking_accepts_the_defaults():
    validate_matchmaking(GameSpec.symmetric(2, OBS, ACT), {"agent_0": ["player"]}, cfg())


def test_layout_and_anchor_names_are_checked():
    spec = GameSpec.symmetric([2, 4], OBS, ACT)
    with pytest.raises(ConfigError, match="unknown layouts"):
        validate_matchmaking(spec, {"agent_0": ["player"]}, cfg(matchmaking={"layouts": {"3p": 1.0}}))
    roles = {"a": ["player"], "b": ["player"], "bot": ["player"]}
    agents = {"a": {}, "b": {}, "bot": BOT}
    with pytest.raises(ConfigError, match="'ghost'"):
        validate_matchmaking(spec, roles, cfg(agents=agents, matchmaking={"anchors": ["ghost"]}))
    with pytest.raises(ConfigError, match="trainable agent 'b'"):
        validate_matchmaking(spec, roles, cfg(agents=agents, matchmaking={"anchors": ["b"]}))
    with pytest.raises(ConfigError, match="agent 'a'"):        # per-agent anchors are checked per agent
        validate_matchmaking(spec, roles, cfg(agents={**agents, "a": {"matchmaking": {"anchors": ["ghost"]}}}))
    validate_matchmaking(spec, roles, cfg(agents=agents, matchmaking={"anchors": ["bot"]}))


def test_every_role_needs_a_trainable_agent_or_an_anchor_of_the_owner():
    spec = hunt_spec()
    with pytest.raises(ConfigError, match="prey"):
        validate_matchmaking(spec, {"h": ["hunter"]}, cfg(agents={"h": {"roles": ["hunter"]}}))
    agents = {"h": {"roles": ["hunter"]}, "bot": {**BOT, "roles": ["prey"]}}
    roles = {"h": ["hunter"], "bot": ["prey"]}
    validate_matchmaking(spec, roles, cfg(agents=agents))                    # anchors: null = every fixed agent
    with pytest.raises(ConfigError, match="prey"):
        validate_matchmaking(spec, roles, cfg(agents=agents, matchmaking={"anchors": []}))
    with pytest.raises(ConfigError, match="prey"):                           # weight 0 at every point
        validate_matchmaking(spec, roles, cfg(agents=agents, matchmaking={"anchors": {"bot": 0.0}}))


def test_role_coverage_holds_at_every_schedule_point_by_anchors_with_a_positive_weight():
    """A role nobody trainable plays needs an anchor with a positive weight at EVERY schedule point."""
    spec = GameSpec(
        roles={"hunter": HUNTER, "prey": PREY},
        layouts={"pairs": (SeatSpec("hunter", 0), SeatSpec("prey", 0), SeatSpec("hunter", 1), SeatSpec("prey", 1))},
    )
    agents = {"h": {"roles": ["hunter"]}, "early": {**BOT, "roles": ["prey"]}, "late": {**BOT, "roles": ["prey"]}}
    roles = {"h": ["hunter"], "early": ["prey"], "late": ["prey"]}
    decaying = cfg(agents=agents, matchmaking={"anchors": {"early": {0: 1.0, 1000: 0.0}}})
    with pytest.raises(ConfigError, match="env step 1000.*prey"):        # the only prey player fades out
        validate_matchmaking(spec, roles, decaying)
    crossing = cfg(agents=agents, matchmaking={"anchors": {"early": {0: 1.0, 1000: 0.0}, "late": {0: 0.0, 1000: 1.0}}})
    validate_matchmaking(spec, roles, crossing)


def test_every_agent_needs_a_playable_layout():
    spec = GameSpec(
        roles={"hunter": HUNTER, "prey": PREY},
        layouts={"1v3": hunt_spec().layouts["1v3"], "prey_only": (SeatSpec("prey", 0), SeatSpec("prey", 1))},
    )
    config = cfg(agents={"h": {"roles": ["hunter"]}, "q": {"roles": ["prey"]}},
                 matchmaking={"layouts": {"prey_only": 1.0}})
    with pytest.raises(ConfigError, match="agent 'h'"):
        validate_matchmaking(spec, {"h": ["hunter"], "q": ["prey"]}, config)


@pytest.mark.parametrize("opponents, anchors, error", [
    ({"latest": {0: 1.0, 1000: 0.0}, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.0}, None, "env step 1000"),
    ({"latest": 0.0, "snapshots": 0.0, "rivals": 1.0, "anchors": 0.0}, None, "env step 0"),
    ({"latest": 0.0, "snapshots": 1.0, "rivals": 0.0, "anchors": 0.0}, None, None),   # snapshots count
    ({"latest": 0.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 1.0}, [], "env step 0"),
    ({"latest": 0.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 1.0}, ["bot"], None),
])
def test_every_schedule_point_needs_a_fillable_category(opponents, anchors, error):
    spec = GameSpec.symmetric(2, OBS, ACT)
    config = cfg(agents={"agent_0": {}, "bot": BOT}, matchmaking={"opponents": opponents, "anchors": anchors})
    roles = {"agent_0": ["player"], "bot": ["player"]}
    if error is None:
        validate_matchmaking(spec, roles, config)
    else:
        with pytest.raises(ConfigError, match=error):
            validate_matchmaking(spec, roles, config)


def test_effective_mix_without_anchors_spreads_their_share():
    ctx = make_matchmaker_context(GameSpec.symmetric(2, OBS, ACT), {"a": ["player"]})
    assert effective_mix(ctx, "a", "2p", 0, 0) == pytest.approx({"latest": 0.7 / 0.9, "snapshots": 0.2 / 0.9})


def test_describe_mix_prints_shares_at_the_start_and_end_of_the_schedules_and_the_anchors():
    config = make_test_config(
        "turns", agents={"agent_0": {}, "bot": BOT},
        matchmaking={"opponents": {"latest": {0: 0.6, 1000: 0.4}, "snapshots": 0.2, "rivals": 0.0,
                                   "anchors": {0: 0.2, 1000: 0.4}}})
    lines = describe_mix(config, env_spec(config))
    text = "\n".join(lines)
    assert lines[0].startswith("agent 'agent_0': opponents by pfsp hard")
    assert ("step 0: latest 0.60, snapshots 0.20, rivals 0.00, anchors 0.20; "
            "step 1000: latest 0.40, snapshots 0.20, rivals 0.00, anchors 0.40") in text
    assert "  anchors: bot 1" in lines


@pytest.mark.parametrize("seat, problem", [
    (SeatAssignment("a", FIXED_NETWORK_ID, False), "trainable agent 'a' plays 'latest' or a stored snapshot"),
    (SeatAssignment("frozen", LATEST_NETWORK_ID, False), "frozen agent 'frozen' must play network 'fixed'"),
    (SeatAssignment("frozen", LATEST_NETWORK_ID, True), "frozen agent 'frozen' must play network 'fixed'"),
    (SeatAssignment("frozen", "ckpt_v1", False), "frozen agent 'frozen' must play network 'fixed'"),
])
def test_fixed_network_belongs_to_fixed_agents_only(seat, problem):
    """A frozen agent can never be seated as latest weights (which would collect and resolve to a
    trainable model), and a trainable agent never as ``fixed``."""
    ctx = make_matchmaker_context(GameSpec.symmetric(2, OBS, ACT), {"a": ["player"], "frozen": ["player"]},
                                  kinds={"frozen": "frozen"}, snapshots={"a": ["ckpt_v1"]})
    with pytest.raises(ValueError, match=re.escape(problem)):
        check_lineup(ctx, Lineup("2p", [SeatAssignment("a"), seat]), "M")
    check_lineup(ctx, Lineup("2p", [SeatAssignment("a"), SeatAssignment("frozen", FIXED_NETWORK_ID, False)]), "M")
