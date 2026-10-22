"""Seatable agents in the matchmaker (SP5 T4.2 review, controller ruling R41; spec 2.6, 7.1): an agent without
playable weights (failed, or pending with nothing parked) is left out of the candidates — its rivals share goes to the
other categories, its snapshots stay playable, a team nobody seatable can fill is ``NoLineupError`` — and with every
agent seatable the draws are SP3's, draw for draw."""
from __future__ import annotations

from collections import Counter

import gymnasium
import numpy as np
import pytest

from colosseum.core.types import LATEST_NETWORK_ID, SOURCE_OWNER
from colosseum.envs.game import GameSpec, RoleSpec, SeatSpec
from colosseum.league.base import NoLineupError
from colosseum.league.lineups import check_lineup
from colosseum.league.mixture import MixtureMatchmaker
from game_helpers import make_matchmaker_context

OBS = gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)
ACT = gymnasium.spaces.Discrete(3)
ROLES = {"a": ["player"], "b": ["player"], "bot": ["player"]}
KINDS = {"bot": "scripted"}


def _core(lineup):
    (core,) = [s for s in lineup.seats if s.source != SOURCE_OWNER]
    return core


def _draw(ctx, n=600):
    mm = MixtureMatchmaker(ctx)
    out = []
    for _ in range(n):
        lineup = mm.lineup_for("a")
        check_lineup(ctx, lineup, "test")
        out.append(lineup)
    return out


def test_an_unseatable_rival_s_share_is_spread_over_the_other_categories():
    spec = GameSpec.symmetric(2, OBS, ACT)
    mix = {"opponents": {"latest": 0.25, "snapshots": 0.0, "rivals": 0.5, "anchors": 0.25}}
    ctx = make_matchmaker_context(spec, ROLES, kinds=KINDS, matchmaking=mix, seatable=lambda agent: agent != "b")
    sources = Counter(_core(lu).source for lu in _draw(ctx))
    assert set(sources) == {"latest", "anchors"}                         # b's latest weights never seated
    assert abs(sources["latest"] / sum(sources.values()) - 0.5) < 0.08   # 0.25 / (0.25 + 0.25)


def test_a_rivals_only_mix_falls_back_to_self_play_and_snapshots_stay_playable():
    spec = GameSpec.symmetric(2, OBS, ACT)
    rivals = {"opponents": {"latest": 0.0, "snapshots": 0.0, "rivals": 1.0, "anchors": 0.0}}
    ctx = make_matchmaker_context(spec, ROLES, kinds=KINDS, matchmaking=rivals, seatable=lambda agent: agent != "b")
    cores = {(_core(lu).source, _core(lu).agent_id, _core(lu).network_id) for lu in _draw(ctx, 50)}
    assert cores == {("fallback", "a", LATEST_NETWORK_ID)}
    snaps = {"opponents": {"latest": 0.0, "snapshots": 0.5, "rivals": 0.5, "anchors": 0.0}}
    ctx = make_matchmaker_context(spec, ROLES, kinds=KINDS, matchmaking=snaps, snapshots={"b": ["ckpt_v4"]},
                                  seatable=lambda agent: agent != "b")
    cores = {(_core(lu).source, _core(lu).agent_id, _core(lu).network_id) for lu in _draw(ctx, 50)}
    assert cores == {("snapshots", "b", "ckpt_v4")}                     # the failed agent's snapshot still plays


def test_every_agent_seatable_draws_exactly_as_sp3():
    spec = GameSpec.symmetric(2, OBS, ACT)
    mix = {"opponents": {"latest": 0.3, "snapshots": 0.2, "rivals": 0.3, "anchors": 0.2}, "shuffle_seats": True}
    snaps = {"a": ["ckpt_v1"], "b": ["ckpt_v2"]}
    plain = make_matchmaker_context(spec, ROLES, kinds=KINDS, matchmaking=mix, snapshots=snaps)
    asked = []
    seatable = make_matchmaker_context(spec, ROLES, kinds=KINDS, matchmaking=mix, snapshots=snaps,
                                       seatable=lambda agent: asked.append(agent) or True)
    assert [lu.seats for lu in _draw(plain, 300)] == [lu.seats for lu in _draw(seatable, 300)]
    assert asked                                                         # consulted at every draw, not cached


def test_a_team_nobody_seatable_can_fill_is_no_lineup():
    hunter = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (4,), np.float32), gymnasium.spaces.Discrete(5))
    prey = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (3,), np.float32), gymnasium.spaces.Discrete(4))
    spec = GameSpec(roles={"hunter": hunter, "prey": prey},
                    layouts={"1v1": (SeatSpec("hunter", 0), SeatSpec("prey", 1))})
    ctx = make_matchmaker_context(spec, {"h": ["hunter"], "p": ["prey"]}, seatable=lambda agent: agent != "p")
    with pytest.raises(NoLineupError, match="nobody can play this team now"):
        MixtureMatchmaker(ctx).lineup_for("h")
