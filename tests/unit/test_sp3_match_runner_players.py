"""MatchRunner player pool (SP3 T1.3, spec blocks 2-3): scripted seats with infos, bot RNG and lifetime,
the legality gate with context, frozen players under 'fixed', collect rules, source, on_episode_start."""
from __future__ import annotations

import functools

import numpy as np
import pytest

from colosseum.core.errors import PlayerError
from colosseum.core.types import FIXED_NETWORK_ID, OPPONENT_CATEGORIES, SOURCE_OWNER, Lineup, SeatAssignment
from colosseum.envs.vector import VectorEnv
from colosseum.players import ScriptedBot
from colosseum.players.registry import BotSpec, make_bot
from colosseum.worker.match_runner import MatchRunner, ScriptedPlayer
from game_helpers import (
    DictModelPool,
    RecordingBot,
    RecordingObserver,
    Tick,
    TickGame,
    UnitsGame,
    make_test_model,
    scripted_player,
)

TURNS = [Tick(acting={0}), Tick(acting={1}), Tick(acting={0}), Tick(over=True, rewards={0: 1.0, 1: -1.0})]
MASK = np.array([True, False, True])


class IllegalBot(ScriptedBot):
    def act(self, obs, mask, info):
        return np.int64(1)                    # masked by MASK


class CrashingBot(ScriptedBot):
    def __init__(self, where: str = "act") -> None:
        self.where = where

    def reset(self, *, role, seat, layout, rng):
        if self.where == "reset":
            raise RuntimeError("reset bug")

    def act(self, obs, mask, info):
        raise RuntimeError("act bug")


class WrongShapeBot(ScriptedBot):
    def act(self, obs, mask, info):
        return {"x": 0}


class UnitsTargetBot(ScriptedBot):
    """Unit 0 moves 3 and targets unit 3, which does not exist at step 0 of UnitsGame."""

    def act(self, obs, mask, info):
        return {"base": 0, "units": {"move": np.array([3, 0, 0, 0]), "target": np.array([3, 0, 0, 0])}}


class EpisodeStartObserver(RecordingObserver):
    def on_episode_start(self, env, layout, episode_seed):
        self.events.append(("start", env, layout, episode_seed))


def _role(num_seats=2, **kwargs):
    return next(iter(TickGame([Tick(acting={0})], num_seats, **kwargs).spec.roles.values()))


def _runner(lineups, players, *, script=TURNS, num_seats=2, num_envs=1, seed=0, observer=None, **game_kwargs):
    games = []

    def env_fn():
        games.append(TickGame(script, num_seats, **game_kwargs))
        return games[-1]

    vec = VectorEnv(env_fn, num_envs)
    observer = observer if observer is not None else RecordingObserver()
    runner = MatchRunner(vec_env=vec, lineups=lineups, models=DictModelPool(players), observer=observer,
                         seed=seed, context="worker 0, ", match_id_prefix="w0_e")
    return runner, observer, games


def _spec(num_seats=2, **kwargs):
    return TickGame([Tick(acting={0})], num_seats, **kwargs).spec


def _bot_lineup(*seats):
    return Lineup("2p", [SeatAssignment(a, FIXED_NETWORK_ID, False) if a.startswith("bot") else SeatAssignment(a)
                         for a in seats])


@pytest.fixture(autouse=True)
def _clear_recording_bots():
    RecordingBot.instances.clear()
    yield
    RecordingBot.instances.clear()


def test_types_name_the_fixed_network_and_the_categories():
    assert FIXED_NETWORK_ID == "fixed" and SOURCE_OWNER == "owner"
    assert OPPONENT_CATEGORIES == ("latest", "snapshots", "rivals", "anchors", "fallback")
    assert SeatAssignment("a").source == ""


def test_a_scripted_seat_gets_obs_mask_and_infos_and_is_reset_every_episode():
    spec = _spec(mask_fn=lambda k, t, s: MASK, infos=True)
    recording = BotSpec("game_helpers.RecordingBot", {})
    players = {("a", "latest"): make_test_model(_role()),
               ("bot", FIXED_NETWORK_ID): ScriptedPlayer(functools.partial(make_bot, recording, spec))}
    runner, obs, _games = _runner([_bot_lineup("a", "bot")], players, mask_fn=lambda k, t, s: MASK, infos=True)
    (bot,) = RecordingBot.instances                      # created at the first reset of its (agent, env, seat)
    assert [r["seat"] for r in bot.resets] == [1] and bot.resets[0]["role"] == "player"
    assert bot.resets[0]["layout"] == "2p"
    for _ in range(3):
        runner.step()
    assert len(RecordingBot.instances) == 1               # kept between episodes
    assert len(bot.resets) == 2                           # reset again after the episode end
    (act,) = bot.acts
    assert act["obs"].dtype == np.float32 and act["obs"].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
    assert act["mask"].tolist() == MASK.tolist()
    assert act["info"] == {"k": 0, "t": 1, "seat": 1}
    bot_records = [e[3] for e in obs.events if e[0] == "act" and e[2] == 1]
    assert len(bot_records) == 1
    record = bot_records[0]
    assert (record.agent_id, record.network_id) == ("bot", FIXED_NETWORK_ID)
    assert record.log_prob == 0.0 and record.unit_log_probs is None and record.pre_state is None
    assert record.info == {"k": 0, "t": 1, "seat": 1} and int(record.action) in (0, 2)
    result = next(e[2].result for e in obs.events if e[0] == "end")
    assert [s.network_id for s in result.seats] == ["latest", FIXED_NETWORK_ID]


def test_bot_rngs_follow_the_episode_seed_seat_and_agent():
    def draws(seed):
        RecordingBot.instances.clear()
        spec = _spec()
        factory = ScriptedPlayer(functools.partial(make_bot, BotSpec("game_helpers.RecordingBot", {}), spec))
        _runner([_bot_lineup("bot_a", "bot_a"), _bot_lineup("bot_a", "bot_b")],
                {("bot_a", FIXED_NETWORK_ID): factory, ("bot_b", FIXED_NETWORK_ID): factory}, num_envs=2, seed=seed)
        return [b.resets[0]["draw"] for b in RecordingBot.instances]

    first, again, other = draws(7), draws(7), draws(8)
    assert first == again and first != other
    assert len(set(first)) == 4                           # env, seat and agent all change the stream


def test_bot_instances_are_per_agent_env_and_seat_and_survive_lineup_changes():
    spec = _spec()
    factory = ScriptedPlayer(functools.partial(make_bot, BotSpec("game_helpers.RecordingBot", {}), spec))
    players = {("a", "latest"): make_test_model(_role()), ("bot", FIXED_NETWORK_ID): factory}
    runner, _obs, _games = _runner([_bot_lineup("a", "bot")] * 2, players, num_envs=2)
    assert len(RecordingBot.instances) == 2               # one per env
    first_env0 = RecordingBot.instances[0]
    runner.set_next_lineup(0, _bot_lineup("bot", "a"))
    assert runner.next_lineup(0) == _bot_lineup("bot", "a") and runner.next_lineup(1) is None
    for _ in range(3):
        runner.step()                                     # episode end: env 0 moves the bot to seat 0
    assert len(RecordingBot.instances) == 3               # a new (bot, env 0, seat 0) instance
    runner.set_next_lineup(0, _bot_lineup("a", "bot"))
    for _ in range(3):
        runner.step()
    assert len(RecordingBot.instances) == 3 and len(first_env0.resets) == 2   # seat 1 of env 0 is back


@pytest.mark.parametrize("bot_cls, kwargs, message", [
    (IllegalBot, {}, r"^worker 0, env 0, seat 1, episode step 1, layout 2p: agent 'bot': illegal action: "
                     r"action 1 at <root> is illegal"),
    (CrashingBot, {}, r"^worker 0, env 0, seat 1, episode step 1, layout 2p: agent 'bot': act raised "
                      r"RuntimeError: act bug"),
    (WrongShapeBot, {}, r"agent 'bot': the action does not have the structure"),
])
def test_a_bot_that_breaks_the_rules_is_a_player_error_with_context(bot_cls, kwargs, message):
    spec = _spec(mask_fn=lambda k, t, s: MASK)
    players = {("a", "latest"): make_test_model(_role()),
               ("bot", FIXED_NETWORK_ID): scripted_player(bot_cls, spec, **kwargs)}
    runner, _obs, _games = _runner([_bot_lineup("a", "bot")], players, mask_fn=lambda k, t, s: MASK)
    runner.step()                                          # seat 0 acts
    with pytest.raises(PlayerError, match=message) as info:
        runner.step()                                      # seat 1 (the bot) acts at episode step 1
    if bot_cls is CrashingBot:
        assert isinstance(info.value.__cause__, RuntimeError)


def test_a_bot_whose_reset_raises_fails_at_the_episode_start():
    spec = _spec()
    with pytest.raises(PlayerError, match=r"env 0, seat 1, episode step 0, layout 2p: agent 'bot': reset raised "
                                          r"RuntimeError: reset bug"):
        _runner([_bot_lineup("a", "bot")], {("a", "latest"): make_test_model(_role()),
                                            ("bot", FIXED_NETWORK_ID): scripted_player(CrashingBot, spec,
                                                                                       where="reset")})


def test_units_actions_of_bots_pass_the_same_gate():
    def run(bot_cls_or_spec):
        vec = VectorEnv(lambda: UnitsGame(), 1)
        player = (ScriptedPlayer(functools.partial(make_bot, bot_cls_or_spec, vec.spec))
                  if isinstance(bot_cls_or_spec, BotSpec) else scripted_player(bot_cls_or_spec, vec.spec))
        runner = MatchRunner(vec_env=vec, lineups=[Lineup("solo", [SeatAssignment("bot", FIXED_NETWORK_ID, False)])],
                             models=DictModelPool({("bot", FIXED_NETWORK_ID): player}), seed=0, context="worker 2, ")
        try:
            for _ in range(12):                            # two UnitsGame episodes
                runner.step()
        finally:
            runner.close()
        return runner.episodes_finished

    assert run(BotSpec("colosseum.players.RandomBot", {})) == 2
    with pytest.raises(PlayerError, match=r"worker 2, env 0, seat 0, episode step 0, layout solo: agent 'bot': "
                                          r"illegal action: unit 0 component 'target'"):
        run(UnitsTargetBot)


def test_only_latest_seats_may_collect_and_fixed_seats_need_their_player():
    spec = _spec()
    players = {("a", "latest"): make_test_model(_role()),
               ("bot", FIXED_NETWORK_ID): scripted_player(RecordingBot, spec)}
    with pytest.raises(ValueError, match=r"env 0, seat 1: agent 'bot' plays network 'fixed' with collect=True"):
        _runner([Lineup("2p", [SeatAssignment("a"), SeatAssignment("bot", FIXED_NETWORK_ID, True)])], players)
    with pytest.raises(ValueError, match=r"no model for agent 'bot' \(network 'latest'\)"):
        _runner([Lineup("2p", [SeatAssignment("a"), SeatAssignment("bot", collect=False)])], players)
    with pytest.raises(ValueError, match=r"no model for agent 'ghost' \(network 'fixed'\)"):
        _runner([Lineup("2p", [SeatAssignment("a"), SeatAssignment("ghost", FIXED_NETWORK_ID, False)])], players)


def test_a_frozen_player_of_another_architecture_is_batched_under_fixed():
    small, wide = make_test_model(_role()), make_test_model(_role(), hidden=32)
    players = {("a", "latest"): small, ("old", FIXED_NETWORK_ID): wide}
    lineups = [Lineup("2p", [SeatAssignment("a"), SeatAssignment("old", FIXED_NETWORK_ID, False)]),
               Lineup("2p", [SeatAssignment("old", FIXED_NETWORK_ID, False), SeatAssignment("a")])]
    runner, obs, _games = _runner(lineups, players, num_envs=2,
                                  script=[Tick(acting={0, 1}), Tick(over=True, rewards={0: 1.0, 1: -1.0})])
    runner.step()
    acts = [(e[1], e[2], e[3].agent_id, e[3].network_id) for e in obs.events if e[0] == "act"]
    assert acts == [(0, 0, "a", "latest"), (0, 1, "old", "fixed"), (1, 0, "old", "fixed"), (1, 1, "a", "latest")]
    assert all(e[3].pre_state is None for e in obs.events if e[0] == "act")   # stateless models
    results = [e[2].result for e in obs.events if e[0] == "end"]
    assert [[s.network_id for s in r.seats] for r in results] == [["latest", "fixed"], ["fixed", "latest"]]


def test_source_travels_from_the_lineup_into_the_result():
    spec = _spec()
    players = {("a", "latest"): make_test_model(_role()),
               ("bot", FIXED_NETWORK_ID): scripted_player(RecordingBot, spec)}
    lineup = Lineup("2p", [SeatAssignment("a", source=SOURCE_OWNER),
                           SeatAssignment("bot", FIXED_NETWORK_ID, False, source="anchors")])
    runner, obs, _games = _runner([lineup], players)
    assert [s.source for s in runner.lineup(0).seats] == [SOURCE_OWNER, "anchors"]
    for _ in range(3):
        runner.step()
    result = next(e[2].result for e in obs.events if e[0] == "end")
    assert [s.source for s in result.seats] == [SOURCE_OWNER, "anchors"]


def test_on_episode_start_is_called_after_every_reset_with_the_episode_seed():
    players = {("a", "latest"): make_test_model(_role())}
    observer = EpisodeStartObserver()
    runner, _obs, games = _runner([Lineup("2p", [SeatAssignment("a")] * 2)], players, observer=observer, seed=3)
    for _ in range(3):
        runner.step()
    starts = [e for e in observer.events if e[0] == "start"]
    reset_seeds = [entry[2] for entry in games[0].log if entry[3] is None]
    assert [(e[1], e[2]) for e in starts] == [(0, "2p"), (0, "2p")]
    assert [e[3] for e in starts] == reset_seeds and None not in reset_seeds
    assert observer.kinds().index("start") == 0           # before the first act


def test_neural_seats_records_carry_their_infos_entry():
    """Amendment A15: every acting seat's ActRecord.info is infos.get(seat), neural seats included."""
    players = {("a", "latest"): make_test_model(_role())}
    runner, obs, _games = _runner([Lineup("2p", [SeatAssignment("a")] * 2)], players, infos=True)
    for _ in range(3):
        runner.step()
    infos = [(e[2], e[3].info) for e in obs.events if e[0] == "act"]
    assert infos == [(0, {"k": 0, "t": 0, "seat": 0}), (1, {"k": 0, "t": 1, "seat": 1}),
                     (0, {"k": 0, "t": 2, "seat": 0})]
    runner, obs, _games = _runner([Lineup("2p", [SeatAssignment("a")] * 2)], players)
    runner.step()
    assert [e[3].info for e in obs.events if e[0] == "act"] == [None]   # no infos entry -> None
