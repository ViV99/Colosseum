"""EpisodeTracker: every env-contract rule of SP2 spec block 1, seat phases, returns (SP2 T1.5)."""
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from colosseum.core.errors import EnvContractError
from colosseum.sp2.envs.contract import EpisodeTracker, SeatPhase
from colosseum.sp2.envs.game import GameSpec, Outcome, StepResult
from game_helpers import (
    TOY_GAMES,
    EliminationFFA,
    GlobalStateGame,
    SoloCounterGame,
    TeamDeadTeammateGame,
    UnitsGame,
    play_episode,
)

# 2p and 3p layouts: in "2p" seat 2 is empty.
FFA = GameSpec.symmetric([2, 3], Box(-1.0, 1.0, (2,), dtype=np.float32), Discrete(3))
OBS = np.zeros(2, dtype=np.float32)


def _obs(*seats):
    return {s: OBS for s in seats}


def _tracker(layout="3p", acting=(0, 1, 2), **kwargs):
    tracker = EpisodeTracker(FFA, context="worker 0, env 3", **kwargs)
    masks = tracker.on_reset(layout, StepResult(acting=set(acting), obs=_obs(*acting)))
    return tracker, masks


def test_reset_sets_phases_and_returns_full_masks():
    tracker, masks = _tracker("2p", acting=(0, 1))
    assert [tracker.phase(s) for s in range(3)] == [SeatPhase.LIVE, SeatPhase.LIVE, SeatPhase.EMPTY]
    assert tracker.phase(7) is SeatPhase.EMPTY
    assert sorted(masks) == [0, 1] and masks[0].tolist() == [True, True, True]
    assert tracker.acting() == {0, 1} and tracker.live_seats() == [0, 1]
    assert tracker.layout == "2p" and tracker.episode_step == 0 and not tracker.episode_over
    no_masks = EpisodeTracker(GameSpec.solo(Box(-1.0, 1.0, (2,)), Box(-1.0, 1.0, (2,))))
    assert no_masks.on_reset("solo", StepResult(acting={0}, obs={0: np.zeros(2)})) == {0: None}


@pytest.mark.parametrize("result, message", [
    (StepResult(acting={0}, obs=_obs(0), rewards={0: 1.0}), "reset must not give rewards"),
    (StepResult(acting={0}, obs=_obs(0), terminated={1}), "reset must not terminate"),
    (StepResult(acting=set(), obs={}, episode_over=True), "reset must not end the episode"),
    (StepResult(acting={0}, obs=_obs(0), outcome=Outcome()), "must not report an outcome"),
    (StepResult(acting={0, 1}, obs=_obs(0)), "seat 1, episode step 0, layout 3p: an acting seat has no observation"),
    ("not a result", "must return a StepResult"),
])
def test_reset_rules(result, message):
    with pytest.raises(EnvContractError, match=message):
        EpisodeTracker(FFA).on_reset("3p", result)


def test_unknown_layout():
    with pytest.raises(EnvContractError, match=r"worker 1, episode step 0: layout '5p' is not one of"):
        EpisodeTracker(FFA, context="worker 1").on_reset("5p", StepResult(acting={0}, obs=_obs(0)))


def test_actions_must_be_exactly_the_acting_seats():
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match=r"actions sent are for seats \[0, 1\], but the acting seats were "
                                               r"\[0, 1, 2\]"):
        tracker.on_step({0: 0, 1: 0}, StepResult(acting={0}, obs=_obs(0)))


@pytest.mark.parametrize("result, message", [
    (StepResult(acting={0, 2}, obs=_obs(0, 2)), "seat 2, episode step 1, layout 2p: an empty seat is in acting"),
    (StepResult(acting={0}, obs=_obs(0, 2)), "seat 2.*an observation for an empty seat"),
    (StepResult(acting={0}, obs=_obs(0), action_masks={2: np.ones(3, bool)}), "an action mask for an empty seat"),
    (StepResult(acting={0}, obs=_obs(0), rewards={2: 1.0}), "seat 2.*a reward for an empty seat"),
    (StepResult(acting={0}, obs=_obs(0), terminated={2}), "terminated for an empty seat"),
    (StepResult(acting={0}, obs=_obs(0), rewards={9: 1.0}), "seat 9.*a reward for an empty seat"),
    (StepResult(acting={0}, obs=_obs(0), final_obs=_obs(2)), "seat 2.*final_obs for an empty seat"),
    # The final step checks empty seats too.
    (StepResult(acting=set(), obs=_obs(2), episode_over=True), "seat 2.*an observation for an empty seat"),
    (StepResult(acting=set(), obs={}, action_masks={2: np.ones(3, bool)}, episode_over=True),
     "seat 2.*an action mask for an empty seat"),
    (StepResult(acting=set(), obs={}, global_state={2: OBS}, episode_over=True),
     "seat 2.*a global_state for an empty seat"),
    (StepResult(acting=set(), obs={}, final_obs=_obs(2), episode_over=True), "seat 2.*final_obs for an empty seat"),
    (StepResult(acting=set(), obs={}, final_obs=_obs(0, 1, 2), episode_over=True, truncated=True),
     "seat 2.*final_obs for an empty seat"),
    (StepResult(acting=set(), obs={}, rewards={2: 1.0}, episode_over=True), "seat 2.*a reward for an empty seat"),
])
def test_empty_seats_get_nothing(result, message):
    tracker, _ = _tracker("2p", acting=(0, 1))
    with pytest.raises(EnvContractError, match=message):
        tracker.on_step({0: 0, 1: 0}, result)


@pytest.mark.parametrize("result, message", [
    (StepResult(acting={"0"}, obs=_obs(0)), r"StepResult.acting: seats must be ints, got '0' \(str\)"),
    (StepResult(acting={0}, obs={"0": OBS}), r"StepResult.obs: seats must be ints, got '0'"),
    (StepResult(acting={0}, obs=_obs(0), rewards={"1": 1.0}), r"StepResult.rewards: seats must be ints"),
    (StepResult(acting={0}, obs=_obs(0), terminated={1.0}), r"StepResult.terminated: seats must be ints"),
    (StepResult(acting={0}, obs=_obs(0), action_masks={True: np.ones(3, bool)}),
     r"StepResult.action_masks: seats must be ints, got True \(bool\)"),
    (StepResult(acting={0}, obs=_obs(0), global_state={"0": OBS}), r"StepResult.global_state: seats must be ints"),
    (StepResult(acting=set(), obs={}, episode_over=True, truncated=True, final_obs={"0": OBS}),
     r"StepResult.final_obs: seats must be ints"),
    (StepResult(acting={0}, obs=_obs(0), action_masks=None), r"StepResult.action_masks must be a dict of seat"),
    (StepResult(acting={0}, obs=None), r"StepResult.obs must be a dict of seat -> value, got NoneType"),
    (StepResult(acting=0, obs=_obs(0)), r"StepResult.acting must be a set of seats, got int"),
])
def test_malformed_fields_are_contract_errors(result, message):
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match=r"^worker 0, env 3, episode step 1, layout 3p: " + message):
        tracker.on_step({0: 0, 1: 0, 2: 0}, result)
    with pytest.raises(EnvContractError, match=r"^worker 0, env 3, episode step 0, layout 3p: " + message):
        EpisodeTracker(FFA, context="worker 0, env 3").on_reset("3p", result)


def test_numpy_integer_seats_are_accepted():
    tracker, masks = _tracker(acting=(np.int64(0), np.int64(1), np.int64(2)))
    assert sorted(masks) == [0, 1, 2]
    tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting={np.int64(1)}, obs={np.int64(1): OBS},
                                                   rewards={np.int64(2): 1.0}))
    assert tracker.acting() == {1} and tracker.seat_returns() == [0.0, 0.0, 1.0]


def test_reward_in_the_elimination_step_is_allowed_and_later_ones_are_not():
    tracker, _ = _tracker()
    tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting={0, 2}, obs=_obs(0, 2), rewards={1: -1.0, 0: 0.5},
                                                   terminated={1}))
    assert tracker.phase(1) is SeatPhase.ELIMINATED and tracker.eliminated_step(1) == 1
    assert tracker.seat_returns() == [0.5, -1.0, 0.0] and tracker.live_seats() == [0, 2]
    act = {0: 0, 2: 0}
    with pytest.raises(EnvContractError, match="seat 1, episode step 2, layout 3p: a reward for a seat eliminated "
                                               "at episode step 1"):
        tracker.on_step(act, StepResult(acting={0, 2}, obs=_obs(0, 2), rewards={1: 0.1}))


@pytest.mark.parametrize("result, message", [
    (StepResult(acting={0, 1, 2}, obs=_obs(0, 1, 2)), "seat 1.*a seat eliminated at episode step 1 is in acting"),
    (StepResult(acting={0, 2}, obs=_obs(0, 2), terminated={1}), "terminated for a seat eliminated"),
])
def test_eliminated_seats_cannot_act_or_be_terminated_again(result, message):
    tracker, _ = _tracker()
    tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting={0, 2}, obs=_obs(0, 2), terminated={1}))
    with pytest.raises(EnvContractError, match=message):
        tracker.on_step({0: 0, 2: 0}, result)


def test_terminated_and_acting_in_the_same_step():
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match=r"seats \[1\] are terminated and acting"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting={0, 1}, obs=_obs(0, 1), terminated={1}))


def test_waiting_seats_observations_and_masks_are_ignored():
    tracker, _ = _tracker()
    garbage = np.zeros((7, 7))
    masks = tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(
        acting={0}, obs={0: OBS, 1: garbage, 2: garbage},
        action_masks={1: np.zeros(3, dtype=bool)}, rewards={1: 1.0, 2: 2.0}))
    assert list(masks) == [0] and tracker.acting() == {0}
    assert tracker.seat_returns() == [0.0, 1.0, 2.0]


def test_observation_and_mask_checks_name_the_context():
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match=r"^worker 0, env 3, seat 1, episode step 1, layout 3p: observation: "
                                               r"leaf <root> has shape \(3,\), expected \(2,\)"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting={1}, obs={1: np.zeros(3)}))
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match="seat 0.*action mask: action mask <root> must be a bool array"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting={0}, obs=_obs(0),
                                                       action_masks={0: np.ones(3, dtype=np.int8)}))
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match="seat 0.*has no legal action for an acting seat"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting={0}, obs=_obs(0),
                                                       action_masks={0: np.zeros(3, dtype=bool)}))


def test_units_empty_rows_are_not_an_error():
    env = UnitsGame(max_units=3)
    tracker = EpisodeTracker(env.spec)
    masks = tracker.on_reset("solo", env.reset(None, "solo"))
    assert masks[0]["units"]["unit"].tolist() == [True, False, False]
    assert not masks[0]["units"]["action"][1].any()


@pytest.mark.parametrize("result, message", [
    (StepResult(acting={0}, obs=_obs(0), episode_over=True), r"episode_over with a non-empty acting set \[0\]"),
    (StepResult(acting={0}, obs=_obs(0), outcome=Outcome()), "outcome is only allowed with episode_over"),
    (StepResult(acting={0}, obs=_obs(0), truncated=True), "truncated requires episode_over"),
    (StepResult(acting={0}, obs=_obs(0), rewards={0: float("nan")}), "seat 0.*reward must be finite"),
    (StepResult(acting={0}, obs=_obs(0), rewards={0: "x"}), "reward must be a number"),
])
def test_step_result_rules(result, message):
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match=message):
        tracker.on_step({0: 0, 1: 0, 2: 0}, result)


def test_truncation_needs_final_obs_for_live_seats_only():
    # Seat 2 is terminated in the truncation step: it gets terminal, so it needs no final_obs.
    tracker, _ = _tracker()
    end = StepResult(acting=set(), obs={}, terminated={2}, rewards={2: -1.0}, episode_over=True, truncated=True,
                     final_obs=_obs(0, 1))
    assert tracker.on_step({0: 0, 1: 0, 2: 0}, end) == {}
    assert tracker.episode_over and tracker.live_seats() == [0, 1] and tracker.eliminated_step(2) == 1
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match="seat 1.*truncated episode without final_obs for this live seat"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting=set(), obs={}, episode_over=True, truncated=True,
                                                       final_obs=_obs(0, 2)))
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match="final_obs: leaf <root> has shape"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting=set(), obs={}, episode_over=True, truncated=True,
                                                       final_obs={0: OBS, 1: OBS, 2: np.zeros(5)}))


def test_episode_end_by_the_rules_needs_no_final_obs():
    tracker, _ = _tracker()
    tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting=set(), obs={}, episode_over=True))
    assert tracker.episode_over and tracker.acting() == set()
    with pytest.raises(RuntimeError, match="after episode_over"):
        tracker.on_step({}, StepResult(acting=set(), obs={}))


def test_global_state_rules():
    spec = GlobalStateGame().spec
    gs = np.zeros(4, dtype=np.float32)
    tracker = EpisodeTracker(spec)
    with pytest.raises(EnvContractError, match="seat 1.*acting seat got no global_state"):
        tracker.on_reset("2p", StepResult(acting={0, 1}, obs=_obs(0, 1), global_state={0: gs}))
    tracker = EpisodeTracker(spec)
    tracker.on_reset("2p", StepResult(acting={0, 1}, obs=_obs(0, 1), global_state={0: gs, 1: gs}))
    with pytest.raises(EnvContractError, match="seat 0.*truncated episode without the final global_state"):
        tracker.on_step({0: 0, 1: 0}, StepResult(acting=set(), obs={}, episode_over=True, truncated=True,
                                                 final_obs=_obs(0, 1), global_state={1: gs}))
    plain = EpisodeTracker(FFA)
    with pytest.raises(EnvContractError, match="declares no global_state_space"):
        plain.on_reset("2p", StepResult(acting={0, 1}, obs=_obs(0, 1), global_state={0: gs}))
    plain.on_reset("2p", StepResult(acting={0, 1}, obs=_obs(0, 1)))
    with pytest.raises(EnvContractError, match="seat 0, episode step 1, layout 2p: .*declares no global_state_space"):
        plain.on_step({0: 0, 1: 0}, StepResult(acting=set(), obs={}, episode_over=True, global_state={0: gs}))


def test_global_state_of_waiting_seats_is_ignored():
    gs = np.zeros(4, dtype=np.float32)
    tracker = EpisodeTracker(GlobalStateGame().spec)
    tracker.on_reset("2p", StepResult(acting={0, 1}, obs=_obs(0, 1), global_state={0: gs, 1: gs}))
    tracker.on_step({0: 0, 1: 0}, StepResult(acting={0}, obs=_obs(0), global_state={0: gs, 1: np.zeros((7, 7))}))
    assert tracker.acting() == {0}


def test_max_idle_steps():
    tracker, _ = _tracker("2p", acting=(0, 1), max_idle_steps=2)
    idle = StepResult(acting=set(), obs={})
    tracker.on_step({0: 0, 1: 0}, idle)
    tracker.on_step({}, idle)
    tracker.on_step({}, StepResult(acting={0}, obs=_obs(0)))   # an acting step resets the count
    tracker.on_step({0: 1}, idle)
    tracker.on_step({}, idle)
    with pytest.raises(EnvContractError, match="more than env.max_idle_steps=2 steps in a row"):
        tracker.on_step({}, idle)


def test_reset_without_acting_seats_counts_as_the_first_idle_step():
    tracker = EpisodeTracker(FFA, max_idle_steps=2)
    tracker.on_reset("2p", StepResult(acting=set(), obs={}))
    tracker.on_step({}, StepResult(acting=set(), obs={}))
    with pytest.raises(EnvContractError, match="episode step 2, layout 2p: more than env.max_idle_steps=2"):
        tracker.on_step({}, StepResult(acting=set(), obs={}))


def test_team_result_default_and_explicit():
    tracker, _ = _tracker()
    with pytest.raises(RuntimeError):
        tracker.team_result()
    tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting=set(), obs={}, rewards={0: 1.0, 1: 3.0, 2: 3.0},
                                                   episode_over=True))
    assert tracker.team_result() == ({0: 3.0, 1: 1.0, 2: 1.0}, {0: 1.0, 1: 3.0, 2: 3.0})
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match=r"episode step 1, layout 3p: outcome.team_rank keys \[0, 1\]"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting=set(), obs={}, episode_over=True,
                                                       outcome=Outcome(team_rank={0: 1, 1: 2})))


def test_dead_teammate_keeps_rewards_and_stays_live():
    env = TeamDeadTeammateGame(length=5, dead_at=2)
    tracker, results = play_episode(env, "2v2", rng=np.random.default_rng(1))
    assert all(1 not in r.acting for r in results[2:])
    assert tracker.live_seats() == [0, 1, 2, 3] and tracker.eliminated_step(1) is None
    assert tracker.seat_returns()[1] == pytest.approx(sum(r.rewards.get(1, 0.0) for r in results))


def test_ffa_elimination_bookkeeping():
    tracker, _ = play_episode(EliminationFFA(max_players=4), "4p")
    assert [tracker.eliminated_step(s) for s in range(4)] == [None, 3, 2, 1]
    assert tracker.live_seats() == [0]
    assert tracker.team_result()[0] == {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}


@pytest.mark.parametrize("name", sorted(TOY_GAMES))
def test_every_toy_game_satisfies_the_contract(name):
    env = TOY_GAMES[name]()
    rng = np.random.default_rng(0)
    for layout in env.spec.layouts:
        for episode in range(2):
            tracker, results = play_episode(env, layout, seed=episode, rng=rng)
            assert tracker.episode_over and len(results) >= 2


@pytest.mark.parametrize("env", [SoloCounterGame(truncate_at=3), GlobalStateGame(truncate_at=2)],
                         ids=["solo", "global_state"])
def test_truncating_toy_games_satisfy_the_contract(env):
    layout = next(iter(env.spec.layouts))
    tracker, results = play_episode(env, layout)
    assert results[-1].truncated and tracker.live_seats() == sorted(results[-1].final_obs)
