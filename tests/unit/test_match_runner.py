"""MatchRunner: event order, grouped inference, results, lineups, seeds, fallbacks (T3.3)."""

from __future__ import annotations

import logging
import re

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.core.errors import EnvContractError
from colosseum.networks.state import tree_leaves
from colosseum.sp2.core.types import Lineup, SeatAssignment
from colosseum.sp2.envs.game import GameSpec, Outcome, StepResult
from colosseum.sp2.envs.spaces import Units
from colosseum.sp2.envs.vector import VectorEnv
from colosseum.sp2.networks.model import PolicyModel
from colosseum.sp2.worker.match_runner import MatchRunner
from game_helpers import (
    SCRIPT_GS_SPACE,
    SCRIPT_OBS_SPACE,
    DictModelPool,
    RecordingObserver,
    ScriptedGame,
    Tick,
    TickGame,
    make_test_model,
)

TURNS = [
    Tick(acting={0}),
    Tick(acting={1}, rewards={1: 0.5}),
    Tick(acting={0}, rewards={0: 1.0}),
    Tick(over=True, rewards={0: 1.0, 1: -1.0}),
]


class CallCountingModel(PolicyModel):
    """Delegates to ``inner`` and records the batch size of every ``step``."""

    def __init__(self, inner: PolicyModel) -> None:
        super().__init__()
        self.inner = inner
        self.batch_sizes: list[int] = []

    def initial_state(self, batch_size, device="cpu"):
        return self.inner.initial_state(batch_size, device)

    def step(self, obs, state, action_mask=None):
        self.batch_sizes.append(int(obs.shape[0]))
        return self.inner.step(obs, state, action_mask)

    def unroll(self, *args, **kwargs):
        return self.inner.unroll(*args, **kwargs)

    def reset_state(self, state, done):
        return self.inner.reset_state(state, done)


def _env_fn(script, num_seats=1, **kwargs):
    return lambda: TickGame(script, num_seats, **kwargs)


def _runner(script, num_seats, lineups, models, *, num_envs=1, observer=None, seed=0, **game_kwargs):
    """A MatchRunner over a sync VectorEnv of TickGames; returns (runner, observer, games)."""
    games = []

    def env_fn():
        games.append(TickGame(script, num_seats, **game_kwargs))
        return games[-1]

    vec = VectorEnv(env_fn, num_envs)
    observer = observer if observer is not None else RecordingObserver()
    runner = MatchRunner(vec_env=vec, lineups=lineups, models=DictModelPool(models), observer=observer,
                         seed=seed, context="worker 0, ", match_id_prefix="w0_e")
    return runner, observer, games


def _role(num_seats=1, **game_kwargs):
    return next(iter(TickGame([Tick(acting={0})], num_seats, **game_kwargs).spec.roles.values()))


def test_events_follow_the_contract_order_and_the_result_is_per_team():
    model = make_test_model(_role(2))
    runner, obs, games = _runner(TURNS, 2, [Lineup("2p", [SeatAssignment("a"), SeatAssignment("b")])],
                               {("a", "latest"): model, ("b", "latest"): model})
    for _ in range(3):
        runner.step()
    assert obs.kinds() == ["act", "rewards", "act", "rewards", "act", "rewards", "end"]
    assert [(e[2], e[3].agent_id) for e in obs.events if e[0] == "act"] == [(0, "a"), (1, "b"), (0, "a")]
    assert [e[2] for e in obs.events if e[0] == "rewards"] == [{1: 0.5}, {0: 1.0}, {0: 1.0, 1: -1.0}]
    end = obs.events[-1][2]
    assert not end.truncated and end.live_seats == [0, 1] and end.final_obs is None
    res = end.result
    assert res.match_id == "w0_e0_ep0" and res.layout == "2p" and res.outcome_kind == "wdl"
    assert res.episode_length == 3
    assert [(s.seat, s.agent_id, s.network_id, s.reward, s.team) for s in res.seats] == [
        (0, "a", "latest", 2.0, 0), (1, "b", "latest", -0.5, 1)]
    assert [(t.team, t.rank, t.score) for t in res.teams] == [(0, 1.0, 2.0), (1, 2.0, -0.5)]
    assert runner.episodes_finished == 1
    # the env was reset right away for the next episode, and acting seat 0 sees its obs
    assert games[0].log[-1][:2] == (1, 0)


def test_act_records_carry_obs_mask_action_logprob_and_pre_state():
    role = _role(1)
    model = make_test_model(role, core="lstm")
    mask = np.array([True, False, True])
    script = [Tick(acting={0}), Tick(acting={0}), Tick(over=True)]
    runner, obs, games = _runner(script, 1, [Lineup("solo", [SeatAssignment("a")])], {("a", "latest"): model},
                               mask_fn=lambda k, t, s: mask)
    runner.step()
    runner.step()
    acts = [e[3] for e in obs.events if e[0] == "act"]
    assert len(acts) == 2
    first, second = acts
    assert first.obs.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0] and second.obs.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert first.mask.tolist() == mask.tolist()
    assert int(first.action) in (0, 2) and int(second.action) in (0, 2)
    assert games[0].log[1][3] == {0: first.action}
    assert first.unit_log_probs is None and first.global_state is None
    assert all(float(x.abs().sum()) == 0.0 for x in tree_leaves(first.pre_state))   # episode start
    assert any(float(x.abs().sum()) > 0.0 for x in tree_leaves(second.pre_state))   # advanced by act 1
    with torch.no_grad():
        step = model.step(torch.from_numpy(first.obs[None]), model.initial_state(1),
                          torch.from_numpy(mask[None]))
    assert first.log_prob == pytest.approx(float(step.dist.log_prob(torch.tensor([int(first.action)]))), abs=1e-6)


def test_units_actions_record_unit_log_probs_and_global_state_reaches_records():
    space = Units(3, gymnasium.spaces.Discrete(2))
    role = _role(1, action_space=space, global_state=True)
    model = make_test_model(role)
    script = [Tick(acting={0}), Tick(over=True)]
    runner, obs, _ = _runner(script, 1, [Lineup("solo", [SeatAssignment("a")])], {("a", "latest"): model},
                             action_space=space, global_state=True)
    runner.step()
    rec = obs.events[0][3]
    assert rec.unit_log_probs.shape == (3,) and rec.unit_log_probs.dtype == np.float32
    assert rec.log_prob == pytest.approx(float(rec.unit_log_probs.sum()), abs=1e-5)
    assert rec.action.shape == (3,) and rec.action.dtype == np.int64
    assert rec.global_state.tolist() == [0.0, 0.0]


def test_inference_is_grouped_by_agent_and_network():
    role = _role(2)
    a, b, b_old = (CallCountingModel(make_test_model(role)) for _ in range(3))
    script = [Tick(acting={0, 1}), Tick(acting={0, 1}), Tick(over=True)]
    lineups = [
        Lineup("2p", [SeatAssignment("a"), SeatAssignment("b")]),
        Lineup("2p", [SeatAssignment("a"), SeatAssignment("b", "ckpt_v1", collect=False)]),
        Lineup("2p", [SeatAssignment("b"), SeatAssignment("a")]),
    ]
    runner, obs, _ = _runner(script, 2, lineups, {("a", "latest"): a, ("b", "latest"): b,
                                                    ("b", "ckpt_v1"): b_old}, num_envs=3)
    runner.step()
    assert a.batch_sizes == [3] and b.batch_sizes == [2] and b_old.batch_sizes == [1]
    acts = [(e[1], e[2]) for e in obs.events if e[0] == "act"]
    assert acts == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]          # (env, seat) order
    assert [e[3].network_id for e in obs.events if e[0] == "act"] == ["latest"] * 3 + ["ckpt_v1", "latest", "latest"]


def test_rewards_of_the_elimination_step_come_before_on_terminated():
    script = [
        Tick(acting={0, 1, 2}),
        Tick(acting={0, 1}, rewards={2: -1.0, 0: 0.25}, terminated={2}),
        Tick(acting={0}, rewards={1: -1.0}, terminated={1}),
        Tick(over=True, rewards={0: 1.0}, outcome=Outcome(team_rank={0: 1, 1: 2, 2: 3})),
    ]
    model = make_test_model(_role(3))
    runner, obs, _ = _runner(script, 3, [Lineup("3p", [SeatAssignment("a")] * 3)], {("a", "latest"): model})
    for _ in range(3):
        runner.step()
    assert obs.kinds() == ["act"] * 3 + ["rewards", "terminated"] + ["act"] * 2 + ["rewards", "terminated"] + [
        "act", "rewards", "end"]
    assert [e[2] for e in obs.events if e[0] == "terminated"] == [[2], [1]]
    end = obs.events[-1][2]
    assert end.live_seats == [0]
    res = end.result
    assert res.outcome_kind == "rank"
    assert [s.eliminated_step for s in res.seats] == [None, 2, 1]
    assert [t.rank for t in res.teams] == [1.0, 2.0, 3.0]


def test_truncation_reports_final_obs_and_global_state_of_live_seats_only():
    script = [
        Tick(acting={0, 1, 2}),
        Tick(acting={0, 1}, terminated={2}),
        Tick(over=True, truncated=True, terminated={1}),
    ]
    model = make_test_model(_role(3, global_state=True))
    runner, obs, _ = _runner(script, 3, [Lineup("3p", [SeatAssignment("a")] * 3)], {("a", "latest"): model},
                             global_state=True)
    runner.step()
    runner.step()
    end = obs.events[-1][2]
    assert end.truncated and end.live_seats == [0]
    assert sorted(end.final_obs) == [0] and end.final_obs[0].tolist() == [0.0, 0.0, 2.0, 0.0, 1.0]
    assert sorted(end.final_global_state) == [0] and end.final_global_state[0].tolist() == [0.0, 2.0]


def test_next_lineup_is_applied_at_the_episode_end_and_states_reset():
    role = _role(2)
    model_a, model_b = make_test_model(role, core="gru"), make_test_model(role, core="gru")
    script = [Tick(acting={0, 1}), Tick(acting={0, 1}), Tick(over=True)]
    runner, obs, _ = _runner(script, 2, [Lineup("2p", [SeatAssignment("a"), SeatAssignment("a")])],
                             {("a", "latest"): model_a, ("b", "latest"): model_b})
    runner.step()
    new = Lineup("2p", [SeatAssignment("b"), SeatAssignment("a", collect=False)])
    runner.set_next_lineup(0, new)
    assert runner.lineup(0).seats[0].agent_id == "a"          # not mid-episode
    runner.step()
    kinds = obs.kinds()
    assert kinds[-2:] == ["end", "lineup"]
    _, _, old, applied = obs.events[-1]
    assert [s.agent_id for s in old.seats] == ["a", "a"]
    assert applied == new and runner.lineup(0) == new
    runner.step()                                             # first step of the next episode
    acts = [e[3] for e in obs.events if e[0] == "act"][-2:]
    assert [r.agent_id for r in acts] == ["b", "a"]
    assert all(float(x.abs().sum()) == 0.0 for r in acts for x in tree_leaves(r.pre_state))


def test_missing_network_falls_back_to_latest_collecting_with_one_warning(caplog):
    model = make_test_model(_role(2))
    script = [Tick(acting={0, 1}), Tick(over=True)]
    lineup = Lineup("2p", [SeatAssignment("a"), SeatAssignment("a", "ckpt_v9", collect=False)])
    with caplog.at_level(logging.WARNING, logger="colosseum.sp2.worker.match_runner"):
        runner, _, _ = _runner(script, 2, [lineup, lineup], {("a", "latest"): model}, num_envs=2)
    assert runner.lineup(0).seats[1] == SeatAssignment("a", "latest", True)
    assert runner.lineup(1).seats[1] == SeatAssignment("a", "latest", True)
    assert sum("ckpt_v9" in r.getMessage() for r in caplog.records) == 1


def test_lineups_are_validated():
    model = make_test_model(_role(2))
    script = [Tick(acting={0, 1}), Tick(over=True)]
    with pytest.raises(ValueError, match="layout"):
        _runner(script, 2, [Lineup("9p", [SeatAssignment("a")] * 9)], {("a", "latest"): model})
    with pytest.raises(ValueError, match="seats"):
        _runner(script, 2, [Lineup("2p", [SeatAssignment("a")])], {("a", "latest"): model})
    with pytest.raises(ValueError, match="no model"):
        _runner(script, 2, [Lineup("2p", [SeatAssignment("zzz")] * 2)], {("a", "latest"): model})
    with pytest.raises(ValueError, match="one lineup per env"):
        _runner(script, 2, [Lineup("2p", [SeatAssignment("a")] * 2)], {("a", "latest"): model}, num_envs=2)


def test_episode_seeds_are_deterministic_per_env_and_episode():
    script = [Tick(acting={0}), Tick(over=True)]
    model = make_test_model(_role(1))

    def seeds(seed):
        runner, _, games = _runner(script, 1, [Lineup("solo", [SeatAssignment("a")])] * 2,
                                 {("a", "latest"): model}, num_envs=2, seed=seed)
        for _ in range(2):
            runner.step()
        return [[entry[2] for entry in env.log if entry[3] is None] for env in games]

    first, again, other, none = seeds(7), seeds(7), seeds(8), seeds(None)
    assert first == again and first != other
    assert len({s for env in first for s in env}) == 6       # 2 envs x 3 resets, all distinct
    assert none == [[None] * 3, [None] * 3]


def test_match_ids_count_episodes_per_env():
    script = [Tick(acting={0}), Tick(over=True)]
    model = make_test_model(_role(1))
    runner, obs, _ = _runner(script, 1, [Lineup("solo", [SeatAssignment("a")])] * 2, {("a", "latest"): model},
                             num_envs=2)
    for _ in range(2):
        runner.step()
    ids = [e[2].result.match_id for e in obs.events if e[0] == "end"]
    assert ids == ["w0_e0_ep0", "w0_e1_ep0", "w0_e0_ep1", "w0_e1_ep1"]
    assert all(e[2].result.outcome_kind == "score" for e in obs.events if e[0] == "end")


def test_idle_ticks_step_the_env_with_no_actions():
    script = [Tick(acting={0}), Tick(acting=()), Tick(acting=()), Tick(acting={0}), Tick(over=True)]
    model = make_test_model(_role(1))
    runner, obs, games = _runner(script, 1, [Lineup("solo", [SeatAssignment("a")])], {("a", "latest"): model})
    for _ in range(4):
        runner.step()
    log = games[0].log
    assert log[1][3] == {0: obs.events[0][3].action}
    assert log[2][3] == {} and log[3][3] == {}            # idle ticks: step({})
    assert list(log[4][3]) == [0]
    assert obs.kinds().count("act") == 2


def test_env_contract_errors_carry_the_worker_and_env_context():
    script = [Tick(acting={0})] + [Tick(acting=())] * 5 + [Tick(over=True)]
    model = make_test_model(_role(1))
    vec = VectorEnv(_env_fn(script, 1), 2)
    runner = MatchRunner(vec_env=vec, lineups=[Lineup("solo", [SeatAssignment("a")])] * 2,
                         models=DictModelPool({("a", "latest"): model}), max_idle_steps=3, context="worker 4, ")
    with pytest.raises(EnvContractError, match="worker 4, env 0"):
        for _ in range(6):
            runner.step()


def test_deterministic_runner_takes_the_mode():
    role = _role(1)
    model = make_test_model(role)
    script = [Tick(acting={0})] * 20 + [Tick(over=True)]
    vec = VectorEnv(_env_fn(script, 1), 1)
    obs = RecordingObserver()
    runner = MatchRunner(vec_env=vec, lineups=[Lineup("solo", [SeatAssignment("a")])],
                         models=DictModelPool({("a", "latest"): model}), observer=obs, deterministic=True)
    for _ in range(5):
        runner.step()
    for rec in (e[3] for e in obs.events if e[0] == "act"):
        with torch.no_grad():
            dist = model.step(torch.from_numpy(rec.obs[None]), None).dist
        assert int(rec.action) == int(dist.mode()[0])


# ---------------------------------------------------------------------------
# Amendment R14: every EpisodeTracker error of spec block 1, driven through MatchRunner.
# Env 0 replays a valid script, env 1 the faulty one, so the error must name env 1.
# Not reachable through MatchRunner (covered by EpisodeTracker's own tests): actions that do
# not match the acting set (the runner always answers exactly the acting seats) and an unknown
# layout (MatchRunner rejects such a lineup with ValueError before any reset,
# test_lineups_are_validated).
# ---------------------------------------------------------------------------


def _seats_obs(*seats):
    return {s: np.zeros(5, np.float32) for s in seats}


def _acts(*seats, gs=False, **fields):
    """A StepResult where ``seats`` act (with observations, and global states when ``gs``)."""
    if gs:
        fields.setdefault("global_state", {s: np.zeros(2, np.float32) for s in seats})
    return StepResult(acting=set(seats), obs=_seats_obs(*seats), **fields)


def _over(**fields):
    return StepResult(acting=set(), obs={}, episode_over=True, **fields)


_W = "worker 7, env 1, "
CONTRACT_CASES = {
    "acting seat without observation": (
        lambda: [StepResult(acting={0, 1}, obs=_seats_obs(0))],
        f"{_W}seat 1, episode step 0, layout 2p: ", r"an acting seat has no observation"),
    "empty seat in acting": (
        lambda: [_acts(0, 1), _acts(2)],
        f"{_W}seat 2, episode step 1, layout 2p: ", r"an empty seat is in acting"),
    "eliminated seat in acting": (
        lambda: [_acts(0, 1), _acts(0, terminated={1}), _acts(0, 1)],
        f"{_W}seat 1, episode step 2, layout 2p: ", r"a seat eliminated at episode step 1 is in acting"),
    "reward for an empty seat": (
        lambda: [_acts(0, 1), _acts(0, 1, rewards={2: 1.0})],
        f"{_W}seat 2, episode step 1, layout 2p: ", r"a reward for an empty seat"),
    "reward after elimination": (
        lambda: [_acts(0, 1), _acts(0, terminated={1}), _acts(0, rewards={1: -1.0})],
        f"{_W}seat 1, episode step 2, layout 2p: ", r"a reward for a seat eliminated at episode step 1"),
    "terminated again": (
        lambda: [_acts(0, 1), _acts(0, terminated={1}), _acts(0, terminated={1})],
        f"{_W}seat 1, episode step 2, layout 2p: ", r"terminated for a seat eliminated at episode step 1"),
    "episode_over with acting seats": (
        lambda: [_acts(0, 1), _acts(0, episode_over=True)],
        f"{_W}episode step 1, layout 2p: ", r"episode_over with a non-empty acting set \[0\]"),
    "outcome not by the layout's teams": (
        lambda: [_acts(0, 1), _over(outcome=Outcome(team_rank={0: 1, 1: 2, 2: 3}))],
        f"{_W}episode step 1, layout 2p: ", r"outcome\.team_rank keys \[0, 1, 2\] must be exactly"),
    "truncation without final_obs of a live seat": (
        lambda: [_acts(0, 1), _over(truncated=True, final_obs=_seats_obs(0))],
        f"{_W}seat 1, episode step 1, layout 2p: ", r"truncated episode without final_obs"),
    "observation does not fit the role's space": (
        lambda: [_acts(0, 1), StepResult(acting={0}, obs={0: np.zeros(4, np.float32)})],
        f"{_W}seat 0, episode step 1, layout 2p: ", r"observation: .*has shape \(4,\), expected \(5,\)"),
    "mask does not fit the role's space": (
        lambda: [_acts(0, 1), _acts(0, action_masks={0: np.ones(2, bool)})],
        f"{_W}seat 0, episode step 1, layout 2p: ", r"action mask.*has shape \(2,\), expected \(3,\)"),
    "acting seat without the declared global_state": (
        lambda: [_acts(0, 1, gs=True, global_state={0: np.zeros(2, np.float32)})],
        f"{_W}seat 1, episode step 0, layout 2p: ", r"declares a global_state_space but the acting seat got no"),
    "idle overflow": (
        lambda: [_acts(0, 1), _acts(), _acts(), _acts()],
        f"{_W}episode step 3, layout 2p: ", r"more than env\.max_idle_steps=2 steps in a row"),
}


@pytest.mark.parametrize("case", list(CONTRACT_CASES))
def test_episode_tracker_errors_surface_through_the_runner_with_its_context(case):
    bad_script, prefix, pattern = CONTRACT_CASES[case]
    with_gs = "global_state" in case
    spec = GameSpec.symmetric([2, 3], SCRIPT_OBS_SPACE, gymnasium.spaces.Discrete(3),
                              SCRIPT_GS_SPACE if with_gs else None)
    scripts = iter([[_acts(0, 1, gs=with_gs) for _ in range(6)], bad_script()])
    vec = VectorEnv(lambda: ScriptedGame(spec, next(scripts)), 2)
    model = make_test_model(spec.roles["player"])
    with pytest.raises(EnvContractError) as excinfo:
        runner = MatchRunner(vec_env=vec, lineups=[Lineup("2p", [SeatAssignment("a")] * 2)] * 2,
                             models=DictModelPool({("a", "latest"): model}), max_idle_steps=2,
                             context="worker 7, ")
        for _ in range(5):
            runner.step()
    message = str(excinfo.value)
    assert message.startswith(prefix), message
    assert re.search(pattern, message[len(prefix):]), message
