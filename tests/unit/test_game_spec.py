"""GameSpec helpers, outcome kinds and validation (SP2 T1.4)."""
import pickle

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete, Tuple

from colosseum.core.errors import EnvContractError
from colosseum.envs.game import GameSpec, MultiAgentEnv, RoleSpec, SeatSpec, StepResult

OBS = Box(-1.0, 1.0, (3,), dtype=np.float32)
ACT = Discrete(4)


def test_solo():
    spec = GameSpec.solo(OBS, ACT)
    assert list(spec.roles) == ["player"] and spec.layouts == {"solo": (SeatSpec("player", 0),)}
    assert spec.max_seats == 1 and spec.teams("solo") == [[0]] and spec.outcome_kind("solo") == "score"
    spec.validate()


def test_symmetric_ffa_layouts():
    spec = GameSpec.symmetric(range(2, 5), OBS, ACT)
    assert list(spec.layouts) == ["2p", "3p", "4p"]
    assert spec.layouts["3p"] == (SeatSpec("player", 0), SeatSpec("player", 1), SeatSpec("player", 2))
    assert spec.max_seats == 4 and spec.layout_size("2p") == 2
    assert spec.teams("4p") == [[0], [1], [2], [3]]
    assert spec.outcome_kind("2p") == "wdl" and spec.outcome_kind("3p") == "rank"
    assert list(GameSpec.symmetric(2, OBS, ACT).layouts) == ["2p"]
    with pytest.raises(ValueError, match=">= 1"):
        GameSpec.symmetric(0, OBS, ACT)
    with pytest.raises(ValueError, match="duplicate"):
        GameSpec.symmetric([2, 2], OBS, ACT)


def test_teams_of_names_and_seat_order():
    spec = GameSpec.teams_of([[2, 2], [2, 1, 1], [3]], OBS, ACT, global_state=Box(-1.0, 1.0, (5,)))
    assert list(spec.layouts) == ["2v2", "2v1v1", "coop3"]
    assert [s.team for s in spec.layouts["2v1v1"]] == [0, 0, 1, 2]
    assert spec.teams("2v1v1") == [[0, 1], [2], [3]]
    assert spec.outcome_kind("2v2") == "wdl" and spec.outcome_kind("2v1v1") == "rank"
    assert spec.outcome_kind("coop3") == "score" and spec.num_teams("coop3") == 1
    assert spec.roles["player"].global_state_space is not None
    assert list(GameSpec.teams_of([3, 3], OBS, ACT).layouts) == ["3v3"]
    with pytest.raises(ValueError, match="duplicate"):
        GameSpec.teams_of([[2, 2], [2, 2]], OBS, ACT)


def test_role_of_and_unknown_layout():
    spec = GameSpec(roles={"hunter": RoleSpec(OBS, ACT), "prey": RoleSpec(OBS, Discrete(2))},
                    layouts={"1v2": [SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1)]})
    assert isinstance(spec.layouts["1v2"], tuple)          # lists are stored as tuples
    assert spec.role_of("1v2", 0) == "hunter" and spec.role_of("1v2", 2) == "prey"
    with pytest.raises(EnvContractError, match="seats 0..2, not 3"):
        spec.role_of("1v2", 3)
    with pytest.raises(EnvContractError, match="unknown layout '2v2'"):
        spec.teams("2v2")


@pytest.mark.parametrize("roles, layouts, message", [
    ({}, {"x": (SeatSpec("player", 0),)}, "no roles"),
    ({"player": RoleSpec(OBS, ACT)}, {}, "no layouts"),
    ({"player": RoleSpec(OBS, ACT)}, {"x": ()}, "has no seats"),
    ({"player": RoleSpec(OBS, ACT)}, {"x": (SeatSpec("ghost", 0),)}, "unknown role 'ghost'"),
    ({"player": RoleSpec(OBS, ACT)}, {"x": (SeatSpec("player", 0), SeatSpec("player", 2))},
     r"teams \[0, 2\]; team numbers must be exactly 0..T-1"),
    ({"player": RoleSpec(OBS, ACT)}, {"x": (SeatSpec("player", 1),)}, "team numbers must be exactly"),
    ({"pl.ayer": RoleSpec(OBS, ACT)}, {"x": (SeatSpec("pl.ayer", 0),)}, "role name 'pl.ayer'"),
    ({"player": RoleSpec(OBS, ACT)}, {"2p/x": (SeatSpec("player", 0),)}, "layout name '2p/x'"),
    ({"player": RoleSpec(Tuple([OBS]), ACT)}, {"x": (SeatSpec("player", 0),)}, "role 'player': unsupported"),
    ({"player": RoleSpec(OBS, Box(-1.0, 1.0, (2, 2)))}, {"x": (SeatSpec("player", 0),)}, "1-D float"),
    ({"player": (OBS, ACT)}, {"x": (SeatSpec("player", 0),)}, "must be a RoleSpec"),
    ({"player": RoleSpec(OBS, ACT)}, {"x": (("player", 0),)}, "must be a SeatSpec"),
])
def test_validate_names_the_problem(roles, layouts, message):
    with pytest.raises(EnvContractError, match=message):
        GameSpec(roles=roles, layouts=layouts).validate()


def test_specs_compare_by_value_and_pickle():
    a = GameSpec.symmetric([2, 4], OBS, ACT)
    assert a == GameSpec.symmetric([2, 4], OBS, ACT)
    assert a != GameSpec.symmetric([2, 3], OBS, ACT)
    assert pickle.loads(pickle.dumps(a)) == a


def test_step_result_defaults_and_abstract_env():
    r = StepResult(acting={0}, obs={0: np.zeros(3)})
    assert r.rewards == {} and r.terminated == set() and r.action_masks == {}
    assert not r.episode_over and not r.truncated and r.final_obs is None and r.outcome is None
    with pytest.raises(TypeError):
        MultiAgentEnv()
