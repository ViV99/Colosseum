"""Role signatures and agents.<id>.roles resolution (SP2 T2.4)."""
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.roles import agent_role_spec, resolve_agent_roles, role_signature
from colosseum.envs.game import GameSpec, RoleSpec, SeatSpec
from game_helpers import AsymmetricGame, EliminationFFA, GlobalStateGame

ASYM = AsymmetricGame().spec
NETWORKS = {"encoder_class": "game_helpers.GenericEncoder", "policy_class": "game_helpers.TreePolicyHead",
            "value_class": "game_helpers.GenericValue"}


def _config(agents=None, env="game_helpers.AsymmetricGame"):
    data = {"env": {"env_class": env}, "networks": NETWORKS}
    if agents is not None:
        data["agents"] = agents
    return ColosseumConfig.model_validate(data)


def test_role_signature_names_every_space():
    hunter, prey = ASYM.roles["hunter"], ASYM.roles["prey"]
    assert role_signature(hunter) == "obs=<root>:float32[4]|act=<root>:discrete(5)|gs=none"
    assert role_signature(prey) != role_signature(hunter)
    with_gs = GlobalStateGame().spec.roles["player"]
    assert role_signature(with_gs).endswith("|gs=<root>:float32[4]")
    assert role_signature(RoleSpec(Box(-1.0, 1.0, (4,), dtype=np.float32), Discrete(5))) == role_signature(hunter)


def test_single_role_games_default_to_every_role():
    assert resolve_agent_roles(_config(env="game_helpers.EliminationFFA"), EliminationFFA().spec) == \
        {"agent_0": ["player"]}


def test_explicit_roles_for_an_asymmetric_game():
    cfg = _config({"hunter": {"roles": ["hunter"]}, "prey": {"roles": ["prey"]}})
    roles = resolve_agent_roles(cfg, ASYM)
    assert roles == {"hunter": ["hunter"], "prey": ["prey"]}
    assert agent_role_spec(ASYM, roles["prey"]) is ASYM.roles["prey"]


def test_omitted_roles_with_different_spaces_need_separate_agents():
    with pytest.raises(ConfigError, match="set agents.agent_0.roles; roles with different spaces need separate agents"):
        resolve_agent_roles(_config(), ASYM)


def test_unknown_role():
    with pytest.raises(ConfigError, match=r"agents.a.roles: unknown roles \['wolf'\]; the game has roles "
                                          r"\['hunter', 'prey'\]"):
        resolve_agent_roles(_config({"a": {"roles": ["wolf"]}}), ASYM)


def test_roles_of_one_agent_must_share_spaces():
    with pytest.raises(ConfigError, match="roles 'hunter' and 'prey' have different spaces"):
        resolve_agent_roles(_config({"a": {"roles": ["hunter", "prey"]}}), ASYM)


def test_roles_with_equal_spaces_can_share_an_agent():
    space = RoleSpec(Box(-1.0, 1.0, (3,), dtype=np.float32), Discrete(2))
    spec = GameSpec(roles={"attacker": space, "defender": space},
                    layouts={"1v1": (SeatSpec("attacker", 0), SeatSpec("defender", 1))})
    assert resolve_agent_roles(_config(), spec) == {"agent_0": ["attacker", "defender"]}
    assert resolve_agent_roles(_config({"x": {"roles": ["defender"]}}), spec) == {"x": ["defender"]}
