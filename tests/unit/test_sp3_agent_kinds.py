"""Agent kinds in the config (SP3 T1.1, spec block 1): trainable / scripted / frozen, implicit agent_0, --set."""
from __future__ import annotations

import pytest
import yaml

from colosseum.core.config import (
    AgentOverride,
    ColosseumConfig,
    FrozenAgent,
    ScriptedAgent,
    TrainableAgent,
    apply_overrides,
    load_config,
)
from colosseum.core.errors import ConfigError

RANDOM_BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}


def _data(**sections) -> dict:
    data = {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}}
    data.update(sections)
    return data


def _write(tmp_path, data, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


MIXED = {
    "main": None,
    "beta": {"algorithm": {"learning_rate": 1.0e-4}},
    "greedy": {"kind": "scripted", "class": "my_game.bots.Greedy", "kwargs": {"aggr": 0.7}},
    "prev": {"kind": "frozen", "path": "runs/sub12/checkpoints/main/ckpt_v9000"},
    "bc_net": {"kind": "frozen", "path": "runs/bc/main.pt", "networks": {"kwargs": {"hidden": 32}},
               "roles": ["player"]},
}


def test_entries_default_to_trainable_and_keep_their_kind():
    cfg = ColosseumConfig.model_validate(_data(agents=MIXED))
    assert cfg.agent_ids() == ["main", "beta", "greedy", "prev", "bc_net"]
    assert cfg.get_trainable_agent_ids() == ["main", "beta"]
    assert cfg.fixed_agent_ids() == ["greedy", "prev", "bc_net"]
    assert [cfg.agent_kind(a) for a in cfg.agent_ids()] == ["trainable", "trainable", "scripted", "frozen", "frozen"]
    assert cfg.agent_entry("main") == TrainableAgent() and AgentOverride is TrainableAgent
    greedy = cfg.agent_entry("greedy")
    assert isinstance(greedy, ScriptedAgent)
    assert greedy.class_path == "my_game.bots.Greedy" and greedy.kwargs == {"aggr": 0.7} and greedy.roles is None
    bc_net = cfg.agent_entry("bc_net")
    assert isinstance(bc_net, FrozenAgent) and bc_net.networks == {"kwargs": {"hidden": 32}}
    assert cfg.agent_roles("greedy") is None and cfg.agent_roles("bc_net") == ["player"]
    assert cfg.get_agent_config("beta").algorithm.learning_rate == 1.0e-4


def test_the_resolved_dump_round_trips_with_aliases(tmp_path):
    cfg = ColosseumConfig.model_validate(_data(agents=MIXED))
    path = _write(tmp_path, cfg.model_dump(mode="json", by_alias=True), "resolved.yaml")
    dumped = yaml.safe_load(path.read_text())
    assert dumped["agents"]["greedy"]["class"] == "my_game.bots.Greedy"
    assert dumped["agents"]["main"]["kind"] == "trainable"
    assert load_config(path) == cfg
    assert ColosseumConfig.model_validate(cfg.model_dump()) == cfg   # field names validate too


@pytest.mark.parametrize("agents", [{}, {"rnd": RANDOM_BOT}, {"old": {"kind": "frozen", "path": "x.pt"}}])
def test_an_implicit_trainable_agent_0_exists_without_trainable_agents(agents):
    cfg = ColosseumConfig.model_validate(_data(agents=agents))
    assert cfg.get_trainable_agent_ids() == ["agent_0"]
    assert cfg.agent_ids() == ["agent_0", *agents]
    assert cfg.agent_kind("agent_0") == "trainable" and cfg.agent_entry("agent_0") == TrainableAgent()
    assert cfg.get_agent_config("agent_0").agents == {}
    assert cfg.fixed_agent_ids() == list(agents)


def test_agent_0_is_not_implicit_when_a_trainable_agent_exists():
    cfg = ColosseumConfig.model_validate(_data(agents={"main": {}, "rnd": RANDOM_BOT}))
    with pytest.raises(ConfigError, match="Unknown agent 'agent_0'"):
        cfg.agent_entry("agent_0")


def test_a_fixed_agent_named_agent_0_without_trainable_agents_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="implicit trainable agent"):
        load_config(_write(tmp_path, _data(agents={"agent_0": RANDOM_BOT})))


@pytest.mark.parametrize("entry, words", [
    ({"kind": "scripted", "class": "a.B", "path": "x.pt"}, ["path", "scripted", "kwargs"]),
    ({"kind": "frozen", "path": "x.pt", "class": "a.B"}, ["class", "frozen", "networks"]),
    ({"class": "a.B"}, ["class", "trainable", "kind: scripted"]),
    ({"path": "x.pt"}, ["path", "trainable", "kind: frozen"]),
    ({"algoritm": {}}, ["algoritm", "trainable"]),
    ({"kind": "bot"}, ["bot"]),
    ({"kind": "scripted"}, ["class"]),
    ({"kind": "scripted", "class": "NoDot"}, ["dotted"]),
    ({"kind": "frozen"}, ["path"]),
    ({"kind": "scripted", "class": "a.B", "roles": []}, ["empty"]),
    ({"kind": "frozen", "path": "x.pt", "roles": ["p", "p"]}, ["more than once"]),
    ({"matchmaking": {"anchors": []}}, ["not supported yet"]),
    ({"init": {"from": "bc.pt"}}, ["not supported yet"]),
    ({"kickstart": {"teacher": "x"}}, ["not supported yet"]),
])
def test_bad_agent_entries_are_config_errors_with_a_hint(tmp_path, entry, words):
    with pytest.raises(ConfigError) as info:
        load_config(_write(tmp_path, _data(agents={"main": {}, "x": entry})))
    for word in words:
        assert word in str(info.value), str(info.value)


def test_get_agent_config_is_for_trainable_agents_only():
    cfg = ColosseumConfig.model_validate(_data(agents=MIXED))
    with pytest.raises(ConfigError, match="'greedy' is a scripted agent"):
        cfg.get_agent_config("greedy")
    with pytest.raises(ConfigError, match="'prev' is a frozen agent"):
        cfg.get_agent_config("prev")
    with pytest.raises(ConfigError, match="Unknown agent"):
        cfg.agent_kind("nobody")


def test_set_reaches_the_fields_of_every_kind(tmp_path):
    path = _write(tmp_path, _data(agents={"main": {}, "greedy": {"kind": "scripted", "class": "a.B"}}))
    cfg = load_config(path, {
        "agents.greedy.kwargs.aggr": 0.9, "agents.greedy.class": "c.D", "agents.greedy.roles": ["player"],
        "agents.prev.kind": "frozen", "agents.prev.path": "old.pt", "agents.prev.networks.kwargs.hidden": 32,
        "agents.main.algorithm.learning_rate": 1.0e-4,
    })
    greedy = cfg.agent_entry("greedy")
    assert greedy.kwargs == {"aggr": 0.9} and greedy.class_path == "c.D" and greedy.roles == ["player"]
    assert cfg.agent_kind("prev") == "frozen" and cfg.agent_entry("prev").networks == {"kwargs": {"hidden": 32}}
    assert cfg.get_agent_config("main").algorithm.learning_rate == 1.0e-4
    raw = yaml.safe_load(path.read_text())
    for key in ("agents.greedy.clas", "agents.greedy.kind.x", "agents.main.trainin.x"):
        with pytest.raises(ConfigError, match="agents"):
            apply_overrides(raw, {key: 1})
