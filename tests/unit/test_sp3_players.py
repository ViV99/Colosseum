"""colosseum.players (SP3 T1.2, spec blocks 1-3): ScriptedBot, RandomBot, bot RNG, the legality gate,
roles of every agent kind, fixed-player loading and the shared frozen loader."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
import torch

from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.config import NetworkConfig, load_config
from colosseum.core.errors import ColosseumError, ConfigError, PlayerError
from colosseum.core.registry import build_model, env_spec
from colosseum.core.roles import role_signature
from colosseum.envs.contract import EpisodeTracker
from colosseum.players import RandomBot, ScriptedBot
from colosseum.players.registry import (
    BotSpec,
    FixedPlayers,
    build_frozen_model,
    load_fixed_players,
    load_frozen,
    make_bot,
    resolve_player_roles,
)
from colosseum.players.scripted import bot_rng, check_bot_action
from game_helpers import (
    SP2_CHECKPOINT,
    SP2_TTT_TINY,
    ConstantBot,
    TurnTakingGame,
    UnitsGame,
    agent_role_of,
    frozen_agent,
    make_test_config,
    scripted_agent,
)

WIDE = {"model_class": "game_helpers.GameTestModel", "kwargs": {"core": "none", "hidden": 32}}


def numpy_state(model) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def save_checkpoint(base_dir, agent_id, config, trainable="agent_0", networks=None, signature=None):
    """A signed SP2-style checkpoint dir with ``trainable``'s roles in the game of ``config``; the model is
    built from ``networks`` (a raw networks dict, default WIDE), which also goes into meta.json."""
    roles, role = agent_role_of(config, trainable)
    nets = WIDE if networks is None else networks
    model = build_model(config.model_copy(update={"networks": NetworkConfig.model_validate(nets)}), role)
    ckpt_id = CheckpointManager(base_dir).save(agent_id, 2, numpy_state(model), meta_extra={
        "networks": nets, "roles": roles, "role_signature": signature or role_signature(role)})
    return base_dir / agent_id / ckpt_id


def test_player_error_is_a_colosseum_error():
    assert issubclass(PlayerError, ColosseumError) and not issubclass(PlayerError, ConfigError)


def test_scripted_bot_is_abstract_and_reset_is_a_no_op():
    with pytest.raises(TypeError):
        ScriptedBot()

    class Echo(ScriptedBot):
        def act(self, obs, mask, info):
            return 0

    assert Echo().reset(role="player", seat=0, layout="2p", rng=np.random.default_rng(0)) is None


def test_bot_rng_depends_on_episode_seed_seat_and_agent():
    draw = bot_rng(123, 0, "rnd").random(3)
    assert np.array_equal(draw, bot_rng(123, 0, "rnd").random(3))
    for other in (bot_rng(123, 1, "rnd"), bot_rng(124, 0, "rnd"), bot_rng(123, 0, "greedy")):
        assert not np.array_equal(draw, other.random(3))
    assert not np.array_equal(bot_rng(None, 0, "rnd").random(3), bot_rng(None, 0, "rnd").random(3))


@pytest.mark.parametrize("game_cls", [TurnTakingGame, UnitsGame])
def test_random_bot_plays_legal_actions_that_pass_the_gate(game_cls):
    game = game_cls()
    layout = next(iter(game.spec.layouts))
    bots = {}
    for seat, seat_spec in enumerate(game.spec.layouts[layout]):
        bots[seat] = make_bot(BotSpec("colosseum.players.RandomBot", {}), game.spec)
        assert isinstance(bots[seat], RandomBot)
        bots[seat].reset(role=seat_spec.role, seat=seat, layout=layout, rng=bot_rng(7, seat, "rnd"))
    tracker = EpisodeTracker(game.spec)
    result = game.reset(0, layout)
    masks = tracker.on_reset(layout, result)
    chosen = []
    while not result.episode_over:
        actions = {}
        for seat in sorted(result.acting):
            role = game.spec.roles[game.spec.role_of(layout, seat)]
            actions[seat] = check_bot_action(role, bots[seat].act(result.obs[seat], masks[seat], None), masks[seat],
                                             "test")
            chosen.append(actions[seat])
        result = game.step(actions)
        masks = tracker.on_step(actions, result)
    assert chosen
    if game_cls is TurnTakingGame:
        assert all(int(a) in (0, 1) for a in chosen)   # action 2 is always masked


def test_check_bot_action_casts_and_rejects_bad_actions():
    turns = TurnTakingGame().spec.roles["player"]
    mask = np.array([True, False, True])
    out = check_bot_action(turns, 2, mask, "w")
    assert isinstance(out, np.ndarray) and out.dtype == np.int64 and out.shape == () and int(out) == 2
    with pytest.raises(PlayerError, match=r"^w: illegal action: action 1 at <root> is illegal"):
        check_bot_action(turns, 1, mask, "w")
    with pytest.raises(PlayerError, match="not in the role's action space"):
        check_bot_action(turns, 5, None, "w")
    with pytest.raises(PlayerError, match="structure"):
        check_bot_action(turns, {"move": 1}, None, "w")
    with pytest.raises(PlayerError, match="integer"):
        check_bot_action(turns, 1.5, None, "w")
    with pytest.raises(PlayerError, match="shape"):
        check_bot_action(turns, [1, 2], None, "w")
    with pytest.raises(PlayerError, match="None"):
        check_bot_action(turns, None, None, "w")

    units = UnitsGame()
    role = units.spec.roles["player"]
    masks = EpisodeTracker(units.spec).on_reset("solo", units.reset(0, "solo"))   # only unit 0 exists
    legal = {"units": {"target": [0, 0, 0, 0], "move": [3, 0, 0, 0]}, "base": 1}   # key order does not matter
    cast = check_bot_action(role, legal, masks[0], "w")
    assert list(cast) == ["base", "units"] and cast["units"]["move"].dtype == np.int64
    illegal = {"base": 1, "units": {"move": [3, 0, 0, 0], "target": [3, 0, 0, 0]}}
    with pytest.raises(PlayerError, match="unit 0 component 'target'"):
        check_bot_action(role, illegal, masks[0], "w")


def test_make_bot_imports_constructs_and_sets_the_game_spec():
    spec = TurnTakingGame().spec
    bot = make_bot(BotSpec("game_helpers.ConstantBot", {"value": 1}), spec)
    assert isinstance(bot, ConstantBot) and bot.game_spec is spec and bot.value == 1
    with pytest.raises(ConfigError, match="cannot be imported"):
        make_bot(BotSpec("no_such_module.Bot", {}), spec)
    with pytest.raises(ConfigError, match="must subclass"):
        make_bot(BotSpec("game_helpers.TurnTakingGame", {}), spec)
    with pytest.raises(ConfigError, match="kwargs"):
        make_bot(BotSpec("game_helpers.ConstantBot", {"nope": 1}), spec)
    assert pickle.loads(pickle.dumps(BotSpec("a.B", {"x": 1}))) == BotSpec("a.B", {"x": 1})


def test_resolve_player_roles_covers_every_kind_in_config_order(tmp_path):
    base = make_test_config("asymmetric")
    old_hunter = save_checkpoint(tmp_path / "ck", "hunter", base, trainable="hunter",
                                 networks=base.networks.model_dump(mode="json", by_alias=True))
    _roles, prey_role = agent_role_of(base, "prey")
    prey_pt = tmp_path / "prey.pt"
    torch.save(build_model(base.get_agent_config("prey"), prey_role).state_dict(), prey_pt)
    cfg = make_test_config("asymmetric", agents={
        "rnd": scripted_agent(), "hunter": {"roles": ["hunter"]}, "prey": {"roles": ["prey"]},
        "prey_bot": scripted_agent(roles=["prey"]), "old_hunter": frozen_agent(str(old_hunter)),
        "old_prey": frozen_agent(str(prey_pt), roles=["prey"]),
    })
    spec = env_spec(cfg)
    assert resolve_player_roles(cfg, spec) == {
        "rnd": ["hunter", "prey"], "hunter": ["hunter"], "prey": ["prey"], "prey_bot": ["prey"],
        "old_hunter": ["hunter"], "old_prey": ["prey"],
    }
    for agents, message in (
        ({"h": {"roles": ["hunter"]}, "p": {"roles": ["prey"]}, "z": frozen_agent(str(prey_pt))}, "different spaces"),
        ({"h": {"roles": ["hunter"]}, "p": {"roles": ["prey"]}, "z": scripted_agent(roles=["wolf"])}, "unknown roles"),
        ({"h": {"roles": ["hunter"]}, "p": {"roles": ["prey"]}, "z": frozen_agent(str(tmp_path / "nope"))},
         "expected a checkpoint dir"),
    ):
        with pytest.raises(ConfigError, match=message):
            resolve_player_roles(make_test_config("asymmetric", agents=agents), spec)


def test_load_fixed_players_loads_bots_and_frozen_weights_of_their_own_architecture(tmp_path):
    base = make_test_config("turns")
    wide_dir = save_checkpoint(tmp_path / "ck", "wide_agent", base)
    _roles, role = agent_role_of(base, "agent_0")
    mid_pt = tmp_path / "mid.pt"
    mid_cfg = make_test_config("turns", networks={"kwargs": {"core": "none", "hidden": 24}})
    torch.save(build_model(mid_cfg.get_agent_config("agent_0"), role).state_dict(), mid_pt)
    cfg = make_test_config("turns", agents={
        "agent_0": {}, "rnd": scripted_agent(), "wide": frozen_agent(str(wide_dir)),
        "mid": frozen_agent(str(mid_pt), networks={"kwargs": {"hidden": 24}}),
    })
    spec = env_spec(cfg)
    fixed = load_fixed_players(cfg, spec)
    assert isinstance(fixed, FixedPlayers)
    assert fixed.bots == {"rnd": BotSpec("colosseum.players.RandomBot", {})}
    assert list(fixed.frozen) == ["wide", "mid"]
    assert fixed.roles == {"rnd": ("player",), "wide": ("player",), "mid": ("player",)}
    wide = fixed.frozen["wide"]
    assert wide.agent_id == "wide" and wide.roles == ("player",) and wide.source == str(wide_dir)
    assert wide.networks["kwargs"]["hidden"] == 32
    assert wide.networks_source == f"Checkpoint {wide_dir}: meta.json networks"
    assert all(isinstance(v, np.ndarray) for v in wide.model_state.values())
    model = build_frozen_model(None, wide, spec)          # the architecture comes from the spec alone
    assert not model.training
    for key, value in model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), wide.model_state[key])
    mid = build_frozen_model(cfg, fixed.frozen["mid"], spec)
    assert fixed.frozen["mid"].networks_source == f"{mid_pt}: networks"
    assert sum(p.numel() for p in mid.parameters()) < sum(p.numel() for p in model.parameters())
    again = pickle.loads(pickle.dumps(fixed))             # crosses process boundaries as Process kwargs
    assert again.bots == fixed.bots and list(again.frozen) == ["wide", "mid"]


def test_frozen_agent_errors_name_the_agent_or_the_path(tmp_path):
    base = make_test_config("turns")
    good = save_checkpoint(tmp_path / "ck", "wide_agent", base)
    other = save_checkpoint(tmp_path / "ck2", "other_agent", base, signature="obs=other-game")
    spec = env_spec(base)
    with pytest.raises(ConfigError, match="meta.json gives"):
        load_frozen(make_test_config("turns", agents={"agent_0": {}, "w": frozen_agent(str(good), roles=["player"])}),
                    "w", str(good), spec)
    with pytest.raises(ConfigError, match="role signature"):
        load_frozen(base, "x", str(other), spec)
    with pytest.raises(ConfigError, match="scripted agent"):
        load_frozen(make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent()}), "rnd", str(good), spec)
    narrow_pt = tmp_path / "narrow.pt"
    torch.save(torch.load(good / "model.pt", weights_only=True), narrow_pt)   # wide weights, default networks
    frozen = load_frozen(base, "x", str(narrow_pt), spec)
    with pytest.raises(ConfigError, match="do not match"):
        build_frozen_model(base, frozen, spec)
    meta = json.loads((good / "meta.json").read_text())
    (good / "meta.json").write_text(json.dumps({**meta, "networks": {"bogus_key": 1}}))
    with pytest.raises(ConfigError, match="invalid meta.json networks"):
        load_frozen(base, "x", str(good), spec)


def test_the_sp2_checkpoint_fixture_loads_as_a_frozen_agent():
    cfg = load_config(SP2_TTT_TINY, {"agents.old.kind": "frozen", "agents.old.path": str(SP2_CHECKPOINT)})
    spec = env_spec(cfg)
    assert cfg.get_trainable_agent_ids() == ["agent_0"] and cfg.fixed_agent_ids() == ["old"]
    fixed = load_fixed_players(cfg, spec)
    assert fixed.roles == {"old": ("player",)}
    model = build_frozen_model(cfg, fixed.frozen["old"], spec)
    obs = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
    out = model.step(obs, model.initial_state(1), torch.ones((1, 9), dtype=torch.bool))
    assert tuple(out.dist.sample().shape) == (1,)


def test_a_frozen_checkpoint_dir_with_roles_or_networks_fails_already_when_resolving_roles(tmp_path):
    base = make_test_config("turns")
    good = save_checkpoint(tmp_path / "ck", "wide_agent", base)
    for entry in (frozen_agent(str(good), roles=["player"]), frozen_agent(str(good), networks={"kwargs": {}})):
        cfg = make_test_config("turns", agents={"agent_0": {}, "w": entry})
        with pytest.raises(ConfigError, match=r"agents\.w: .*meta.json gives"):
            resolve_player_roles(cfg, env_spec(cfg))


def test_check_bot_action_rejects_a_ragged_component():
    turns = TurnTakingGame().spec.roles["player"]
    with pytest.raises(PlayerError, match="^w: an action component is not an array of one shape"):
        check_bot_action(turns, [[1, 2], [3]], None, "w")
