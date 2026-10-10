"""``colosseum record`` (spec block 7, T5.1): lineups with and without --against, per-seat contiguous
seat-episodes per role in the BC format, record.json, discarded episodes, errors."""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from gymnasium.spaces import Box, Discrete

from colosseum.bc.offline_bc import OfflineBCTrainer
from colosseum.cli import main
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.envs.game import GameSpec, RoleSpec, SeatSpec
from colosseum.eval import load_player, play_lineups
from colosseum.record import RECORD_FILE, _schedule, record
from game_helpers import RandomPolicy, TurnTakingGame, agent_role_of, make_test_config, write_test_config

RANDOM_BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}
ASYM_AGENTS = {
    "hunter": {"roles": ["hunter"]},
    "prey": {"roles": ["prey"]},
    "preybot": {**RANDOM_BOT, "roles": ["prey"]},
    "hunterbot": {**RANDOM_BOT, "roles": ["hunter"]},
}


def _turns(**agents):
    """TurnTakingGame (6 moves, seats alternate: 3 decisions per seat and episode) with RandomBot agents."""
    return make_test_config("turns", agents=agents or {"bot": RANDOM_BOT})


def _part(directory) -> dict:
    files = sorted(directory.glob("part-*.pt"))
    assert [f.name for f in files] == ["part-00000.pt"], files
    return torch.load(files[0], weights_only=True)


def _seat_episodes(data) -> list[torch.Tensor]:
    """Observations split after every ``done`` (one piece per seat-episode)."""
    dones = data["dones"]
    assert bool(dones[-1]), "a part file ends with a finished seat-episode"
    ends = torch.nonzero(dones).flatten().tolist()
    starts = [0] + [e + 1 for e in ends[:-1]]
    return [data["observations"][s:e + 1] for s, e in zip(starts, ends, strict=True)]


def test_a_bot_against_itself_records_every_seat_contiguously(tmp_path):
    out = tmp_path / "data"
    content = record(_turns(), "bot", [], layouts=None, num_matches=10, output=out, num_envs=4, seed=0)
    data = _part(out / "player")
    # 10 matches x 2 seats x 3 decisions; with 4 envs for 10 lineups some envs play extra
    # episodes after the last lineup started: the eval engine discards them, so does record
    assert data["observations"].shape == (60, 3) and data["observations"].dtype == torch.float32
    assert data["actions"].shape == (60,) and data["actions"].dtype == torch.int64
    assert data["action_masks"].shape == (60, 3) and data["action_masks"].dtype == torch.bool
    assert bool(data["action_masks"][torch.arange(60), data["actions"]].all())     # every action legal
    episodes = _seat_episodes(data)
    assert len(episodes) == 20
    for obs in episodes:                    # obs = [t / length, seat, 1]: one seat, time increasing
        assert obs.shape[0] == 3
        assert bool((obs[:, 1] == obs[0, 1]).all())
        assert bool((obs[1:, 0] > obs[:-1, 0]).all())
    assert content["player"] == "bot" and content["player_source"] is None and content["against"] == []
    assert (content["matches"], content["seat_episodes"], content["decisions"]) == (10, 20, 60)
    assert content["roles"] == {"player": {"seat_episodes": 20, "decisions": 60, "files": ["part-00000.pt"]}}
    assert content["layouts"] == ["2p"] and content["summary"]["layouts"]["2p"]["n"] == 10
    saved = json.loads((out / RECORD_FILE).read_text())
    assert {k: saved[k] for k in ("format", "player", "against", "decisions", "roles")} == {
        "format": 1, "player": "bot", "against": [], "decisions": 60, "roles": content["roles"]}


def test_against_records_only_the_players_seats(tmp_path):
    out = tmp_path / "data"
    content = record(_turns(bot=RANDOM_BOT, other=RANDOM_BOT), "bot", ["other"], layouts=None, num_matches=10,
                     output=out, num_envs=4, seed=0)
    data = _part(out / "player")
    assert data["observations"].shape[0] == 30 and int(data["dones"].sum()) == 10
    assert {int(obs[0, 1]) for obs in _seat_episodes(data)} == {0, 1}     # the pair rotates over the sides
    assert content["against"] == ["other"] and content["seat_episodes"] == 10
    (pair,) = [r for r in content["summary"]["layouts"]["2p"]["pairs"] if r["agent_a"] == "bot"]
    assert pair["agent_b"] == "other" and pair["n"] == 10


def test_a_player_without_every_role_of_a_layout_needs_against(tmp_path):
    config = make_test_config("asymmetric", agents=ASYM_AGENTS)
    with pytest.raises(ConfigError, match="--against"):
        record(config, "preybot", [], layouts=None, num_matches=2, output=tmp_path / "a", seed=0)
    with pytest.raises(ConfigError, match="--against"):
        record(config, "preybot", [], layouts=["1v2"], num_matches=2, output=tmp_path / "b", seed=0)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    content = record(config, "preybot", ["hunterbot"], layouts=None, num_matches=4, output=tmp_path / "c", seed=0)
    assert sorted(p.name for p in (tmp_path / "c").iterdir()) == ["prey", RECORD_FILE]
    data = _part(tmp_path / "c" / "prey")
    assert data["observations"].shape == (40, 3)            # 4 matches x 2 prey seats x 5 steps
    assert int(data["dones"].sum()) == 8
    # Discrete(3) has a mask group: the env sends none, so every recorded row is the normalized full mask
    assert data["action_masks"].shape == (40, 3) and bool(data["action_masks"].all())
    assert content["roles"]["prey"]["seat_episodes"] == 8 and "hunter" not in content["roles"]


def test_units_actions_and_uint8_observations_load_into_the_bc_trainer(tmp_path):
    config = make_test_config("units", agents={"bot": RANDOM_BOT})
    out = tmp_path / "data"
    record(config, "bot", [], layouts=None, num_matches=3, output=out, num_envs=2, seed=0)
    data = _part(out / "player")
    assert data["observations"]["grid"].dtype == torch.uint8          # the role's dtypes are kept
    assert data["observations"]["entity_mask"].dtype == torch.int8
    assert data["actions"]["base"].shape == (18,)                     # 3 episodes x 6 steps
    assert data["actions"]["units"]["move"].shape == (18, 4)          # U = 4, ActionSpec.allocate_actions layout
    assert data["action_masks"]["units"]["action"].dtype == torch.bool
    _roles, role = agent_role_of(config, "agent_0")
    model = build_model(config.get_agent_config("agent_0"), role)
    trainer = OfflineBCTrainer(model, ActionSpec.from_space(role.action_space),
                               ObsSpec.from_space(role.observation_space))
    assert trainer.load_data(out / "player") == 18                    # shapes, dtypes, legality under the masks


def test_a_frozen_agent_by_name_and_a_pt_as_name_equals_path(tmp_path):
    base = make_test_config("turns")
    _roles, role = agent_role_of(base, "agent_0")
    torch.manual_seed(0)
    pt = tmp_path / "net.pt"
    torch.save(build_model(base.get_agent_config("agent_0"), role).state_dict(), pt)
    config = make_test_config("turns", agents={"fz": {"kind": "frozen", "path": str(pt)}})
    by_name = record(config, "fz", [], layouts=None, num_matches=2, output=tmp_path / "a", seed=0)
    by_path = record(config, f"net={pt}", [], layouts=None, num_matches=2, output=tmp_path / "b", seed=0)
    assert by_name["decisions"] == by_path["decisions"] == 12
    assert by_path["player"] == "net" and by_path["player_source"] == str(pt)


def test_bad_players_and_a_non_empty_output_are_config_errors(tmp_path):
    config = _turns(bot=RANDOM_BOT)
    with pytest.raises(ConfigError, match=r"trainable agent.*--player agent_0=<checkpoint dir or \.pt>"):
        record(config, "agent_0", [], layouts=None, num_matches=1, output=tmp_path / "a")
    with pytest.raises(ConfigError, match="more than once"):
        record(config, "bot", ["bot"], layouts=None, num_matches=1, output=tmp_path / "b")
    with pytest.raises(ConfigError, match="nobody"):
        record(config, "nobody", [], layouts=None, num_matches=1, output=tmp_path / "c")
    with pytest.raises(ConfigError, match="not layouts of the game"):
        record(config, "bot", [], layouts=["9p"], num_matches=1, output=tmp_path / "d")
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "old.txt").write_text("x")
    with pytest.raises(ConfigError, match="not empty"):
        record(config, "bot", [], layouts=None, num_matches=1, output=tmp_path / "e")


def test_a_path_that_is_neither_a_dir_nor_a_pt_is_a_config_error(tmp_path):
    """Pre-flight ruling P7: ``load_player`` checks an explicit path itself (no FileNotFoundError)."""
    config = _turns()
    (tmp_path / "w.txt").write_text("x")
    for path in (tmp_path / "missing", tmp_path / "w.txt"):
        with pytest.raises(ConfigError, match=r"expected a checkpoint dir or a \.pt file"):
            record(config, f"net={path}", [], layouts=None, num_matches=1, output=tmp_path / "out")
        with pytest.raises(ConfigError, match=r"^--against net="):
            load_player(config, "net", str(path), spec=TurnTakingGame().spec, option="--against")
    assert not (tmp_path / "out").exists()


def test_default_layouts_the_player_cannot_fill_alone_are_skipped_with_one_info_line(caplog):
    """Pre-flight ruling P8: without --against and --layout, unfillable layouts are skipped and named once."""
    vec = Box(-np.inf, np.inf, (3,), dtype=np.float32)
    spec = GameSpec(roles={"hunter": RoleSpec(vec, Discrete(3)), "prey": RoleSpec(vec, Discrete(3))},
                    layouts={"1v2": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1)),
                             "2h": (SeatSpec("hunter", 0), SeatSpec("hunter", 1)),
                             "pp": (SeatSpec("prey", 0), SeatSpec("prey", 1))})
    with caplog.at_level(logging.INFO, logger="colosseum.record"):
        lineups = _schedule(spec, "bot", {"bot": ["prey"]}, [], None, 3)
    assert {lineup.layout for lineup in lineups} == {"pp"} and len(lineups) == 3
    lines = [r.getMessage() for r in caplog.records if r.name == "colosseum.record"]
    assert len(lines) == 1 and "['1v2', '2h']" in lines[0] and "--against" in lines[0]
    with pytest.raises(ConfigError, match="add --against"):
        _schedule(spec, "bot", {"bot": ["nobody"]}, [], None, 3)
    with pytest.raises(ConfigError, match="--against"):
        _schedule(spec, "bot", {"bot": ["prey"]}, [], ["pp", "2h"], 3)


def test_a_seed_makes_the_recording_reproducible(tmp_path):
    for name in ("a", "b"):
        record(_turns(), "bot", [], layouts=None, num_matches=6, output=tmp_path / name, num_envs=3, seed=7)
    a, b = _part(tmp_path / "a" / "player"), _part(tmp_path / "b" / "player")
    assert all(torch.equal(a[k], b[k]) for k in ("observations", "actions", "action_masks", "dones"))


class _CountingObserver:
    def __init__(self) -> None:
        self.acts = self.ends = self.discarded = 0

    def on_act(self, env, seat, record) -> None:
        self.acts += 1

    def on_rewards(self, env, rewards) -> None:
        pass

    def on_terminated(self, env, seats) -> None:
        pass

    def on_lineup_applied(self, env, old, new) -> None:
        pass

    def on_episode_end(self, env, end) -> None:
        self.ends += 1

    def on_episode_discarded(self, env) -> None:
        self.discarded += 1


def test_play_lineups_observer_sees_scheduled_episodes_and_the_discarded_rest():
    role = TurnTakingGame().spec.roles["player"]
    observer = _CountingObserver()
    lineups = [Lineup("2p", [SeatAssignment("r"), SeatAssignment("r")]) for _ in range(3)]
    results = play_lineups(env_fn=TurnTakingGame, models={"r": RandomPolicy(role)}, lineups=lineups, num_envs=2,
                           seed=0, observer=observer)
    # Both envs end their first episode at step 6: env 0 gets the third lineup, env 1 has none left
    # and plays an unscheduled episode that ends together with env 0's at step 12.
    assert len(results) == 3
    assert (observer.ends, observer.discarded, observer.acts) == (3, 1, 24)


def test_record_cli(tmp_path, restore_root_logging):
    cfg = write_test_config(tmp_path / "cfg.yaml", "turns", agents={"bot": RANDOM_BOT})
    out = tmp_path / "data"
    result = CliRunner().invoke(main, ["record", "-c", str(cfg), "--player", "bot", "-n", "4", "-o", str(out),
                                       "--seed", "0", "--num-envs", "2"])
    assert result.exit_code == 0, result.output
    assert "Recorded 24 decisions" in result.output and "player: 8 seat-episodes" in result.output
    assert (out / "player" / "part-00000.pt").is_file() and (out / RECORD_FILE).is_file()
    result = CliRunner().invoke(main, ["record", "-c", str(cfg), "--player", "nobody", "-o", str(tmp_path / "x")])
    assert result.exit_code == 1 and result.stderr.startswith("Config error:")
    result = CliRunner().invoke(main, ["record", "-c", str(cfg), "--player", "bot", "-o", str(tmp_path / "y"),
                                       "--num-matches", "0"])
    assert result.exit_code == 2


def test_an_odd_num_matches_against_opponents_is_rounded_up_on_two_team_layouts(tmp_path, caplog):
    config = _turns(bot=RANDOM_BOT, other=RANDOM_BOT)
    with caplog.at_level(logging.INFO, logger="colosseum.record"):
        content = record(config, "bot", ["other"], layouts=None, num_matches=5, output=tmp_path / "a",
                         num_envs=2, seed=0)
    notes = [r for r in caplog.records if "rounded up" in r.getMessage()]
    assert len(notes) == 1 and notes[0].levelno == logging.INFO and "using 6" in notes[0].getMessage()
    assert (content["num_matches"], content["matches"], content["seat_episodes"]) == (6, 6, 6)
    data = _part(tmp_path / "a" / "player")
    assert sorted(int(obs[0, 1]) for obs in _seat_episodes(data)) == [0, 0, 0, 1, 1, 1]   # balanced sides
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="colosseum.record"):     # without --against: no pairs, no rounding
        content = record(_turns(), "bot", [], layouts=None, num_matches=5, output=tmp_path / "b", num_envs=2, seed=0)
    assert content["num_matches"] == 5 and content["matches"] == 5
    assert not [r for r in caplog.records if "rounded up" in r.getMessage()]
