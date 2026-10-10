"""``colosseum bc`` reads ``record`` output dirs and several --data paths (spec block 7, T5.2)."""
from __future__ import annotations

import json

import pytest
import torch
from click.testing import CliRunner

from colosseum.bc.offline_bc import OfflineBCTrainer, bc_data_sources
from colosseum.cli import main
from colosseum.core.errors import DataError
from colosseum.core.registry import build_model
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.record import RECORD_FILE, record
from game_helpers import agent_role_of, make_test_config, write_test_config

RANDOM_BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}
ASYM_AGENTS = {
    "hunter": {"roles": ["hunter"]},
    "prey": {"roles": ["prey"]},
    "preybot": {**RANDOM_BOT, "roles": ["prey"]},
    "hunterbot": {**RANDOM_BOT, "roles": ["hunter"]},
}

pytestmark = pytest.mark.usefixtures("restore_root_logging")


def bc(*args):
    return CliRunner().invoke(main, ["bc", *map(str, args)])


def test_sources_expand_record_dirs_into_the_agents_role_folders(tmp_path):
    rec = tmp_path / "rec"
    for role in ("hunter", "prey"):
        (rec / role).mkdir(parents=True)
    (rec / RECORD_FILE).write_text(json.dumps({"roles": {"hunter": {}, "prey": {}}}))
    plain = tmp_path / "plain"
    plain.mkdir()
    one = tmp_path / "one.pt"
    one.write_bytes(b"")
    assert bc_data_sources([rec, plain, one], ["prey"]) == [rec / "prey", plain, one]
    assert bc_data_sources([str(rec)], ["prey", "hunter"]) == [rec / "prey", rec / "hunter"]
    with pytest.raises(DataError, match="the agent plays"):
        bc_data_sources([rec], ["scout"])
    (rec / RECORD_FILE).write_text("{not json")
    with pytest.raises(DataError, match=RECORD_FILE):
        bc_data_sources([rec], ["prey"])


def test_bc_trains_on_a_record_dir(tmp_path):
    cfg = make_test_config("turns", agents={"bot": RANDOM_BOT})
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns", agents={"bot": RANDOM_BOT})
    record(cfg, "bot", [], layouts=None, num_matches=10, output=tmp_path / "rec", num_envs=4, seed=0)
    out = tmp_path / "bc.pt"
    result = bc("-c", cfg_path, "-d", tmp_path / "rec", "-o", out, "--epochs", 2, "--batch-size", 16)
    assert result.exit_code == 0, result.output
    assert "(agent_0): 60 decisions, final-epoch NLL" in result.output
    _roles, role = agent_role_of(cfg, "agent_0")
    build_model(cfg.get_agent_config("agent_0"), role).load_state_dict(torch.load(out, weights_only=True))


def test_bc_takes_several_data_paths(tmp_path):
    cfg = make_test_config("turns", agents={"bot": RANDOM_BOT})
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns", agents={"bot": RANDOM_BOT})
    for seed in (0, 1):
        record(cfg, "bot", [], layouts=None, num_matches=10, output=tmp_path / f"rec{seed}", num_envs=4, seed=seed)
    part = tmp_path / "rec1" / "player" / "part-00000.pt"
    result = bc("-c", cfg_path, "-d", tmp_path / "rec0", "-d", part, "-o", tmp_path / "bc.pt", "--epochs", 1)
    assert result.exit_code == 0, result.output
    assert "(agent_0): 120 decisions" in result.output


def test_bc_reads_only_the_agents_roles_of_a_record(tmp_path):
    cfg = make_test_config("asymmetric", agents=ASYM_AGENTS)
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "asymmetric", agents=ASYM_AGENTS)
    record(cfg, "preybot", ["hunterbot"], layouts=None, num_matches=4, output=tmp_path / "rec", seed=0)
    result = bc("-c", cfg_path, "-d", tmp_path / "rec", "-o", tmp_path / "prey.pt", "--agent", "prey", "--epochs", 1)
    assert result.exit_code == 0, result.output
    assert "(prey): 40 decisions" in result.output
    result = bc("-c", cfg_path, "-d", tmp_path / "rec", "-o", tmp_path / "hunter.pt", "--agent", "hunter")
    assert result.exit_code == 1
    assert result.stderr.startswith("Config error:") and "prey" in result.stderr


def test_a_record_split_into_several_part_files_loads_whole(tmp_path):
    cfg = make_test_config("turns", agents={"bot": RANDOM_BOT})
    out = tmp_path / "rec"
    # 10 matches x 2 seats x 3 decisions; a part file is written once 7 decisions are pending,
    # i.e. after every third seat-episode (9 decisions), the last one holds the remaining 6
    content = record(cfg, "bot", [], layouts=None, num_matches=10, output=out, num_envs=4, seed=0,
                     decisions_per_file=7)
    names = [f"part-{i:05d}.pt" for i in range(7)]
    assert sorted(p.name for p in (out / "player").iterdir()) == names
    assert content["roles"]["player"]["files"] == names
    assert json.loads((out / RECORD_FILE).read_text())["roles"]["player"]["files"] == names
    total = 0
    for name in names:
        data = torch.load(out / "player" / name, weights_only=True)
        n = data["dones"].shape[0]
        assert n in (6, 9) and bool(data["dones"][-1])           # whole seat-episodes only
        assert data["dones"].nonzero().flatten().tolist() == list(range(2, n, 3))
        obs = data["observations"]
        for start in range(0, n, 3):                               # one seat per episode, time increasing
            episode = obs[start:start + 3]
            assert bool((episode[:, 1] == episode[0, 1]).all()) and bool((episode[1:, 0] > episode[:-1, 0]).all())
        total += n
    assert total == content["decisions"] == 60
    _roles, role = agent_role_of(cfg, "agent_0")
    trainer = OfflineBCTrainer(build_model(cfg.get_agent_config("agent_0"), role),
                               ActionSpec.from_space(role.action_space), ObsSpec.from_space(role.observation_space))
    for source in bc_data_sources([out], ["player"]):
        trainer.load_data(source)
    assert trainer.num_samples == 60
