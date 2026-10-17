"""Tournament players (SP4 T5.1, spec blocks 2 and 3): entity names, kinds, snapshot selection (all, last, every,
top by the run journal's lower bound), candidates vs field, the lazy LRU pool, (de)serialization."""
from __future__ import annotations

import numpy as np
import pytest

from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog, RunInfo
from colosseum.arena.tournament.engine import LazyPlayerPool
from colosseum.arena.tournament.players import (
    PlayerRequest,
    RunSelection,
    config_player,
    entity_hash,
    parse_snapshots,
    player_from_dict,
    player_to_dict,
    resolve_players,
    select_snapshots,
    user_player,
)
from colosseum.coordinator.checkpoint_manager import read_snapshots
from colosseum.core.errors import ConfigError
from colosseum.core.registry import env_spec
from colosseum.core.types import MatchResult, SeatResult, TeamResult
from colosseum.players.registry import load_fixed_players
from colosseum.worker.match_runner import ScriptedPlayer
from game_helpers import make_test_config, scripted_agent, write_ckpt_dir, write_fake_run


def _cfg(**agents):
    return make_test_config("turns", agents={"main": {}, "rnd": scripted_agent(), **agents})


def test_user_and_config_players_have_the_entity_names_of_block_2(tmp_path):
    cfg = _cfg()
    spec = env_spec(cfg)
    ckpt = write_ckpt_dir(tmp_path / "x", cfg, "main", policy_version=5)
    p = user_player(cfg, spec, "prev", str(ckpt), validated=set())
    assert p.name == f"prev#{entity_hash(ckpt)}" and len(entity_hash(ckpt)) == 8 and p.display == "prev"
    assert (p.agent_id, p.network_id, p.kind, p.lineage) == (p.name, "fixed", "checkpoint", p.name)
    assert p.source.weights == str(ckpt / "model.pt") and p.roles == ("player",)
    bot = config_player(cfg, spec, "rnd", load_fixed_players(cfg, spec))
    assert (bot.name, bot.agent_id, bot.network_id, bot.kind) == ("rnd", "rnd", "fixed", "scripted")
    assert bot.source.bot.class_path == "colosseum.players.RandomBot"
    with pytest.raises(ConfigError, match="trainable"):
        config_player(cfg, spec, "main", load_fixed_players(cfg, spec))
    assert player_from_dict(player_to_dict(p)) == p and player_from_dict(player_to_dict(bot)) == bot


def test_snapshot_selection_rules(tmp_path):
    root, _cfg_ = write_fake_run(tmp_path / "run", versions=(10, 20, 30, 40, 50))
    infos = read_snapshots(root / "checkpoints" / "main")

    def pick(text, lcb=None):
        return [c.policy_version for c in select_snapshots(infos, *parse_snapshots(text), lcb=lcb)]

    assert pick("all") == [10, 20, 30, 40, 50] and pick("last:2") == [40, 50] and pick("every:2") == [10, 30, 50]
    assert pick("top:2", {"ckpt_v20": 3.0, "ckpt_v40": 9.0, "ckpt_v10": 1.0}) == [20, 40]
    for bad in ("last", "last:0", "top:x", "first:2"):
        with pytest.raises(ConfigError, match="--snapshots"):
            parse_snapshots(bad)


def _match(i, version, wins) -> MatchResult:
    """A 1v1 training match of main@v<version> against rnd; sides alternate."""
    side = i % 2
    seats = [None, None]
    seats[side] = SeatResult(side, "player", side, "main", "latest", 0.0, version_start=version, version_end=version)
    seats[1 - side] = SeatResult(1 - side, "player", 1 - side, "rnd", "fixed", 0.0)
    ranks = {side: 1.0 if wins else 2.0, 1 - side: 2.0 if wins else 1.0}
    return MatchResult(match_id=f"m{i}", layout="2p", outcome_kind="wdl", seats=seats,
                       teams=[TeamResult(t, ranks[t], 0.0) for t in (0, 1)], episode_length=6, uid=f"u{version}-{i}",
                       source="train", env_steps=i)


def test_top_snapshots_come_from_the_run_journal_and_need_it(tmp_path):
    root, cfg = write_fake_run(tmp_path / "run", versions=(10, 20, 30), agents={"main": {}, "rnd": scripted_agent()},
                               checkpoint={"interval": 10})
    spec = env_spec(cfg)
    fixed = load_fixed_players(cfg, spec)
    request = PlayerRequest(user=(), runs=(RunSelection(root, snapshots="top:1"),), field=None)
    with pytest.raises(ConfigError, match="match log"):
        resolve_players(cfg, spec, request, fixed=fixed, rng=np.random.default_rng(0))
    rng = np.random.default_rng(0)
    log = MatchLog(root / MATCH_LOG_FILE, RunInfo("r", "run", None, cfg.env.env_class, "h", 10))
    rates = {10: 0.2, 20: 0.85, 30: 0.5}
    log.append([_match(i, v, rng.random() < rate) for v, rate in rates.items() for i in range(300)])
    log.close()
    candidates, field = resolve_players(cfg, spec, request, fixed=fixed, rng=np.random.default_rng(1))
    assert [p.name for p in candidates] == ["main@v20"] and [p.name for p in field] == ["rnd"]


def test_candidates_and_field_aliases_and_explicit_field_members(tmp_path):
    first, cfg = write_fake_run(tmp_path / "a", versions=(10, 20), agents={"main": {}, "rnd": scripted_agent()})
    second, _ = write_fake_run(tmp_path / "b", versions=(5,), agents={"main": {}, "rnd": scripted_agent()})
    spec = env_spec(cfg)
    fixed = load_fixed_players(cfg, spec)
    ckpt = write_ckpt_dir(tmp_path / "x", cfg, "main", policy_version=7)
    request = PlayerRequest(user=(("prev", str(ckpt)), ("rnd", None)),
                            runs=(RunSelection(first), RunSelection(second, alias="old")),
                            field=("rnd", "main@v10", "old/main@v5"))
    candidates, field = resolve_players(cfg, spec, request, fixed=fixed)
    assert [p.name for p in candidates] == [f"prev#{entity_hash(ckpt)}", "main@v20"]
    assert [p.name for p in field] == ["rnd", "main@v10", "old/main@v5"]
    old = field[2]
    assert (old.agent_id, old.network_id, old.lineage, old.version, old.run_agent) == (
        "old/main", "ckpt_v5", "old/main", 5, "main")
    default = resolve_players(cfg, spec, PlayerRequest(user=(), runs=(RunSelection(first),), field=None), fixed=fixed)
    assert [p.name for p in default[1]] == ["rnd"]
    with pytest.raises(ConfigError, match="nobody"):
        resolve_players(cfg, spec, PlayerRequest(user=(), runs=(RunSelection(first),), field=("nobody",)), fixed=fixed)
    with pytest.raises(ConfigError, match="field is empty"):
        resolve_players(make_test_config("turns", agents={"main": {}}), spec,
                        PlayerRequest(user=(), runs=(RunSelection(first),), field=None), fixed=fixed)
    with pytest.raises(ConfigError, match="alias"):
        resolve_players(cfg, spec, PlayerRequest(user=(), runs=(RunSelection(first), RunSelection(second, alias="a/b")),
                                                 field=None), fixed=fixed)


def test_the_lazy_pool_loads_on_first_use_and_keeps_reserved_models(tmp_path):
    root, cfg = write_fake_run(tmp_path / "run", versions=(1, 2, 3), agents={"main": {}, "rnd": scripted_agent()})
    spec = env_spec(cfg)
    fixed = load_fixed_players(cfg, spec)
    candidates, field = resolve_players(cfg, spec, PlayerRequest(user=(), runs=(RunSelection(root),), field=None),
                                        fixed=fixed)
    pool = LazyPlayerPool([*candidates, *field], spec, capacity=1)
    assert pool.loads == 0 and isinstance(pool.get("rnd", "fixed"), ScriptedPlayer)
    pool.reserve({("main", "ckpt_v1"), ("main", "ckpt_v2")})
    assert pool.loads == 2 and set(pool.cached()) == {("main", "ckpt_v1"), ("main", "ckpt_v2")}
    assert not pool.get("main", "ckpt_v1").training                     # eval mode
    pool.get("main", "ckpt_v1")
    assert pool.loads == 2                                               # cached
    pool.reserve({("main", "ckpt_v3")})
    assert pool.cached() == [("main", "ckpt_v3")]                         # capacity 1: the others leave
    assert pool.get("nobody", "fixed") is None
    with pytest.raises(ValueError, match="nobody"):
        pool.reserve({("nobody", "fixed")})


def test_top_snapshots_survive_a_journal_write_between_reads(tmp_path, monkeypatch):
    """A live journal: a training write lands while ``top:N`` reads. The fit is built from the same bootstrap data
    it is resampled with, so ``bt.bootstrap``'s coverage check never sees two different states of the journal."""
    from colosseum.arena import bt
    from colosseum.arena.tournament.players import top_snapshot_lcb

    root, cfg = write_fake_run(tmp_path / "run", versions=(10, 20), agents={"main": {}, "rnd": scripted_agent()},
                               checkpoint={"interval": 10})
    info = RunInfo("r", "run", None, cfg.env.env_class, "h", 10)
    log = MatchLog(root / MATCH_LOG_FILE, info)
    log.append([_match(i, v, i % 3 != 0) for v in (10, 20) for i in range(40)])
    log.close()
    original = bt.bootstrap_data
    landed = []

    def write_then_read(reader, layout, *, sources):
        writer = MatchLog(root / MATCH_LOG_FILE, info)        # a second writer process: the training run
        writer.append([_match(1000 + len(landed), 20, True)])
        writer.close()
        landed.append(layout)
        return original(reader, layout, sources=sources)

    monkeypatch.setattr(bt, "bootstrap_data", write_then_read)
    lcb = top_snapshot_lcb(root, "main", cfg, env_spec(cfg), rng=np.random.default_rng(0))
    assert landed == ["2p"] and set(lcb) == {"ckpt_v10", "ckpt_v20"}
