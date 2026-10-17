"""Tournament players (SP4 T5.1, spec blocks 2 and 3): entity names, kinds, snapshot selection (all, last, every,
top by the run journal's lower bound), candidates vs field, the lazy LRU pool, (de)serialization."""
from __future__ import annotations

from pathlib import Path

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


def test_top_snapshots_rank_by_the_lower_bound_not_the_mean(tmp_path, monkeypatch):
    """``top:N`` follows the lower bound of the bootstrap, not the BT mean. Deterministic wiring check:
    ``bt.bootstrap`` is replaced by one whose lower bounds reverse the fit's order (a real-data inversion of mean
    and lower bound is not robust over bootstrap seeds: ~1.8 % failures for the best small design found)."""
    from colosseum.arena import bt

    root, cfg = write_fake_run(tmp_path / "run", versions=(10, 20, 30),
                               agents={"main": {}, "rnd": scripted_agent()}, checkpoint={"interval": 10})
    spec = env_spec(cfg)
    log = MatchLog(root / MATCH_LOG_FILE, RunInfo("r", "run", None, cfg.env.env_class, "h", 10))
    plan = {10: (100, 30), 20: (100, 80), 30: (100, 55)}            # version: (games, wins)
    log.append([_match(i, v, i < wins) for v, (n, wins) in plan.items() for i in range(n)])
    log.close()
    seen = []

    def reversed_bounds(fit_, data, **kwargs):
        seen.append(dict(fit_.rating))
        lcb = {entity: -rating for entity, rating in fit_.rating.items()}
        return bt.Uncertainty(ci95={}, lcb=lcb, p_best_lineage={}, p_best_group={}, p_top_k={}, p_top_k_group={})

    monkeypatch.setattr(bt, "bootstrap", reversed_bounds)
    request = PlayerRequest(user=(), runs=(RunSelection(root, snapshots="top:1"),), field=None)
    candidates, _field = resolve_players(cfg, spec, request, fixed=load_fixed_players(cfg, spec),
                                         rng=np.random.default_rng(0))
    (rating,) = seen
    snapshots = sorted(("main@v10", "main@v20", "main@v30"), key=rating.get)
    assert snapshots == ["main@v10", "main@v30", "main@v20"]          # the mean prefers v20
    assert [p.name for p in candidates] == ["main@v10"]               # the (reversed) lower bound wins


def test_top_snapshots_of_a_score_game_use_the_mean_score_bound(tmp_path):
    """A one-team (``score``) game: ``top:N`` ranks by the lower bound of the mean team score."""
    root, cfg = write_fake_run(tmp_path / "run", "solo", versions=(10, 20, 30), checkpoint={"interval": 10})
    spec = env_spec(cfg)
    rng = np.random.default_rng(0)
    means = {10: 1.0, 20: 3.0, 30: 2.0}
    results = []
    for version, mean in means.items():
        for i in range(30):
            score = float(mean + rng.normal(0.0, 0.3))
            results.append(MatchResult(
                match_id=f"s{version}-{i}", layout="solo", outcome_kind="score",
                seats=[SeatResult(0, "player", 0, "main", "latest", score, version_start=version, version_end=version)],
                teams=[TeamResult(0, 1.0, score)], episode_length=3, uid=f"s{version}-{i}", source="train",
                env_steps=i))
    log = MatchLog(root / MATCH_LOG_FILE, RunInfo("r", "run", None, cfg.env.env_class, "h", 10))
    log.append(results)
    log.close()
    request = PlayerRequest(user=(), runs=(RunSelection(root, snapshots="top:2"),), field=None)
    candidates, field = resolve_players(cfg, spec, request, fixed=load_fixed_players(cfg, spec),
                                        rng=np.random.default_rng(0), allow_empty_field=True)
    assert [p.name for p in candidates] == ["main@v20", "main@v30"] and field == []


def test_an_a_player_named_like_a_config_agent_stays_a_candidate(tmp_path):
    """``-a bc=other`` next to the config's frozen ``bc`` (default field): both play, under distinct names and,
    since their displays would collide, the -a player shows its full name."""
    from game_helpers import frozen_agent

    base = _cfg()
    own = write_ckpt_dir(tmp_path / "own", base, "main", policy_version=3)
    other = write_ckpt_dir(tmp_path / "other", base, "main", policy_version=4, seed=1)
    cfg = _cfg(bc=frozen_agent(own))
    spec = env_spec(cfg)
    request = PlayerRequest(user=(("bc", str(other)),), runs=(), field=None)
    candidates, field = resolve_players(cfg, spec, request, fixed=load_fixed_players(cfg, spec))
    key = f"bc#{entity_hash(other)}"
    assert [(p.name, p.display) for p in candidates] == [(key, key)]
    assert [(p.name, p.display) for p in field] == [("rnd", "rnd"), ("bc", "bc")]
    alone = resolve_players(cfg, spec, PlayerRequest(user=(("x", str(other)),), runs=(), field=None),
                            fixed=load_fixed_players(cfg, spec))[0]
    assert [(p.name, p.display) for p in alone] == [(f"x#{entity_hash(other)}", "x")]   # no collision: short


def test_repeated_names_and_aliases_are_refused(tmp_path, monkeypatch):
    first, cfg = write_fake_run(tmp_path / "a" / "run", versions=(10,), agents={"main": {}, "rnd": scripted_agent()})
    second, _ = write_fake_run(tmp_path / "b" / "run", versions=(5,), agents={"main": {}, "rnd": scripted_agent()})
    spec = env_spec(cfg)
    fixed = load_fixed_players(cfg, spec)
    ckpt = write_ckpt_dir(tmp_path / "x", cfg, "main", policy_version=7)
    again = str(first / "checkpoints" / "main" / "ckpt_v10")
    twice = PlayerRequest(user=(("p", str(ckpt)), ("rnd", None), ("p", again)), runs=(), field=None)
    with pytest.raises(ConfigError, match="-a 'p' is given more than once"):
        resolve_players(cfg, spec, twice, fixed=fixed)
    same_dir_name = PlayerRequest(user=(), runs=(RunSelection(first), RunSelection(second)), field=None)
    with pytest.raises(ConfigError, match="alias 'run' is used twice"):
        resolve_players(cfg, spec, same_dir_name, fixed=fixed)
    monkeypatch.chdir(second)                                         # --run . : the alias is the dir's name
    here = PlayerRequest(user=(), runs=(RunSelection(first, alias="a"), RunSelection(Path("."))), field=None)
    candidates, _field = resolve_players(cfg, spec, here, fixed=fixed)
    assert [p.name for p in candidates] == ["main@v10", "run/main@v5"]
