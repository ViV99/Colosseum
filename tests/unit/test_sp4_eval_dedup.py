"""Shared player loading of eval / record / tournament (SP4 T5.1; SP3 appendix dedup items): one -a parser, one
rounding rule, frozen agents loaded once, no .pt overlay of a frozen entry on a user path, play_lineups with
explicit seeds, a pool and a source."""
from __future__ import annotations

import pytest
import torch
from click.testing import CliRunner

import colosseum.players.registry as registry
from colosseum.cli import main
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model, env_spec, make_env
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.eval import balanced_match_count, parse_agent_spec, play_lineups
from colosseum.players import RandomBot
from colosseum.worker.match_runner import MatchRunner
from game_helpers import (
    DictModelPool,
    EliminationFFA,
    TurnTakingGame,
    agent_role_of,
    frozen_agent,
    make_test_config,
    scripted_agent,
    scripted_player,
    write_test_config,
)


def _pt(tmp_path, name, hidden=16, seed=0):
    cfg = make_test_config("turns", networks={"kwargs": {"core": "none", "hidden": hidden}})
    torch.manual_seed(seed)
    _roles, role = agent_role_of(cfg, "agent_0")
    path = tmp_path / f"{name}.pt"
    torch.save(build_model(cfg.get_agent_config("agent_0"), role).state_dict(), path)
    return path


def test_parse_agent_spec_and_the_rounding_rule():
    assert parse_agent_spec("greedy") == ("greedy", None)
    assert parse_agent_spec("a=runs/x/ckpt_v3") == ("a", "runs/x/ckpt_v3")
    for bad in ("", "=x", "a="):
        with pytest.raises(ConfigError, match="expected an agent name"):
            parse_agent_spec(bad, option="--against")
    spec = EliminationFFA(4).spec
    assert balanced_match_count(spec, None, 3) == 4 and balanced_match_count(spec, ["2p"], 4) == 4
    solo = TurnTakingGame().spec
    assert balanced_match_count(solo, ["2p"], 5) == 6


def test_eval_loads_a_frozen_agent_once_and_record_uses_the_shared_parser(tmp_path, monkeypatch):
    pt = _pt(tmp_path, "old")
    cfg = write_test_config(tmp_path / "cfg.yaml", "turns",
                            agents={"agent_0": {}, "rnd": scripted_agent(), "old": frozen_agent(pt)})
    calls = []
    original = registry.load_frozen
    monkeypatch.setattr(registry, "load_frozen", lambda *a, **k: calls.append(a[1]) or original(*a, **k))
    result = CliRunner().invoke(main, ["eval", "-c", str(cfg), "-a", "old", "-a", "rnd", "-n", "2", "--num-envs", "2"])
    assert result.exit_code == 0, result.output
    assert calls == ["old"]
    bad = CliRunner().invoke(main, ["record", "-c", str(cfg), "--player", "=x", "-o", str(tmp_path / "rec")])
    assert bad.exit_code == 1 and "expected an agent name" in bad.output


def test_a_user_path_ignores_the_frozen_entry_of_the_same_name(tmp_path):
    wide, narrow = _pt(tmp_path, "wide", hidden=32), _pt(tmp_path, "narrow", hidden=16)
    cfg = write_test_config(tmp_path / "cfg.yaml", "turns", agents={
        "agent_0": {}, "rnd": scripted_agent(), "wide": frozen_agent(wide, networks={"kwargs": {"hidden": 32}})})
    by_name = CliRunner().invoke(main, ["eval", "-c", str(cfg), "-a", "wide", "-a", "rnd", "-n", "2",
                                        "--num-envs", "2"])
    assert by_name.exit_code == 0, by_name.output                       # the entry's networks for its own path
    by_path = CliRunner().invoke(main, ["eval", "-c", str(cfg), "-a", f"wide={narrow}", "-a", "rnd", "-n", "2",
                                        "--num-envs", "2"])
    assert by_path.exit_code == 0, by_path.output                       # SP4: global networks for a user path


def test_play_lineups_keeps_explicit_seeds_uids_and_the_source():
    lineups = [Lineup("2p", [SeatAssignment("r1", "fixed"), SeatAssignment("r2", "fixed")], seed=1234 + i % 2,
                      block=f"t/A/2p/x/-/{i % 2}", uid=f"t/A/2p/x/-/{i % 2}/r{i // 2}") for i in range(4)]
    models = {"r1": RandomBot(), "r2": RandomBot()}
    results = play_lineups(env_fn=TurnTakingGame, models=models, lineups=lineups, num_envs=2, seed=0,
                           source="tournament")
    assert sorted(r.uid for r in results) == sorted(lu.uid for lu in lineups)
    by_uid = {r.uid: r for r in results}
    assert all(by_uid[lu.uid].seed == lu.seed and by_uid[lu.uid].block == lu.block for lu in lineups)
    assert {r.source for r in results} == {"tournament"}
    # the same seed gives the same bot games (bots are seeded from the episode seed)
    assert [t.rank for t in by_uid[lineups[0].uid].teams] == [t.rank for t in by_uid[lineups[2].uid].teams]
    assert [s.reward for s in by_uid[lineups[0].uid].seats] == [s.reward for s in by_uid[lineups[2].uid].seats]
    with pytest.raises(ValueError, match="exactly one"):
        play_lineups(env_fn=TurnTakingGame, lineups=lineups, num_envs=2)


def test_play_lineups_with_a_pool_serves_snapshots_by_network_id():
    cfg = make_test_config("turns")
    _roles, role = agent_role_of(cfg, "agent_0")
    torch.manual_seed(0)
    v5, v9 = build_model(cfg.get_agent_config("agent_0"), role), build_model(cfg.get_agent_config("agent_0"), role)
    pool = DictModelPool({("main", "ckpt_v5"): v5.eval(), ("main", "ckpt_v9"): v9.eval(),
                          ("rnd", "fixed"): scripted_player(RandomBot, env_spec(cfg))})
    lineups = [Lineup("2p", [SeatAssignment("main", "ckpt_v5"), SeatAssignment("main", "ckpt_v9")], seed=7,
                      uid="u0", block="b0"),
               Lineup("2p", [SeatAssignment("main", "ckpt_v9"), SeatAssignment("rnd", "fixed")], seed=8, uid="u1",
                      block="b1")]
    results = play_lineups(env_fn=lambda: make_env(cfg), pool=pool, lineups=lineups, num_envs=2, seed=0,
                           source="tournament")
    versions = {r.uid: [s.version_start for s in r.seats] for r in results}
    assert versions == {"u0": [5, 9], "u1": [9, None]}


def test_match_runner_without_the_snapshot_fallback_needs_the_snapshot_itself():
    from colosseum.envs.vector import VectorEnv

    pool = DictModelPool({("rnd", "fixed"): scripted_player(RandomBot, TurnTakingGame().spec)})
    lineup = Lineup("2p", [SeatAssignment("main", "ckpt_v5", collect=False), SeatAssignment("rnd", "fixed", False)])
    vec_env = VectorEnv(TurnTakingGame, 1)
    try:
        with pytest.raises(ValueError, match="ckpt_v5"):
            MatchRunner(vec_env=vec_env, lineups=[lineup], models=pool, snapshot_fallback=False)
    finally:
        vec_env.close()
