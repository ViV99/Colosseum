"""The tournament engine (SP4 T5.1, spec block 3): whole blocks in two spawn workers, explicit seeds, results
back in the main process, errors with their context, a dead worker named with its log."""
from __future__ import annotations

import multiprocessing as mp
import queue

import pytest

from colosseum.arena.tournament.engine import TournamentError, make_tasks, play_blocks
from colosseum.arena.tournament.players import PlayerRequest, RunSelection, resolve_players
from colosseum.arena.tournament.schedule import Block, stage_seed
from colosseum.core.errors import PlayerError
from colosseum.core.registry import env_spec
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.players.registry import load_fixed_players
from game_helpers import scripted_agent, write_fake_run


def _block(index, a, b, seed_index=None, prefix="t") -> Block:
    """A 1v1 block: ``a`` vs ``b`` on both sides (players as (agent_id, network_id))."""
    bid = f"{prefix}/A/2p/{a[0]}{a[1]}/{b[0]}/{index}"
    seed = stage_seed(0, "A", index if seed_index is None else seed_index)
    lineups = tuple(Lineup("2p", [SeatAssignment(*first, collect=False), SeatAssignment(*second, collect=False)],
                           seed=seed, block=bid, uid=f"{bid}/r{r}")
                    for r, (first, second) in enumerate([(a, b), (b, a)]))
    return Block(id=bid, stage="A", layout="2p", candidate=f"{a[0]}@{a[1]}", composition=(b[0],), index=index,
                 seed=seed, lineups=lineups, rotations=(0, 1))


def _setup(tmp_path, **bots):
    root, cfg = write_fake_run(tmp_path / "run", versions=(5, 9),
                               agents={"main": {}, "rnd": scripted_agent(), **bots})
    spec = env_spec(cfg)
    candidates, field = resolve_players(cfg, spec, PlayerRequest(user=(), runs=(RunSelection(root),),
                                                                 field=("rnd", *bots)),
                                        fixed=load_fixed_players(cfg, spec))
    return cfg, spec, {p.name: p for p in [*candidates, *field]}


def test_blocks_are_played_whole_by_two_workers(tmp_path):
    cfg, spec, players = _setup(tmp_path)
    blocks = [_block(k, ("main", "ckpt_v5"), ("rnd", "fixed")) for k in range(6)]
    blocks += [_block(k, ("main", "ckpt_v9"), ("rnd", "fixed")) for k in range(6)]
    blocks.append(_block(99, ("main", "ckpt_v9"), ("rnd", "fixed"), seed_index=0))   # same seed as block 0 of v9
    seen: dict[str, list] = {}
    play_blocks(cfg, spec, players, blocks, num_workers=2, num_envs=2, deterministic=True,
                on_block=lambda block_id, results: seen.setdefault(block_id, []).extend(results),
                log_dir=tmp_path / "logs")
    assert set(seen) == {b.id for b in blocks}
    for b in blocks:
        results = seen[b.id]
        assert [r.uid for r in results] == sorted(lu.uid for lu in b.lineups)
        assert {r.seed for r in results} == {b.seed} and {r.source for r in results} == {"tournament"}
        assert {r.block for r in results} == {b.id} and all(r.env_steps is None for r in results)
        snapshot_seats = [s for r in results for s in r.seats if s.agent_id == "main"]
        version = int(b.lineups[0].seats[0].network_id.removeprefix("ckpt_v"))
        assert {s.version_start for s in snapshot_seats} == {version}
    same = [seen[blocks[6].id], seen[blocks[-1].id]]                      # common seed, deterministic policy
    assert [[t.rank for t in r.teams] for r in same[0]] == [[t.rank for t in r.teams] for r in same[1]]
    assert (tmp_path / "logs" / "tournament-worker-0.log").is_file()
    assert [len(t) for t in make_tasks(blocks, num_envs=2)] == [4, 4, 4, 1]   # >= 4 * 2 matches per task


def _no_workers_left() -> bool:
    return not [p for p in mp.active_children() if p.name.startswith("tournament-worker")]


def test_a_bot_error_in_a_worker_is_a_player_error_in_the_main_process(tmp_path):
    cfg, spec, players = _setup(tmp_path, bad=scripted_agent("game_helpers.PlayerErrorBot"))
    blocks = [_block(0, ("main", "ckpt_v5"), ("bad", "fixed"))]
    with pytest.raises(PlayerError, match="bad") as caught:
        play_blocks(cfg, spec, players, blocks, num_workers=1, num_envs=1, deterministic=True,
                    on_block=lambda *_: None, log_dir=tmp_path / "logs")
    message = str(caught.value)
    assert message.startswith("tournament, worker 0, env 0, seat ")              # SP2 context, tournament prefix
    assert f"match {blocks[0].lineups[0].uid} of block {blocks[0].id}" in message
    assert str(tmp_path / "logs" / "tournament-worker-0.log") in message
    assert _no_workers_left()


def test_a_dead_worker_names_its_log(tmp_path):
    cfg, spec, players = _setup(tmp_path, crash=scripted_agent("game_helpers.CrashBot"))
    blocks = [_block(0, ("main", "ckpt_v5"), ("crash", "fixed"))]
    with pytest.raises(TournamentError, match="tournament-worker-0") as caught:
        play_blocks(cfg, spec, players, blocks, num_workers=1, num_envs=1, deterministic=True,
                    on_block=lambda *_: None, log_dir=tmp_path / "logs")
    assert str(tmp_path / "logs" / "tournament-worker-0.log") in str(caught.value)
    assert _no_workers_left()


def test_blocks_need_a_lineup_of_their_own_per_episode(tmp_path):
    """One lineup is one episode and its uid the journal's duplicate key: a repeated uid, or a lineup without the
    block's seed and id, is refused before any worker starts."""
    from dataclasses import replace

    cfg, spec, players = _setup(tmp_path)
    good = _block(0, ("main", "ckpt_v5"), ("rnd", "fixed"))
    again = replace(_block(1, ("main", "ckpt_v5"), ("rnd", "fixed")), lineups=good.lineups)
    with pytest.raises(ValueError, match="needs block"):
        play_blocks(cfg, spec, players, [good, again], num_workers=1, num_envs=1, deterministic=True,
                    on_block=lambda *_: None)
    copied = replace(good, lineups=(good.lineups[0], good.lineups[0]))
    with pytest.raises(ValueError, match="scheduled twice"):
        play_blocks(cfg, spec, players, [copied], num_workers=1, num_envs=1, deterministic=True,
                    on_block=lambda *_: None)


def test_seats_must_be_tournament_players_and_errors_keep_their_base_class(tmp_path):
    from colosseum.arena.tournament.engine import _drain, _error_kind
    from colosseum.core.errors import ConfigError, DataError

    cfg, spec, players = _setup(tmp_path)
    with pytest.raises(TournamentError, match="not tournament players"):
        play_blocks(cfg, spec, players, [_block(0, ("main", "ckpt_v7"), ("rnd", "fixed"))], num_workers=1,
                    num_envs=1, deterministic=True, on_block=lambda *_: None)
    assert _no_workers_left()
    assert _error_kind(DataError("bad data")) == "ConfigError" and _error_kind(PlayerError("x")) == "PlayerError"
    assert issubclass(DataError, ConfigError)
    q = mp.get_context("spawn").Queue()
    for item in ("a", "b"):
        q.put(item)
    assert _drain(q) == ["a", "b"]                         # the final drain before "tasks unplayed"
    with pytest.raises(queue.Empty):
        q.get_nowait()
    q.close()
