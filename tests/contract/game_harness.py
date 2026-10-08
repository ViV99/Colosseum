"""Drive the real SP2 RolloutLoop (and, from T4.3, the real APPO) in-process.

Chunks come from TickGame (tests/game_helpers.py), whose observations encode
``[env, episode, step, seat, is_final]``, so every slot of a chunk can be traced back to
the env step it came from.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from colosseum.sp2.core.types import (
    SLOT_ACT,
    SLOT_BOOT,
    SLOT_PAD,
    Lineup,
    MatchResult,
    SeatAssignment,
    TrajectoryChunk,
    WeightPayload,
    WorkerCommand,
)
from colosseum.sp2.worker.rollout_loop import LoopIO, RolloutLoop
from game_helpers import TickGame

KIND_LETTER = {SLOT_ACT: "A", SLOT_BOOT: "B", SLOT_PAD: "P"}


@dataclass
class Collected:
    """In-memory LoopIO: everything the loop emits is appended to these lists.

    ``weights[agent_id]`` holds WeightPayloads handed out one per ``poll_weights`` call;
    ``commands`` are handed out one per ``poll_command`` call.
    """

    chunks: list[TrajectoryChunk] = field(default_factory=list)
    results: list[MatchResult] = field(default_factory=list)
    env_steps: list[int] = field(default_factory=list)
    weights: dict[str, list[WeightPayload]] = field(default_factory=dict)
    commands: list[WorkerCommand] = field(default_factory=list)

    def io(self) -> LoopIO:
        def poll_weights(agent_id: str) -> WeightPayload | None:
            pending = self.weights.get(agent_id)
            return pending.pop(0) if pending else None

        def poll_command() -> WorkerCommand | None:
            return self.commands.pop(0) if self.commands else None

        return LoopIO(send_chunk=self.chunks.append, poll_weights=poll_weights,
                      report_result=self.results.append, poll_command=poll_command,
                      add_env_steps=self.env_steps.append)


class GameFactory:
    """Env factory: the i-th call builds ``TickGame(*games[i % n], tag=i)``.

    Each entry of ``games`` is a ``(script, num_seats)`` pair; ``game_kwargs`` go to every game.
    Sync vector envs build their envs in index order, so ``tag`` (the observation's first
    entry) is the env index. ``created`` keeps the built games.
    """

    def __init__(self, *games: tuple[Sequence[Any], int], **game_kwargs: Any) -> None:
        self.games = list(games)
        self.game_kwargs = game_kwargs
        self.created: list[TickGame] = []

    def __call__(self) -> TickGame:
        script, num_seats = self.games[len(self.created) % len(self.games)]
        game = TickGame(script, num_seats, tag=len(self.created), **self.game_kwargs)
        self.created.append(game)
        return game


def lineup(layout: str, *agents: str | SeatAssignment) -> Lineup:
    """``lineup("2p", "a", SeatAssignment("b", collect=False))``; strings are latest + collect."""
    return Lineup(layout, [a if isinstance(a, SeatAssignment) else SeatAssignment(a) for a in agents])


def make_loop(
    env_fn: Callable[[], Any],
    model_factories: Mapping[str, Callable[[], Any]],
    lineups: Sequence[Lineup],
    *,
    chunk_length: int = 4,
    collected: Collected | None = None,
    agent_roles: Mapping[str, Sequence[str]] | None = None,
    seed: int = 0,
    weight_sync_interval: float = 0.0,
    **kwargs: Any,
) -> tuple[RolloutLoop, Collected]:
    """A RolloutLoop with in-memory I/O over ``len(lineups)`` envs.

    ``agent_roles`` defaults to ``["player"]`` for every agent (TickGame's only role).
    ``collected`` may be pre-filled (e.g. ``weights`` for the initial sync).
    """
    col = collected if collected is not None else Collected()
    agent_ids = list(model_factories)
    loop = RolloutLoop(
        worker_id=0, env_fn=env_fn, num_envs=len(lineups), chunk_length=chunk_length,
        agent_ids=agent_ids,
        agent_roles=agent_roles if agent_roles is not None else {a: ["player"] for a in agent_ids},
        model_factories=dict(model_factories), io=col.io(), lineups=list(lineups),
        weight_sync_interval=weight_sync_interval, seed=seed, **kwargs,
    )
    return loop, col


def run_steps(loop: RolloutLoop, n: int) -> None:
    for _ in range(n):
        loop.step()


def run_until_chunks(loop: RolloutLoop, col: Collected, n_chunks: int,
                     max_steps: int = 10_000) -> list[TrajectoryChunk]:
    """Step until at least ``n_chunks`` chunks were sent; return the first ``n_chunks``."""
    for _ in range(max_steps):
        if len(col.chunks) >= n_chunks:
            return col.chunks[:n_chunks]
        loop.step()
    raise AssertionError(f"only {len(col.chunks)} chunks after {max_steps} steps")


def kinds(chunk: TrajectoryChunk) -> str:
    """Slot kinds as letters, e.g. ``"AAAB"``; a terminal ACT is ``"T"``, a BOOT with reset ``"R"``."""
    out = []
    for s in range(chunk.num_slots):
        letter = KIND_LETTER[int(chunk.kind[s])]
        if letter == "A" and bool(chunk.terminal[s]):
            letter = "T"
        elif letter == "B" and bool(chunk.reset_after[s]):
            letter = "R"
        out.append(letter)
    return "".join(out)


class Slot(NamedTuple):
    """A slot's TickGame observation: env tag, episode, step, seat, is_final."""

    env: int
    ep: int
    t: int
    seat: int
    final: int


def slot_steps(chunk: TrajectoryChunk) -> list[Slot]:
    """Decode every slot's TickGame observation."""
    return [Slot(*(int(round(float(x))) for x in chunk.obs[s])) for s in range(chunk.num_slots)]


def seat_chunks(col: Collected, seat: int, env: int = 0) -> list[TrajectoryChunk]:
    """Chunks whose first slot belongs to (``env``, ``seat``)."""
    return [c for c in col.chunks if slot_steps(c)[0].seat == seat and slot_steps(c)[0].env == env]
