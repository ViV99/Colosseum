"""Tournament schedules (SP4 spec block 3): blocks, admissible team rotations, FFA compositions, ``score``
layouts, seed domains, role groups.

A **block** is one seed (``stage_seed(seed, stage, index)``), one composition and every admissible rotation:
- two or more teams (``gauntlet_blocks``, ``round_robin_blocks``): the entity list ``[candidate, *composition]``
  is assigned to the teams by cyclic shifts; shift ``r`` gives team ``t`` the entity ``E[(t - r) % T]`` (so the
  candidate sits in team ``r``) and every seat of a team to that entity. A shift is admissible when every
  team's roles are roles of its entity; a block holds every admissible shift (1v1: two matches with sides
  swapped; T symmetric teams: T matches; asymmetric roles: only the admissible shifts, sometimes one). A
  composition without an admissible shift is not scheduled.
- one team (``score_blocks``): the candidate's homogeneous team (one match, rotation 0; composition ``()``,
  only if the candidate plays every seat's role) and, per field member ``f``, the composition ``(f,)``: ``f``
  takes one seat, the candidate the others, and the block holds one match per seat ``f`` can take (rotation =
  that seat), like SP2's cross-play.

Gauntlet compositions (spec block 3): the candidate takes one team, field members the other ``T - 1``: every
combination of ``T - 1`` field members when ``F >= T - 1``, else every multiset of size ``T - 1``; so every field
member meets the candidate equally often (symmetric roles). ``blocks`` is the number of blocks per
composition: indices ``first_index .. first_index + blocks - 1`` (halving rounds continue with ``first_index``).
``round_robin_blocks`` runs only on two-team layouts (ruling C4): every pair ``(a, b)`` is the candidate ``a``
with the composition ``(b,)``.

Seeds: ``stage_seed(seed, stage, index)`` puts the stage's domain code in the top two bits of a 32-bit seed and
a keyed bijection of ``index`` in the low 30 bits, so the domains A (screening), B (confirmation) and P (pulse)
never share a seed and no seed repeats within a domain (indices < 2**30). Every candidate and every composition
gets the same seed for the same (stage, index): common random numbers.

Ids (amendment A9): ``Block.id = "<prefix>/<stage>/<layout>/<candidate>/<composition or ->/<index>"`` (the
composition's members joined by ``+``) and every match's ``Lineup.uid = "<block id>/r<rotation>"``; ``prefix``
is the tournament name (or ``pulse-<run id>``) and contains no ``/``. Every lineup is a fresh object with fresh
seats (one lineup per episode; ``engine.check_blocks``), and every seat plays ``collect=False``. A name given
twice to one builder is a ValueError (it would schedule the same uid twice).

Role groups (spec block 2): a candidate's group in a layout is the set of the layout's roles it plays.
Candidates are ranked within a group; groups that overlap without being equal (e.g. ``{hunter}`` and
``{hunter, prey}``) cannot share a table: ``check_role_groups`` raises ConfigError listing them.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from colosseum.core.errors import ConfigError
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.envs.game import GameSpec

STAGES = ("A", "B", "P")
_STAGE_CODE = {"A": 1, "B": 2, "P": 3}
_LOW_BITS = 30
_LOW_MASK = (1 << _LOW_BITS) - 1


@dataclass(frozen=True)
class Block:
    """One seed, one composition and every admissible rotation (T5.2): the unit of scheduling, resume and
    bootstrap. ``lineups[i]`` is rotation ``rotations[i]``; every lineup carries ``seed``, ``block`` (= ``id``)
    and ``uid`` (= ``id + "/r<rotation>"``)."""

    id: str
    stage: str
    layout: str
    candidate: str
    composition: tuple[str, ...]
    index: int
    seed: int
    lineups: tuple[Lineup, ...]
    rotations: tuple[int, ...]

    @property
    def stratum(self) -> str:
        """The composition as one key: field members joined by ``+`` (``-`` for the homogeneous team)."""
        return "+".join(self.composition) or "-"


def stage_seed(seed: int, stage: str, index: int) -> int:
    """The episode seed of block ``index`` of ``stage`` (module docstring: disjoint domains, no repeats)."""
    if stage not in _STAGE_CODE:
        raise ValueError(f"stage_seed: stage must be one of {STAGES}, got {stage!r}")
    if not 0 <= int(index) <= _LOW_MASK:
        raise ValueError(f"stage_seed: index must be in [0, 2**30), got {index}")
    code = _STAGE_CODE[stage]
    a, b, c = (int(x) for x in np.random.SeedSequence([int(seed) % 2**63, code]).generate_state(3, np.uint32))
    x = (int(index) * (a | 1) + b) & _LOW_MASK      # odd multiplier + offset: a bijection mod 2**30
    x ^= x >> 15                                     # xorshift: a bijection
    x = (x * (c | 1)) & _LOW_MASK
    x ^= x >> 13
    return (code << _LOW_BITS) | x


class SeatPlayer(Protocol):
    """What a schedule needs of a player (``players.TournamentPlayer`` has it)."""

    @property
    def agent_id(self) -> str: ...
    @property
    def network_id(self) -> str: ...
    @property
    def roles(self) -> tuple[str, ...]: ...


# A rotation: (rotation number, the player name of every seat).
_Rotation = tuple[int, tuple[str, ...]]


def _team_roles(spec: GameSpec, layout: str) -> list[set[str]]:
    seats = spec.layouts[layout]
    return [{seats[s].role for s in members} for members in spec.teams(layout)]


def role_group(spec: GameSpec, layout: str, roles: Sequence[str]) -> frozenset[str]:
    """The roles of ``layout`` that a player with ``roles`` plays (its role group in that layout)."""
    return frozenset({seat.role for seat in spec.layouts[layout]} & set(roles))


def group_label(group: frozenset[str]) -> str:
    """A role group as one key: its roles sorted and joined by ``+``."""
    return "+".join(sorted(group))


def check_role_groups(spec: GameSpec, layout: str, names: Sequence[str],
                      players: Mapping[str, SeatPlayer]) -> dict[str, list[str]]:
    """``{group label: candidates}`` of ``layout`` (groups and members in input order); ConfigError when a
    candidate plays none of the layout's roles or two groups overlap without being equal."""
    groups: dict[frozenset[str], list[str]] = {}
    for name in names:
        groups.setdefault(role_group(spec, layout, players[name].roles), []).append(name)
    empty = groups.pop(frozenset(), None)
    if empty:
        raise ConfigError(f"layout {layout!r}: candidates {empty} play none of its roles "
                          f"{sorted({s.role for s in spec.layouts[layout]})}; choose layouts with --layout")
    keys = list(groups)
    for g1, g2 in itertools.combinations(keys, 2):
        if g1 & g2:
            listing = {group_label(g): groups[g] for g in keys}
            raise ConfigError(f"layout {layout!r}: candidates of overlapping role groups cannot be ranked in one "
                              f"table: {listing}; give every candidate the same roles or disjoint roles "
                              f"(e.g. hunters and prey)")
    return {group_label(g): members for g, members in groups.items()}


def compositions(field: Sequence[str], slots: int) -> list[tuple[str, ...]]:
    """Gauntlet compositions of ``slots`` field members: combinations when ``len(field) >= slots``, else
    multisets (module docstring)."""
    if slots == 0:
        return [()]
    if len(field) >= slots:
        return list(itertools.combinations(field, slots))
    return list(itertools.combinations_with_replacement(field, slots))


def _team_rotations(spec: GameSpec, layout: str, entities: Sequence[str],
                    players: Mapping[str, SeatPlayer]) -> list[_Rotation]:
    """Every admissible cyclic shift of ``entities`` over the layout's teams (module docstring)."""
    teams = spec.teams(layout)
    roles = _team_roles(spec, layout)
    n = len(teams)
    out: list[_Rotation] = []
    for shift in range(n):
        seats = [""] * spec.layout_size(layout)
        for t, members in enumerate(teams):
            name = entities[(t - shift) % n]
            if not roles[t] <= set(players[name].roles):
                break
            for s in members:
                seats[s] = name
        else:
            out.append((shift, tuple(seats)))
    return out


def _seat(player: SeatPlayer) -> SeatAssignment:
    return SeatAssignment(agent_id=player.agent_id, network_id=player.network_id, collect=False)


def _block(prefix: str, stage: str, layout: str, candidate: str, composition: tuple[str, ...], index: int,
           seed: int, rotations: Sequence[_Rotation], players: Mapping[str, SeatPlayer]) -> Block:
    block_id = f"{prefix}/{stage}/{layout}/{candidate}/{'+'.join(composition) or '-'}/{index}"
    s = stage_seed(seed, stage, index)
    lineups = tuple(Lineup(layout=layout, seats=[_seat(players[name]) for name in names], seed=s, block=block_id,
                           uid=f"{block_id}/r{r}")
                    for r, names in rotations)
    return Block(id=block_id, stage=stage, layout=layout, candidate=candidate, composition=composition,
                 index=index, seed=s, lineups=lineups, rotations=tuple(r for r, _ in rotations))


def _check_common(spec: GameSpec, layout: str, stage: str, blocks: int, first_index: int, prefix: str,
                  players: Mapping[str, SeatPlayer], *name_lists: Sequence[str]) -> None:
    if layout not in spec.layouts:
        raise ConfigError(f"layout {layout!r} is not a layout of the game {sorted(spec.layouts)}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if blocks < 1 or first_index < 0:
        raise ValueError(f"need blocks >= 1 and first_index >= 0, got {blocks}, {first_index}")
    if not prefix or "/" in prefix:
        raise ValueError(f"block id prefix must be non-empty without '/', got {prefix!r}")
    for names in name_lists:
        repeated = sorted({n for n in names if list(names).count(n) > 1})
        if repeated:
            raise ValueError(f"names given more than once: {repeated} (every block id and match uid must be unique)")
        unknown = [n for n in names if n not in players]
        if unknown:
            raise ValueError(f"no player for the names {unknown}")


def gauntlet_blocks(spec: GameSpec, layout: str, candidates: Sequence[str], field: Sequence[str],
                    players: Mapping[str, SeatPlayer], *, blocks: int, seed: int, stage: str,
                    first_index: int = 0, prefix: str = "t") -> list[Block]:
    """Blocks of every candidate against the field in a layout of two or more teams (module docstring), in the
    order candidate, composition, index.

    ConfigError when the field is empty or a candidate has no admissible composition (the message lists the
    teams' roles); ValueError for a one-team layout (use ``score_blocks``)."""
    _check_common(spec, layout, stage, blocks, first_index, prefix, players, candidates, field)
    n_teams = spec.num_teams(layout)
    if n_teams < 2:
        raise ValueError(f"gauntlet_blocks: layout {layout!r} has one team; use score_blocks")
    if not field:
        raise ConfigError("the field is empty: name opponents with --field (agents of the config, -a players or "
                          "snapshots), or add scripted / frozen agents to the config")
    out: list[Block] = []
    comps = compositions(list(field), n_teams - 1)
    for cand in candidates:
        found = False
        for comp in comps:
            rotations = _team_rotations(spec, layout, [cand, *comp], players)
            if not rotations:
                continue
            found = True
            for k in range(first_index, first_index + blocks):
                out.append(_block(prefix, stage, layout, cand, tuple(comp), k, seed, rotations, players))
        if not found:
            raise ConfigError(
                f"layout {layout!r}: candidate {cand!r} (roles {list(players[cand].roles)}) and the field "
                f"{list(field)} cannot fill its teams (roles per team: "
                f"{[sorted(r) for r in _team_roles(spec, layout)]}); check the roles or choose --layout")
    return out


def round_robin_blocks(spec: GameSpec, layout: str, participants: Sequence[str],
                       players: Mapping[str, SeatPlayer], *, blocks: int, seed: int, stage: str = "A",
                       first_index: int = 0, prefix: str = "t") -> list[Block]:
    """Every pair (a, b) of ``participants`` (input order, ``a`` first) in a two-team layout: blocks with
    ``candidate=a``, ``composition=(b,)`` and both admissible sides. A pair without an admissible side is
    skipped; ConfigError when no pair can play, or for a layout with other than two teams (ruling C4)."""
    _check_common(spec, layout, stage, blocks, first_index, prefix, players, participants)
    if spec.num_teams(layout) != 2:
        raise ConfigError(f"--mode round-robin needs two-team layouts; {layout!r} has {spec.num_teams(layout)} "
                          f"teams (use gauntlet, halving or select)")
    out: list[Block] = []
    for a, b in itertools.combinations(participants, 2):
        rotations = _team_rotations(spec, layout, [a, b], players)
        if not rotations:
            continue
        for k in range(first_index, first_index + blocks):
            out.append(_block(prefix, stage, layout, a, (b,), k, seed, rotations, players))
    if not out:
        raise ConfigError(f"layout {layout!r}: no pair of {list(participants)} can fill its two teams (roles per "
                          f"team: {[sorted(r) for r in _team_roles(spec, layout)]})")
    return out


def score_blocks(spec: GameSpec, layout: str, candidates: Sequence[str], field: Sequence[str],
                 players: Mapping[str, SeatPlayer], *, blocks: int, seed: int, stage: str,
                 first_index: int = 0, prefix: str = "t") -> list[Block]:
    """Blocks of a one-team layout (module docstring; ruling D8): per candidate its homogeneous team (if it plays
    every role) and, per field member, the cross-play composition, in the order candidate, composition, index.
    ConfigError when a candidate has no block; ValueError for a layout of several teams."""
    _check_common(spec, layout, stage, blocks, first_index, prefix, players, candidates, field)
    if spec.num_teams(layout) != 1:
        raise ValueError(f"score_blocks: layout {layout!r} has {spec.num_teams(layout)} teams")
    seat_roles = [seat.role for seat in spec.layouts[layout]]
    out: list[Block] = []
    for cand in candidates:
        own = set(players[cand].roles)
        plans: list[tuple[tuple[str, ...], list[_Rotation]]] = []
        if set(seat_roles) <= own:
            plans.append(((), [(0, (cand,) * len(seat_roles))]))
        if len(seat_roles) >= 2:
            for f in field:
                rotations: list[_Rotation] = []
                for s, role in enumerate(seat_roles):
                    others = {r for i, r in enumerate(seat_roles) if i != s}
                    if role in players[f].roles and others <= own:
                        rotations.append((s, tuple(f if i == s else cand for i in range(len(seat_roles)))))
                if rotations:
                    plans.append(((f,), rotations))
        if not plans:
            raise ConfigError(f"layout {layout!r}: candidate {cand!r} (roles {sorted(own)}) cannot play its seats "
                              f"(roles {seat_roles}) alone or with a field member; choose --layout")
        for comp, rotations in plans:
            for k in range(first_index, first_index + blocks):
                out.append(_block(prefix, stage, layout, cand, comp, k, seed, rotations, players))
    return out
