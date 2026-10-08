"""Deriving per-player match outcomes from rewards or env-provided signals.

ELO / win-rate tracking needs a per-player outcome in ``[0, 1]`` (1 = win,
0 = loss, 0.5 = draw).  The most robust source is the environment itself: a
game usually knows the authoritative result (winner / final ranking), which
can differ from "who accumulated the most shaped reward".  These helpers prefer
an explicit env signal and fall back to cumulative reward only when none is
provided.

Env convention (all optional, checked in the terminal info dict per player):
  - ``info["outcome"]``: float in [0, 1] — used directly. A value outside [0, 1]
    (or NaN) raises :class:`~colosseum.core.errors.EnvContractError`.
  - ``info["rank"]``:    number, 1 = best — converted to [0, 1] (best→1, worst→0).
    Non-integer ranks are allowed (e.g. 2.5 for a two-way tie for 2nd/3rd); a
    non-finite rank raises :class:`~colosseum.core.errors.EnvContractError`.
If neither is present for all players, outcomes are derived from total reward.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from colosseum.core.errors import EnvContractError


def outcomes_from_rewards(total_rewards: Sequence[float]) -> list[float]:
    """1.0 to the max-reward player(s), 0.0 to others, 0.5 to all if tied."""
    r = np.asarray(total_rewards, dtype=float)
    mx = float(r.max())
    mn = float(r.min())
    if mx == mn:
        return [0.5] * len(r)
    return [1.0 if float(x) == mx else 0.0 for x in r]


def outcomes_from_terminal_infos(
    terminal_infos: dict[int, dict],
    num_players: int,
) -> list[float] | None:
    """Try to derive outcomes from env-provided ``outcome`` or ``rank`` keys.

    Returns a list of per-player outcomes in ``[0, 1]``, or ``None`` if the env
    did not provide an authoritative signal for every player.
    """
    if not terminal_infos:
        return None

    explicit: list[float] = []
    ranks: list[float] = []
    have_explicit = True
    have_rank = True

    for p in range(num_players):
        info = terminal_infos.get(p, {})
        if not isinstance(info, dict):
            return None
        if "outcome" in info:
            outcome = float(info["outcome"])
            if not 0.0 <= outcome <= 1.0:  # also rejects NaN
                raise EnvContractError(
                    f"player {p}: terminal info['outcome'] must be in [0, 1], got {info['outcome']!r}")
            explicit.append(outcome)
        else:
            have_explicit = False
        if "rank" in info:
            rank = float(info["rank"])
            if not math.isfinite(rank):
                raise EnvContractError(
                    f"player {p}: terminal info['rank'] must be finite, got {info['rank']!r}")
            ranks.append(rank)
        else:
            have_rank = False

    if have_explicit and len(explicit) == num_players:
        return explicit

    if have_rank and len(ranks) == num_players:
        rmin, rmax = min(ranks), max(ranks)
        if rmax == rmin:
            return [0.5] * num_players
        # rank 1 (best) → 1.0, worst → 0.0
        return [(rmax - rk) / (rmax - rmin) for rk in ranks]

    return None


def player_outcomes(
    total_rewards: Sequence[float],
    terminal_infos: dict[int, dict] | None = None,
    num_players: int | None = None,
) -> list[float]:
    """Authoritative per-player outcomes: prefer env signal, else reward.

    Args:
        total_rewards: cumulative reward per player slot.
        terminal_infos: optional ``{player_idx: terminal_info_dict}`` from the
            episode's final step (VectorEnv stores this under
            ``info[p]["terminal_info"]`` on auto-reset).
        num_players: number of player slots (defaults to ``len(total_rewards)``).
    """
    n = num_players if num_players is not None else len(total_rewards)
    if terminal_infos is not None:
        out = outcomes_from_terminal_infos(terminal_infos, n)
        if out is not None:
            return out
    return outcomes_from_rewards(total_rewards)
