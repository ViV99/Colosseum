"""Per-seat ``active`` flags and action masks from vectorized env infos.

One implementation of the seat rules shared by the rollout worker and eval:

- ``info[p]["active"]`` marks the seats that act this step; a seat without the
  key acts. Only acting seats run inference.
- ``info[p]["action_mask"]`` (flat bool array, or dict of per-component masks)
  is flattened with the :class:`ActionSpec`; a seat without a mask gets an
  all-true row.
- Empty-row rule (R1-13, ET-08): a mask row with no legal action in some
  discrete component becomes all-true on a non-acting seat (its action is
  ignored); on an acting seat it is an :class:`EnvContractError`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from colosseum.core.action_spec import ActionSpec
from colosseum.core.errors import EnvContractError


def _seat_info(infos: Sequence[dict], e: int, p: int) -> dict | None:
    info_e = infos[e] if e < len(infos) else {}
    info = info_e.get(p) if isinstance(info_e, dict) else None
    return info if isinstance(info, dict) else None


def acting_flags(infos: Sequence[dict], num_envs: int, num_players: int) -> np.ndarray:
    """[E, P] bool: ``info[p]["active"]`` per seat; seats without the key act."""
    acting = np.ones((num_envs, num_players), dtype=bool)
    for e in range(num_envs):
        for p in range(num_players):
            info = _seat_info(infos, e, p)
            if info is not None and "active" in info:
                acting[e, p] = bool(info["active"])
    return acting


def extract_masks(
    infos: Sequence[dict], num_envs: int, num_players: int, action_spec: ActionSpec,
) -> np.ndarray | None:
    """[E*P, flat_mask_size] bool (row ``e * P + p``), or None if no seat has a mask.

    Seats without ``action_mask`` get an all-true row.
    """
    size = action_spec.flat_mask_size
    if size == 0:
        return None
    out: np.ndarray | None = None
    for e in range(num_envs):
        for p in range(num_players):
            info = _seat_info(infos, e, p)
            if info is None or info.get("action_mask") is None:
                continue
            if out is None:
                out = np.ones((num_envs * num_players, size), dtype=bool)
            raw = info["action_mask"]
            if isinstance(raw, dict):
                out[e * num_players + p] = action_spec.flatten_mask(raw)
            else:
                out[e * num_players + p] = np.asarray(raw, dtype=bool).reshape(size)
    return out


def check_masks(
    masks: np.ndarray,
    acting: np.ndarray,
    action_spec: ActionSpec,
    where: Callable[[int, int], str],
) -> None:
    """Apply the empty-row rule in place to ``masks`` ([E*P, M], ``acting`` [E, P]).

    ``where(env, seat)`` prefixes the error message (caller context).
    """
    num_players = acting.shape[1]
    acting_flat = acting.reshape(-1)
    for comp in action_spec.components:
        if comp.mask_size == 0:
            continue
        lo, hi = comp.mask_offset, comp.mask_offset + comp.mask_size
        empty = ~masks[:, lo:hi].any(axis=1)
        bad = np.flatnonzero(empty & acting_flat)
        if bad.size:
            e, p = divmod(int(bad[0]), num_players)
            raise EnvContractError(
                f"{where(e, p)}: action_mask has no legal action "
                f"(component {comp.name!r}) for an acting seat"
            )
        masks[empty, lo:hi] = True
