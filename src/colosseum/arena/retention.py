"""Snapshot retention (SP4 spec block 4): a pure function "entries + scores + policy -> what to keep".

Per agent, ``retain`` builds the kept set in this order:

1. **always** (never evicted, not even by the cap): the newest ``keep_last``; the final snapshot of the
   CURRENT run (``final`` and ``run_id == policy.current_run_id``; finals imported from parent runs and SP3
   snapshots without ``run_id`` are ordinary; with ``current_run_id=None``, e.g. a manager built without a
   run id, every final is protected as in SP3); every pinned snapshot; the snapshot just saved (``protect``);
2. **keep_every**: version is a multiple of ``keep_every_period`` (``interval * keep_every``; 0 = off);
3. **top-k** (``keep_top_k > 0``): the ``keep_top_k`` rated snapshots with the highest ``lcb`` (ties: the
   newer first), chosen among ALL rated entries (a snapshot of rule 1 may hold a top-k place);
4. **new** (only with ``keep_top_k > 0``): a snapshot outside the ``keep_last`` window with fewer than
   ``top_k_min_games`` rated games (``SnapshotScore.games``; no score = 0 games) is kept until it has them.

Everything else is evicted. Then the cap: while the kept set is larger than ``max_snapshots`` (``None`` =
no cap), unprotected snapshots (not rule 1) are evicted in this order:

a. keep_every-only snapshots (rule 2, not 3 or 4): oldest first when ``keep_top_k == 0`` (FIFO, no ratings:
   also the distributed learner), else the unrated ones first (oldest first), then by ascending ``lcb``
   (ties: the older first);
b. new-only snapshots (rule 4, not 3): oldest first;
c. top-k snapshots: ascending ``lcb`` (ties: the older first).

If rule 1 alone exceeds the cap, every other snapshot is evicted, rule 1 stays and ``warnings`` says so
(the caller logs it once per agent).

**No ratings yet** (``scores=None``, with ``keep_top_k > 0``): the first live refit has not happened (a fresh
run, or a resume whose imported snapshots the parent rated). "Not rated yet" is not "rated with 0 games":
every snapshot outside the ``keep_last`` window counts as new whatever ``top_k_min_games`` is (so nothing is
evicted by rules 2-4), and the cap is ``NO_RATINGS_CAP_FACTOR * max_snapshots`` instead of ``max_snapshots``
(the real cap waits for ratings; the raised one still bounds the disk if refits never come). The caller reruns
retention after the first refit. ``scores={}`` means "rated, none of these snapshots has a score".

``reasons`` names, per checkpoint id, why it was kept or evicted.
The function never touches the file system; ``CheckpointManager`` applies the decision (index first, then
``on_evict``), and SP5's storage can call it the same way.

``combine_scores`` / ``snapshot_scores`` turn per-layout BT results (live Laplace or bootstrap) into one
``SnapshotScore`` per snapshot of an agent (spec block 4: the BT rating over the agent's competitive layouts,
averaged with the ``matchmaking.layouts`` weights on every draw; lower bounds in the lineage-contrast scale;
games with ``score`` layouts only use the mean score).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np

KEEP_LAST, FINAL, PINNED, SAVED = "keep_last", "final", "pinned", "just saved"
KEEP_EVERY, TOP_K, NEW = "keep_every", "top_k", "new"
NO_RATINGS = "new (no ratings yet)"
# Before the first live refit the cap waits for ratings, up to this multiple of max_snapshots (module docstring).
NO_RATINGS_CAP_FACTOR = 2


@dataclass(frozen=True)
class SnapshotInfo:
    checkpoint_id: str
    version: int
    final: bool
    run_id: str | None
    pinned: bool


@dataclass(frozen=True)
class SnapshotScore:
    lcb: float | None    # None = not rated yet
    games: int


@dataclass(frozen=True)
class RetentionPolicy:
    keep_last: int
    keep_every_period: int    # interval * keep_every; 0 = off
    keep_top_k: int
    top_k_min_games: int
    max_snapshots: int | None
    current_run_id: str | None


@dataclass(frozen=True)
class RetentionDecision:
    keep: list[str]
    evict: list[str]
    reasons: dict[str, str]   # checkpoint_id -> why kept / evicted (validate, logs)
    warnings: list[str] = field(default_factory=list)


def _lcb(scores: Mapping[str, SnapshotScore], checkpoint_id: str) -> float | None:
    score = scores.get(checkpoint_id)
    return None if score is None else score.lcb


def retain(entries: Sequence[SnapshotInfo], scores: Mapping[str, SnapshotScore] | None, policy: RetentionPolicy,
           *, protect: str | None = None) -> RetentionDecision:
    """Apply the rules of the module docstring to ``entries`` (any order; ids unique); ``scores=None``: no
    ratings yet (module docstring)."""
    if policy.keep_last < 1 or policy.keep_every_period < 0 or policy.keep_top_k < 0 \
            or policy.top_k_min_games < 0 or (policy.max_snapshots is not None and policy.max_snapshots < 1):
        raise ValueError(f"retain: invalid policy {policy}")
    ordered = sorted(entries, key=lambda e: e.version)
    ids = [e.checkpoint_id for e in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError(f"retain: duplicate checkpoint ids in {ids}")
    version = {e.checkpoint_id: e.version for e in ordered}
    newest = {e.checkpoint_id for e in ordered[-policy.keep_last:]}

    always: dict[str, list[str]] = {}
    for e in ordered:
        why = []
        if e.checkpoint_id in newest:
            why.append(KEEP_LAST)
        if e.final and (policy.current_run_id is None or e.run_id == policy.current_run_id):
            why.append(FINAL)
        if e.pinned:
            why.append(PINNED)
        if e.checkpoint_id == protect:
            why.append(SAVED)
        if why:
            always[e.checkpoint_id] = why

    waiting = scores is None and policy.keep_top_k > 0      # no ratings yet
    scores = {} if scores is None else scores
    period = policy.keep_every_period
    every = {e.checkpoint_id for e in ordered if period > 0 and e.version % period == 0}
    top: set[str] = set()
    new: set[str] = set()
    if waiting:
        new = {e.checkpoint_id for e in ordered if e.checkpoint_id not in newest}
    elif policy.keep_top_k > 0:
        rated = [e for e in ordered if _lcb(scores, e.checkpoint_id) is not None]
        rated.sort(key=lambda e: (-_lcb(scores, e.checkpoint_id), -e.version))
        top = {e.checkpoint_id for e in rated[:policy.keep_top_k]}
        for e in ordered:
            score = scores.get(e.checkpoint_id)
            games = score.games if score is not None else 0
            if e.checkpoint_id not in newest and games < policy.top_k_min_games:
                new.add(e.checkpoint_id)

    reasons: dict[str, str] = {}
    keep: set[str] = set()
    for cid in ids:
        rules = list(always.get(cid, []))
        rules += [name for name, members in ((KEEP_EVERY, every), (TOP_K, top),
                                             (NO_RATINGS if waiting else NEW, new)) if cid in members]
        if rules:
            keep.add(cid)
            reasons[cid] = "kept: " + ", ".join(rules)
        else:
            reasons[cid] = "evicted: no retention rule"

    warnings: list[str] = []
    cap = policy.max_snapshots
    if waiting and cap is not None:
        cap *= NO_RATINGS_CAP_FACTOR                        # the real cap waits for the first refit
    if cap is not None and len(keep) > cap:
        if len(always) > cap:
            warnings.append(f"{len(always)} protected snapshots (keep_last, final of this run, pinned, just saved) "
                            f"exceed checkpoint.max_snapshots={cap}; they are all kept")
        unprotected = [cid for cid in ids if cid in keep and cid not in always]
        every_only = [cid for cid in unprotected if cid not in top and cid not in new]
        new_only = [cid for cid in unprotected if cid in new and cid not in top]
        top_only = [cid for cid in unprotected if cid in top]
        if policy.keep_top_k == 0:
            stage_a = sorted(every_only, key=lambda c: version[c])
        else:
            unrated = sorted((c for c in every_only if _lcb(scores, c) is None), key=lambda c: version[c])
            rated_a = sorted((c for c in every_only if _lcb(scores, c) is not None),
                             key=lambda c: (_lcb(scores, c), version[c]))
            stage_a = unrated + rated_a
        stage_b = sorted(new_only, key=lambda c: version[c])
        stage_c = sorted(top_only, key=lambda c: (_lcb(scores, c), version[c]))
        for stage, label in ((stage_a, KEEP_EVERY), (stage_b, NEW), (stage_c, TOP_K)):
            for cid in stage:
                if len(keep) <= cap:
                    break
                keep.discard(cid)
                reasons[cid] = f"evicted: over max_snapshots={cap} ({label})"
    return RetentionDecision(keep=[c for c in ids if c in keep], evict=[c for c in ids if c not in keep],
                             reasons=reasons, warnings=warnings)


def _version_of(entity: str, agent_id: str) -> int | None:
    """``N`` of the entity key ``<agent_id>@v<N>``, else None (other agents, fixed players)."""
    prefix = f"{agent_id}@v"
    rest = entity[len(prefix):] if entity.startswith(prefix) else ""
    return int(rest) if rest.isdigit() else None


def _weighted(layouts: Sequence[str], layout_weights: Mapping[str, float]) -> list[tuple[int, float]]:
    """``(index, weight)`` of the layouts that count: those with a positive ``layout_weights`` entry; when none has
    one, every layout with weight 1. Zero-weight layouts are dropped, so ``0 * inf`` never makes a NaN."""
    weights = [(i, float(layout_weights.get(name, 0.0))) for i, name in enumerate(layouts)]
    positive = [(i, w) for i, w in weights if w > 0]
    return positive or [(i, 1.0) for i in range(len(layouts))]


def combine_scores(agent_id: str, *, fits: Mapping[str, Any], uncertainty: Mapping[str, Any],
                   scores: Mapping[str, Any], layout_weights: Mapping[str, float],
                   quantile: float) -> dict[str, SnapshotScore]:
    """One ``SnapshotScore`` per snapshot entity ``<agent_id>@v<N>``, keyed ``ckpt_v<N>`` (module docstring).

    - Pairwise layouts (``fits``: layout -> ``BTFit``; ``uncertainty``: layout -> ``Uncertainty``): an entity
      rated in one (counted) layout gets that layout's ``Uncertainty.lcb``; rated in several, the ``quantile`` of
      the layout-weighted mean of its ``Uncertainty.contrast_samples`` per draw index (layouts are independent
      fits with their own seeds; the draws are truncated to the shortest layout's count, e.g. 400 iterative
      draws next to the Laplace ones; missing draws in a counted layout are a ValueError: ``bt`` always publishes
      them next to ``lcb``).
      ``games``: ``BTFit.games`` summed over the layouts, i.e. rated COMPARISONS, self-play excluded: one per
      match in a two-team layout, ``T - 1`` per match in a ``T``-team FFA (``top_k_min_games`` counts these).
    - Only when no fit names the agent's snapshots: ``score`` layouts (``scores``: layout -> ``ScoreSummary``
      with ``by_entity[entity] = (n, mean, se)``): the weighted mean and its standard error,
      ``lcb = mean + z_q * se``, ``games`` = matches. ``se = inf`` (fewer than two matches) gives ``lcb = -inf``:
      such a snapshot is rated, never certain, and the first top-k candidate to lose its place.
    Weights: ``layout_weights`` (``matchmaking.layouts``); only layouts with a positive weight count, and when
    none of the entity's layouts has one, all count equally. An entity without an interval yet (rated, but not
    in ``uncertainty``) has ``lcb=None``.
    """
    per_entity: dict[str, list[str]] = {}
    for layout, fit in fits.items():
        for entity in fit.rating:
            if _version_of(entity, agent_id) is not None:
                per_entity.setdefault(entity, []).append(layout)
    out: dict[str, SnapshotScore] = {}
    if per_entity:
        for entity, layouts in per_entity.items():
            games = sum(int(fits[name].games.get(entity, 0)) for name in layouts)
            unc = [uncertainty.get(name) for name in layouts]
            counted = _weighted(layouts, layout_weights)
            if any(unc[i] is None or entity not in unc[i].lcb for i, _w in counted):
                lcb = None
            elif len(counted) == 1:
                lcb = float(unc[counted[0][0]].lcb[entity])
            else:
                total = sum(w for _i, w in counted)
                draws = [unc[i].contrast_samples.get(entity) for i, _w in counted]
                missing = [layouts[i] for (i, _w), d in zip(counted, draws, strict=True) if d is None or len(d) == 0]
                if missing:     # bt always publishes draws next to lcb: a producer bug, not a case to average over
                    raise ValueError(f"combine_scores: {entity} has a lower bound but no contrast_samples in layouts "
                                     f"{missing}; the layout-weighted lower bound needs the draws of every layout")
                arrays = [np.asarray(d, dtype=np.float64) for d in draws]
                n = min(len(d) for d in arrays)
                mixed = sum(w * d[:n] for (_i, w), d in zip(counted, arrays, strict=True)) / total
                lcb = float(np.quantile(mixed, quantile))
            out[f"ckpt_v{_version_of(entity, agent_id)}"] = SnapshotScore(lcb, games)
        return out
    z = NormalDist().inv_cdf(quantile)
    rows: dict[str, list[tuple[str, int, float, float]]] = {}
    for layout, summary in scores.items():
        for entity, (n, mean, se) in summary.by_entity.items():
            if _version_of(entity, agent_id) is not None:
                rows.setdefault(entity, []).append((layout, int(n), float(mean), float(se)))
    for entity, items in rows.items():
        counted = _weighted([layout for layout, *_ in items], layout_weights)
        total = sum(w for _i, w in counted)
        mean = sum(w * items[i][2] for i, w in counted) / total
        se = math.sqrt(sum((w * items[i][3]) ** 2 for i, w in counted)) / total
        lcb = -math.inf if math.isinf(se) else mean + z * se
        out[f"ckpt_v{_version_of(entity, agent_id)}"] = SnapshotScore(float(lcb), sum(n for _l, n, *_ in items))
    return out


def snapshot_scores(snapshot: Any, agent_id: str, *, layout_weights: Mapping[str, float],
                    quantile: float) -> dict[str, SnapshotScore]:
    """``combine_scores`` of a ``RatingsSnapshot`` (T2.2): ``fits``, ``uncertainty`` and ``scores`` per layout."""
    return combine_scores(agent_id, fits=snapshot.fits, uncertainty=snapshot.uncertainty,
                          scores=getattr(snapshot, "scores", {}) or {}, layout_weights=layout_weights,
                          quantile=quantile)


def describe_retention(checkpoint: Any) -> str:
    """The effective retention rule of a ``CheckpointConfig`` in one line (``validate``, spec block 4)."""
    parts = [f"newest {checkpoint.keep_last} (keep_last, with trainer state)"]
    if checkpoint.keep_every > 0:
        parts.append(f"every {checkpoint.interval * checkpoint.keep_every}th version (keep_every "
                     f"{checkpoint.keep_every})")
    parts += ["the final one of this run", "pinned"]
    if checkpoint.keep_top_k > 0:
        parts.append(f"top {checkpoint.keep_top_k} by the rating's lower bound (quantile {checkpoint.top_k_quantile});"
                     f" a snapshot with fewer than {checkpoint.top_k_min_games} rated games is kept until it has them")
        order = ("keep_every snapshots by lowest lower bound (unrated first), then new ones oldest first, then "
                 "top-k by lowest lower bound")
    else:
        parts.append("top-k off")
        order = "keep_every snapshots oldest first"
    cap = checkpoint.max_snapshots
    parts.append("no cap" if cap is None else f"at most {cap} per agent (over the cap: {order})")
    return "snapshots: " + "; ".join(parts)
