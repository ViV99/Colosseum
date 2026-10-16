"""Bradley-Terry ratings of entities from team-pair statistics (SP4 spec block 2).

Model, per layout. Team strength = mean of its members' strengths ``theta`` + a bias ``b[team index]`` (the side
and its roles; ``sum_t b_t = 0``, parametrized as ``b = C beta`` with ``C = [I; -1]``). ``P(a beats b) =
sigmoid(S_a - S_b)``; a draw counts 0.5 (logistic likelihood with fractional outcomes). Each team pair of a match
is one comparison with weight ``1 / (T - 1)`` (``colosseum.arena.comparisons``). Priors: ``theta_e ~ N(0,
prior_sigma^2)`` (finite estimates for an entity that always loses), ``beta ~ N(0, prior_sigma^2)``, and a
random-walk smoothness ``-(smoothness / 2) (theta_{v_k+1} - theta_{v_k})^2`` between consecutive versions of one
lineage. MAP by Newton's method with backtracking on the log posterior: a dense Hessian up to
``BTOptions.dense_limit`` parameters, conjugate gradients with a Jacobi preconditioner above; warm start from a
previous fit.

Groups and components. Entities that played the same roles of the layout (in the data) form a role group; groups
merge when an entity played roles of both (a bridge). Entities linked by comparisons form a component. Ratings are
comparable only within a cell = (role group, component); every cell has its own anchor (rating 0), chosen by
``_anchor``: ``options.anchor`` (an agent of the config: its entity, or for a trainable agent its lowest version)
if it played in the cell; else the config's first ``RandomBot`` agent; else the mean of the scripted / frozen agents
that played in the cell; else the lowest version (``v0``) of the first trainable agent (config order) of the cell;
else the first entity by name. A configured or default anchor that did not play in a cell is noted in
``BTGroup.note`` (callers log each note once).

Scale: Elo units, ``ELO_PER_LOGIT = 400 / ln 10`` per logit.

Uncertainty. ``laplace``: normal intervals from the inverse Hessian at the MAP; ``P(best)``, ``P(top-k)`` and the
lower bound from samples of ``N(theta_hat, H^-1)`` (a diagonal approximation on the iterative path).
``bootstrap``: refits on resampled data (``BootstrapData``), warm-started at the fit, percentile intervals. Both
report ``P(best)`` within the entity's lineage and within its cell, ``P(top-k)`` likewise, and ``lcb``, the
``quantile`` of the entity's lineage contrast ``theta_e - mean(theta of its lineage in the cell)`` (spec block 4:
the anchor's uncertainty does not reorder the snapshots of one agent).

``PRIOR_SIGMA`` and ``SMOOTHNESS`` are provisional; task T8.2 fixes them by cross-validation (controller rulings).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from colosseum.arena.comparisons import ComparisonTable, PairStat, TeamKey, coefficients, match_pairs
from colosseum.arena.entities import EntityKey, entity_version, lineage, snapshot_version

if TYPE_CHECKING:
    from colosseum.core.types import MatchResult

ELO_PER_LOGIT = 400.0 / math.log(10.0)
Z_95 = 1.959963984540054
PRIOR_SIGMA = 4.0          # logits (~695 Elo); provisional (T8.2)
SMOOTHNESS = 0.5           # precision of one version step (sd ~1.41 logits); provisional (T8.2)
DENSE_LIMIT = 2000         # parameters; above it Newton steps use conjugate gradients
MEAN_OF_FIXED = "mean-of-fixed"
TRANSITIVITY_P = 0.01      # the G2 test flags p < 0.01 (spec block 2)


@dataclass(frozen=True)
class BTOptions:
    prior_sigma: float = PRIOR_SIGMA
    smoothness: float = SMOOTHNESS
    anchor: str | None = None
    max_iter: int = 50
    tol: float = 1e-8
    dense_limit: int = DENSE_LIMIT
    agent_order: tuple[str, ...] = ()   # every agent in config order; ``fit(agent_order=...)`` takes precedence


DEFAULT_OPTIONS = BTOptions()


def default_options(anchor: str | None = None) -> BTOptions:
    """The module's prior and smoothness (``PRIOR_SIGMA``, ``SMOOTHNESS``; T8.2 fixes them) with ``anchor``."""
    return BTOptions(anchor=anchor)


@dataclass
class BTGroup:
    """One cell of a layout: the entities of one role group and one component, and their anchor."""

    layout: str
    group: str                       # '+'-joined sorted role names of the role group ("player", "hunter+prey")
    component: int
    entities: list[EntityKey]
    anchor: EntityKey | str          # an entity, or MEAN_OF_FIXED
    anchor_members: list[EntityKey]  # the entities whose mean is 0 (one for an entity anchor)
    rule: str                        # "config" | "random_bot" | "mean_of_fixed" | "first_trainable" | "first"
    note: str | None = None          # why a configured / default anchor was replaced


@dataclass
class _Design:
    """Sparse design of a fit: rows = team pairs, columns = entities then the free bias parameters."""

    entities: list[EntityKey]
    index: dict[EntityKey, int]
    num_teams: int
    keys: list[tuple[TeamKey, TeamKey]]
    row_of: dict[tuple[TeamKey, TeamKey], int]
    nz_row: np.ndarray
    nz_col: np.ndarray
    nz_val: np.ndarray
    sw: np.ndarray
    swy: np.ndarray
    prior: np.ndarray | None         # dense prior precision (None above the dense limit)
    prior_diag: np.ndarray
    chains: list[tuple[int, int]]    # smoothness edges (column pairs)

    @property
    def num_params(self) -> int:
        return len(self.entities) + max(self.num_teams - 1, 0)


@dataclass
class BTFit:
    layout: str
    groups: list[BTGroup]
    rating: dict[EntityKey, float]      # Elo units, anchor = 0 in its cell
    se: dict[EntityKey, float]          # Laplace standard errors of the anchored rating (Elo units)
    bias: dict[int, float]              # team index -> Elo units; sums to 0 over the layout's teams
    bias_se: dict[int, float]
    games: dict[EntityKey, int]         # matches with a non-zero coefficient (self-play excluded)
    iterations: int
    converged: bool
    approximate: bool = False           # iterative path: standard errors from the Hessian's diagonal
    params: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)   # MAP, logits
    design: _Design | None = field(default=None, repr=False)
    options: BTOptions = field(default=DEFAULT_OPTIONS, repr=False)
    hessian: np.ndarray | None = field(default=None, repr=False)   # dense negative Hessian at the MAP

    def group_of(self, entity: EntityKey) -> BTGroup:
        for group in self.groups:
            if entity in group.entities:
                return group
        raise KeyError(entity)


@dataclass
class Uncertainty:
    ci95: dict[EntityKey, tuple[float, float]]
    lcb: dict[EntityKey, float]                   # quantile of the lineage contrast (Elo units)
    p_best_lineage: dict[EntityKey, float]
    p_best_group: dict[EntityKey, float]          # within the cell
    p_top_k: dict[EntityKey, float]               # within the lineage (only with top_k > 0)
    p_top_k_group: dict[EntityKey, float]         # within the cell (only with top_k > 0)
    contrast_samples: dict[EntityKey, np.ndarray] = field(default_factory=dict, repr=False)   # Elo, float32


Contribution = tuple[TeamKey, TeamKey, float, float]   # (a, b, weight, outcome of a), as comparisons.match_pairs


@dataclass
class BootstrapData:
    """What bootstrap replicates resample, a decomposition of the fitted table (shared with the tournament's
    statistics, T5.3):

    - ``counts``: aggregated rows ``(a, b, PairStat)`` (compacted training matches): wins ~ Poisson(swy / w) and
      losses ~ Poisson(n - swy / w) with ``w = sw / n``: exact without draws, a draw's variance is overstated;
    - ``matches``: per detailed match the list of its contributions; every match gets a Poisson(1) weight, so the
      pairs of one FFA match move together;
    - ``blocks``: ``{stratum: {key: [contributions]}}`` (tournament and pulse blocks: a stratum is a field member
      or composition, a key a seed index shared by every candidate); per replicate and stratum the keys are drawn
      with replacement, the same draw for every contribution of a key (paired, stratified).
    """

    counts: list[tuple[TeamKey, TeamKey, PairStat]] = field(default_factory=list)
    matches: list[list[Contribution]] = field(default_factory=list)
    blocks: dict[str, dict[str, list[Contribution]]] = field(default_factory=dict)

    @classmethod
    def from_table(cls, table: ComparisonTable, layout: str, *, sources: Collection[str]) -> BootstrapData:
        """Every team-pair row of ``layout`` as a ``counts`` row (Poisson by row; a detailed FFA match's pairs no
        longer move together: an approximation, e.g. for measurements)."""
        return cls(counts=list(table.pairs(layout, sources)))

    @classmethod
    def from_blocks(cls, results_by_block: Mapping[str, Sequence[MatchResult]], *,
                    strata: Mapping[str, str] | None = None) -> BootstrapData:
        """Whole blocks (tournament, pulse; spec block 2), rating versions from the network ids
        (``snapshot_version``). With ``strata`` (block id -> stratum) a block's key is its id's last ``/``
        component, the seed index that every candidate's block on that seed shares (paired, stratified); without,
        one stratum and every block its own key."""
        blocks: dict[str, dict[str, list[Contribution]]] = {}
        for block_id, results in sorted(results_by_block.items()):
            stratum = "" if strata is None else strata[block_id]
            key = block_id if strata is None else block_id.rsplit("/", 1)[-1]
            bucket = blocks.setdefault(stratum, {}).setdefault(key, [])
            for result in results:
                bucket.extend(match_pairs(result, [snapshot_version(s.network_id) for s in result.seats]))
        return cls(blocks=blocks)


def bootstrap_data(reader: Any, layout: str, *, sources: Collection[str]) -> BootstrapData:
    """The resampling units of a run journal's ``layout`` (spec block 2; ``reader``: a ``MatchLogReader`` or
    ``MatchLog``, through ``bootstrap_units``): compacted ``counts`` rows (Poisson by row), detailed training
    matches (one Poisson weight per match), matches with a block id (pulse) as whole blocks (stratum: the block id
    without its last component; key: that component)."""
    counts, details = reader.bootstrap_units(layout, sources)
    data = BootstrapData(counts=list(counts))
    for block, contributions in details:
        if block is None:
            data.matches.append(list(contributions))
        else:
            stratum, _, key = block.rpartition("/")
            data.blocks.setdefault(stratum, {}).setdefault(key, []).extend(contributions)
    return data


@dataclass
class TransitivityResult:
    g2: float
    dof: int
    p_value: float | None            # None: not testable (no pair with min_games, or dof <= 0)
    flagged: bool                    # p_value < TRANSITIVITY_P
    pairs_tested: int
    worst_pairs: list[tuple[TeamKey, TeamKey, float, int]]   # (a, b, signed deviance residual, matches), largest |r|

    def to_dict(self) -> dict:
        """JSON-ready form (``ratings.json``, reports): teams as ``"+"``-joined member entities."""
        return {"flag": self.flagged, "g2": self.g2, "dof": self.dof, "p_value": self.p_value,
                "pairs_tested": self.pairs_tested,
                "worst_pairs": [{"a": "+".join(a.entities()), "b": "+".join(b.entities()), "team_a": a.team,
                                 "team_b": b.team, "residual": r, "games": n} for a, b, r, n in self.worst_pairs]}


@dataclass
class ScoreCell:
    n: int
    mean: float
    ci_low: float
    ci_high: float


@dataclass
class ScoreSummary:
    layout: str
    entities: dict[EntityKey, ScoreCell]
    compositions: dict[str, ScoreCell]          # "a+a+b": sorted member entities
    homogeneous: dict[str, bool]

    @property
    def by_entity(self) -> dict[EntityKey, tuple[int, float, float]]:
        """``(n, mean, standard error)`` of every entity (every match it sat in; T4.1's top-k in games with
        score layouts only)."""
        return {e: (c.n, c.mean, (c.ci_high - c.ci_low) / (2.0 * Z_95)) for e, c in self.entities.items()}

    def to_dict(self) -> dict:
        """JSON-ready form (reports)."""
        return {"layout": self.layout,
                "entities": {e: {"n": c.n, "mean": c.mean, "ci_low": c.ci_low, "ci_high": c.ci_high}
                             for e, c in self.entities.items()},
                "compositions": {k: {"n": c.n, "mean": c.mean, "ci_low": c.ci_low, "ci_high": c.ci_high,
                                     "homogeneous": self.homogeneous[k]} for k, c in self.compositions.items()}}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class _UnionFind:
    """Union-find whose root is the smallest element (deterministic labels)."""

    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _build_design(rows: Sequence[tuple[TeamKey, TeamKey, PairStat]], options: BTOptions) -> _Design | None:
    coefs = [coefficients(a, b) for a, b, _ in rows]
    entities = sorted({e for c in coefs for e in c})
    if not entities:
        return None
    index = {e: i for i, e in enumerate(entities)}
    num_teams = 1 + max(max(a.team, b.team) for a, b, _ in rows)
    E, F = len(entities), max(num_teams - 1, 0)

    def bias_coef(team: int) -> dict[int, float]:      # b = C beta: team t < T-1 -> beta_t, team T-1 -> -sum(beta)
        if team < num_teams - 1:
            return {E + team: 1.0}
        return {E + j: -1.0 for j in range(F)}

    nz_row, nz_col, nz_val = [], [], []
    for r, ((a, b, _), coef) in enumerate(zip(rows, coefs, strict=True)):
        entry: dict[int, float] = {index[e]: c for e, c in coef.items()}
        for col, val in bias_coef(a.team).items():
            entry[col] = entry.get(col, 0.0) + val
        for col, val in bias_coef(b.team).items():
            entry[col] = entry.get(col, 0.0) - val
        for col, val in entry.items():
            if val != 0.0:
                nz_row.append(r)
                nz_col.append(col)
                nz_val.append(val)
    by_lineage: dict[str, list[tuple[int, EntityKey]]] = {}
    for e in entities:
        version = entity_version(e)
        if version is not None:
            by_lineage.setdefault(lineage(e), []).append((version, e))
    chains: list[tuple[int, int]] = []
    for versions in by_lineage.values():
        versions.sort()
        chains.extend((index[x], index[y]) for (_, x), (_, y) in zip(versions, versions[1:], strict=False))
    precision = 1.0 / options.prior_sigma ** 2
    prior_diag = np.full(E + F, precision)
    for i, j in chains:
        prior_diag[i] += options.smoothness
        prior_diag[j] += options.smoothness
    prior = None
    if E + F <= options.dense_limit:
        prior = np.diag(np.full(E + F, precision))
        for i, j in chains:
            prior[i, i] += options.smoothness
            prior[j, j] += options.smoothness
            prior[i, j] -= options.smoothness
            prior[j, i] -= options.smoothness
    keys = [(a, b) for a, b, _ in rows]
    return _Design(
        entities=entities, index=index, num_teams=num_teams, keys=keys, row_of={k: r for r, k in enumerate(keys)},
        nz_row=np.asarray(nz_row, np.int64), nz_col=np.asarray(nz_col, np.int64), nz_val=np.asarray(nz_val),
        sw=np.asarray([s.sw for _, _, s in rows], np.float64), swy=np.asarray([s.swy for _, _, s in rows], np.float64),
        prior=prior, prior_diag=prior_diag, chains=chains,
    )


def _eta(d: _Design, params: np.ndarray) -> np.ndarray:
    return np.bincount(d.nz_row, weights=d.nz_val * params[d.nz_col], minlength=len(d.keys))


def _prior_times(d: _Design, v: np.ndarray, options: BTOptions) -> np.ndarray:
    if d.prior is not None:
        return d.prior @ v
    out = v / options.prior_sigma ** 2
    for i, j in d.chains:
        diff = options.smoothness * (v[i] - v[j])
        out[i] += diff
        out[j] -= diff
    return out


def _log_post(d: _Design, params: np.ndarray, sw: np.ndarray, swy: np.ndarray, options: BTOptions) -> float:
    eta = _eta(d, params)
    return float(np.sum(swy * eta - sw * np.logaddexp(0.0, eta))) - 0.5 * float(params @ _prior_times(d, params,
                                                                                                       options))


def _gradient(d: _Design, params: np.ndarray, sw: np.ndarray, swy: np.ndarray,
              options: BTOptions) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of the log posterior and the per-row curvature weights ``sw p (1 - p)``."""
    p = _sigmoid(_eta(d, params))
    grad = np.bincount(d.nz_col, weights=d.nz_val * (swy - sw * p)[d.nz_row], minlength=d.num_params)
    return grad - _prior_times(d, params, options), sw * p * (1.0 - p)


def _dense_hessian(d: _Design, w: np.ndarray) -> np.ndarray:
    """Negative Hessian of the log posterior: ``sum_r w_r z_r z_r^T`` + the prior precision."""
    H = d.prior.copy()
    order = np.argsort(d.nz_row, kind="stable")
    rows, cols, vals = d.nz_row[order], d.nz_col[order], d.nz_val[order]
    starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    ends = np.r_[starts[1:], len(rows)]
    sizes = ends - starts
    for k in np.unique(sizes):                    # rows grouped by their number of non-zeros
        sel = starts[sizes == k]
        idx = sel[:, None] + np.arange(k)[None, :]
        c, v = cols[idx], vals[idx]
        np.add.at(H, (np.repeat(c[:, :, None], k, 2), np.repeat(c[:, None, :], k, 1)),
                  w[rows[sel]][:, None, None] * v[:, :, None] * v[:, None, :])
    return H


def _hess_times(d: _Design, w: np.ndarray, v: np.ndarray, options: BTOptions) -> np.ndarray:
    out = np.bincount(d.nz_col, weights=d.nz_val * (w * _eta(d, v))[d.nz_row], minlength=d.num_params)
    return out + _prior_times(d, v, options)


def _hess_diag(d: _Design, w: np.ndarray) -> np.ndarray:
    return np.bincount(d.nz_col, weights=d.nz_val ** 2 * w[d.nz_row], minlength=d.num_params) + d.prior_diag


def _cg(apply: Callable[[np.ndarray], np.ndarray], b: np.ndarray, diag: np.ndarray, tol: float = 1e-10,
        max_iter: int = 500) -> np.ndarray:
    """Preconditioned conjugate gradients for ``apply(x) = b`` (``apply`` symmetric positive definite)."""
    x = np.zeros_like(b)
    r = b.copy()
    z = r / diag
    p = z.copy()
    rz = float(r @ z)
    bnorm = max(float(np.linalg.norm(b)), 1e-300)
    for _ in range(max_iter):
        if float(np.linalg.norm(r)) <= tol * bnorm:
            break
        ap = apply(p)
        alpha = rz / float(p @ ap)
        x += alpha * p
        r -= alpha * ap
        z = r / diag
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def _newton(d: _Design, sw: np.ndarray, swy: np.ndarray, start: np.ndarray,
            options: BTOptions) -> tuple[np.ndarray, int, bool]:
    params = start.copy()
    value = _log_post(d, params, sw, swy, options)
    for iteration in range(1, options.max_iter + 1):
        grad, w = _gradient(d, params, sw, swy, options)
        if d.prior is not None:
            step = np.linalg.solve(_dense_hessian(d, w), grad)
        else:
            step = _cg(lambda v, w=w: _hess_times(d, w, v, options), grad, _hess_diag(d, w))
        t = 1.0
        while True:                               # backtracking: the log posterior never decreases
            candidate = params + t * step
            new_value = _log_post(d, candidate, sw, swy, options)
            if new_value >= value - 1e-12 * max(1.0, abs(value)) or t < 1e-6:
                break
            t *= 0.5
        params, value = candidate, new_value
        if float(np.max(np.abs(t * step))) < options.tol:
            return params, iteration, True
    return params, options.max_iter, False


# ---------------------------------------------------------------------------
# groups, components, anchors
# ---------------------------------------------------------------------------


def _cells(rows: Sequence[tuple[TeamKey, TeamKey, PairStat]],
           entities: Sequence[EntityKey]) -> list[tuple[str, int, list[EntityKey]]]:
    """``(group label, component, entities)`` of every cell (module docstring), sorted."""
    known = set(entities)
    roles, links = _UnionFind(), _UnionFind()
    entity_roles: dict[EntityKey, set[str]] = {}
    for a, b, _ in rows:
        for key in (a, b):
            for entity, role in key.members:
                if entity in known:
                    entity_roles.setdefault(entity, set()).add(role)
        active = list(coefficients(a, b))
        for entity in active[1:]:
            links.union(active[0], entity)
    for played in entity_roles.values():
        first = min(played)
        for role in played:
            roles.union(first, role)
    group_roles: dict[str, set[str]] = {}
    for played in entity_roles.values():
        group_roles.setdefault(roles.find(min(played)), set()).update(played)
    labels = {root: "+".join(sorted(played)) for root, played in group_roles.items()}
    component_ids: dict[EntityKey, int] = {}
    for entity in sorted(entities):
        component_ids.setdefault(links.find(entity), len(component_ids))
    cells: dict[tuple[str, int], list[EntityKey]] = {}
    for entity in entities:
        label = labels[roles.find(min(entity_roles[entity]))]
        cells.setdefault((label, component_ids[links.find(entity)]), []).append(entity)
    return [(label, comp, sorted(members)) for (label, comp), members in sorted(cells.items())]


def _lowest_version(agent: str, entities: Iterable[EntityKey]) -> EntityKey | None:
    versions = [(v, e) for e in entities if lineage(e) == agent and (v := entity_version(e)) is not None]
    return min(versions)[1] if versions else None


def _anchor(cell: Sequence[EntityKey], options: BTOptions, fixed_agents: Collection[str], random_bot: str | None,
            agent_order: Sequence[str]) -> tuple[str, list[EntityKey], str, str | None]:
    """``(anchor, anchor members, rule, note)`` of one cell (module docstring)."""
    present = set(cell)
    notes: list[str] = []
    if options.anchor is not None:
        name = options.anchor
        if name in present:
            return name, [name], "config", None
        lowest = _lowest_version(name, cell)
        if lowest is not None:
            return lowest, [lowest], "config", None
        notes.append(f"ratings.anchor {name!r} did not play here")
    if random_bot is not None:
        if random_bot in present:
            return random_bot, [random_bot], "random_bot", "; ".join(notes) or None
        notes.append(f"RandomBot agent {random_bot!r} did not play here")
    note = "; ".join(notes) or None
    fixed = [f for f in fixed_agents if f in present]
    if fixed:
        return MEAN_OF_FIXED, fixed, "mean_of_fixed", note
    for agent in agent_order:
        if agent in fixed_agents:
            continue
        lowest = _lowest_version(agent, cell)
        if lowest is not None:
            return lowest, [lowest], "first_trainable", note
    first = sorted(cell)[0]
    return first, [first], "first", note


def _contrast(design: _Design, group: BTGroup) -> np.ndarray:
    c = np.zeros(design.num_params)
    for entity in group.anchor_members:
        c[design.index[entity]] += 1.0 / len(group.anchor_members)
    return c


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def fit(table: ComparisonTable, layout: str, *, sources: Collection[str], options: BTOptions = DEFAULT_OPTIONS,
        fixed_agents: Collection[str] = (), random_bot: str | None = None, agent_order: Sequence[str] = (),
        warm: BTFit | None = None) -> BTFit:
    """MAP Bradley-Terry fit of one layout over ``sources`` (module docstring); an empty fit without comparisons.

    ``fixed_agents``: the config's scripted / frozen agents; ``random_bot``: the config's first ``RandomBot`` agent;
    ``agent_order``: every agent in config order (the "first trainable agent" anchor rule)."""
    rows = table.pairs(layout, sources)
    design = _build_design(rows, options) if rows else None
    if design is None:
        return BTFit(layout=layout, groups=[], rating={}, se={}, bias={}, bias_se={}, games={}, iterations=0,
                     converged=True, options=options)
    E, T = len(design.entities), design.num_teams
    start = np.zeros(design.num_params)
    if warm is not None and warm.design is not None:
        for entity, i in design.index.items():
            j = warm.design.index.get(entity)
            if j is not None:
                start[i] = warm.params[j]
        if warm.design.num_teams == T:
            start[E:] = warm.params[len(warm.design.entities):]
    params, iterations, converged = _newton(design, design.sw, design.swy, start, options)
    _, w = _gradient(design, params, design.sw, design.swy, options)
    hessian = _dense_hessian(design, w) if design.prior is not None else None
    groups = []
    for label, component, members in _cells(rows, design.entities):
        anchor, anchor_members, rule, note = _anchor(members, options, fixed_agents, random_bot,
                                                     agent_order or options.agent_order)
        groups.append(BTGroup(layout, label, component, members, anchor, anchor_members, rule, note))
    if hessian is not None:
        cov = np.linalg.inv(hessian)
        cov_diag = np.diag(cov)
    else:
        diag_inv = 1.0 / _hess_diag(design, w)
    rating, se = {}, {}
    for group in groups:
        c = _contrast(design, group)
        cols = np.array([design.index[e] for e in group.entities])
        base = float(c @ params)
        if hessian is not None:
            cov_c = cov @ c
            var = cov_diag[cols] - 2.0 * cov_c[cols] + float(c @ cov_c)
        else:
            var = diag_inv[cols] - 2.0 * c[cols] * diag_inv[cols] + float(np.sum(c * c * diag_inv))
        for k, entity in enumerate(group.entities):
            rating[entity] = (float(params[cols[k]]) - base) * ELO_PER_LOGIT
            se[entity] = math.sqrt(max(float(var[k]), 0.0)) * ELO_PER_LOGIT
    C = np.vstack([np.eye(T - 1), -np.ones((1, T - 1))])         # T >= 2: every row pairs two teams
    bias_var = (np.einsum("ti,ij,tj->t", C, cov[E:, E:], C) if hessian is not None
                else (C * C) @ diag_inv[E:])
    bias_logits = C @ params[E:]
    games = {e: g for e, g in table.entity_games(layout, sources).items() if e in design.index}
    return BTFit(
        layout=layout, groups=groups, rating=rating, se=se,
        bias={t: float(bias_logits[t]) * ELO_PER_LOGIT for t in range(T)},
        bias_se={t: math.sqrt(max(float(bias_var[t]), 0.0)) * ELO_PER_LOGIT for t in range(T)},
        games=games, iterations=iterations, converged=converged, approximate=hessian is None, params=params,
        design=design, options=options, hessian=hessian,
    )


def _summaries(fit_: BTFit, samples: np.ndarray, *, top_k: int, quantile: float,
               ci_from_samples: bool) -> Uncertainty:
    """Probabilities, intervals and lower bounds from parameter samples ``[S, P]`` (logits)."""
    design = fit_.design
    ci: dict = {}
    lcb: dict = {}
    pbl: dict = {}
    pbg: dict = {}
    ptk: dict = {}
    ptkg: dict = {}
    contrasts: dict = {}
    for group in fit_.groups:
        cols = np.array([design.index[e] for e in group.entities])
        anchored = (samples[:, cols] - (samples @ _contrast(design, group))[:, None]) * ELO_PER_LOGIT   # [S, n]
        best = np.argmax(anchored, axis=1)
        ranks = np.argsort(np.argsort(-anchored, axis=1), axis=1) if top_k > 0 else None
        for j, entity in enumerate(group.entities):
            pbg[entity] = float(np.mean(best == j))
            if ci_from_samples:
                lo, hi = np.quantile(anchored[:, j], [0.025, 0.975])
                ci[entity] = (float(lo), float(hi))
            else:
                half = Z_95 * fit_.se[entity]
                ci[entity] = (fit_.rating[entity] - half, fit_.rating[entity] + half)
            if ranks is not None:
                ptkg[entity] = float(np.mean(ranks[:, j] < top_k))
        by_lineage: dict[str, list[int]] = {}
        for j, entity in enumerate(group.entities):
            by_lineage.setdefault(lineage(entity), []).append(j)
        for columns in by_lineage.values():
            sub = anchored[:, columns]
            contrast = sub - sub.mean(axis=1, keepdims=True)
            best_l = np.argmax(sub, axis=1)
            ranks_l = np.argsort(np.argsort(-sub, axis=1), axis=1) if top_k > 0 else None
            for k, j in enumerate(columns):
                entity = group.entities[j]
                pbl[entity] = float(np.mean(best_l == k))
                lcb[entity] = float(np.quantile(contrast[:, k], quantile))
                contrasts[entity] = contrast[:, k].astype(np.float32)
                if ranks_l is not None:
                    ptk[entity] = float(np.mean(ranks_l[:, k] < top_k))
    return Uncertainty(ci95=ci, lcb=lcb, p_best_lineage=pbl, p_best_group=pbg, p_top_k=ptk, p_top_k_group=ptkg,
                       contrast_samples=contrasts)


def _empty_uncertainty() -> Uncertainty:
    return Uncertainty({}, {}, {}, {}, {}, {})


def laplace(fit_: BTFit, *, samples: int = 1000, rng: np.random.Generator, top_k: int = 0,
            quantile: float = 0.1) -> Uncertainty:
    """Fast uncertainty (module docstring): normal intervals from the fit's standard errors; probabilities and
    ``lcb`` from ``samples`` draws of ``N(theta_hat, H^-1)``."""
    if fit_.design is None:
        return _empty_uncertainty()
    z = rng.standard_normal((samples, fit_.design.num_params))
    if fit_.hessian is not None:
        L = np.linalg.cholesky(fit_.hessian)
        delta = np.linalg.solve(L.T, z.T).T              # cov(delta) = (L L^T)^-1 = H^-1
    else:
        _, w = _gradient(fit_.design, fit_.params, fit_.design.sw, fit_.design.swy, fit_.options)
        delta = z / np.sqrt(_hess_diag(fit_.design, w))
    return _summaries(fit_, fit_.params[None, :] + delta, top_k=top_k, quantile=quantile, ci_from_samples=False)


def _contribution_arrays(design: _Design, items: Iterable[tuple[int, TeamKey, TeamKey, float, float]]):
    unit, row, w, wy = [], [], [], []
    for u, a, b, weight, outcome in items:
        if a.team > b.team:
            a, b, outcome = b, a, 1.0 - outcome
        r = design.row_of.get((a, b))
        if r is None:
            raise ValueError(f"bootstrap data has a team pair the fit does not have: {a} vs {b}")
        unit.append(u)
        row.append(r)
        w.append(weight)
        wy.append(weight * outcome)
    return (np.asarray(unit, np.int64), np.asarray(row, np.int64), np.asarray(w, np.float64),
            np.asarray(wy, np.float64))


def bootstrap(fit_: BTFit, data: BootstrapData, *, samples: int, rng: np.random.Generator, top_k: int = 0,
              quantile: float = 0.1) -> Uncertainty:
    """Uncertainty from ``samples`` refits on resampled ``data`` (``BootstrapData``), each warm-started at the fit;
    percentile intervals. The cells and anchors are those of ``fit_``; ``data`` must decompose the table it was fit
    on (a team pair the fit does not know is a ValueError)."""
    design = fit_.design
    if design is None:
        return _empty_uncertainty()
    c_rows, c_w, c_wins, c_losses = [], [], [], []
    for a, b, stat in data.counts:
        if stat.n <= 0:
            continue
        w = stat.sw / stat.n
        wins = stat.swy / w if w > 0 else 0.0           # in matches: a draw counts half
        if a.team > b.team:                              # canonical order: the outcome becomes b's
            a, b, wins = b, a, stat.n - wins
        r = design.row_of.get((a, b))
        if r is None:
            raise ValueError(f"bootstrap data has a team pair the fit does not have: {a} vs {b}")
        wins = min(max(wins, 0.0), float(stat.n))
        c_rows.append(r)
        c_w.append(w)
        c_wins.append(wins)
        c_losses.append(stat.n - wins)
    c_rows_a, c_w_a = np.asarray(c_rows, np.int64), np.asarray(c_w, np.float64)
    c_wins_a, c_losses_a = np.asarray(c_wins, np.float64), np.asarray(c_losses, np.float64)
    m_unit, m_row, m_w, m_wy = _contribution_arrays(
        design, ((m, a, b, w, y) for m, contribs in enumerate(data.matches) for a, b, w, y in contribs))
    strata: list[tuple[int, int]] = []          # (first unit, number of keys) per stratum
    units: list[list[Contribution]] = []
    for _, by_key in sorted(data.blocks.items()):
        keys = sorted(by_key)
        strata.append((len(units), len(keys)))
        units.extend(by_key[k] for k in keys)
    b_unit, b_row, b_w, b_wy = _contribution_arrays(
        design, ((u, a, b, w, y) for u, contribs in enumerate(units) for a, b, w, y in contribs))
    R = len(design.keys)
    replicates = np.empty((samples, design.num_params))
    for s in range(samples):
        sw, swy = np.zeros(R), np.zeros(R)
        if len(c_rows_a):
            wins = rng.poisson(c_wins_a)
            losses = rng.poisson(c_losses_a)
            np.add.at(sw, c_rows_a, c_w_a * (wins + losses))
            np.add.at(swy, c_rows_a, c_w_a * wins)
        if len(m_row):
            weight = rng.poisson(1.0, size=len(data.matches))[m_unit]
            np.add.at(sw, m_row, m_w * weight)
            np.add.at(swy, m_row, m_wy * weight)
        if len(b_row):
            unit_weight = np.zeros(len(units))
            for first, k in strata:
                unit_weight[first:first + k] = rng.multinomial(k, np.full(k, 1.0 / k))
            weight = unit_weight[b_unit]
            np.add.at(sw, b_row, b_w * weight)
            np.add.at(swy, b_row, b_wy * weight)
        replicates[s], _, _ = _newton(design, sw, swy, fit_.params, fit_.options)
    return _summaries(fit_, replicates, top_k=top_k, quantile=quantile, ci_from_samples=True)


def chi2_sf(x: float, dof: int) -> float:
    """Survival function of chi^2(dof) at ``x``: the regularized upper incomplete gamma ``Q(dof / 2, x / 2)``
    (series below ``a + 1``, Lentz's continued fraction above; Numerical Recipes 6.2)."""
    if x <= 0:
        return 1.0
    a, z = 0.5 * dof, 0.5 * x
    log_front = -z + a * math.log(z) - math.lgamma(a)
    if z < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(10_000):
            ap += 1.0
            term *= z / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(log_front))
    tiny = 1e-300
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(log_front) * h)


def transitivity_test(table: ComparisonTable, fit_: BTFit, *, sources: Collection[str],
                      min_games: int = 20) -> TransitivityResult:
    """Deviance ``G2 = 2 (LL_pairs - LL_BT)`` over the team pairs with at least ``min_games`` matches, against
    ``chi^2`` with ``pairs - (entities - cells) - (T - 1)`` degrees of freedom; flagged at p < 0.01 (spec block 2).
    With fractional weights (FFA) and draws the reference distribution is approximate."""
    design = fit_.design
    if design is None:
        return TransitivityResult(0.0, 0, None, False, 0, [])
    eta = _eta(design, fit_.params)
    g2 = 0.0
    residuals: list[tuple[TeamKey, TeamKey, float, int]] = []
    involved: set[EntityKey] = set()
    tested = 0
    for a, b, stat in table.pairs(fit_.layout, sources):
        r = design.row_of.get((a, b))
        if stat.n < min_games or stat.sw <= 0 or r is None:
            continue
        tested += 1
        involved.update(coefficients(a, b))
        p = min(max(float(_sigmoid(eta[r])), 1e-12), 1.0 - 1e-12)
        y, n = stat.swy, stat.sw
        dev = 0.0
        if y > 0:
            dev += y * math.log((y / n) / p)
        if n - y > 0:
            dev += (n - y) * math.log(((n - y) / n) / (1.0 - p))
        dev = max(2.0 * dev, 0.0)
        g2 += dev
        residuals.append((a, b, math.copysign(math.sqrt(dev), y / n - p), int(stat.n)))
    cells = {(g.group, g.component) for g in fit_.groups if set(g.entities) & involved}
    dof = tested - (len(involved) - len(cells)) - (design.num_teams - 1)
    p_value = chi2_sf(g2, dof) if tested and dof > 0 else None
    residuals.sort(key=lambda item: -abs(item[2]))
    return TransitivityResult(g2=g2, dof=dof, p_value=p_value,
                              flagged=p_value is not None and p_value < TRANSITIVITY_P, pairs_tested=tested,
                              worst_pairs=residuals[:5])


def _score_cell(n: int, total: float, total_sq: float) -> ScoreCell:
    mean = total / n
    var = (total_sq - n * mean * mean) / (n - 1) if n >= 2 else 0.0
    half = Z_95 * math.sqrt(max(var, 0.0) / n) if n >= 2 else 0.0
    return ScoreCell(n=n, mean=mean, ci_low=mean - half, ci_high=mean + half)


def score_summary(table: ComparisonTable, layout: str, *, sources: Collection[str]) -> ScoreSummary:
    """One-team layouts (spec block 2): mean team score with a 95 % normal CI per entity (every match it sat in) and
    per team composition (``"a+a+b"``: sorted member entities)."""
    per_entity: dict[EntityKey, list[float]] = {}
    per_comp: dict[str, list[float]] = {}
    homogeneous: dict[str, bool] = {}
    for key, stat in table.scores(layout, sources):
        members = sorted(key.entities())
        comp = "+".join(members)
        homogeneous[comp] = len(set(members)) == 1
        for cell in [per_comp.setdefault(comp, [0, 0.0, 0.0])] + [per_entity.setdefault(e, [0, 0.0, 0.0])
                                                                  for e in sorted(set(members))]:
            cell[0] += stat.n
            cell[1] += stat.total
            cell[2] += stat.total_sq
    return ScoreSummary(layout=layout,
                        entities={e: _score_cell(*v) for e, v in sorted(per_entity.items())},
                        compositions={c: _score_cell(*v) for c, v in sorted(per_comp.items())},
                        homogeneous=homogeneous)
