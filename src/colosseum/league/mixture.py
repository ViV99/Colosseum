"""The built-in matchmaker (spec block 5): opponent categories mixed by shares.

For an env whose training data belongs to owner O (the coordinator rotates owners):

1. **Layout**: by O's ``layouts`` weights among the enabled layouts with a seat of O's roles.
2. **O's team**: uniform among the teams with a seat of O's roles; O@latest takes a random seat
   of its role there (source ``owner``).
3. **The core of every other team T**, drawn independently. E_T = the trainable agents that play
   a role of T. Categories:
   - ``latest``: O@latest if O is in E_T, else the latest weights of an agent of E_T by PFSP;
   - ``snapshots``: every stored snapshot of the agents of E_T by PFSP (own, the opponent's in an
     asymmetric game, other agents');
   - ``rivals``: if O is in E_T, the latest weights of the other agents of E_T by PFSP (these
     seats collect too); otherwise empty, and its share is added to ``latest``;
   - ``anchors``: O's anchors (scripted / frozen agents) that play a role of T, by weight.
   The category is drawn by O's ``opponents`` shares at the coordinator's env step among the
   non-empty categories (the shares of empty ones are spread proportionally); its name is the
   source of every seat of T. If every category with a positive share is empty (e.g.
   ``snapshots: 1`` before the first snapshot), the core comes from ``latest``, else from
   ``anchors``, with source ``fallback`` and one warning per (agent, layout). The core takes a
   random seat of its role in T.
4. **The other seats of every team**: a seat of a role the core plays follows ``teammates``
   (``self``: the core; ``mixed``: the core with probability ``teammate_self_prob``, else uniformly
   another trainable agent's latest weights with that role, a snapshot of the core's agent or one
   of O's anchors with that role); a seat of a role the core does not play gets the latest
   weights of a uniformly drawn trainable agent with that role, else one of O's anchors with it
   (by weight).
5. Only seats with the latest weights of trainable agents collect.
6. ``shuffle_seats``: ``permute_seats`` (teams of equal role composition, seats of equal role).

PFSP weight of a candidate X: ``pfsp_weight(score of O@latest against X, weighting, exponent)``
with O's ``pfsp`` config.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence

from colosseum.core.config import ColosseumConfig, MatchmakingConfig
from colosseum.core.errors import ConfigError
from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID, SOURCE_OWNER, Lineup, SeatAssignment
from colosseum.envs.game import GameSpec
from colosseum.league.base import BaseMatchmaker, MatchmakerContext
from colosseum.league.lineups import enabled_layouts, permute_seats, playable_layouts
from colosseum.league.pfsp import PlayerKey, pfsp_weight
from colosseum.league.schedule import schedule_points, schedule_value

logger = logging.getLogger(__name__)

CATEGORIES = ("latest", "snapshots", "rivals", "anchors")
FALLBACK = "fallback"


# ---------------------------------------------------------------------------
# Shares
# ---------------------------------------------------------------------------


def category_shares(config: MatchmakingConfig, env_steps: int) -> dict[str, float]:
    """The ``opponents`` shares at ``env_steps``."""
    return {c: schedule_value(getattr(config.opponents, c), env_steps) for c in CATEGORIES}


def fold_rivals(shares: Mapping[str, float], owner_in_team: bool) -> dict[str, float]:
    """For a team the owner cannot play, ``rivals`` counts as ``latest`` (latest of other agents)."""
    out = dict(shares)
    if not owner_in_team:
        out["latest"] += out["rivals"]
        out["rivals"] = 0.0
    return out


def spread(shares: Mapping[str, float], available: set[str]) -> dict[str, float]:
    """Positive shares of the available categories, normalized ({} if none)."""
    positive = {c: s for c, s in shares.items() if c in available and s > 0}
    total = sum(positive.values())
    return {c: s / total for c, s in positive.items()} if total > 0 else {}


def schedule_points_of(config: MatchmakingConfig) -> list[int]:
    """0 and every breakpoint of the share and anchor-weight schedules, ascending."""
    points = {0}
    for category in CATEGORIES:
        points.update(schedule_points(getattr(config.opponents, category)))
    if isinstance(config.anchors, dict):
        for weight in config.anchors.values():
            points.update(schedule_points(weight))
    return sorted(points)


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------


def team_roles(spec: GameSpec, layout: str, team: int) -> frozenset[str]:
    seats = spec.layouts[layout]
    return frozenset(seats[s].role for s in spec.teams(layout)[team])


def owner_teams(spec: GameSpec, layout: str, roles: frozenset[str]) -> list[int]:
    """Teams with a seat of one of ``roles``."""
    seats = spec.layouts[layout]
    return [t for t, members in enumerate(spec.teams(layout)) if any(seats[s].role in roles for s in members)]


def opposing_teams(spec: GameSpec, layout: str, roles: frozenset[str]) -> list[int]:
    """Teams that oppose the owner in at least one draw of its team."""
    own = owner_teams(spec, layout, roles)
    return [t for t in range(spec.num_teams(layout)) if any(o != t for o in own)]


def _team_view(context: MatchmakerContext, owner: str, layout: str, team: int,
               env_steps: int) -> tuple[list[str], bool, dict[str, float]]:
    """``(E_T in config order, owner in E_T, owner's anchors that play a role of T -> weight)``."""
    roles = team_roles(context.spec, layout, team)
    players = [a for a in context.trainable if context.agents[a].roles & roles]
    anchors = {a: w for a, w in context.anchors_at(owner, env_steps).items()
               if a in context.agents and context.agents[a].roles & roles}
    return players, owner in players, anchors


def _structural(context: MatchmakerContext, owner: str, layout: str, team: int,
                env_steps: int) -> tuple[dict[str, float], set[str]]:
    """Folded shares and the categories that can ever be filled (snapshots count as available)."""
    players, owner_in, anchors = _team_view(context, owner, layout, team, env_steps)
    available: set[str] = set()
    if players:
        available.update(("latest", "snapshots"))
    if owner_in and len(players) > 1:
        available.add("rivals")
    if any(w > 0 for w in anchors.values()):
        available.add("anchors")
    return fold_rivals(category_shares(context.matchmaking[owner], env_steps), owner_in), available


def effective_mix(context: MatchmakerContext, owner: str, layout: str, team: int, env_steps: int) -> dict[str, float]:
    """The category shares for opposing team ``team`` of ``owner`` in ``layout`` at ``env_steps``:
    ``rivals`` folded into ``latest`` when the owner cannot play the team, shares of categories that
    can never be filled spread proportionally. Snapshots count as available (their temporary
    absence is the runtime fallback's job). ``{}``: nothing with a positive share can be filled."""
    shares, available = _structural(context, owner, layout, team, env_steps)
    return spread(shares, available)


# ---------------------------------------------------------------------------
# The matchmaker
# ---------------------------------------------------------------------------


class MixtureMatchmaker(BaseMatchmaker):
    """The built-in matchmaker (module docstring)."""

    def __init__(self, context: MatchmakerContext) -> None:
        super().__init__(context)
        self._rng: random.Random = context.rng
        self._warned: set[tuple[str, str]] = set()

    def lineup_for(self, owner: str) -> Lineup:
        ctx = self.context
        if owner not in ctx.matchmaking:
            raise KeyError(f"{owner!r} is not a trainable agent; trainable agents: {ctx.trainable}")
        config = ctx.matchmaking[owner]
        spec = ctx.spec
        env_steps = ctx.env_steps()
        anchors = ctx.anchors_at(owner, env_steps)
        roles = ctx.agents[owner].roles
        weights = enabled_layouts(spec, config)
        layouts = playable_layouts(spec, config, roles)
        if not layouts:
            raise KeyError(f"agent {owner!r} has no playable layout (validate_matchmaking should have caught it)")
        layout = self._rng.choices(layouts, weights=[weights[name] for name in layouts], k=1)[0]
        own_team = self._rng.choice(owner_teams(spec, layout, roles))
        seats: list[SeatAssignment | None] = [None] * spec.layout_size(layout)
        for team, members in enumerate(spec.teams(layout)):
            if team == own_team:
                core, source = (owner, LATEST_NETWORK_ID), SOURCE_OWNER
            else:
                core, source = self._opponent_core(owner, config, layout, team, env_steps)
            self._fill_team(config, anchors, core, source, layout, members, seats)
        lineup_seats: list[SeatAssignment] = seats  # type: ignore[assignment]
        if config.shuffle_seats:
            lineup_seats = permute_seats(spec, layout, lineup_seats, self._rng)
        return Lineup(layout=layout, seats=lineup_seats)

    # -- cores ---------------------------------------------------------

    def _opponent_core(self, owner: str, config: MatchmakingConfig, layout: str, team: int,
                       env_steps: int) -> tuple[PlayerKey, str]:
        ctx = self.context
        players, owner_in, anchors = _team_view(ctx, owner, layout, team, env_steps)
        candidates: dict[str, list[PlayerKey]] = {
            "latest": [(owner, LATEST_NETWORK_ID)] if owner_in else [(a, LATEST_NETWORK_ID) for a in players],
            "snapshots": [(a, c) for a in players for c in ctx.snapshots(a)],
            "rivals": [(a, LATEST_NETWORK_ID) for a in players if a != owner] if owner_in else [],
            "anchors": [(a, FIXED_NETWORK_ID) for a, w in anchors.items() if w > 0],
        }
        filled = {c for c, items in candidates.items() if items}
        mix = spread(fold_rivals(category_shares(config, env_steps), owner_in), filled)
        if mix:
            names = list(mix)
            category = self._rng.choices(names, weights=[mix[c] for c in names], k=1)[0]
            source = category
        else:
            category = "latest" if candidates["latest"] else "anchors"
            if not candidates["anchors"]:
                candidates["anchors"] = [(a, FIXED_NETWORK_ID) for a in anchors]
            if not candidates[category]:
                raise RuntimeError(f"agent {owner!r}, layout {layout!r}, team {team}: nobody can play this team "
                                   f"(validate_matchmaking should have caught it)")
            source = FALLBACK
            if (owner, layout) not in self._warned:
                self._warned.add((owner, layout))
                logger.warning(
                    f"agent {owner!r}, layout {layout!r}: every opponent category with a positive share is empty "
                    f"now (e.g. no snapshot yet); the core comes from {category!r} (source 'fallback'). Logged once "
                    f"per agent and layout."
                )
        return self._pick(owner, config, layout, category, candidates[category], anchors), source

    def _pick(self, owner: str, config: MatchmakingConfig, layout: str, category: str,
              candidates: Sequence[PlayerKey], anchors: Mapping[str, float]) -> PlayerKey:
        if len(candidates) == 1:
            return candidates[0]
        if category == "anchors":
            return self._weighted(candidates, [anchors.get(a, 0.0) for a, _ in candidates])
        weights = [pfsp_weight(self.context.pfsp_score(layout, owner, c), config.pfsp.weighting,
                               config.pfsp.exponent) for c in candidates]
        return self._rng.choices(list(candidates), weights=weights, k=1)[0]

    def _weighted(self, items: Sequence[PlayerKey], weights: Sequence[float]) -> PlayerKey:
        if sum(weights) <= 0:
            return self._rng.choice(list(items))
        return self._rng.choices(list(items), weights=list(weights), k=1)[0]

    # -- teams ---------------------------------------------------------

    def _fill_team(self, config: MatchmakingConfig, anchors: Mapping[str, float], core: PlayerKey, source: str,
                   layout: str, members: Sequence[int], seats: list[SeatAssignment | None]) -> None:
        ctx = self.context
        seat_specs = ctx.spec.layouts[layout]
        core_roles = ctx.agents[core[0]].roles
        core_seat = self._rng.choice([s for s in members if seat_specs[s].role in core_roles])
        for s in members:
            role = seat_specs[s].role
            if s == core_seat or (role in core_roles and config.teammates == "self"):
                player = core
            elif role in core_roles:
                player = self._mixed_teammate(config, anchors, core, role)
            else:
                player = self._role_player(anchors, role)
            seats[s] = self._seat(player, source)

    def _mixed_teammate(self, config: MatchmakingConfig, anchors: Mapping[str, float], core: PlayerKey,
                        role: str) -> PlayerKey:
        if self._rng.random() < config.teammate_self_prob:
            return core
        ctx = self.context
        agent = core[0]
        candidates: list[PlayerKey] = [(a, LATEST_NETWORK_ID) for a in ctx.trainable
                                       if a != agent and role in ctx.agents[a].roles]
        candidates += [(agent, c) for c in ctx.snapshots(agent)]
        candidates += [(a, FIXED_NETWORK_ID) for a, w in anchors.items()
                       if w > 0 and a != agent and a in ctx.agents and role in ctx.agents[a].roles]
        return self._rng.choice(candidates) if candidates else core

    def _role_player(self, anchors: Mapping[str, float], role: str) -> PlayerKey:
        ctx = self.context
        players = [a for a in ctx.trainable if role in ctx.agents[a].roles]
        if players:
            return self._rng.choice(players), LATEST_NETWORK_ID
        options = [a for a in anchors if a in ctx.agents and role in ctx.agents[a].roles]
        if not options:
            raise RuntimeError(f"nobody plays role {role!r} (validate_matchmaking should have caught it)")
        return self._weighted([(a, FIXED_NETWORK_ID) for a in options], [anchors[a] for a in options])

    def _seat(self, player: PlayerKey, source: str) -> SeatAssignment:
        agent, network = player
        collect = network == LATEST_NETWORK_ID and self.context.agents[agent].kind == "trainable"
        return SeatAssignment(agent_id=agent, network_id=network, collect=collect, source=source)


# ---------------------------------------------------------------------------
# Config checks and the validate printout
# ---------------------------------------------------------------------------


def _fmt_shares(shares: Mapping[str, float]) -> str:
    return ", ".join(f"{c} {shares.get(c, 0.0):.2f}" for c in CATEGORIES)


def validate_matchmaking(spec: GameSpec, player_roles: Mapping[str, Sequence[str]], config: ColosseumConfig) -> None:
    """Spec block 5 «Проверки конфига»; ConfigError with a fix hint. Per trainable agent O (its
    effective config):

    - ``layouts`` names layouts of the game; ``anchors`` names scripted or frozen agents;
    - O has at least one playable layout;
    - every role of every playable layout is played by a trainable agent or by an anchor of O
      (with a positive weight at some schedule point);
    - for every playable layout, opposing team and schedule point some category with a positive
      share can be filled (snapshots count as available).
    """
    context = MatchmakerContext.from_config(config, spec, player_roles, rng=random.Random(0))
    fixed = set(context.fixed)
    trainable_roles = {role for a in context.trainable for role in context.agents[a].roles}
    for owner in context.trainable:
        m = context.matchmaking[owner]
        where = f"agent {owner!r}"
        unknown = sorted(set(m.layouts) - set(spec.layouts))
        if unknown:
            raise ConfigError(f"{where}: matchmaking.layouts names unknown layouts {unknown}; the game's layouts are "
                              f"{sorted(spec.layouts)}")
        for name in list(m.anchors) if m.anchors is not None else []:
            if name not in context.agents:
                raise ConfigError(f"{where}: matchmaking.anchors names {name!r}, which is not an agent of the config; "
                                  f"anchors are scripted or frozen agents ({sorted(fixed)})")
            if name not in fixed:
                raise ConfigError(f"{where}: matchmaking.anchors names trainable agent {name!r}; anchors are scripted "
                                  f"or frozen agents: use opponents.rivals to meet other trainable agents, or declare "
                                  f"a snapshot as a frozen agent with path")
        roles = context.agents[owner].roles
        layouts = playable_layouts(spec, m, roles)
        if not layouts:
            raise ConfigError(f"{where} (roles {sorted(roles)}) has no seat in its enabled layouts "
                              f"{sorted(enabled_layouts(spec, m))}; check agents.{owner}.roles and matchmaking.layouts")
        points = schedule_points_of(m)
        live_anchors = {a for p in points for a, w in context.anchors_at(owner, p).items() if w > 0}
        covered = trainable_roles | {role for a in live_anchors for role in context.agents[a].roles}
        for layout in layouts:
            missing = sorted({seat.role for seat in spec.layouts[layout]} - covered)
            if missing:
                raise ConfigError(
                    f"{where}, layout {layout!r}: no trainable agent and no anchor of {owner!r} plays role(s) "
                    f"{missing}; add an agent or a scripted/frozen anchor with these roles, or leave {layout!r} out "
                    f"of matchmaking.layouts"
                )
            for team in opposing_teams(spec, layout, roles):
                for point in points:
                    shares, available = _structural(context, owner, layout, team, point)
                    if not spread(shares, available):
                        raise ConfigError(
                            f"{where}, layout {layout!r}, opposing team {team}: at env step {point} no opponent "
                            f"category with a positive share can be filled (shares: {_fmt_shares(shares)}; "
                            f"fillable: {sorted(available) or 'none'}); give a positive share to one of "
                            f"{sorted(available) or ['anchors (add an anchor)']}"
                        )


def describe_mix(config: ColosseumConfig, spec: GameSpec) -> list[str]:
    """``validate`` lines: every trainable agent's effective mix per playable layout and opposing
    team at env step 0 and at the last schedule point, and its anchors with weights."""
    from colosseum.players.registry import resolve_player_roles

    context = MatchmakerContext.from_config(config, spec, resolve_player_roles(config, spec), rng=random.Random(0))
    lines: list[str] = []
    for owner in context.trainable:
        m = context.matchmaking[owner]
        last = schedule_points_of(m)[-1]
        steps = [0, last] if last > 0 else [0]
        lines.append(f"agent {owner!r}: opponents by pfsp {m.pfsp.weighting} (exponent {m.pfsp.exponent:g}, "
                     f"half-life {m.pfsp.halflife_games:g} games), teammates {m.teammates}")
        roles = context.agents[owner].roles
        for layout in playable_layouts(spec, m, roles):
            teams = opposing_teams(spec, layout, roles)
            if not teams:
                lines.append(f"  {layout}: no opposing team")
            for team in teams:
                parts = [f"step {p}: {_fmt_shares(effective_mix(context, owner, layout, team, p))}" for p in steps]
                lines.append(f"  {layout}, team {team}: " + "; ".join(parts))
        first, final = context.anchors_at(owner, steps[0]), context.anchors_at(owner, steps[-1])
        if not first:
            lines.append("  anchors: none")
        else:
            lines.append("  anchors: " + ", ".join(
                f"{a} {first[a]:g}" if first[a] == final[a] else f"{a} {first[a]:g} -> {final[a]:g}" for a in first))
    return lines
