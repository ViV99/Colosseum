"""Protocol v1 constants and the only Python <-> protobuf converters (SP5 spec block 4.9 "Управляющие сообщения").

Session messages are used as their generated classes (``colosseum.net.proto.colosseum_v1_pb2``) everywhere; only the
types that the SP3 / SP4 code keeps as dataclasses are converted here:
- ``MatchResult`` / ``SeatResult`` / ``TeamResult`` <-> ``MatchResultProto`` (every SP4 field of the match record, plus
  ``MatchResult.node`` and ``SeatResult.generation``); ``None`` <-> an unset ``optional`` field;
- ``Lineup`` / ``SeatAssignment`` <-> ``LineupProto`` (``seed`` / ``block`` / ``uid`` / ``epoch`` optional);
- :class:`VersionInfo` (the hub's version registry entry, spec 4.5) <-> ``VersionInfo``.

Evaluation (T5.1, spec 5.3). :class:`PlayerRef` (one player of an evaluation block as an evaluator builds it) and
:class:`BlockTask` (one evaluation block as the hub queues it) live here because the hub, the node and the evaluator
all need them; ``player_ref_to_proto`` / ``player_ref_from_proto``, ``block_task_to_proto`` /
``block_task_from_proto`` and ``lease_to_proto`` convert them. A neural player's weights are the eval blob
``blob_key("eval", agent_id, blob_id)`` of the machine cache (A18): the wire carries the raw ``blob_id``
(``BlobRequest(kind="eval")``), ``player_ref_from_proto`` rebuilds the key and raises ValueError for a malformed id
(the node's untrusted input check). Bot kwargs and networks travel as JSON (ValueError for values JSON cannot hold).

The converters copy values verbatim and never raise on a well-formed protobuf: a result from a node is checked by
the hub's intake (``colosseum.hub.state.ResultIntake.check``), not here. ``to_proto`` raises ValueError for values a
field cannot hold (an int beyond int64).

``PROTOCOL_VERSION`` = (major, minor): a peer with another major is refused at ``Bootstrap/Info`` and ``Hello``
(:func:`protocol_compatible`); a minor bump adds fields old peers ignore.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from colosseum.core.types import Lineup, MatchResult, SeatAssignment, SeatResult, TeamResult
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node.weights import CacheKey, blob_key, split_blob_key
from colosseum.players.registry import BotSpec

PROTOCOL_VERSION = (1, 0)

TOURNAMENT_PRIORITY = 0      # spec 5.3: tournament blocks go before pulse blocks
PULSE_PRIORITY = 1

# Every value of ``Event.kind`` (amendment A1): the one validity list of node events.
EVENT_KINDS = ("proc_died", "child_error", "slot_disabled", "learner_failed", "warning", "run_stopped",
               "node_stopped", "node_leaving", "code_updated", "code_update_failed")


def protocol_compatible(major: int, minor: int) -> bool:
    """Whether a peer speaking ``major.minor`` can talk to this code (same major; any minor)."""
    return int(major) == PROTOCOL_VERSION[0]


@dataclass(frozen=True)
class VersionInfo:
    """One announced weights version of an agent (spec 4.5): where a node fetches it from. ``digest`` is the blake2b-128
    hex of the weights frame, ``node`` the learner's node number (-1: parked, hub-held weights), ``addresses`` its
    data-port addresses (empty or ``relay_only``: through the hub)."""

    agent_id: str
    generation: int
    version: int
    size: int
    digest: str
    node: int
    addresses: tuple[str, ...] = ()
    relay_only: bool = False
    teacher_active: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["addresses"] = list(self.addresses)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> VersionInfo:
        return cls(agent_id=str(d["agent_id"]), generation=int(d["generation"]), version=int(d["version"]),
                   size=int(d["size"]), digest=str(d["digest"]), node=int(d["node"]),
                   addresses=tuple(d.get("addresses", ())), relay_only=bool(d.get("relay_only", False)),
                   teacher_active=bool(d.get("teacher_active", False)))


def _opt(message, field: str):
    return getattr(message, field) if message.HasField(field) else None


def _set_opt(message, field: str, value) -> None:
    if value is not None:
        setattr(message, field, value)


def seat_result_to_proto(s: SeatResult) -> pb.SeatResultProto:
    p = pb.SeatResultProto(seat=s.seat, role=s.role, team=s.team, agent_id=s.agent_id, network_id=s.network_id,
                           reward=float(s.reward), source=s.source)
    _set_opt(p, "eliminated_step", s.eliminated_step)
    _set_opt(p, "version_start", s.version_start)
    _set_opt(p, "version_end", s.version_end)
    _set_opt(p, "generation", s.generation)
    return p


def seat_result_from_proto(p: pb.SeatResultProto) -> SeatResult:
    return SeatResult(seat=p.seat, role=p.role, team=p.team, agent_id=p.agent_id, network_id=p.network_id,
                      reward=p.reward, eliminated_step=_opt(p, "eliminated_step"), source=p.source,
                      version_start=_opt(p, "version_start"), version_end=_opt(p, "version_end"),
                      generation=_opt(p, "generation"))


def result_to_proto(r: MatchResult) -> pb.MatchResultProto:
    p = pb.MatchResultProto(match_id=r.match_id, layout=r.layout, outcome_kind=r.outcome_kind,
                            seats=[seat_result_to_proto(s) for s in r.seats],
                            teams=[pb.TeamResultProto(team=t.team, rank=float(t.rank), score=float(t.score))
                                   for t in r.teams],
                            episode_length=r.episode_length, uid=r.uid, source=r.source)
    _set_opt(p, "seed", r.seed)
    _set_opt(p, "env_steps", r.env_steps)
    _set_opt(p, "block", r.block)
    _set_opt(p, "node", r.node)
    return p


def result_from_proto(p: pb.MatchResultProto) -> MatchResult:
    return MatchResult(match_id=p.match_id, layout=p.layout, outcome_kind=p.outcome_kind,  # type: ignore[arg-type]
                       seats=[seat_result_from_proto(s) for s in p.seats],
                       teams=[TeamResult(team=t.team, rank=t.rank, score=t.score) for t in p.teams],
                       episode_length=p.episode_length, uid=p.uid, source=p.source, seed=_opt(p, "seed"),
                       env_steps=_opt(p, "env_steps"), block=_opt(p, "block"), node=_opt(p, "node"))


def results_to_batch(seq: int, results: Iterable[MatchResult]) -> pb.ResultBatch:
    return pb.ResultBatch(seq=int(seq), results=[result_to_proto(r) for r in results])


def results_from_batch(p: pb.ResultBatch) -> list[MatchResult]:
    return [result_from_proto(r) for r in p.results]


def lineup_to_proto(lineup: Lineup) -> pb.LineupProto:
    p = pb.LineupProto(layout=lineup.layout,
                       seats=[pb.SeatProto(agent_id=s.agent_id, network_id=s.network_id, collect=s.collect,
                                           source=s.source) for s in lineup.seats])
    _set_opt(p, "seed", lineup.seed)
    _set_opt(p, "block", lineup.block)
    _set_opt(p, "uid", lineup.uid)
    _set_opt(p, "epoch", lineup.epoch)
    return p


def lineup_from_proto(p: pb.LineupProto) -> Lineup:
    return Lineup(layout=p.layout,
                  seats=[SeatAssignment(agent_id=s.agent_id, network_id=s.network_id, collect=s.collect,
                                        source=s.source) for s in p.seats],
                  seed=_opt(p, "seed"), block=_opt(p, "block"), uid=_opt(p, "uid"), epoch=_opt(p, "epoch"))


def version_to_proto(v: VersionInfo) -> pb.VersionInfo:
    return pb.VersionInfo(agent_id=v.agent_id, generation=v.generation, version=v.version, size=v.size,
                          digest=v.digest, node=v.node, addresses=list(v.addresses), relay_only=v.relay_only,
                          teacher_active=v.teacher_active)


def version_from_proto(p: pb.VersionInfo) -> VersionInfo:
    return VersionInfo(agent_id=p.agent_id, generation=p.generation, version=p.version, size=p.size,
                       digest=p.digest, node=p.node, addresses=tuple(p.addresses), relay_only=p.relay_only,
                       teacher_active=p.teacher_active)


# ---------------------------------------------------------------------------------------------------------------
# Evaluation (T5.1)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerRef:
    """One player of an evaluation block as an evaluator builds it (spec 5.3): a neural player's weights are the
    machine-cache blob ``blob`` (``blob_key("eval", agent_id, blob_id)``, fetched from the hub with
    ``BlobRequest(kind="eval")``) built with ``networks`` (a ``NetworkConfig`` dump) for ``roles``; a scripted player
    is ``bot``. ``(agent_id, network_id)`` is how the block's seats name it; ``name`` is the entity name used in
    messages."""

    name: str
    agent_id: str
    network_id: str
    roles: tuple[str, ...]
    kind: Literal["neural", "scripted"]
    blob: CacheKey | None = None
    networks: dict | None = None
    bot: BotSpec | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("neural", "scripted"):
            raise ValueError(f"player {self.name!r}: kind must be neural or scripted, got {self.kind!r}")
        if self.kind == "neural" and not _is_eval_key(self.blob):
            raise ValueError(f"player {self.name!r}: a neural player needs an eval blob cache key "
                             f"(blob_key('eval', ...)), got {self.blob!r}")
        if self.kind == "scripted" and self.bot is None:
            raise ValueError(f"player {self.name!r}: a scripted player needs a bot")


def _is_eval_key(key: CacheKey | None) -> bool:
    if key is None:
        return False
    try:
        return split_blob_key(key)[0] == "eval"
    except ValueError:
        return False


@dataclass(frozen=True)
class BlockTask:
    """One evaluation block (an SP4 schedule block: one seed, one composition, every rotation) as the hub queues
    it: ``lineups`` carry the block's seed, id and uids; ``players`` every player its seats name; ``owner`` the pulse
    snapshot (``pulse:<agent>:<checkpoint id>``) or tournament (``tournament:<name>``) it belongs to; ``priority``
    ``TOURNAMENT_PRIORITY`` runs before ``PULSE_PRIORITY``; ``num_envs`` the envs an evaluator plays its lease
    with. On a node (``block_task_from_proto``) ``owner`` is "" and ``priority`` 0."""

    block_id: str
    source: Literal["pulse", "tournament"]
    owner: str
    lineups: tuple[Lineup, ...]
    players: tuple[PlayerRef, ...]
    priority: int
    deterministic: bool
    num_envs: int = 8

    @property
    def matches(self) -> int:
        return len(self.lineups)


def _json(value: Any, what: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what} is not JSON ({e}); hub-run evaluation sends it to nodes as JSON") from e


def player_ref_to_proto(ref: PlayerRef) -> pb.PlayerRefProto:
    return pb.PlayerRefProto(
        name=ref.name, agent_id=ref.agent_id, network_id=ref.network_id, roles=list(ref.roles), kind=ref.kind,
        blob_id="" if ref.blob is None else split_blob_key(ref.blob)[1],
        networks_json="" if ref.networks is None else _json(ref.networks, f"the networks of {ref.name}"),
        bot_class="" if ref.bot is None else ref.bot.class_path,
        bot_kwargs_json="" if ref.bot is None else _json(dict(ref.bot.kwargs), f"the bot kwargs of {ref.name}"))


def player_ref_from_proto(p: pb.PlayerRefProto) -> PlayerRef:
    """ValueError for a malformed blob id, kind or JSON (node-side input check)."""
    return PlayerRef(
        name=p.name, agent_id=p.agent_id, network_id=p.network_id, roles=tuple(p.roles), kind=p.kind,  # type: ignore[arg-type]
        blob=blob_key("eval", p.agent_id, p.blob_id) if p.blob_id else None,
        networks=json.loads(p.networks_json) if p.networks_json else None,
        bot=BotSpec(p.bot_class, json.loads(p.bot_kwargs_json) if p.bot_kwargs_json else {}) if p.bot_class else None)


def block_task_to_proto(task: BlockTask) -> pb.BlockSpec:
    return pb.BlockSpec(block_id=task.block_id, source=task.source,
                        lineups=[lineup_to_proto(lu) for lu in task.lineups],
                        players=[player_ref_to_proto(p) for p in task.players], deterministic=task.deterministic)


def block_task_from_proto(p: pb.BlockSpec, *, num_envs: int) -> BlockTask:
    return BlockTask(block_id=p.block_id, source=p.source, owner="",  # type: ignore[arg-type]
                     lineups=tuple(lineup_from_proto(x) for x in p.lineups),
                     players=tuple(player_ref_from_proto(x) for x in p.players), priority=0,
                     deterministic=p.deterministic, num_envs=int(num_envs))


def lease_to_proto(lease_id: str, lease_sec: float, tasks: Sequence[BlockTask]) -> pb.BlockLease:
    return pb.BlockLease(lease_id=lease_id, lease_sec=float(lease_sec), blocks=[block_task_to_proto(t) for t in tasks],
                         num_envs=tasks[0].num_envs if tasks else 0)
