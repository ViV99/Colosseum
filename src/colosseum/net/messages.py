"""Protocol v1 constants and the only Python <-> protobuf converters (SP5 spec block 4.9 "Управляющие сообщения").

Session messages are used as their generated classes (``colosseum.net.proto.colosseum_v1_pb2``) everywhere; only the
types that the SP3 / SP4 code keeps as dataclasses are converted here:
- ``MatchResult`` / ``SeatResult`` / ``TeamResult`` <-> ``MatchResultProto`` (every SP4 field of the match record, plus
  ``MatchResult.node`` and ``SeatResult.generation``); ``None`` <-> an unset ``optional`` field;
- ``Lineup`` / ``SeatAssignment`` <-> ``LineupProto`` (``seed`` / ``block`` / ``uid`` / ``epoch`` optional);
- :class:`VersionInfo` (the hub's version registry entry, spec 4.5) <-> ``VersionInfo``.

The converters copy values verbatim and never raise on a well-formed protobuf: a result from a node is checked by
the hub's intake (``colosseum.hub.state.ResultIntake.check``), not here. ``to_proto`` raises ValueError for values a
field cannot hold (an int beyond int64).

``PROTOCOL_VERSION`` = (major, minor): a peer with another major is refused at ``Bootstrap/Info`` and ``Hello``
(:func:`protocol_compatible`); a minor bump adds fields old peers ignore.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from colosseum.core.types import Lineup, MatchResult, SeatAssignment, SeatResult, TeamResult
from colosseum.net.proto import colosseum_v1_pb2 as pb

PROTOCOL_VERSION = (1, 0)

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
