"""Evaluation messages (SP5 T5.1, spec 5.3): PlayerRef / BlockTask round trips through protobuf, their checks,
and the protocol fields this task uses (T1.2 declared them, A1)."""
from __future__ import annotations

import pytest

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.net.messages import (
    BlockTask,
    PlayerRef,
    block_task_from_proto,
    lease_to_proto,
    player_ref_from_proto,
    player_ref_to_proto,
)
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node.weights import blob_key
from colosseum.players.registry import BotSpec

NETWORKS = {"model_class": "game_helpers.GameTestModel", "kwargs": {"core": "none", "hidden": 16}}
NEURAL = PlayerRef("main@v3", "main", "ckpt_v3", ("player",), "neural",
                   blob=blob_key("eval", "main", "pulse-main-ckpt_v3"), networks=NETWORKS)
BOT = PlayerRef("ones", "ones", "fixed", ("player",), "scripted", bot=BotSpec("game_helpers.ConstantBot", {"value": 1}))


def _lineups(block: str) -> tuple[Lineup, ...]:
    return tuple(Lineup("2p", [SeatAssignment("main", "ckpt_v3", False), SeatAssignment("ones", "fixed", False)],
                        seed=11, block=block, uid=f"{block}/r{r}") for r in range(2))


def test_player_refs_round_trip_through_protobuf():
    for ref in (NEURAL, BOT):
        assert player_ref_from_proto(player_ref_to_proto(ref)) == ref
    p = player_ref_to_proto(NEURAL)
    assert p.blob_id == "pulse-main-ckpt_v3" and p.kind == "neural" and not p.bot_class   # the raw blob id (A18)
    assert player_ref_from_proto(p).blob.ident == "eval-pulse-main-ckpt_v3"


def test_player_refs_check_their_kind():
    with pytest.raises(ValueError, match="blob"):
        PlayerRef("x", "x", "fixed", ("player",), "neural")
    with pytest.raises(ValueError, match="blob"):                     # an eval blob key only (P-PF4-2)
        PlayerRef("x", "x", "fixed", ("player",), "neural", blob=blob_key("fixed", "x", "x"))
    with pytest.raises(ValueError, match="bot"):
        PlayerRef("x", "x", "fixed", ("player",), "scripted")
    with pytest.raises(ValueError, match="kind"):
        PlayerRef("x", "x", "fixed", ("player",), "human", bot=BOT.bot)


def test_a_malformed_blob_id_from_the_wire_is_refused():
    p = player_ref_to_proto(NEURAL)
    p.blob_id = "../escape"
    with pytest.raises(ValueError, match="blob id"):
        player_ref_from_proto(p)


def test_non_json_bot_kwargs_are_refused_with_the_player_name():
    ref = PlayerRef("odd", "odd", "fixed", ("player",), "scripted",
                    bot=BotSpec("game_helpers.ConstantBot", {"v": {1, 2}}))
    with pytest.raises(ValueError, match="bot kwargs of odd"):
        player_ref_to_proto(ref)


def test_block_tasks_and_leases_round_trip():
    tasks = [BlockTask(f"t/A/2p/main@v3/ones/{k}", "tournament", "tournament:t", _lineups(f"t/A/2p/main@v3/ones/{k}"),
                       (NEURAL, BOT), 0, True, 3) for k in range(2)]
    lease = lease_to_proto("L1", 600.0, tasks)
    assert (lease.lease_id, lease.lease_sec, lease.num_envs, len(lease.blocks)) == ("L1", 600.0, 3, 2)
    back = [block_task_from_proto(b, num_envs=lease.num_envs) for b in lease.blocks]
    for got, want in zip(back, tasks):
        assert (got.block_id, got.source, got.deterministic, got.num_envs) == (want.block_id, "tournament", True, 3)
        assert (got.owner, got.priority) == ("", 0)                         # hub-side fields stay on the hub
        assert got.lineups == want.lineups and got.players == want.players
        assert got.matches == 2


def test_the_session_fields_exist():
    hello = pb.Hello(leases=["L1", "L2"])
    assert list(hello.leases) == ["L1", "L2"]
    assert pb.BlockLease(num_envs=4).num_envs == 4
    assert pb.BlockResult(error_kind="abandoned").error_kind == "abandoned"
    msg = pb.HubMessage(block_ack=pb.BlockAck(lease_id="L1", block_ids=["b0"]))
    assert msg.WhichOneof("m") == "block_ack"
    assert list(pb.Command(kind="cancel_leases", lease_ids=["L1"]).lease_ids) == ["L1"]
