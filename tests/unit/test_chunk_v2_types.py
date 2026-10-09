"""SP2 core types: chunk v2 and its numpy payload, lineups, results, commands (T3.1)."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest
import torch

from colosseum.core.ipc import assert_no_tensors
from colosseum.sp2.core.types import (
    LATEST_NETWORK_ID,
    SLOT_ACT,
    SLOT_BOOT,
    SLOT_PAD,
    Lineup,
    MatchResult,
    SeatAssignment,
    SeatResult,
    TeamResult,
    TrajectoryChunk,
    WeightPayload,
    WorkerCommand,
    state_dict_from_numpy,
    state_dict_to_numpy,
    validate_slot_structure,
)
from game_helpers import chunk_v2_payload


def _chunk(*, with_gs: bool, with_masks: bool, with_units: bool, state: str) -> TrajectoryChunk:
    S = 4
    obs = {
        "grid": torch.randint(0, 255, (S, 3, 3), dtype=torch.uint8),
        "vec": torch.randn(S, 2),
    }
    actions = {"move": torch.randint(0, 3, (S,)), "units": torch.randint(0, 4, (S, 5))}
    masks = None
    if with_masks:
        masks = {
            "move": torch.ones(S, 3, dtype=torch.bool),
            "units": {"unit": torch.ones(S, 5, dtype=torch.bool), "action": torch.ones(S, 5, 4, dtype=torch.bool)},
        }
    initial_state = {
        "none": None,
        "dict": {"h": torch.randn(1, 1, 8), "c": torch.randn(1, 1, 8)},
        "tuple": (torch.randn(1, 3), torch.zeros(1, dtype=torch.long)),
    }[state]
    return TrajectoryChunk(
        agent_id="a",
        policy_version=7,
        initial_state=initial_state,
        obs=obs,
        global_state={"map": torch.randint(0, 2, (S, 4, 4), dtype=torch.int8)} if with_gs else None,
        actions=actions,
        action_masks=masks,
        kind=torch.tensor([SLOT_ACT, SLOT_ACT, SLOT_BOOT, SLOT_PAD], dtype=torch.int8),
        reward=torch.tensor([1.0, 0.5, 0.0, 0.0]),
        terminal=torch.tensor([False, False, False, False]),
        reset_after=torch.tensor([False, False, True, True]),
        behavior_logp=torch.tensor([-1.0, -2.0, 0.0, 0.0]),
        behavior_unit_logp=torch.randn(S, 6) if with_units else None,
    )


def _assert_trees_equal(a, b) -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b)
        for key in a:
            _assert_trees_equal(a[key], b[key])
    elif isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        for x, y in zip(a, b):
            _assert_trees_equal(x, y)
    elif a is None:
        assert b is None
    else:
        assert isinstance(b, torch.Tensor) and b.dtype == a.dtype, (a.dtype, getattr(b, "dtype", None))
        assert torch.equal(a, b)


def test_slot_kind_values_and_latest_id():
    assert (SLOT_ACT, SLOT_BOOT, SLOT_PAD) == (0, 1, 2)
    assert LATEST_NETWORK_ID == "latest"


@pytest.mark.parametrize("state", ["none", "dict", "tuple"])
@pytest.mark.parametrize("with_gs,with_masks,with_units", [
    (False, False, False), (True, True, True), (False, True, False), (True, False, True),
])
def test_chunk_payload_roundtrip_preserves_trees_and_dtypes(state, with_gs, with_masks, with_units):
    chunk = _chunk(with_gs=with_gs, with_masks=with_masks, with_units=with_units, state=state)
    payload = chunk.to_payload()
    assert_no_tensors(payload)
    payload = pickle.loads(pickle.dumps(payload))      # what an mp.Queue does
    assert payload["obs"]["grid"].dtype == np.uint8
    back = TrajectoryChunk.from_payload(payload)
    assert back.agent_id == "a" and back.policy_version == 7
    for f in dataclasses.fields(TrajectoryChunk):
        if f.name not in ("agent_id", "policy_version"):
            _assert_trees_equal(getattr(chunk, f.name), getattr(back, f.name))


def test_chunk_counts_slots_and_acts():
    chunk = _chunk(with_gs=False, with_masks=False, with_units=False, state="none")
    assert chunk.num_slots == 4
    assert chunk.num_acts == 2


def test_chunk_to_moves_every_tensor_including_state_leaves():
    chunk = _chunk(with_gs=True, with_masks=True, with_units=True, state="dict")
    moved = chunk.to("meta")
    leaves = [moved.kind, moved.reward, moved.terminal, moved.reset_after, moved.behavior_logp,
              moved.behavior_unit_logp, moved.obs["grid"], moved.obs["vec"], moved.global_state["map"],
              moved.actions["move"], moved.actions["units"], moved.action_masks["move"],
              moved.action_masks["units"]["unit"], moved.action_masks["units"]["action"],
              moved.initial_state["h"], moved.initial_state["c"]]
    assert all(t.device.type == "meta" for t in leaves)
    assert moved.obs["grid"].dtype == torch.uint8
    assert chunk.kind.device.type == "cpu"            # the original is untouched


def test_lineup_and_results_are_plain_picklable_dataclasses():
    lineup = Lineup(layout="2v2", seats=[SeatAssignment("a"), SeatAssignment("b", "ckpt_v3", collect=False)])
    assert lineup.seats[0].network_id == LATEST_NETWORK_ID and lineup.seats[0].collect is True
    result = MatchResult(
        match_id="w0_e1_ep2", layout="2p", outcome_kind="wdl",
        seats=[SeatResult(0, "player", 0, "a", "latest", 1.0),
               SeatResult(1, "player", 1, "b", "ckpt_v3", -1.0, eliminated_step=4)],
        teams=[TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)],
        episode_length=4,
    )
    assert result.seats[0].eliminated_step is None
    for obj in (lineup, result):
        assert_no_tensors(obj)
        assert pickle.loads(pickle.dumps(obj)) == obj


def test_worker_command_carries_lineups_and_numpy_checkpoints():
    cmd = WorkerCommand(
        lineups=[None, Lineup("solo", [SeatAssignment("a")])],
        new_checkpoints={"a": {"ckpt_v1": state_dict_to_numpy({"w": torch.ones(2)})}},
    )
    assert_no_tensors(cmd)
    assert WorkerCommand(lineups=[None]).new_checkpoints == {}


def test_weight_payload_and_state_dict_helpers_roundtrip():
    model = torch.nn.Linear(3, 2)
    payload = WeightPayload.from_model("a", 5, model)
    assert_no_tensors(payload)
    other = torch.nn.Linear(3, 2)
    other.load_state_dict(payload.to_torch_state_dict())
    assert torch.equal(other.weight, model.weight)
    sd = {"x": torch.ones(2, dtype=torch.bfloat16)}
    assert state_dict_from_numpy(state_dict_to_numpy(sd))["x"].dtype == torch.float32


def _chunk_of(pattern: str, **overrides) -> TrajectoryChunk:
    payload = chunk_v2_payload(pattern=pattern, agent_id="hero")
    payload.update(overrides)
    return TrajectoryChunk.from_payload(payload)


@pytest.mark.parametrize("pattern", ["AAAB", "ARTP", "ATPP", "TTTP", "ARAR", "AR", "AB"])
def test_validate_slot_structure_accepts_worker_shaped_chunks(pattern):
    validate_slot_structure(_chunk_of(pattern))


@pytest.mark.parametrize("pattern,slot,rule", [
    ("TTA", 2, "an ACT in the last slot"),
    ("ATAP", 2, "an open ACT followed by a PAD"),
    ("APTB", 0, "an open ACT followed by a PAD"),
    ("BAAB", 0, "a BOOT that does not follow an open ACT"),
    ("TBAB", 1, "a BOOT that does not follow an open ACT"),
    ("PAAB", 0, "a PAD that does not follow the end of an episode"),
    ("PPTP", 0, "a PAD that does not follow the end of an episode"),
    ("ABAB", 1, "a BOOT without reset_after before the last slot"),
])
def test_validate_slot_structure_names_the_agent_slot_and_rule(pattern, slot, rule):
    with pytest.raises(ValueError, match=rf"agent 'hero'.*slots '{pattern}'.*slot {slot}: {rule}"):
        validate_slot_structure(_chunk_of(pattern))


def test_validate_slot_structure_rejects_unknown_kinds_and_inconsistent_flags():
    with pytest.raises(ValueError, match=r"agent 'hero'.*slots 'A\?TP'.*slot 1: unknown slot kind 7"):
        validate_slot_structure(_chunk_of("AATP", kind=np.array([0, 7, 0, 2], np.int8)))
    with pytest.raises(ValueError, match=r"agent 'hero'.*kind, terminal and reset_after"):
        validate_slot_structure(_chunk_of("AAAB", terminal=np.zeros(3, np.bool_)))
    with pytest.raises(ValueError, match=r"agent 'hero'.*no slots"):
        validate_slot_structure(_chunk_of(""))


def _flags(pattern: str, name: str, slot: int, value: bool) -> np.ndarray:
    flags = chunk_v2_payload(pattern=pattern)[name].copy()
    flags[slot] = value
    return flags


@pytest.mark.parametrize("pattern,name,slot,value,rule", [
    ("ATAB", "reset_after", 1, False, "a terminal ACT without reset_after"),
    ("AAAB", "reset_after", 1, True, "an open ACT with reset_after"),
    ("ATPP", "reset_after", 3, False, "a PAD without reset_after"),
    ("AAAB", "terminal", 3, True, "a terminal flag on a non-ACT slot"),
    ("AAAR", "terminal", 3, True, "a terminal flag on a non-ACT slot"),
    ("ATPP", "terminal", 2, True, "a terminal flag on a non-ACT slot"),
])
def test_validate_slot_structure_checks_flags_against_the_slot_kind(pattern, name, slot, value, rule):
    chunk = _chunk_of(pattern, **{name: _flags(pattern, name, slot, value)})
    with pytest.raises(ValueError, match=rf"agent 'hero'.*slots '{pattern}'.*slot {slot}: {rule}$"):
        validate_slot_structure(chunk)
