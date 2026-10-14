"""DAgger labels in chunk v2 (spec block 6): buffers write them on ACT slots only, payloads and gRPC
validation carry them with their dtypes, the slot rules reject a label elsewhere; WeightPayload's
teacher flag (T4.5)."""
from __future__ import annotations

import pickle

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.core.ipc import assert_no_tensors
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.types import SLOT_BOOT, TrajectoryChunk, WeightPayload, validate_slot_structure
from colosseum.envs.game import RoleSpec
from colosseum.transport.serialization import validate_chunk_payload
from colosseum.worker.buffers import BufferSpec, RolloutBuffer
from game_helpers import chunk_v2_payload, make_test_model

OBS = gymnasium.spaces.Box(-1.0, 1.0, (3,), np.float32)
ACT = gymnasium.spaces.Dict({"move": gymnasium.spaces.Discrete(4),
                             "aim": gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32)})


def buffer(teacher: bool, slots: int = 4) -> RolloutBuffer:
    return RolloutBuffer(slots, BufferSpec(ObsSpec.from_space(OBS), ActionSpec.from_space(ACT), None, teacher=teacher))


def act(buf: RolloutBuffer, label=None) -> None:
    action = {"move": np.int64(1), "aim": np.zeros(2, np.float32)}
    buf.write_act(np.zeros(3, np.float32), None, None, action, -1.0, None, 0.0, teacher_action=label)


def test_a_teacher_buffer_labels_act_slots_only_and_keeps_dtypes():
    buf = buffer(teacher=True)
    buf.begin(None, 0)
    act(buf, {"move": np.int64(3), "aim": np.array([0.5, -0.5], np.float32)})
    act(buf)                                       # the teacher was not asked (e.g. its flag is off)
    buf.mark_terminal()
    buf.write_pad()
    buf.write_pad()
    chunk = buf.build_chunk("a")
    assert chunk.has_teacher.tolist() == [True, False, False, False]
    assert chunk.teacher_action["move"].dtype == torch.int64 and chunk.teacher_action["aim"].dtype == torch.float32
    assert chunk.teacher_action["move"].tolist() == [3, 0, 0, 0]
    assert chunk.teacher_action["aim"][0].tolist() == [0.5, -0.5] and not chunk.teacher_action["aim"][1:].any()
    validate_slot_structure(chunk)


def test_buffers_without_a_teacher_have_no_label_fields():
    buf = buffer(teacher=False, slots=2)
    buf.begin(None, 0)
    with pytest.raises(ValueError, match="teacher"):
        act(buf, {"move": np.int64(3), "aim": np.zeros(2, np.float32)})
    act(buf)
    buf.write_boot(np.zeros(3, np.float32), None, reset_after=False)
    chunk = buf.build_chunk("a")
    assert chunk.teacher_action is None and chunk.has_teacher is None


def test_payload_round_trip_carries_the_labels():
    buf = buffer(teacher=True, slots=2)
    buf.begin(None, 3)
    act(buf, {"move": np.int64(2), "aim": np.array([0.25, 0.0], np.float32)})
    buf.write_boot(np.zeros(3, np.float32), None, reset_after=False)
    payload = buf.build_chunk("a").to_payload()
    assert_no_tensors(payload)
    payload = pickle.loads(pickle.dumps(payload))
    assert payload["teacher_action"]["move"].dtype == np.int64 and payload["has_teacher"].dtype == np.bool_
    validate_chunk_payload(payload)
    back = TrajectoryChunk.from_payload(payload)
    assert back.has_teacher.tolist() == [True, False] and back.teacher_action["move"].tolist() == [2, 0]
    moved = back.to("cpu")
    assert torch.equal(moved.has_teacher, back.has_teacher)


def test_payloads_without_the_fields_still_load():
    chunk = TrajectoryChunk.from_payload(chunk_v2_payload(pattern="AAB"))
    assert chunk.teacher_action is None and chunk.has_teacher is None
    validate_chunk_payload(chunk.to_payload())


def test_a_label_on_a_non_act_slot_breaks_the_slot_rules():
    payload = chunk_v2_payload(pattern="AAB")
    payload["teacher_action"] = np.zeros(3, np.int64)
    payload["has_teacher"] = np.array([True, False, True])
    assert payload["kind"][2] == SLOT_BOOT
    with pytest.raises(ValueError, match="teacher label on a non-ACT slot"):
        validate_slot_structure(TrajectoryChunk.from_payload(payload))
    with pytest.raises(ValueError, match="teacher label on a non-ACT slot"):
        validate_chunk_payload(payload)
    half = chunk_v2_payload(pattern="AAB")
    half["has_teacher"] = np.array([True, False, False])
    with pytest.raises(ValueError, match="come together"):
        validate_chunk_payload(half)


def test_weight_payload_carries_the_teacher_flag():
    model = make_test_model(RoleSpec(OBS, ACT))
    assert WeightPayload("a", 0).teacher_active is False
    assert WeightPayload.from_model("a", 1, model).teacher_active is False
    payload = WeightPayload.from_model("a", 1, model, teacher_active=True)
    assert payload.teacher_active is True
    assert_no_tensors(payload)
