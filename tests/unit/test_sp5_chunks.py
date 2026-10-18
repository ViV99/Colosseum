"""Chunk, weights and checkpoint frames and the learner's chunk checks (SP5 T1.1, spec block 4.9): chunk v2 round
trips with every optional field, the same content as the SP1 npz format, decode_chunk's field checks, every
ChunkValidator rule, buffer-built chunks of several role kinds passing, weights and checkpoint frames."""
from __future__ import annotations

import io
from collections import namedtuple

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.types import TrajectoryChunk
from colosseum.envs.game import RoleSpec
from colosseum.envs.spaces import Units
from colosseum.net.chunks import (
    ChunkError,
    ChunkMeta,
    ChunkValidator,
    WeightsMeta,
    decode_checkpoint,
    decode_chunk,
    decode_weights,
    encode_checkpoint,
    encode_chunk,
    encode_weights,
)
from colosseum.net.frame import FrameError, encode_frame, payload_byte_cap
from colosseum.networks.state import state_to_numpy
from colosseum.worker.buffers import BufferSpec, RolloutBuffer
from game_helpers import chunk_v2_payload, learner_role

CAP = payload_byte_cap(64)
LSTM = namedtuple("LSTM", ["h", "c"])
ROLE = learner_role(3)                                    # Box(4) obs, Discrete(3) actions (masks: bool[3])


def _chunk(**changes) -> TrajectoryChunk:
    payload = chunk_v2_payload(pattern="AATP")
    payload.update(changes)
    return TrajectoryChunk.from_payload(payload)


def _validator(**kwargs) -> ChunkValidator:
    return ChunkValidator(ROLE, agent_id="a", teacher=False, **kwargs)


def _round_trip(chunk: TrajectoryChunk, **kwargs) -> tuple[TrajectoryChunk, ChunkMeta]:
    return decode_chunk(encode_chunk(chunk, generation=2, node=5, worker=1, **kwargs), cap=CAP)


def _equal(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, tuple | list):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _chunks_equal(a: TrajectoryChunk, b: TrajectoryChunk) -> bool:
    fields = ("agent_id", "policy_version", "initial_state", "obs", "global_state", "actions", "action_masks", "kind",
              "reward", "terminal", "reset_after", "behavior_logp", "behavior_unit_logp", "teacher_action",
              "has_teacher")
    return all(_equal(getattr(a, f), getattr(b, f)) for f in fields)


@pytest.mark.parametrize("compress", [False, True])
def test_a_chunk_round_trips_with_every_optional_field(compress):
    payload = chunk_v2_payload(version=7, pattern="AATP")
    payload["obs"] = {"grid": np.arange(4 * 6, dtype=np.uint8).reshape(4, 2, 3), "vec": np.ones((4, 3), np.float32)}
    payload["global_state"] = np.zeros((4, 5), np.float32)
    payload["behavior_unit_logp"] = np.full((4, 2), -0.5, np.float32)
    payload["teacher_action"] = np.array([1, 2, 0, 0], np.int64)
    payload["has_teacher"] = np.array([True, True, False, False])
    state = LSTM(np.ones((1, 1, 8), np.float32), np.zeros((1, 1, 8), np.float32))
    payload["initial_state"] = {"core": state, "extra": [np.ones((1, 2), np.float32), (np.zeros(1, np.int64),)]}
    chunk = TrajectoryChunk.from_payload(payload)
    back, meta = _round_trip(chunk, compress=compress)
    assert meta == ChunkMeta(agent_id="a", policy_version=7, generation=2, node=5, worker=1)
    assert _chunks_equal(back, chunk)
    assert type(back.initial_state["core"]) is LSTM and torch.equal(back.initial_state["core"].h,
                                                                     torch.from_numpy(state.h))
    assert back.obs["grid"].dtype == torch.uint8 and back.kind.dtype == torch.int8


def test_frame_and_npz_formats_decode_to_the_same_content():
    """The frame replaces the SP1 npz format (removed in T7.2): the same payload decodes to the same chunk."""
    from colosseum.transport.serialization import deserialize_chunk_payload, serialize_chunk_payload

    payload = chunk_v2_payload(version=3, pattern="ATPP")
    payload["obs"] = {"a": np.ones((4, 2), np.float32), "b": {"c": np.arange(4, dtype=np.int16)}}
    payload["initial_state"] = state_to_numpy((torch.ones(1, 3), [torch.zeros(1, 2, dtype=torch.int64)]))
    chunk = TrajectoryChunk.from_payload(payload)
    via_npz = TrajectoryChunk.from_payload(deserialize_chunk_payload(*serialize_chunk_payload(chunk.to_payload())))
    via_frame, _ = _round_trip(chunk)
    assert _chunks_equal(via_frame, via_npz)


def test_the_meta_names_the_agent_and_never_needs_decoding_the_body():
    from colosseum.net.frame import peek_meta

    buf = encode_chunk(_chunk(), generation=0, node=-1, worker=3)
    assert peek_meta(buf) == ("chunk", {"agent_id": "a", "policy_version": 0, "generation": 0, "node": -1,
                                        "worker": 3})


@pytest.mark.parametrize("meta, detail", [
    ({"agent_id": "a b", "policy_version": 0, "generation": 0, "node": 0, "worker": 0}, "agent_id"),
    ({"agent_id": "a", "policy_version": -1, "generation": 0, "node": 0, "worker": 0}, "policy_version"),
    ({"agent_id": "a", "policy_version": True, "generation": 0, "node": 0, "worker": 0}, "policy_version"),
    ({"agent_id": "a", "policy_version": 0, "generation": 0, "node": -2, "worker": 0}, "node"),
    ({"agent_id": "a", "policy_version": 0, "generation": 0, "node": 0}, "meta keys"),
])
def test_decode_chunk_refuses_malformed_meta(meta, detail):
    payload = chunk_v2_payload()
    payload.pop("agent_id"), payload.pop("policy_version")
    payload.update(teacher_action=None, has_teacher=None)
    with pytest.raises(ChunkError) as info:
        decode_chunk(encode_frame(payload, kind="chunk", meta=meta), cap=CAP)
    assert info.value.reason == "invalid" and detail in info.value.detail


def _framed(**changes) -> bytes:
    payload = chunk_v2_payload()
    payload.pop("agent_id"), payload.pop("policy_version")
    payload.update(teacher_action=None, has_teacher=None)
    payload.update(changes)
    return encode_frame(payload, kind="chunk", meta={"agent_id": "a", "policy_version": 0, "generation": 0,
                                                     "node": 0, "worker": 0})


@pytest.mark.parametrize("changes, detail", [
    ({"kind": None}, "'kind'"),
    ({"reward": np.zeros(3, np.float32)}, "'reward'"),
    ({"obs": {"grid": np.zeros((3, 2), np.uint8)}}, "obs/grid"),
    ({"obs": [np.zeros(4)]}, "'obs'"),
    ({"obs": {1: np.zeros(4)}}, "'obs'"),
    ({"behavior_unit_logp": np.zeros(3, np.float32)}, "behavior_unit_logp"),
    ({"has_teacher": np.zeros(4, bool)}, "come together"),
    ({"initial_state": "x"}, "initial_state"),
    ({"kind": np.zeros(1, np.int8), "reward": np.zeros(1, np.float32), "terminal": np.zeros(1, bool),
      "reset_after": np.zeros(1, bool), "behavior_logp": np.zeros(1, np.float32), "obs": np.zeros((1, 4)),
      "actions": np.zeros(1, np.int64), "action_masks": np.ones((1, 3), bool)}, "at least 2 slots"),
])
def test_decode_chunk_refuses_malformed_fields(changes, detail):
    with pytest.raises(ChunkError) as info:
        decode_chunk(_framed(**changes), cap=CAP)
    assert info.value.reason == "invalid" and detail in info.value.detail


def test_decode_chunk_refuses_other_frames_and_missing_fields():
    with pytest.raises(ChunkError, match="frame:kind"):
        decode_chunk(encode_weights({"w": np.zeros(1, np.float32)}, WeightsMeta("a", "latest", 1, 0)), cap=CAP)
    with pytest.raises(ChunkError, match="frame:magic"):
        decode_chunk(b"nonsense-bytes-nonsense", cap=CAP)
    payload = chunk_v2_payload()
    payload.pop("agent_id"), payload.pop("policy_version")
    with pytest.raises(ChunkError, match="payload fields"):                 # no teacher fields at all
        decode_chunk(encode_frame(payload, kind="chunk", meta={"agent_id": "a", "policy_version": 0,
                                                               "generation": 0, "node": 0, "worker": 0}), cap=CAP)


# ---------------------------------------------------------------------------
# ChunkValidator (spec 4.9 "Проверка чанка на лёрнере")
# ---------------------------------------------------------------------------


def test_a_good_chunk_passes():
    _validator(chunk_length=4).validate(_chunk())


@pytest.mark.parametrize("chunk, detail", [
    (_chunk(agent_id="b"), "a chunk of agent 'b'"),
    (_chunk(obs=np.zeros((4, 5), np.float32)), "obs/<root>"),
    (_chunk(obs=np.zeros((4, 4), np.float64)), "obs/<root>"),
    (_chunk(obs={"x": np.zeros((4, 4), np.float32)}), "obs: expected an array"),
    (_chunk(global_state=np.zeros((4, 2), np.float32)), "global_state present"),
    (_chunk(action_masks=np.ones((4, 3), np.int8)), "action_masks/<root>"),
    (_chunk(action_masks=None), "action_masks missing"),
    (_chunk(actions=np.array([0, 3, 0, 0], np.int64)), "outside the action space"),
    (_chunk(actions=np.array([0, -1, 0, 0], np.int64)), "outside the action space"),
    (_chunk(actions=np.zeros(4, np.int32)), "actions/<root>"),
    (_chunk(behavior_logp=np.array([np.nan, 0, 0, 0], np.float32)), "non-finite behavior_logp"),
    (_chunk(behavior_logp=np.array([0, -np.inf, 0, 0], np.float32)), "non-finite behavior_logp"),
    (_chunk(reward=np.array([0, 0, 0, 1], np.float32)), "non-zero reward on a BOOT or PAD"),
    (_chunk(reward=np.array([np.inf, 0, 0, 0], np.float32)), "non-finite reward"),
    (_chunk(reward=np.zeros(4, np.float64)), "reward"),
    (_chunk(behavior_unit_logp=np.zeros((4, 2), np.float32)), "behavior_unit_logp present"),
    (_chunk(teacher_action=np.zeros(4, np.int64), has_teacher=np.zeros(4, bool)), "DAgger labels on a chunk"),
    (_chunk(kind=np.array([0, 0, 0, 0], np.int8), terminal=np.zeros(4, bool), reset_after=np.zeros(4, bool)),
     "an ACT in the last slot"),
])
def test_the_validator_refuses_each_broken_rule(chunk, detail):
    with pytest.raises(ChunkError) as info:
        _validator().validate(chunk)
    assert info.value.reason == "invalid" and detail in str(info.value)


def test_a_non_finite_logp_off_act_slots_and_the_chunk_length():
    _validator().validate(_chunk(behavior_logp=np.array([0, 0, 0, np.nan], np.float32)))   # the PAD slot
    with pytest.raises(ChunkError, match="expected rollout.chunk_length 8"):
        _validator(chunk_length=8).validate(_chunk())


def test_the_initial_state_must_match_the_models_template():
    template = state_to_numpy(LSTM(torch.zeros(1, 1, 8), torch.zeros(1, 1, 8)))
    good = _chunk(initial_state=state_to_numpy(LSTM(torch.ones(1, 1, 8), torch.ones(1, 1, 8))))
    _validator(state_template=template).validate(good)
    for bad in (None, (np.zeros((1, 1, 8), np.float32), np.zeros((1, 1, 8), np.float32)),
                LSTM(np.zeros((1, 1, 4), np.float32), np.zeros((1, 1, 8), np.float32)),
                LSTM(np.zeros((1, 1, 8), np.float64), np.zeros((1, 1, 8), np.float32))):
        with pytest.raises(ChunkError, match="initial_state"):
            _validator(state_template=template).validate(_chunk(initial_state=bad))


DICT_ACT = gymnasium.spaces.Dict({"move": gymnasium.spaces.Discrete(4),
                                  "aim": gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32),
                                  "pick": gymnasium.spaces.MultiDiscrete([2, 3])})
UNITS_ACT = Units(3, gymnasium.spaces.Dict({"accel": gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32),
                                            "push": gymnasium.spaces.Discrete(2)}))
OBS = gymnasium.spaces.Dict({"ships": gymnasium.spaces.Box(-1, 1, (3, 9), np.float32),
                             "mask": gymnasium.spaces.MultiBinary(5), "turn": gymnasium.spaces.Discrete(7)})


def _buffer_chunk(role: RoleSpec, *, teacher: bool = False, slots: int = 4) -> TrajectoryChunk:
    """A chunk the worker's own buffer builds: ACT, ACT, terminal ACT, PAD."""
    spec = BufferSpec(ObsSpec.from_space(role.observation_space), ActionSpec.from_space(role.action_space),
                      None if role.global_state_space is None else ObsSpec.from_space(role.global_state_space),
                      teacher=teacher)
    buf = RolloutBuffer(slots, spec)
    buf.begin(None, 1)
    obs = spec.obs.allocate(())
    action = spec.action.allocate_actions(())
    mask = spec.action.full_mask(())
    gs = None if spec.global_state is None else spec.global_state.allocate(())
    unit_logp = np.full(spec.action.num_deciders, -0.25, np.float32) if spec.action.num_deciders > 1 else None
    for _ in range(slots - 1):
        buf.write_act(obs, gs, mask, action, -0.5, unit_logp, 0.0, teacher_action=action if teacher else None)
    buf.mark_terminal()
    buf.write_pad()
    return buf.build_chunk("a")


@pytest.mark.parametrize("role, teacher", [
    (RoleSpec(OBS, DICT_ACT), True),
    (RoleSpec(OBS, UNITS_ACT, gymnasium.spaces.Box(-1, 1, (6,), np.float32)), False),
    (RoleSpec(gymnasium.spaces.Discrete(5), gymnasium.spaces.Box(-1, 1, (3,), np.float32)), False),
])
def test_chunks_the_worker_buffers_build_pass_after_a_round_trip(role, teacher):
    chunk = _buffer_chunk(role, teacher=teacher)
    back, _ = _round_trip(chunk)
    ChunkValidator(role, agent_id="a", teacher=teacher, chunk_length=4).validate(back)
    with pytest.raises(ChunkError):
        ChunkValidator(role, agent_id="a", teacher=not teacher).validate(back)


def test_units_and_dict_actions_are_range_checked_per_component():
    units_role = RoleSpec(OBS, UNITS_ACT)
    chunk = _buffer_chunk(units_role)
    chunk.actions["push"][0, 1] = 2                                          # Discrete(2) component
    with pytest.raises(ChunkError, match="component 'push'"):
        ChunkValidator(units_role, agent_id="a", teacher=False).validate(chunk)
    chunk = _buffer_chunk(units_role)
    chunk.actions["accel"][1, 0, 0] = float("nan")
    with pytest.raises(ChunkError, match="component 'accel'"):
        ChunkValidator(units_role, agent_id="a", teacher=False).validate(chunk)
    dict_role = RoleSpec(OBS, DICT_ACT)
    chunk = _buffer_chunk(dict_role, teacher=True)
    chunk.teacher_action["pick"][0, 1] = 3                                   # MultiDiscrete([2, 3])
    with pytest.raises(ChunkError, match="teacher_action/pick"):
        ChunkValidator(dict_role, agent_id="a", teacher=True).validate(chunk)
    chunk = _buffer_chunk(dict_role, teacher=True)
    chunk.has_teacher = chunk.has_teacher.to(torch.int8)
    with pytest.raises(ChunkError, match="has_teacher"):
        ChunkValidator(dict_role, agent_id="a", teacher=True).validate(chunk)


# ---------------------------------------------------------------------------
# Weights and checkpoints
# ---------------------------------------------------------------------------


def test_weights_round_trip_with_their_meta_and_are_never_compressed():
    sd = {"w": np.random.default_rng(0).standard_normal((3, 3)).astype(np.float32), "n": np.arange(2, dtype=np.int64)}
    meta = WeightsMeta("main", "fixed", 0, 40, teacher_active=True, checkpoint_id="ckpt_v40",
                       architecture={"encoder_class": "x.Y", "kwargs": {"h": 8}}, roles=("player",))
    buf = encode_weights(sd, meta)
    from colosseum.net.frame import frame_sizes

    assert frame_sizes(buf)[0] == frame_sizes(buf)[1]                         # plain
    back, back_meta = decode_weights(buf, cap=CAP)
    assert back_meta == meta and set(back) == {"w", "n"} and np.array_equal(back["w"], sd["w"])
    with pytest.raises(FrameError, match="kind"):
        decode_weights(encode_frame({"w": np.zeros(1)}), cap=CAP)
    with pytest.raises(FrameError, match="weights_meta"):
        decode_weights(encode_frame({"w": np.zeros(1)}, kind="weights", meta={"agent_id": "a"}), cap=CAP)
    with pytest.raises(FrameError, match="weights"):
        decode_weights(encode_frame([np.zeros(1)], kind="weights", meta={}), cap=CAP)
    with pytest.raises(ValueError, match="kind"):
        encode_weights(sd, WeightsMeta("a", "nonsense", 0, 0))                 # type: ignore[arg-type]


@pytest.mark.parametrize("with_trainer", [True, False])
def test_checkpoints_round_trip_with_the_trainer_state_as_bytes(with_trainer):
    buf = io.BytesIO()
    torch.save({"step": torch.tensor(3), "opt": {"lr": 0.1}}, buf)
    payload = {"agent_id": "main", "policy_version": 120, "model_state": {"w": np.ones((2, 2), np.float32)},
               "trainer_state_bytes": buf.getvalue() if with_trainer else None, "final": True}
    back, meta = decode_checkpoint(encode_checkpoint(payload, kind="final", generation=3), cap=CAP)
    assert meta == {"agent_id": "main", "policy_version": 120, "final": True, "kind": "final", "generation": 3}
    assert back["trainer_state_bytes"] == payload["trainer_state_bytes"] and back["final"] is True
    assert np.array_equal(back["model_state"]["w"], payload["model_state"]["w"])
    if with_trainer:
        state = torch.load(io.BytesIO(back["trainer_state_bytes"]), weights_only=True)
        assert int(state["step"]) == 3
    with pytest.raises(FrameError, match="checkpoint"):
        decode_checkpoint(encode_frame({"model_state": {}}, kind="checkpoint", meta={}), cap=CAP)


def test_validate_refuses_a_model_state_that_cannot_be_framed():
    """core.validation checks a model's state on the frame codec (T1.1): nested containers and importable
    namedtuples pass, a namedtuple defined in a function does not."""
    from colosseum.core.errors import ConfigError
    from colosseum.core.validation import _check_state_payload

    _check_state_payload({"core": LSTM(torch.zeros(2, 1, 4), torch.ones(2, 1, 4)), "n": [torch.zeros(2, 3)]}, "m")
    Local = namedtuple("Local", ["h"])
    with pytest.raises(ConfigError, match="initial_state cannot be sent between processes.*module level"):
        _check_state_payload(Local(torch.zeros(2, 4)), "m")
