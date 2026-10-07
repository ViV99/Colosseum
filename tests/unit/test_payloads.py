"""Numpy payload forms of chunks, weights, commands and optimizer states (T2.2)."""
import numpy as np
import pytest
import torch

from colosseum.core.ipc import assert_no_tensors, from_numpy_tree, to_numpy_tree
from colosseum.core.types import (
    TrajectoryChunk,
    WeightPayload,
    WorkerCommand,
    state_dict_from_numpy,
    state_dict_to_numpy,
)
from colosseum.weight_store.shared_memory import InMemoryWeightStore
from dataflow_helpers import TinyModel


def _chunk(initial_state, masks: bool, T: int = 5) -> TrajectoryChunk:
    return TrajectoryChunk(
        agent_id="a", observations=torch.randn(T, 4), actions=torch.randint(0, 3, (T,)),
        action_log_probs=torch.randn(T), rewards=torch.randn(T),
        dones=torch.tensor([False, True, False, False, True]), values=torch.randn(T),
        bootstrap_value=torch.tensor(0.25), behavior_policy_version=7,
        initial_state=initial_state,
        action_masks=(torch.rand(T, 3) > 0.3) if masks else None,
    )


STATES = {
    "none": None,
    "lstm_like": {"h": torch.randn(1, 1, 8), "c": torch.randn(1, 1, 8)},
    "window_like": {"mem": torch.randn(1, 4, 8), "len": torch.tensor([3])},
}


def _assert_same_state(a, b) -> None:
    if a is None:
        assert b is None
        return
    assert set(a) == set(b)
    for key in a:
        assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key])


@pytest.mark.parametrize("state_name", list(STATES))
@pytest.mark.parametrize("masks", [True, False])
def test_chunk_payload_roundtrip(state_name, masks):
    chunk = _chunk(STATES[state_name], masks)
    payload = chunk.to_payload()
    assert_no_tensors(payload)
    back = TrajectoryChunk.from_payload(payload)
    for name in ("observations", "actions", "action_log_probs", "rewards", "dones", "values"):
        assert torch.equal(getattr(chunk, name), getattr(back, name)), name
    assert back.dones.dtype == torch.bool
    assert float(back.bootstrap_value) == pytest.approx(0.25)
    assert back.behavior_policy_version == 7 and back.agent_id == "a"
    _assert_same_state(chunk.initial_state, back.initial_state)
    if masks:
        assert torch.equal(chunk.action_masks, back.action_masks)
    else:
        assert back.action_masks is None


def test_weight_payload_is_numpy_and_loads_into_a_model():
    model = TinyModel()
    wp = WeightPayload.from_model("a", 3, model)
    assert_no_tensors(wp)
    assert all(isinstance(v, np.ndarray) for v in wp.state_dict.values())
    other = TinyModel()
    other.load_state_dict(wp.to_torch_state_dict())
    for key, value in model.state_dict().items():
        assert torch.equal(value, other.state_dict()[key])


def test_state_dict_numpy_roundtrip_handles_bfloat16():
    sd = {"w": torch.randn(2, 2).to(torch.bfloat16), "n": torch.tensor(5)}
    np_sd = state_dict_to_numpy(sd)
    assert np_sd["w"].dtype == np.float32 and np_sd["n"].dtype == np.int64
    back = state_dict_from_numpy(np_sd)
    assert torch.equal(back["w"], sd["w"].float()) and int(back["n"]) == 5


def test_worker_command_with_numpy_checkpoints_has_no_tensors():
    cmd = WorkerCommand(
        slot_agent_map=[["a"]], slot_network_map=[["ckpt_v1"]], collect_mask=[[False]],
        new_checkpoints={"a": {"ckpt_v1": state_dict_to_numpy(TinyModel().state_dict())}},
    )
    assert_no_tensors(cmd)


def test_optimizer_state_numpy_tree_roundtrip():
    model = TinyModel()
    opt = torch.optim.Adam(model.parameters())
    model.v(torch.randn(2, 4)).sum().backward()
    opt.step()
    tree = to_numpy_tree(opt.state_dict())
    assert_no_tensors(tree)
    restored = torch.optim.Adam(TinyModel().parameters())
    restored.load_state_dict(from_numpy_tree(tree))
    key = next(iter(opt.state_dict()["state"]))
    assert torch.equal(restored.state_dict()["state"][key]["exp_avg"],
                       opt.state_dict()["state"][key]["exp_avg"])


def test_assert_no_tensors_reports_the_path():
    with pytest.raises(TypeError, match=r"item\['x'\]\[1\]"):
        assert_no_tensors({"x": [1, torch.zeros(1)]})


def test_in_memory_weight_store_keeps_numpy():
    store = InMemoryWeightStore()
    store.put("a", WeightPayload.from_model("a", 2, TinyModel()))
    got = store.get("a")
    assert got.policy_version == 2 and store.get_version("a") == 2
    assert all(isinstance(v, np.ndarray) for v in got.state_dict.values())
