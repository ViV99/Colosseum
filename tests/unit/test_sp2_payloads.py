"""Numpy payload forms of SP2 commands, optimizer states and the in-memory weight store (SP1 guarantees, T7.1;
chunk and weight payload round trips are Part B's tests/unit/test_chunk_v2_types.py)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from colosseum.core.ipc import assert_no_tensors, from_numpy_tree, to_numpy_tree
from colosseum.sp2.core.types import (
    Lineup,
    SeatAssignment,
    WeightPayload,
    WorkerCommand,
    state_dict_to_numpy,
)
from colosseum.weight_store.shared_memory import InMemoryWeightStore


def tiny() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))


def test_worker_command_with_lineups_and_numpy_checkpoints_has_no_tensors():
    cmd = WorkerCommand(lineups=[Lineup("2p", [SeatAssignment("a"), SeatAssignment("a", "ckpt_v1", False)]), None],
                        new_checkpoints={"a": {"ckpt_v1": state_dict_to_numpy(tiny().state_dict())}})
    assert_no_tensors(cmd)


def test_optimizer_state_numpy_tree_roundtrip():
    model = tiny()
    opt = torch.optim.Adam(model.parameters())
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    tree = to_numpy_tree(opt.state_dict())
    assert_no_tensors(tree)
    restored = torch.optim.Adam(tiny().parameters())
    restored.load_state_dict(from_numpy_tree(tree))
    key = next(iter(opt.state_dict()["state"]))
    assert torch.equal(restored.state_dict()["state"][key]["exp_avg"], opt.state_dict()["state"][key]["exp_avg"])


def test_assert_no_tensors_reports_the_path():
    with pytest.raises(TypeError, match=r"item\['x'\]\[1\]"):
        assert_no_tensors({"x": [1, torch.zeros(1)]})


def test_in_memory_weight_store_keeps_numpy():
    store = InMemoryWeightStore()
    store.put("a", WeightPayload.from_model("a", 2, tiny()))
    got = store.get("a")
    assert got.policy_version == 2 and store.get_version("a") == 2
    assert all(isinstance(v, np.ndarray) for v in got.state_dict.values())
