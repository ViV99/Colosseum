"""Learners on the GPU of ``train``'s local node (SP5 T4.1; spec 1, 2.6; ruling R18; P-PF3-4): ``learner.device: auto``
places the learner on ``cuda:0`` when torch here can use CUDA, and the local node takes every learner on its one GPU.
GPU only (``-m gpu``); skipped without CUDA."""
from __future__ import annotations

import json

import pytest
import torch

from cli_runner import TTT_CONFIG, TTT_MULTI_CONFIG, run_train

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device torch can use")]


def _learners(run) -> dict:
    return json.loads((run.root / "hub_state.json").read_text())["learners"]["agents"]


def test_auto_puts_the_local_learner_on_the_gpu(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, "gpu", overrides={"learner.device": "auto"})
    assert run.returncode == 0, run.stderr[-3000:]
    learner = _learners(run)["agent_0"]
    assert (learner["device"], learner["node"], learner["generation"], learner["local"]) == ("cuda:0", 0, 1, True)
    assert "device=cuda:0" in run.log("learner-agent_0")
    train = run.records("train")
    assert train and all(r["generation"] == 1 and r["learner_node"] == 0 for r in train)


def test_two_agents_share_the_local_gpu(tmp_path):
    run = run_train(TTT_MULTI_CONFIG, tmp_path, "gpu2", overrides={"learner.device": "auto"})
    assert run.returncode == 0, run.stderr[-3000:]
    learners = _learners(run)
    assert len(learners) >= 2 and {entry["device"] for entry in learners.values()} == {"cuda:0"}
