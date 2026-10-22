"""Recovery points from the learner (SP5 T4.2, spec 5.2): one every learner.recovery_interval_sec on the learner's
monotonic clock, never twice without a new train step, sent non-blocking as kind "recovery"; none without an
interval; the final snapshot on stop unless the learner was superseded; a learner restored from a recovery point
continues its optimizer state and publishes max(restored version, floor) and keeps its progress floor (R29); chunks of
the previous generation below the new floor are trained with their lag (the dispatch's relocation question: the
intake's R16 rule keeps them, ``test_sp5_local_connections``); a trainer state that needs pickle is refused
(weights_only=True)."""
from __future__ import annotations

import io
import pickle
import queue
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from colosseum.core.config import LearnerConfig
from colosseum.core.types import TrajectoryChunk
from colosseum.learner.learner import (
    PROGRESS_ENV_STEPS_KEY,
    apply_resume_state,
    learner_process,
    make_checkpoint_payload,
)
from game_helpers import chunk_v2_payload, learner_appo


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class ListIntake:
    """``ChunkIntake`` over prepared chunks; every ``get`` advances the fake clock by one second."""

    def __init__(self, payloads, clock: FakeClock) -> None:
        self._chunks = [TrajectoryChunk.from_payload(p) for p in payloads]
        self._clock = clock
        self.stop_reason = None

    @property
    def depth(self) -> int:
        return len(self._chunks)

    @property
    def dropped(self) -> dict[str, int]:
        return {}

    def note_policy_version(self, version: int) -> None:
        pass

    def get(self, timeout: float) -> TrajectoryChunk:
        self._clock.t += 1.0
        if not self._chunks:
            time.sleep(min(timeout, 0.01))
            raise queue.Empty
        return self._chunks.pop(0)


class RecordingOutputs:
    checkpoints = True                   # LearnerOutputs.checkpoints: this sink takes checkpoint payloads

    def __init__(self) -> None:
        self.weights, self.sent, self.metrics = [], [], []

    def publish_weights(self, payload) -> None:
        self.weights.append(payload)

    def send_checkpoint(self, payload: dict, *, kind: str, block: bool) -> bool:
        self.sent.append((kind, block, payload))
        return True

    def send_metrics(self, metrics: dict) -> None:
        self.metrics.append(metrics)

    def finish(self, stop_event) -> None:
        pass


def _run(n_chunks: int, *, interval, algo=None, resume_state=None, version_floor: int = 0, send_final=None,
         versions=None, counter=None, total_timesteps: int = 0):
    clock, outputs, stop = FakeClock(), RecordingOutputs(), threading.Event()
    versions = range(n_chunks) if versions is None else versions
    intake = ListIntake([chunk_v2_payload(version=v) for v in versions], clock)
    algo = algo if algo is not None else learner_appo()
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, intake=intake, outputs=outputs,
        config=LearnerConfig(batch_chunks=1, device="cpu"), stop_event=stop, resume_state=resume_state,
        version_floor=version_floor, recovery_interval_sec=interval, clock=clock, send_final=send_final,
        progress_counter=counter, total_timesteps=total_timesteps))
    thread.start()
    deadline = time.monotonic() + 60
    while len(outputs.metrics) < n_chunks and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    return outputs, algo


def _restore(point: dict) -> dict:
    return {"model_state": point["model_state"], "trainer_state": point["trainer_state_bytes"],
            "policy_version": point["policy_version"], "env_steps": 0,
            "source": f"recovery point v{point['policy_version']}"}


def test_recovery_points_follow_the_interval_and_a_new_train_step():
    outputs, _ = _run(10, interval=2.5)
    kinds = [(kind, block, payload["policy_version"]) for kind, block, payload in outputs.sent]
    assert kinds == [("recovery", False, 3), ("recovery", False, 6), ("recovery", False, 9), ("final", True, 10)]
    recovery = outputs.sent[0][2]
    assert recovery["final"] is False and isinstance(recovery["trainer_state_bytes"], bytes)


def test_no_recovery_points_without_an_interval():
    outputs, _ = _run(4, interval=None)
    assert [kind for kind, _, _ in outputs.sent] == ["final"]


def test_a_superseded_learner_sends_no_final_snapshot():
    outputs, _ = _run(2, interval=None, send_final=lambda: False)
    assert outputs.sent == []


def test_a_restored_recovery_point_continues_the_optimizer_and_the_versions():
    first, _ = _run(7, interval=2.5)
    point = first.sent[1][2]                                  # the recovery point at version 6
    second, algo = _run(0, interval=None, resume_state=_restore(point), version_floor=8)
    assert second.weights[0].policy_version == 8                     # max(restored 6, floor 8)
    final = second.sent[-1][2]
    saved = torch.load(io.BytesIO(point["trainer_state_bytes"]), weights_only=True)
    again = torch.load(io.BytesIO(final["trainer_state_bytes"]), weights_only=True)
    assert saved["optimizer"]["state"].keys() == again["optimizer"]["state"].keys()
    for key, tensors in saved["optimizer"]["state"].items():
        for name, value in tensors.items():
            assert torch.equal(value, again["optimizer"]["state"][key][name]), (key, name)


def test_a_learner_restarted_from_a_recovery_point_keeps_its_progress_floor():
    """FX2 / R29: the recovery point records the env-step value the learner used; a learner restarted from it never
    trains with a lower one although the counter it now reads is lower (the hub's stamp trails the node's estimate)."""
    first, _ = _run(3, interval=2.5, counter=SimpleNamespace(value=500), total_timesteps=1000)
    point = first.sent[0][2]
    assert first.sent[0][0] == "recovery"
    assert torch.load(io.BytesIO(point["trainer_state_bytes"]), weights_only=True)[PROGRESS_ENV_STEPS_KEY] == 500
    second, _ = _run(1, interval=None, resume_state=_restore(point), version_floor=4,
                     counter=SimpleNamespace(value=100), total_timesteps=1000)
    assert [m["progress"] for m in second.metrics] == [0.5]


def test_chunks_of_the_previous_generation_below_the_floor_are_trained_with_their_lag():
    """The relocation question (dispatch item 5, ruled «accept»): a new generation starting from the recovery point v4
    with floor 5 trains on chunks the old generation's weights (v3, v4) produced, each with its policy lag; the intake
    keeps such chunks only below the floor (R16; ``test_sp5_local_connections``)."""
    first, _ = _run(5, interval=2.5)
    point = first.sent[0][2]                                  # v3
    second, _ = _run(2, interval=None, resume_state=_restore(point), version_floor=5, versions=[3, 4])
    assert [(m["train_step"], m["policy_lag_max"]) for m in second.metrics] == [(6, 2.0), (7, 2.0)]


class _Evil:
    def __reduce__(self):
        return (print, ("unpickled",))


def test_a_trainer_state_that_needs_pickle_is_refused():
    payload = make_checkpoint_payload("a", learner_appo())
    buf = io.BytesIO()
    torch.save({"x": _Evil()}, buf)
    with pytest.raises(pickle.UnpicklingError):
        apply_resume_state(learner_appo(), {"model_state": payload["model_state"], "trainer_state": buf.getvalue(),
                                            "policy_version": 3, "source": "x"})
