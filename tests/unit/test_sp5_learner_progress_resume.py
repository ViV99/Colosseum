"""The learner's env-step value for LR progress never goes backward, also across a resume (SP5 FX2; controller ruling
R29): the learner records the value it last used for progress in its trainer state (``progress_env_steps``) and a
resumed learner uses ``max(counter, recorded)``; the value is in env steps, so a resume with another
``total_timesteps`` follows the new budget; a trainer state without the key resumes as before."""
from __future__ import annotations

import io
import threading
import time

import pytest
import torch

from colosseum.core.config import LearnerConfig
from colosseum.core.types import TrajectoryChunk
from colosseum.learner.learner import learner_process, make_checkpoint_payload
from game_helpers import NumpyOnlyQueue, RecordingLearnerAlgorithm, chunk_v2_payload, learner_appo, queue_learner_io

BASE_LR = 1e-3


class _Counter:
    """A ``progress_counter`` whose ``.value`` the test sets (a hub count through ``ConnectionIntake.value``)."""

    def __init__(self, value: int) -> None:
        self.value = value


def _run(algo, counter, total_timesteps, *, resume_state=None, num_batches=1, timeout=60.0):
    """Run ``learner_process`` on ``num_batches`` batches of 2 chunks; returns (train metrics, final checkpoint)."""
    traj, mq, ckq, stop = NumpyOnlyQueue(), NumpyOnlyQueue(), NumpyOnlyQueue(), threading.Event()
    for v in range(2 * num_batches):
        traj.put(chunk_v2_payload(version=v))
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo,
        **queue_learner_io(traj, weight_queues=[NumpyOnlyQueue(maxsize=1)], metrics_queue=mq, checkpoint_queue=ckq),
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"), stop_event=stop,
        resume_state=resume_state, progress_counter=counter, total_timesteps=total_timesteps,
    ))
    thread.start()
    deadline = time.monotonic() + timeout
    while mq.qsize() < num_batches and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    metrics = [mq.get_nowait() for _ in range(mq.qsize())]
    finals = [p for p in (ckq.get_nowait() for _ in range(ckq.qsize())) if p["final"]]
    assert len(metrics) == num_batches and len(finals) == 1
    return metrics, finals[0]


def _trainer_state(payload: dict) -> dict:
    return torch.load(io.BytesIO(payload["trainer_state_bytes"]), weights_only=True)


def _resume_state(payload: dict, *, trainer_state: dict | None = None) -> dict:
    blob = payload["trainer_state_bytes"]
    if trainer_state is not None:
        buf = io.BytesIO()
        torch.save(trainer_state, buf)
        blob = buf.getvalue()
    return {"model_state": payload["model_state"], "trainer_state": blob,
            "policy_version": payload["policy_version"], "env_steps": 0, "source": "test"}


@pytest.fixture(scope="module")
def first_run():
    """A first life: the learner trained with the node's estimate 600 of 1000 env steps (LR at progress 0.6)."""
    metrics, final = _run(learner_appo(lr_schedule="linear", learning_rate=BASE_LR), _Counter(600), 1000)
    return metrics[-1], final


def test_the_value_used_for_progress_is_recorded_in_the_trainer_state(first_run):
    metrics, final = first_run
    assert metrics["progress"] == pytest.approx(0.6) and metrics["lr"] == pytest.approx(BASE_LR * 0.4)
    assert _trainer_state(final)["progress_env_steps"] == 600


def test_a_resume_below_the_recorded_value_keeps_the_recorded_progress(first_run):
    """The hub stamped the checkpoint at 450 (its budget counter lags the node's estimate): the resumed learner's
    counter starts there, but its progress continues from the recorded 600, so the LR does not go back up."""
    last, final = first_run
    algo = learner_appo(lr_schedule="linear", learning_rate=BASE_LR)
    metrics, resumed_final = _run(algo, _Counter(450), 1000, resume_state=_resume_state(final))
    assert metrics[0]["progress"] == pytest.approx(0.6)
    assert metrics[0]["lr"] <= last["lr"] + 1e-15
    assert _trainer_state(resumed_final)["progress_env_steps"] == 600


def test_a_resume_with_a_larger_budget_follows_the_new_budget(first_run):
    last, final = first_run
    algo = learner_appo(lr_schedule="linear", learning_rate=BASE_LR)
    metrics, _ = _run(algo, _Counter(450), 2000, resume_state=_resume_state(final))
    assert metrics[0]["progress"] == pytest.approx(600 / 2000)
    assert metrics[0]["lr"] == pytest.approx(BASE_LR * (1 - 0.3))


def test_a_counter_above_the_recorded_value_wins(first_run):
    _, final = first_run
    metrics, resumed_final = _run(learner_appo(lr_schedule="linear", learning_rate=BASE_LR), _Counter(800), 1000,
                                  resume_state=_resume_state(final))
    assert metrics[0]["progress"] == pytest.approx(0.8)
    assert _trainer_state(resumed_final)["progress_env_steps"] == 800


def test_a_trainer_state_without_the_key_resumes_as_before(first_run):
    """SP2-SP4 trainer states have no ``progress_env_steps``: progress follows the counter, as before R29."""
    _, final = first_run
    old = _trainer_state(final)
    del old["progress_env_steps"]
    algo = learner_appo(lr_schedule="linear", learning_rate=BASE_LR)
    metrics, _ = _run(algo, _Counter(450), 1000, resume_state=_resume_state(final, trainer_state=old))
    assert metrics[0]["progress"] == pytest.approx(0.45)
    assert algo.policy_version == final["policy_version"] + 1


def test_a_stop_before_any_train_step_carries_the_recorded_value_over(first_run):
    _, final = first_run
    algo = learner_appo(lr_schedule="linear", learning_rate=BASE_LR)
    ckq, stop = NumpyOnlyQueue(), threading.Event()
    stop.set()
    learner_process(agent_id="a", algorithm_factory=lambda: algo,
                    **queue_learner_io(NumpyOnlyQueue(), weight_queues=[NumpyOnlyQueue(maxsize=1)],
                                       checkpoint_queue=ckq),
                    config=LearnerConfig(batch_chunks=2, device="cpu"), stop_event=stop,
                    resume_state=_resume_state(final), progress_counter=_Counter(450), total_timesteps=1000)
    assert _trainer_state(ckq.get_nowait())["progress_env_steps"] == 600


def test_within_one_life_the_value_used_never_decreases():
    class _Falling:
        def __init__(self) -> None:
            self._values = iter([700, 300, 900])

        @property
        def value(self) -> int:
            return next(self._values)

    algo = RecordingLearnerAlgorithm()
    traj, stop = NumpyOnlyQueue(), threading.Event()
    for v in range(6):
        traj.put(chunk_v2_payload(version=v))
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo,
        **queue_learner_io(traj, weight_queues=[NumpyOnlyQueue(maxsize=1)]),
        config=LearnerConfig(batch_chunks=2, device="cpu"), stop_event=stop,
        progress_counter=_Falling(), total_timesteps=1000))
    thread.start()
    deadline = time.monotonic() + 30
    while len(algo.progress_at_train) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=10)
    assert algo.progress_at_train == pytest.approx([0.7, 0.7, 0.9])


def test_without_a_counter_nothing_is_recorded():
    """SP2's distributed roles (``progress_counter`` None): progress = consumed_samples / total_timesteps, unchanged;
    the trainer state has no ``progress_env_steps``."""
    source = learner_appo()
    source.train_step([TrajectoryChunk.from_payload(chunk_v2_payload(version=v)) for v in range(2)])
    assert "progress_env_steps" not in _trainer_state(make_checkpoint_payload("a", source))
    ckq, stop = NumpyOnlyQueue(), threading.Event()
    stop.set()
    learner_process(agent_id="a", algorithm_factory=learner_appo,
                    **queue_learner_io(NumpyOnlyQueue(), weight_queues=[NumpyOnlyQueue(maxsize=1)],
                                       checkpoint_queue=ckq),
                    config=LearnerConfig(batch_chunks=2, device="cpu"), stop_event=stop, total_timesteps=1000)
    assert "progress_env_steps" not in _trainer_state(ckq.get_nowait())
