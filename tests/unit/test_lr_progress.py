"""APPO learning rate as a function of training progress (T2.5)."""
import pytest

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig, LRSchedule
from colosseum.core.types import TrajectoryChunk
from dataflow_helpers import TinyModel, chunk_payload


def _lr(algo) -> float:
    return algo._optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize("schedule, progress, expected", [
    (LRSchedule.CONSTANT, 0.7, 1e-3),
    (LRSchedule.LINEAR, 0.0, 1e-3),
    (LRSchedule.LINEAR, 0.25, 0.75e-3),
    (LRSchedule.LINEAR, 1.0, 0.0),
    (LRSchedule.LINEAR, 1.5, 0.0),
    (LRSchedule.COSINE, 0.5, 0.5e-3),
    (LRSchedule.COSINE, 1.0, 0.0),
])
def test_lr_is_a_function_of_progress(schedule, progress, expected):
    algo = APPO(TinyModel(), AlgorithmConfig(lr_schedule=schedule, learning_rate=1e-3), device="cpu")
    assert _lr(algo) == pytest.approx(1e-3)
    algo.set_progress(progress)
    assert _lr(algo) == pytest.approx(expected, abs=1e-12)


def test_train_step_does_not_change_the_lr():
    algo = APPO(TinyModel(), AlgorithmConfig(lr_schedule=LRSchedule.LINEAR, learning_rate=1e-3),
                device="cpu")
    algo.set_progress(0.25)
    algo.train_step([TrajectoryChunk.from_payload(chunk_payload(T=4)) for _ in range(2)])
    assert _lr(algo) == pytest.approx(0.75e-3)


def test_step_based_lr_schedule_is_gone():
    assert not hasattr(APPO, "setup_lr_schedule")
