"""`colosseum train` with a scripted kickstart teacher (DAgger): the worker labels the student's
decisions and the learner's label loss is in the train metrics (spec block 6, T4.5)."""
from __future__ import annotations

from pathlib import Path

from cli_runner import run_train
from game_helpers import write_test_config


def test_a_scripted_teacher_labels_the_students_decisions(tmp_path: Path):
    config = write_test_config(
        tmp_path / "dagger.yaml", "turns",
        agents={"agent_0": {}, "teacher_bot": {"kind": "scripted", "class": "game_helpers.LabelBot",
                                               "kwargs": {"action": 1}}},
        matchmaking={"anchors": []},
        kickstart={"teacher": "teacher_bot", "lambda": 1.0, "decay_steps": 100_000},
    )
    run = run_train(config, tmp_path, name="dagger")
    assert run.returncode == 0, run.stderr[-3000:]
    train = [r for r in run.records("train") if r["agent"] == "agent_0"]
    assert train and max(r["kickstart_loss"] for r in train) > 0
    assert max(r["kickstart_label_frac"] for r in train) > 0.5
