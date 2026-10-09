"""End-to-end league behaviour through the SP2 `train` (SP1 guarantees on lineups, T7.1)."""
from __future__ import annotations

import json
import os
import signal
from collections import Counter
from pathlib import Path

import pytest

from cli_runner import TINY_SP2, TTT_SP2_CONFIG, TTT_SP2_MULTI_CONFIG, TrainRun, run_train, training_process, wait_for
from colosseum.sp2.core.config import load_config

SP2 = "colosseum.sp2"


def checkpoint_versions(run: TrainRun, agent_id: str) -> list[int]:
    agent_dir = run.root / "checkpoints" / agent_id
    return sorted(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))


def final_meta(run: TrainRun, agent_id: str) -> dict:
    version = checkpoint_versions(run, agent_id)[-1]
    return json.loads((run.root / "checkpoints" / agent_id / f"ckpt_v{version}" / "meta.json").read_text())


def lr_of(record: dict) -> float:
    value = record.get("lr", record.get("learning_rate"))
    assert value is not None, f"no learning-rate metric in {sorted(record)}"
    return float(value)


def max_system_env_steps(root: Path) -> int:
    """Largest ``env_steps`` of the ``system`` records written so far (a line still being
    written is skipped)."""
    path = root / "metrics.jsonl"
    if not path.exists():
        return 0
    best = 0
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "system":
            best = max(best, int(record["env_steps"]))
    return best


def test_two_agent_self_play_reaches_budget_and_both_learners_train(tmp_path):
    run = run_train(TTT_SP2_MULTI_CONFIG, tmp_path, name="sp2", module=SP2, overrides={
        "matchmaking.mode": "self_play",
        "training.total_timesteps": "6000",
        "rollout.num_workers": "2",
    })
    assert run.returncode == 0, run.stderr[-3000:]
    assert "Training budget reached" in run.log("main")
    assert run.records("system")[-1]["env_steps"] >= 6000
    for agent_id in ("agent_alpha", "agent_beta"):
        steps = [r["train_step"] for r in run.records("train") if r["agent"] == agent_id]
        assert steps and max(steps) >= 1, f"{agent_id} never trained"
        assert sum(r["episodes"] for r in run.records("episodes") if r["agent"] == agent_id) > 0
        meta = final_meta(run, agent_id)
        assert meta["final"] is True and meta["agent_id"] == agent_id
        assert meta["networks"]["encoder_class"].endswith("TicTacToeEncoder") and meta["roles"] == ["player"]
        # Periodic checkpoints were saved during training, and the final one is the newest.
        metas = [json.loads((run.root / "checkpoints" / agent_id / f"ckpt_v{v}" / "meta.json").read_text())
                 for v in checkpoint_versions(run, agent_id)]
        periodic = [m["policy_version"] for m in metas if m["final"] is False]
        assert periodic, f"{agent_id}: no periodic checkpoint: {metas}"
        assert meta["policy_version"] == max(m["policy_version"] for m in metas) > max(periodic)


def test_three_agent_league_all_pairs_meet_and_seats_balanced(tmp_path):
    run = run_train(TTT_SP2_MULTI_CONFIG, tmp_path, name="league3", module=SP2, overrides={
        "matchmaking.mode": "league",
        "matchmaking.self_play_ratio": "0.0",
        "agents.agent_gamma": "{}",
        # Masked tic-tac-toe has no illegal-move forfeits, so episodes run ~7.6 moves (T8.1);
        # 20000 steps keep the ~2000 seats per agent that the bound below was measured on.
        "training.total_timesteps": "20000",
        # Seats are shuffled per env and refresh, and kept until the env's next episode boundary,
        # so episodes of one env are clustered. With 256 envs and a 0.2 s refresh an env plays
        # about one episode per seat draw, so the ~2000 seats per agent are nearly independent.
        "rollout.num_workers": "2",
        "rollout.envs_per_worker": "128",
        "rollout.match_refresh_interval_sec": "0.2",
    })
    assert run.returncode == 0, run.stderr[-3000:]
    agents = ["agent_alpha", "agent_beta", "agent_gamma"]
    games = run.ratings()["layouts"]["2p"]["games"]
    for a in agents:
        for b in agents:
            if a != b:
                assert games[a][b] > 0, f"{a} never met {b}: {games}"
    seats: dict[str, Counter] = {a: Counter() for a in agents}
    for record in run.records("episodes"):
        for seat_index, count in enumerate(record["seat_counts"]):
            seats[record["agent"]][seat_index] += count
    for agent_id, counts in seats.items():
        total = sum(counts.values())
        assert total >= 200, (agent_id, counts)
        # ±5% is asserted on ~10^4 lineups in tests/unit/test_sp2_matchmaker.py
        # (test_seats_are_balanced_within_5_percent). Here, measured (SP1)
        # over 20 runs x 3 agents: ~1940 seats per agent, sigma of the seat-0 share 1.28% (the
        # binomial floor is 1.14%), max deviation 3.2%. ±8% is 6.3 sigma.
        assert abs(counts[0] / total - 0.5) <= 0.08, (agent_id, counts)


def test_resume_continues_versions_env_steps_and_lr(tmp_path):
    """A run interrupted (Ctrl-C) part-way through its budget, resumed from its run dir with
    the same budget: versions, train steps, the global env-step count and the linear LR all
    continue from the resumed checkpoint's own numbers.

    Both runs share the budget, because the linear LR is a function of
    ``env_steps / total_timesteps``: only then is "the LR at the end of the first run" a
    point on the schedule the resumed run continues.
    """
    budget = 10000
    interrupt_at = 4000  # leaves room for the steps made while the interrupt lands
    common = {"algorithm.lr_schedule": "linear", "checkpoint.interval": "10",
              "training.total_timesteps": str(budget)}

    with training_process(TTT_SP2_CONFIG, tmp_path, name="first", overrides=common, module=SP2) as (proc, root):
        assert wait_for(lambda: max_system_env_steps(root) >= interrupt_at or proc.poll() is not None, 180), \
            (tmp_path / "first.stderr").read_text()[-3000:]
        assert proc.poll() is None, (tmp_path / "first.stderr").read_text()[-3000:]
        os.killpg(proc.pid, signal.SIGINT)  # like Ctrl-C: the final checkpoint is still saved
        returncode = proc.wait(60)
    first = TrainRun(returncode, "", (tmp_path / "first.stderr").read_text(), root)
    assert first.returncode == 130, first.stderr[-3000:]
    first_final = checkpoint_versions(first, "agent_0")[-1]
    first_meta = final_meta(first, "agent_0")
    resumed_env_steps = first_meta["env_steps"]
    assert first_meta["final"] is True and first_meta["policy_version"] == first_final
    assert interrupt_at <= resumed_env_steps < budget, f"interrupt did not land mid-budget: {first_meta}"
    first_train = [r for r in first.records("train") if r["agent"] == "agent_0"]
    first_lr, last_lr = lr_of(first_train[0]), lr_of(first_train[-1])

    second = run_train(TTT_SP2_CONFIG, tmp_path, name="second", module=SP2, overrides={
        **common, "training.resume_from": str(first.root)})
    assert second.returncode == 0, second.stderr[-3000:]
    assert f"(policy_version {first_final})" in second.log("main")
    assert f"env-step counter continues from {resumed_env_steps}" in second.log("main")

    # train_step / policy_version continue from the resumed ckpt_v<N>
    second_train = [r for r in second.records("train") if r["agent"] == "agent_0"]
    assert second_train[0]["train_step"] == first_final + 1, (first_final, second_train[0])
    versions = checkpoint_versions(second, "agent_0")
    assert versions and min(versions) > first_final, (first_final, versions)
    assert versions[-1] == max(r["train_step"] for r in second_train), (versions, second_train[-1])
    second_meta = final_meta(second, "agent_0")
    assert second_meta["final"] is True and second_meta["env_steps"] >= budget

    # The global env-step count continues, and the first rate measures only new steps (no spike):
    # a non-forced tick comes >= console_interval_sec after the baseline, so
    # rate * console_interval_sec <= rate * dt == env_steps - resumed_env_steps.
    first_system = second.records("system")[0]
    assert first_system["env_steps"] >= resumed_env_steps, (resumed_env_steps, first_system)
    interval = float(TINY_SP2["metrics.console_interval_sec"])
    new_steps = first_system["env_steps"] - resumed_env_steps
    assert first_system["env_steps_per_sec"] * interval <= new_steps + 1e-6, (resumed_env_steps, first_system)

    # The linear LR continues from the resumed progress instead of restarting at the base LR:
    # the first step after resume is at progress >= resumed_env_steps / budget, so its LR is at
    # most the schedule's value at the first run's final checkpoint, which is at most the LR of
    # the first run's last train step (the global counter only grows).
    base_lr = load_config(second.root / "config.resolved.yaml").algorithm.learning_rate
    resumed_progress = resumed_env_steps / budget
    second_lr = lr_of(second_train[0])
    assert second_train[0]["progress"] >= resumed_progress, (resumed_progress, second_train[0])
    assert second_lr == pytest.approx(base_lr * (1.0 - second_train[0]["progress"]))
    assert second_lr <= base_lr * (1.0 - resumed_progress) + 1e-12, (base_lr, resumed_progress, second_lr)
    assert second_lr <= last_lr, (last_lr, second_lr)
    # progress starts at resumed_env_steps / budget >= 0.4, so the linear LR has already decayed
    assert second_lr < 0.75 * first_lr, (first_lr, second_lr)
