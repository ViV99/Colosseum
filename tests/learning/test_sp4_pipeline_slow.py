"""Slow SP4 pipeline test (criterion 3.4) on unit_harvest_league trained from scratch: ``train`` with pulse
evaluation -> the snapshots' BT rating rises with the version (Spearman rho and gain thresholds: T8.2
rulings) -> ``tournament --mode select`` over the run's snapshots with the anchors as the field picks a
winner that was re-evaluated on fresh seeds (its stage B blocks, whose seeds no stage A or pulse match used)
-> the winner beats ``RandomBot`` at least as often as the SP3 slow test requires (0.95) -> ``colosseum
report`` of the run builds. ``confirmed`` is printed, not asserted (ruling C14 (a): the best snapshots of a
converged run are legitimately indistinguishable; the ladder test asserts it). 1 rollout worker + the pulse
process (2 processes, test rule); the tournament runs after training with 2 workers."""
from __future__ import annotations

import json
import sys

import pytest

import arena_kit as kit
import pipeline_kit as pk
from cli_runner import run_in_session
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
from demo_learning import LEARNING_SEED
from report_kit import select_groups

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800), pytest.mark.usefixtures("restore_global_rng")]

# T8.2 rulings (docs/benchmarks/sp4-pulse-cadence.json, sp4-pipeline-calibration.json): every=1 is the smallest
# cadence with 0 skipped and >= 6 evaluated snapshots (9 of 10 stored); calibration seeds 0 / 1 / 2: rho 0.988 /
# 0.982 / 1.000, gain 1731 / 1375 / 1379 Elo, every select winner 1.00 vs random -> RHO_MIN = max(0.3,
# floor_0.05(min rho - 0.1)) = 0.85.
SETTINGS = kit.PulseSettings(every=1, blocks=25)
RHO_MIN = 0.85
GAIN_MIN = 0.0                                     # the newest third rates above the oldest third
VS_RANDOM_MIN_WIN = 0.95                           # SP3 slow pipeline test (T6.4 ruling), not tuned here
TOURNAMENT = "sel"                                 # arena_kit.select_snapshots' default name
LAYOUT = "2p"


def _stage_b_freshness(root, winner: str) -> tuple[int, int, int]:
    """(the winner's stage B matches, stage B seeds, stage B seeds also used by a stage A or pulse match), read from
    the run's journal: block ids are ``<tournament>/<stage>/<layout>/<candidate>/<composition>/<index>``."""
    with MatchLog.open_readonly(root / MATCH_LOG_FILE) as reader:
        rows = reader.matches(source="tournament", tournament=TOURNAMENT, limit=None)
        pulse_seeds = {r.seed for r in reader.matches(source="pulse", limit=None)}
    seeds = {stage: {r.seed for r in rows if r.block.startswith(f"{TOURNAMENT}/{stage}/")} for stage in "AB"}
    winner_b = sum(1 for r in rows if r.block.startswith(f"{TOURNAMENT}/B/{LAYOUT}/{winner}/"))
    return winner_b, len(seeds["B"]), len(seeds["B"] & (seeds["A"] | pulse_seeds))


def test_pulse_trend_select_and_report(tmp_path):
    run = kit.train_with_pulse(tmp_path, seed=LEARNING_SEED, settings=SETTINGS)
    print(f"[sp4 pipeline] train {run.seconds['train']:.0f}s; stored {run.stored_versions}; pulse evaluated "
          f"{run.pulse_versions}; trend rho {run.trend.spearman:.3f} gain {run.trend.gain:.1f}")
    assert run.pulse_versions, "pulse evaluated no snapshot"
    assert run.data["sections"]["anchors"]["source"] == "pulse"
    assert run.trend.spearman >= RHO_MIN and run.trend.gain > GAIN_MIN

    report = kit.select_snapshots(run, seed=7, name=TOURNAMENT)
    group = kit.group_of(report, LAYOUT)
    select = group["select"]
    print(f"[sp4 pipeline] select: winner {select['winner']} confirmed {select['confirmed']} "
          f"indistinguishable {select.get('indistinguishable')}\n{json.dumps(group['ranking'])}")
    winner_b, b_seeds, reused = _stage_b_freshness(run.run.root, select["winner"])
    print(f"[sp4 pipeline] stage B: winner played {winner_b} matches on {b_seeds} seeds, {reused} of them "
          f"used by stage A or pulse")
    assert winner_b > 0 and b_seeds > 0 and reused == 0
    version = int(select["winner"].rsplit("@v", 1)[1])
    checkpoint = run.run.root / "checkpoints" / pk.MAIN / f"ckpt_v{version}"
    pairs = pk.evaluate(run.config, checkpoint, [pk.RANDOM], tmp_path / "winner-eval.json", layout=LAYOUT,
                        num_matches=60)
    vs_random = pairs[pk.RANDOM]
    print(f"[sp4 pipeline] winner vs random: win {vs_random['win_rate']:.3f} ({vs_random['n']} matches)")
    assert vs_random["win_rate"] >= VS_RANDOM_MIN_WIN

    proc = run_in_session([sys.executable, "-m", "colosseum", "report", str(run.run.root)], timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    page = (run.run.root / "report.html").read_text()
    assert 'id="sec-tournaments"' in page
    # the embedded data (not any JSON string equal to the name): the tournament's item with its select group
    assert [g["select"]["winner"] for g in select_groups(page, TOURNAMENT)] == [select["winner"]]
