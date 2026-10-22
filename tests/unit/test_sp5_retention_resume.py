"""Retention across in-place resumes (SP5 T4.3): the run id stays, so only the NEWEST final snapshot of the current
run is protected; earlier finals of the same run are ordinary snapshots (every stop / resume cycle would otherwise
add one more protected snapshot). Without a run id the SP3 rule (every final) stays."""
from __future__ import annotations

from colosseum.arena.retention import RetentionPolicy, SnapshotInfo, retain

RUN = "run-now"


def _policy(**kw) -> RetentionPolicy:
    base = dict(keep_last=2, keep_every_period=0, keep_top_k=0, top_k_min_games=0, max_snapshots=None,
                current_run_id=RUN)
    return RetentionPolicy(**{**base, **kw})


def _kept(decision) -> list[int]:
    return sorted(int(c.removeprefix("ckpt_v")) for c in decision.keep)


def test_only_the_newest_final_of_the_current_run_is_protected():
    entries = [SnapshotInfo("ckpt_v11", 11, True, RUN, False), SnapshotInfo("ckpt_v31", 31, True, RUN, False),
               SnapshotInfo("ckpt_v7", 7, True, "parent", False),
               *[SnapshotInfo(f"ckpt_v{v}", v, False, RUN, False) for v in (40, 50)]]
    decision = retain(entries, {}, _policy())
    assert _kept(decision) == [31, 40, 50]
    assert "final" in decision.reasons["ckpt_v31"] and decision.reasons["ckpt_v11"].startswith("evicted")
    assert _kept(retain(entries, {}, _policy(current_run_id=None))) == [7, 11, 31, 40, 50]


def test_a_resumed_run_keeps_its_previous_final_until_the_next_one():
    entries = [SnapshotInfo("ckpt_v31", 31, True, RUN, False),
               *[SnapshotInfo(f"ckpt_v{v}", v, False, RUN, False) for v in (40, 50, 60)]]
    assert _kept(retain(entries, {}, _policy())) == [31, 50, 60]
    entries.append(SnapshotInfo("ckpt_v65", 65, True, RUN, False))
    assert _kept(retain(entries, {}, _policy())) == [60, 65]
