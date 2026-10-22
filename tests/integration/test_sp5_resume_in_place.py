"""In-place resume through the CLI (SP5 T4.3, spec 5.4): the same dir and run id, counters, epochs, journal rows and
snapshot versions continue; refused while the run lives; strict checks with -c; an SP4 run dir continued in place.
Budgets of a resume are relative to the parent's final env steps (the hub stops about one heartbeat of env steps past
its budget)."""
from __future__ import annotations

import json
import re
import signal
import sqlite3
import subprocess
import sys

import yaml

from cli_runner import (
    REPO_ROOT,
    TINY,
    TTT_CONFIG,
    child_env,
    read_records,
    run_in_session,
    run_train,
    training_process,
    wait_for,
)
from colosseum.arena.matchlog import MATCH_LOG_FILE
from colosseum.core.run_state import read_run_state
from game_helpers import copy_sp4_run, sp4_manifest

_EPOCH_RE = re.compile(r"_g(\d+)_")
_STARTED = "main started (pid"


def _resume(root, *, config=None, overrides=None, timeout: float = 240.0) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "colosseum", "train", "--resume", str(root)]
    if config is not None:
        cmd += ["-c", str(config)]
    for key, value in (overrides or {}).items():
        cmd += ["--set", f"{key}={value}"]
    return run_in_session(cmd, timeout)


def _hub_state(root) -> dict:
    return json.loads((root / "hub_state.json").read_text())


def _sql(root, query: str) -> list[tuple]:
    conn = sqlite3.connect(root / MATCH_LOG_FILE)
    try:
        return conn.execute(query).fetchall()
    finally:
        conn.close()


def _versions(root, agent: str) -> set[int]:
    return {int(p.name.removeprefix("ckpt_v")) for p in (root / "checkpoints" / agent).glob("ckpt_v*")}


def _epochs(uids) -> list[int]:
    return [int(m.group(1)) for uid in uids if (m := _EPOCH_RE.search(uid))]


def test_train_resume_continues_the_run_in_place(tmp_path):
    first = run_train(TTT_CONFIG, tmp_path)
    assert first.returncode == 0, first.stderr[-3000:]
    root = first.root
    run_json = (root / "run.json").read_text()
    resolved = (root / "config.resolved.yaml").read_text()
    state = _hub_state(root)
    records_before = len(read_records(root))
    systems_before = read_records(root, "system")
    rows_before = _sql(root, "SELECT id, uid FROM matches WHERE uid LIKE '%/train/%' ORDER BY id")
    versions_before = _versions(root, "agent_0")
    budget = state["env_steps"] + 3000

    second = _resume(root, overrides={"training.total_timesteps": budget})
    assert second.returncode == 0, second.stderr[-3000:]
    assert (root / "run.json").read_text() == run_json
    assert (root / "config.resolved.1.yaml").read_text() == resolved
    assert yaml.safe_load((root / "config.resolved.yaml").read_text())["training"]["total_timesteps"] == budget
    after = read_records(root)[records_before:]
    systems_after = [r for r in after if r["kind"] == "system"]
    assert systems_after and systems_after[0]["env_steps"] >= systems_before[-1]["env_steps"]
    for key in ("dropped_reward_episodes", "dropped_match_results"):           # P-PF3-5: monotonic per run
        assert systems_after[0][key] >= systems_before[-1][key], key
    assert _sql(root, "SELECT value FROM meta WHERE key = 'schema_version'") == [("2",)]
    rows_after = _sql(root, "SELECT id, uid FROM matches WHERE uid LIKE '%/train/%' ORDER BY id")[len(rows_before):]
    assert rows_after and min(r[0] for r in rows_after) > max(r[0] for r in rows_before)
    assert min(_epochs(u for _, u in rows_after)) > max(_epochs(u for _, u in rows_before))     # R7
    resumed = _hub_state(root)
    assert resumed["env_steps"] >= budget and resumed["epoch_reserved_until"] > state["epoch_reserved_until"]
    new_versions = _versions(root, "agent_0") - versions_before
    assert new_versions and min(new_versions) > max(versions_before)
    log = (root / "logs" / "main.log").read_text()
    assert log.count(_STARTED) == 2
    assert "Resume: replayed 0 training matches" in log                     # the clean stop wrote the watermark
    assert "Resume [agent_0]: learners start from snapshot ckpt_v" in log
    run_state = read_run_state(root)
    assert run_state.status == "finished" and run_state.exit_code == 0


def test_resume_is_refused_while_the_run_is_alive(tmp_path):
    with training_process(TTT_CONFIG, tmp_path, overrides={"training.total_timesteps": "100000000"}) as (proc, root):
        assert wait_for(lambda: (s := read_run_state(root)) is not None and s.status == "running"
                        and (root / "hub_state.json").exists() and (root / MATCH_LOG_FILE).exists()
                        and _sql(root, "SELECT COUNT(*) FROM matches")[0][0] > 0, 120)
        refused = _resume(root, overrides={"training.total_timesteps": 10**9})
        assert refused.returncode == 1 and "is running (pid" in refused.stderr, refused.stderr[-2000:]
        assert not (root / "config.resolved.1.yaml").exists()
        proc.send_signal(signal.SIGINT)
        assert proc.wait(120) == 130
    budget = _hub_state(root)["env_steps"] + 1000
    top = _sql(root, "SELECT MAX(id) FROM matches")[0][0]
    # Seam 2 of T4.3(a): with the journal off the resumed run still writes a watermark (never None while rows are
    # accounted), so a later resume with the journal on does not replay them again.
    again = _resume(root, overrides={"training.total_timesteps": budget, "match_log.enabled": "false"})
    assert again.returncode == 0, again.stderr[-3000:]
    assert json.loads((root / "league_state.json").read_text())["journal_row"] == top


def test_an_sp4_run_dir_continues_in_place(tmp_path):
    manifest = sp4_manifest()
    root = copy_sp4_run(tmp_path)
    old_top = _sql(root, "SELECT MAX(id) FROM matches")[0][0]
    # P-PF3-9: no pulse child (two loaded processes at most in a fast test).
    run = _resume(root, overrides={"training.total_timesteps": manifest["env_steps"] + 1500,
                                   "evaluation.enabled": "false"})
    assert run.returncode == 0, run.stderr[-3000:]
    assert _sql(root, "SELECT value FROM meta WHERE key = 'schema_version'") == [("2",)]
    assert _sql(root, f"SELECT COUNT(*) FROM matches WHERE id <= {old_top} AND node IS NOT NULL") == [(0,)]
    new_nodes = _sql(root, f"SELECT DISTINCT node FROM matches WHERE id > {old_top} AND uid LIKE '%/train/%'")
    assert new_nodes == [(0,)]                                       # every new training row: the local node
    assert (root / "hub_state.json").is_file()
    assert json.loads((root / "run.json").read_text())["run_id"] == manifest["run_id"]
    log = (root / "logs" / "main.log").read_text()
    assert "schema version 1 migrated to 2" in log
    assert "has no hub_state.json (an SP4 run dir)" in log and "Resume: replayed" in log
    assert "league state (ratings, PFSP statistics) restored" in log
    newest_with_trainer = max(manifest["with_trainer_state"])
    learner = (root / "logs" / f"learner-{manifest['agent']}.log").read_text()
    assert f"Resumed from ckpt_v{newest_with_trainer} of the hub" in learner
    new_versions = _versions(root, manifest["agent"]) - set(manifest["versions"])
    assert new_versions and min(new_versions) > manifest["final_version"]


def test_a_changed_architecture_is_refused_before_anything_is_written(tmp_path):
    first = run_train(TTT_CONFIG, tmp_path)
    assert first.returncode == 0, first.stderr[-3000:]
    root = first.root
    resolved = (root / "config.resolved.yaml").read_text()
    data = yaml.safe_load(resolved)
    data["networks"]["kwargs"] = {"hidden": 7}
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(data))
    refused = _resume(root, config=changed, overrides={"training.total_timesteps": 10**9})
    assert refused.returncode == 1 and "cannot continue with this config" in refused.stderr, refused.stderr[-2000:]
    assert (root / "config.resolved.yaml").read_text() == resolved
    assert not (root / "config.resolved.1.yaml").exists()
    assert read_run_state(root).status == "finished"


def test_with_c_the_diff_is_logged_and_an_agent_can_be_added(tmp_path):
    first = run_train(TTT_CONFIG, tmp_path)
    assert first.returncode == 0, first.stderr[-3000:]
    root = first.root
    data = yaml.safe_load((root / "config.resolved.yaml").read_text())
    data["learner"]["batch_chunks"] = int(TINY["learner.batch_chunks"]) + 1
    data["agents"]["rnd"] = {"kind": "scripted", "class": "colosseum.players.RandomBot", "kwargs": {}}
    data["matchmaking"]["opponents"] = {"latest": 0.7, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.3}
    more = tmp_path / "more.yaml"
    more.write_text(yaml.safe_dump(data))
    top = _sql(root, "SELECT MAX(id) FROM matches")[0][0]
    # R8: a league state that holds the journal only up to 40 rows before its end (a hub killed before its tick):
    # exactly those 40 rows are replayed, and the next league state's watermark covers them.
    league = json.loads((root / "league_state.json").read_text())
    assert league["journal_row"] == top
    league["journal_row"] = top - 40
    (root / "league_state.json").write_text(json.dumps(league))
    run = _resume(root, config=more, overrides={"training.total_timesteps": _hub_state(root)["env_steps"] + 3000})
    assert run.returncode == 0, run.stderr[-3000:]
    log = (root / "logs" / "main.log").read_text()
    assert "differs from the run's saved config in" in log and "learner.batch_chunks:" in log
    assert f"Resume: replayed 40 training matches of the journal into the ratings and PFSP statistics (after row " \
           f"{top - 40})" in log
    assert json.loads((root / "league_state.json").read_text())["journal_row"] >= top
    with_rnd = _sql(root, f"SELECT COUNT(*) FROM matches m JOIN seats s ON s.match = m.id JOIN entities e "
                          f"ON e.id = s.agent WHERE m.id > {top} AND e.name = 'rnd'")
    assert with_rnd[0][0] > 0


def test_resume_without_a_config_needs_resume():
    proc = subprocess.run([sys.executable, "-m", "colosseum", "train"], cwd=REPO_ROOT, env=child_env(),
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2 and "Missing option '-c' / '--config'." in proc.stderr
