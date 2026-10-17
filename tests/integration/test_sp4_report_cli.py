"""``colosseum report`` on short real runs of demo games of every outcome kind (SP4 criterion 3.7, T7.2):
tic_tac_toe (wdl), tron (wdl + rank), coin_grid (score). The file opens offline: data embedded, every
section present, no external resources."""
from __future__ import annotations

import json
import re
import sys

import pytest

from cli_runner import REPO_ROOT, run_in_session, run_train
from colosseum.arena.report.data import SECTIONS

EXAMPLES = REPO_ROOT / "configs" / "examples"
URL = re.compile(r"(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@pytest.mark.parametrize(("config", "outcome"), [("tic_tac_toe", "wdl"), ("tron", "rank"), ("coin_grid", "score")],
                         ids=["wdl", "rank", "score"])
def test_report_of_a_demo_run(tmp_path, config, outcome):
    run = run_train(EXAMPLES / f"{config}.yaml", tmp_path, name=config)
    assert run.returncode == 0, run.stderr[-2000:]
    proc = run_in_session([sys.executable, "-m", "colosseum", "report", str(run.root)], timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    page = (run.root / "report.html").read_text()
    assert f"Report written to {run.root / 'report.html'}" in proc.stdout
    for name in SECTIONS:
        assert f'id="sec-{name}"' in page
    data = json.loads(re.search(r'id="colosseum-data">(.*?)</script>', page, re.S).group(1))
    layouts = data["sections"]["ratings"]["layouts"]
    assert outcome in {entry["outcome_kind"] for entry in layouts.values()}
    for entry in layouts.values():                     # score layouts: mean score, never BT
        assert entry["kind"] == ("score" if entry["outcome_kind"] == "score" else "bt")
    assert data["sections"]["summary"]["status"] == "finished"
    assert data["sections"]["matches"]["rows"] and data["sections"]["training"]["agents"]
    assert not URL.search(page) and len(page) < 5_000_000


def test_report_exit_codes(tmp_path):
    proc = run_in_session([sys.executable, "-m", "colosseum", "report", str(tmp_path)], timeout=60)
    assert proc.returncode == 1 and "Config error:" in proc.stderr
    proc = run_in_session([sys.executable, "-m", "colosseum", "report", str(tmp_path / "missing")], timeout=60)
    assert proc.returncode == 2
    (tmp_path / "metrics.jsonl").write_text("")
    proc = run_in_session([sys.executable, "-m", "colosseum", "report", str(tmp_path), "-o",
                           str(tmp_path / "no" / "such" / "dir.html")], timeout=60)
    assert proc.returncode == 2 and "does not exist" in proc.stderr
