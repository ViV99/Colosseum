"""``colosseum dashboard`` (SP4 spec block 5, T7.3): the live page and ``/api/<section>`` JSON on an
ephemeral port, filters, errors that do not stop the server, live BT while running, slow builds that never
block other requests, a dead builder thread, the Host guard, bind errors and the non-loopback warning, the
CLI's port handling and Ctrl+C (exit 130)."""
from __future__ import annotations

import json
import logging
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from cli_runner import REPO_ROOT, child_env
from colosseum.arena.report.data import SECTIONS
from colosseum.arena.report.server import make_server
from colosseum.core.errors import ConfigError
from report_kit import write_run_dir


@contextmanager
def serving(root, **kwargs):
    server = make_server(root, port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.port}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


JSON = "application/json; charset=utf-8"


def _get_full(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict, str]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=30) as r:
            return r.status, dict(r.headers), r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8")


def _get(url: str) -> tuple[int, str, str]:
    code, headers, body = _get_full(url)
    return code, headers.get("Content-Type", ""), body


def test_page_and_every_section_over_http(tmp_path):
    run = write_run_dir(tmp_path, "wdl", pulse_versions=(200,))
    with serving(run.root, refresh_sec=3) as (base, _):
        code, ctype, page = _get(base + "/")
        assert code == 200 and ctype.startswith("text/html")
        config = json.loads(re.search(r'id="colosseum-config">(.*?)</script>', page, re.S).group(1))
        assert config == {"live": True, "refresh_sec": 3}
        data = json.loads(re.search(r'id="colosseum-data">(.*?)</script>', page, re.S).group(1))
        assert data["title"] == "synthetic" and set(data["sections"]) == set(SECTIONS)
        assert not any(s.get("building") for s in data["sections"].values())
        for name in SECTIONS:
            code, ctype, body = _get(f"{base}/api/{name}")
            assert code == 200 and ctype == JSON, name
            assert isinstance(json.loads(body), dict)
        code, _, body = _get(base + "/api/matches?source=pulse&agent=main&bogus=1")
        rows = json.loads(body)["rows"]
        assert rows and {r["source"] for r in rows} == {"pulse"}
        code, _, body = _get(base + "/api/nope")
        assert code == 404 and "summary" in json.loads(body)["sections"]
        assert _get(base + "/../etc/passwd")[0] == 404


def test_a_failing_section_answers_500_and_the_server_keeps_serving(tmp_path, monkeypatch):
    run = write_run_dir(tmp_path, "wdl")
    with serving(run.root) as (base, server):
        monkeypatch.setattr(server.builder, "_matrix", lambda **kw: 1 / 0)
        code, _, body = _get(base + "/api/matrix")
        assert code == 500 and "division" in json.loads(body)["error"]
        assert _get(base + "/api/summary")[0] == 200


def test_a_running_run_is_rated_from_ratings_json(tmp_path):
    run = write_run_dir(tmp_path, "wdl", status="running", live_bt=True)
    with serving(run.root) as (base, _):
        ratings = json.loads(_get(base + "/api/ratings")[2])
        assert ratings["source"] == "live"
        assert json.loads(_get(base + "/api/summary")[2])["status"] == "running"


def test_a_slow_build_answers_building_and_never_blocks_other_requests(tmp_path, monkeypatch):
    """One build at a time (the builder is not thread-safe): while a section builds, a request for it answers
    202 with a "building" placeholder after ``request_wait_sec``, concurrent requests for it share that one
    build, another section is answered from its last result (marked stale), and the page renders at once with
    placeholders. Released, the section is served normally."""
    run = write_run_dir(tmp_path, "wdl")
    with serving(run.root) as (base, server):
        assert _get(base + "/api/system")[0] == 200            # a last result to fall back on
        server.request_wait_sec = 0.3
        release, calls = threading.Event(), []
        real = server.builder._matrix

        def slow_matrix(**kw):
            calls.append(kw)
            assert release.wait(timeout=30)
            return real(**kw)

        monkeypatch.setattr(server.builder, "_matrix", slow_matrix)
        try:
            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=3) as pool:
                answers = list(pool.map(_get_full, [base + "/api/matrix"] * 3))
            assert time.monotonic() - t0 < 10
            for code, _, body in answers:
                assert code == 202 and json.loads(body)["building"] is True
                assert "building" in json.loads(body)["notes"][0].lower()
            assert len(calls) == 1                              # one build shared by the three requests
            code, headers, body = _get_full(base + "/api/system")
            assert code == 200 and headers.get("X-Colosseum-Stale") == "1" and "latest" in json.loads(body)
            code, _, page = _get(base + "/")
            assert code == 200
            data = json.loads(re.search(r'id="colosseum-data">(.*?)</script>', page, re.S).group(1))
            assert data["sections"]["matrix"]["building"] is True and "latest" in data["sections"]["system"]
        finally:
            release.set()
        server.request_wait_sec = 30.0
        code, _, body = _get(base + "/api/matrix")
        assert code == 200 and "layouts" in json.loads(body)
        assert len(calls) <= 2


def test_make_server_refuses_a_non_run_dir_and_a_busy_port(tmp_path):
    with pytest.raises(ConfigError, match="not a run directory"):
        make_server(tmp_path)
    run = write_run_dir(tmp_path, "wdl")
    with serving(run.root) as (_, server):
        with pytest.raises(ConfigError, match="in use; choose another --port"):
            make_server(run.root, port=server.port)
    with pytest.raises(ConfigError, match="IPv4 only"):
        make_server(run.root, host="::1", port=0)


def test_a_non_loopback_host_warns_and_foreign_host_headers_are_refused(tmp_path, caplog):
    run = write_run_dir(tmp_path, "wdl")
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.report.server"):
        make_server(run.root, host="0.0.0.0", port=0).server_close()
    assert "listening on 0.0.0.0 without authentication" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.report.server"):
        with serving(run.root) as (base, server):
            assert not caplog.records                          # 127.0.0.1: no warning
            code, _, body = _get_full(base + "/api/summary", {"Host": "evil.example:80"})
            assert code == 403 and "DNS rebinding" in json.loads(body)["error"]
            assert _get_full(base + "/", {"Host": "evil.example"})[0] == 403
            for host in (f"localhost:{server.port}", f"127.0.0.1:{server.port}", "[::1]"):
                assert _get_full(base + "/api/summary", {"Host": host})[0] == 200, host


class _Fatal(BaseException):
    pass


def test_a_dead_builder_thread_is_logged_and_answers_500(tmp_path, monkeypatch, caplog):
    run = write_run_dir(tmp_path, "wdl")
    with serving(run.root) as (base, server):
        def fatal(**kw):
            raise _Fatal("boom")

        monkeypatch.setattr(server.builder, "_matrix", fatal)
        with caplog.at_level(logging.ERROR, logger="colosseum.arena.report.server"):
            code, _, body = _get(base + "/api/matrix")
        assert code == 500 and "builder stopped" in json.loads(body)["error"]
        assert "builder thread stopped" in caplog.text and "_Fatal" in caplog.text
        code, _, body = _get(base + "/api/summary")
        assert code == 500 and "builder stopped" in json.loads(body)["error"]
        code, _, page = _get(base + "/")
        assert code == 200 and "builder stopped" in page


def test_cli_prints_the_address_serves_and_exits_130_on_ctrl_c(tmp_path):
    run = write_run_dir(tmp_path, "wdl")
    proc = subprocess.Popen([sys.executable, "-m", "colosseum", "dashboard", str(run.root), "--port", "0",
                             "--refresh", "5"], cwd=REPO_ROOT, env=child_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        deadline, line = time.monotonic() + 60, ""
        while time.monotonic() < deadline and "Dashboard" not in line:
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready:
                line = proc.stdout.readline()
        m = re.search(r"http://127\.0\.0\.1:(\d+)/", line)
        assert m, line
        assert _get(f"http://127.0.0.1:{m.group(1)}/api/summary")[0] == 200
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=30) == 130
        assert "Interrupted" in proc.stderr.read()
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        proc.stdout.close()
        proc.stderr.close()


def test_cli_exit_codes_for_a_non_run_dir_and_bad_arguments(tmp_path):
    from cli_runner import run_in_session

    res = run_in_session([sys.executable, "-m", "colosseum", "dashboard", str(tmp_path), "--port", "0"], timeout=60)
    assert res.returncode == 1 and "not a run directory" in res.stderr
    run = write_run_dir(tmp_path, "wdl")
    res = run_in_session([sys.executable, "-m", "colosseum", "dashboard", str(run.root), "--refresh", "0"],
                         timeout=60)
    assert res.returncode == 2
