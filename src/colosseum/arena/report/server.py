"""``colosseum dashboard``: the run report served live (SP4 spec block 5).

``GET /`` is the report page in live mode (it refreshes every section from ``api/<section>`` every
``refresh_sec`` seconds); ``GET /api/<section>`` is one section as JSON (``data.SECTIONS``; filters as
query parameters: ``agent``, ``layout``, ``source``, ``outcome``, only those of the section, ``data.FILTERS``).
The journal is opened read-only; BT tables come from ``ratings.json`` while the run is running, else from the
journal, cached until the journal files change (``ReportBuilder``; a running run's comparison table is
re-read at most every ``TABLE_MIN_AGE_SEC``). Binds 127.0.0.1 by default and has no authentication: reach it
from another machine or a phone through an SSH tunnel (docs/ARENA_GUIDE.md).

Concurrency. ``ReportBuilder`` is not thread-safe, and a cold build can be long (a finished run's first BT fit
with bootstrap reads the whole journal). One builder thread therefore owns the builder and runs one build at
a time (under ``DashboardServer.lock``); request threads queue a build of their (section, filters) key and
wait at most ``request_wait_sec`` for it. Requests for a key already queued or building share that build. A
request whose build is not done in time is answered from the key's last successful result (HTTP 200 with
``X-Colosseum-Stale: 1``), or, if there is none yet, with HTTP 202 and a placeholder section
``{"building": true, "section", "notes"}`` that the page draws as a note until the next refresh; the page
itself (``GET /``) is rendered after the same wait with placeholders for the sections still building. A
failed build answers 500 with ``{"error"}``; no request error stops the server. If the builder thread itself
dies (a ``BaseException`` outside a section build, logged once), every later request answers 500 "builder
stopped" instead of waiting for builds that never come.

Security. No authentication; listening on a non-loopback ``host`` logs a WARNING. Against DNS rebinding, a
request whose ``Host`` header names neither a loopback address nor the bound host answers 403 (a wildcard
bind, ``0.0.0.0``, accepts any ``Host``). The server is IPv4 only.
"""

from __future__ import annotations

import errno
import ipaddress
import json
import logging
import socket
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from colosseum.arena.report.data import FILTERS, REPORT_FORMAT, SECTIONS, ReportBuilder
from colosseum.arena.report.render import embed_json, is_run_dir, render_html
from colosseum.core.errors import ConfigError

logger = logging.getLogger(__name__)

# A running run's journal changes every second; its comparison table is re-read at most this often.
TABLE_MIN_AGE_SEC = 30.0
# How long a request waits for its build before it is answered from the last result or as "building".
REQUEST_WAIT_SEC = 5.0
# Distinct (section, filters) keys waiting for the builder; more answer from the last result or 503.
MAX_PENDING = 32
# Last results kept (one per (section, filters) key, least recently built dropped first).
MAX_RESULTS = 64
# server_close waits this long for a build in progress; the builder thread closes the journal when it ends.
CLOSE_WAIT_SEC = 0.5
BUILDING_NOTE = ("Building this section: the first build of a large match log (BT fit and bootstrap) can take "
                 "a while. It is drawn at the next refresh.")

JSON = "application/json; charset=utf-8"
_LOOPBACK_NAMES = ("localhost",)
_WILDCARD = ("", "0.0.0.0")

_Key = tuple[str, tuple[tuple[str, str], ...]]


def is_loopback(host: str) -> bool:
    """``localhost`` or a loopback IP literal (``127.0.0.0/8``, ``::1``)."""
    host = host.strip().strip("[]").lower()
    if host in _LOOPBACK_NAMES or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _host_of(header: str) -> str:
    """The host part of a ``Host`` header (``name:port``, ``[v6]:port``)."""
    header = header.strip().lower()
    if header.startswith("["):
        return header[1:header.find("]")] if "]" in header else header
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


def _key(name: str, filters: dict[str, str] | None = None) -> _Key:
    return name, tuple(sorted((filters or {}).items()))


def placeholder(name: str) -> dict:
    """The section shape answered while ``name`` is still building (the page draws its note)."""
    return {"building": True, "section": name, "notes": [BUILDING_NOTE]}


@dataclass
class _Result:
    seq: int                     # completion order of the build
    error: str | None            # the build failed (body: the last successful result, if any)
    body: bytes | None           # JSON of the last successful build of the key


class _Builds:
    """The one thread that calls the ``ReportBuilder``, a queue of distinct keys and their last results."""

    def __init__(self, builder: ReportBuilder, lock: threading.Lock) -> None:
        self._builder, self._lock = builder, lock
        self._cond = threading.Condition()
        self._pending: OrderedDict[_Key, None] = OrderedDict()
        self._running: _Key | None = None
        self._results: OrderedDict[_Key, _Result] = OrderedDict()
        self._seq = 0
        self._closed = False
        self.failed: str | None = None          # why the builder thread stopped (None while it runs)
        self._thread = threading.Thread(target=self._loop, name="dashboard-builder", daemon=True)
        self._thread.start()

    def get(self, keys: Sequence[_Key], timeout: float) -> dict[_Key, tuple[str, bytes | str | None]]:
        """``key -> (state, payload)`` for every key, waiting at most ``timeout`` seconds in all. States:
        ``fresh`` (JSON bytes of a build that ended after this call began), ``error`` (its message), ``stale``
        (JSON bytes of the last successful build), ``building`` (none yet), ``busy`` (queue full, none yet),
        ``closed``, ``failed`` (the builder thread died; payload: why)."""
        deadline = time.monotonic() + timeout
        out: dict[_Key, tuple[str, bytes | str | None]] = {}
        with self._cond:
            if self.failed is not None:
                return {key: ("failed", self.failed) for key in keys}
            since = self._seq
            full: set[_Key] = set()
            for key in keys:
                if self._closed or key == self._running or key in self._pending:
                    continue
                if len(self._pending) >= MAX_PENDING:
                    full.add(key)
                    continue
                self._pending[key] = None
            self._cond.notify_all()
            while True:
                for key in keys:
                    r = self._results.get(key)
                    if key not in out and r is not None and r.seq > since:
                        out[key] = ("error", r.error) if r.error is not None else ("fresh", r.body)
                remaining = deadline - time.monotonic()
                if len(out.keys() | full) == len(keys) or remaining <= 0 or self._closed or self.failed:
                    break
                self._cond.wait(remaining)
            if self.failed is not None:
                return {key: out.get(key) or ("failed", self.failed) for key in keys}
            for key in keys:
                if key in out:
                    continue
                r = self._results.get(key)
                if r is not None and r.body is not None:
                    out[key] = ("stale", r.body)
                else:
                    out[key] = ("closed" if self._closed else "busy" if key in full else "building", None)
        return out

    def stop(self, timeout: float) -> None:
        with self._cond:
            self._closed = True
            self._pending.clear()
            self._cond.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.info("dashboard: a section build is still running; the match log is closed when it ends")

    def _loop(self) -> None:
        try:
            while True:
                with self._cond:
                    while not self._pending and not self._closed:
                        self._cond.wait()
                    if self._closed:
                        return
                    key, _ = self._pending.popitem(last=False)
                    self._running = key
                body, error = None, None
                t0 = time.monotonic()
                try:
                    with self._lock:
                        section = self._builder.section(key[0], **dict(key[1]))
                    body = embed_json(section).encode("utf-8")
                except Exception as e:  # noqa: BLE001 - one bad section never stops the dashboard
                    logger.exception(f"dashboard: building section {key[0]} {dict(key[1])} failed")
                    error = f"{type(e).__name__}: {e}"
                logger.debug(f"dashboard: built {key[0]} {dict(key[1])} in {time.monotonic() - t0:.3f} s")
                with self._cond:
                    self._seq += 1
                    prev = self._results.pop(key, None)
                    keep = body if error is None else (prev.body if prev is not None else None)
                    self._results[key] = _Result(self._seq, error, keep)
                    while len(self._results) > MAX_RESULTS:
                        self._results.popitem(last=False)
                    self._running = None
                    self._cond.notify_all()
        except BaseException as e:  # noqa: BLE001 - logged and answered as 500 instead of 202 forever
            logger.exception("dashboard: the builder thread stopped; every request now answers 500")
            with self._cond:
                self.failed = f"builder stopped ({type(e).__name__}: {e})"
                self._cond.notify_all()
        finally:
            with self._lock:
                self._builder.close()


class DashboardServer(ThreadingHTTPServer):
    """A bound HTTP server over one run dir (``make_server``): ``port``, ``builder`` (its ``ReportBuilder``,
    called only by the builder thread), ``lock`` (held during every build), ``request_wait_sec``."""

    daemon_threads = True

    def __init__(self, run_dir: Path, address: tuple[str, int], refresh_sec: int) -> None:
        super().__init__(address, _Handler)
        try:
            self.run_dir = Path(run_dir)
            self.refresh_sec = int(refresh_sec)
            self.request_wait_sec = REQUEST_WAIT_SEC
            self.builder = ReportBuilder(self.run_dir, table_min_age_sec=TABLE_MIN_AGE_SEC)
            self.lock = threading.Lock()          # ReportBuilder is not thread-safe
            self.title = self.builder.title()       # read before the builder thread owns the builder
            self.page_sections = tuple(n for n in SECTIONS if n != "tournaments" or self.builder.include_tournaments)
            host = str(address[0])
            self.allowed_hosts = None if host in _WILDCARD else {host.lower()}
            self._builds = _Builds(self.builder, self.lock)
        except BaseException:
            self.server_close()
            raise

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def sections(self, keys: Sequence[_Key]) -> dict[_Key, tuple[str, bytes | str | None]]:
        return self._builds.get(keys, self.request_wait_sec)

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            builds = getattr(self, "_builds", None)       # absent when binding failed in __init__
            if builds is not None:
                builds.stop(CLOSE_WAIT_SEC)


class _Handler(BaseHTTPRequestHandler):
    server: DashboardServer

    def _host_allowed(self) -> bool:
        header = self.headers.get("Host")
        allowed = self.server.allowed_hosts
        if header is None or allowed is None:
            return True
        host = _host_of(header)
        return is_loopback(host) or host in allowed

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        url = urlsplit(self.path)
        try:
            if not self._host_allowed():
                self._json(403, {"error": f"Host {self.headers.get('Host')!r} is not this dashboard's address "
                                          f"(DNS rebinding guard); open it by its loopback or bound address"})
            elif url.path in ("/", "/index.html"):
                self._page()
            elif url.path.startswith("/api/") and url.path[5:] in SECTIONS:
                name = url.path[5:]
                query = parse_qs(url.query)
                filters = {k: query[k][-1] for k in FILTERS.get(name, ()) if k in query and query[k][-1]}
                self._section(name, filters)
            else:
                self._json(404, {"error": f"not found: {url.path}", "sections": list(SECTIONS)})
        except (BrokenPipeError, ConnectionResetError):
            pass                                            # the client went away
        except Exception as e:  # noqa: BLE001 - one bad request never stops the dashboard
            logger.exception(f"dashboard: {url.path} failed")
            try:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
            except OSError:
                pass

    def _page(self) -> None:
        names = self.server.page_sections
        answers = self.server.sections([_key(n) for n in names])
        sections = {}
        for name in names:
            state, payload = answers[_key(name)]
            if state in ("fresh", "stale"):
                sections[name] = json.loads(payload)
            elif state in ("error", "failed"):
                sections[name] = {"notes": [f"This section failed: {payload}"]}
            else:
                sections[name] = placeholder(name)
        data = {"format": REPORT_FORMAT, "title": self.server.title, "sections": sections}
        page = render_html(data, live=True, refresh_sec=self.server.refresh_sec)
        self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))

    def _section(self, name: str, filters: dict[str, str]) -> None:
        key = _key(name, filters)
        state, payload = self.server.sections([key])[key]
        retry = {"Retry-After": str(max(1, round(self.server.request_wait_sec)))}
        if state == "fresh":
            self._send(200, JSON, payload)
        elif state == "stale":
            self._send(200, JSON, payload, {"X-Colosseum-Stale": "1"})
        elif state in ("error", "failed"):
            self._json(500, {"error": payload})
        elif state == "building":
            self._json(202, placeholder(name), retry)
        elif state == "busy":
            self._json(503, {"error": "the dashboard is busy building other sections; retry", "building": True},
                       retry)
        else:
            self._json(503, {"error": "the dashboard is shutting down"})

    def _json(self, code: int, obj: dict, headers: dict[str, str] | None = None) -> None:
        self._send(code, JSON, embed_json(obj).encode("utf-8"), headers)

    def _send(self, code: int, ctype: str, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        logger.debug("dashboard %s - " + fmt, self.address_string(), *args)


def make_server(run_dir: str | Path, *, host: str = "127.0.0.1", port: int = 8765,
                refresh_sec: int = 10) -> DashboardServer:
    """A bound, not yet serving dashboard; ``port=0`` picks a free port (``server.port``)."""
    run_dir = Path(run_dir)
    if not is_run_dir(run_dir):
        raise ConfigError(f"{run_dir} is not a run directory (no metrics.jsonl, config.resolved.yaml, matches.db "
                          f"or tournaments/)")
    if not is_loopback(host):
        logger.warning(f"dashboard: listening on {host} without authentication; prefer 127.0.0.1 and an SSH "
                       f"tunnel (docs/ARENA_GUIDE.md)")
    try:
        return DashboardServer(run_dir, (host, int(port)), refresh_sec)
    except OSError as e:
        raise ConfigError(_bind_error(host, int(port), e)) from None


def _bind_error(host: str, port: int, e: OSError) -> str:
    if isinstance(e, socket.gaierror):
        return (f"cannot listen on {host!r}: {e}; the dashboard listens on IPv4 only: give --host an IPv4 address "
                f"(127.0.0.1 by default) or a name that resolves to one")
    if e.errno == errno.EADDRINUSE:
        return f"cannot listen on {host}:{port}: the port is in use; choose another --port (0: any free port)"
    if e.errno == errno.EADDRNOTAVAIL:
        return f"cannot listen on {host}:{port}: {host} is not an address of this machine; check --host"
    if e.errno == errno.EACCES:
        return (f"cannot listen on {host}:{port}: permission denied (ports below 1024 need privileges); choose "
                f"another --port")
    return f"cannot listen on {host}:{port}: {e}; check --host and --port"


def serve(run_dir: str | Path, *, host: str = "127.0.0.1", port: int = 8765, refresh_sec: int = 10) -> None:
    """Serve until KeyboardInterrupt (Ctrl+C), which propagates after the socket is closed."""
    server = make_server(run_dir, host=host, port=port, refresh_sec=refresh_sec)
    try:
        server.serve_forever()
    finally:
        server.server_close()
