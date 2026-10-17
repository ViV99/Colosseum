"""Self-contained HTML of a run report (SP4 spec block 5): ``assets/page.{html,css,js}`` (package data) with
the report data embedded as JSON. No external resources: no CDN, fonts or libraries; charts are SVG drawn
by ``page.js``. The same page serves ``colosseum report`` (static), ``colosseum dashboard`` (``live=True``:
it refreshes every section from ``api/<section>``) and tournament reports (only the tournaments section).
"""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from colosseum.core.errors import ConfigError
from colosseum.utils.fs import write_text_atomic

REPORT_FILE = "report.html"
_PLACEHOLDER = re.compile(r"\{\{([A-Z]+)\}\}")


class ReportSectionsFailed(Exception):
    """``write_report`` wrote the page, but some sections failed: each is drawn as a note in its place and its
    traceback is logged. ``path``: the page; ``failed``: section -> error."""

    def __init__(self, path: Path, failed: Mapping[str, str]) -> None:
        self.path, self.failed = Path(path), dict(failed)
        listed = "; ".join(f"{name} ({error})" for name, error in self.failed.items())
        super().__init__(f"{self.path} was written, but these sections failed: {listed}")


def asset(name: str) -> str:
    """Text of ``colosseum/arena/report/assets/<name>`` (installed as package data)."""
    return resources.files("colosseum.arena.report").joinpath("assets", name).read_text(encoding="utf-8")


def _clean(obj: Any) -> Any:
    """JSON-safe copy: non-finite floats -> None, tuples -> lists, unknown objects -> str."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if hasattr(obj, "item"):                       # numpy scalars
        return _clean(obj.item())
    return str(obj)


def embed_json(data: Any) -> str:
    """JSON safe inside ``<script type="application/json">``: ``<``, ``>``, ``&`` and U+2028/9 are escaped,
    so no string in the data can close the script element."""
    text = json.dumps(_clean(data), separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    return (text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def render_html(data: dict, *, live: bool = False, refresh_sec: int = 10) -> str:
    """The page for report ``data`` (``data.build_report_data``); placeholders are filled in one pass, so
    nothing inside the CSS, JS or data is ever substituted again."""
    title = str(data.get("title") or "Colosseum run")
    values = {
        "TITLE": html.escape(title),
        "CSS": asset("page.css"),
        "JS": asset("page.js"),
        "CONFIG": embed_json({"live": bool(live), "refresh_sec": int(refresh_sec)}),
        "DATA": embed_json(data),
    }
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], asset("page.html"))


def is_tournament_dir(path: str | Path) -> bool:
    """A finished tournament's dir (``tournament.yaml`` and ``report.json``)."""
    path = Path(path)
    return (path / "tournament.yaml").exists() and (path / "report.json").exists()


def is_run_dir(path: str | Path) -> bool:
    """A training run dir (``metrics.jsonl`` or ``config.resolved.yaml``), or a ``colosseum tournament --output``
    dir, which is laid out like one (``matches.db``, ``tournaments/``; ruling C5)."""
    from colosseum.arena.matchlog import MATCH_LOG_FILE

    path = Path(path)
    return ((path / "metrics.jsonl").exists() or (path / "config.resolved.yaml").exists()
            or (path / MATCH_LOG_FILE).exists() or (path / "tournaments").is_dir())


def tournament_page_data(tournament_dir: str | Path) -> dict:
    """Report data with only the tournaments section, for one tournament dir."""
    from colosseum.arena.report.data import REPORT_FORMAT, ReportBuilder

    tdir = Path(tournament_dir)
    builder = ReportBuilder(tdir)
    try:
        return {"format": REPORT_FORMAT, "title": f"tournament {tdir.name}",
                "sections": {"tournaments": builder.section("tournaments")}}
    finally:
        builder.close()


def write_tournament_html(tournament_dir: str | Path) -> Path:
    """``<tournament dir>/report.html`` next to its ``report.json`` (called by the tournament when it finishes)."""
    tdir = Path(tournament_dir)
    return write_text_atomic(tdir / REPORT_FILE, render_html(tournament_page_data(tdir)))


def write_report(run_dir: str | Path, out: str | Path | None = None) -> Path:
    """``colosseum report``: the HTML of a run dir (default ``<run>/report.html``) or of a tournament dir. Sections
    are isolated: when some fail, the page is written with a note in their place and ``ReportSectionsFailed``
    is raised afterwards (the CLI exits 1)."""
    from colosseum.arena.report.data import ReportBuilder

    root = Path(run_dir)
    failed: dict[str, str] = {}
    if is_tournament_dir(root):
        if out is None:                        # one writer of a tournament dir's report.html
            return write_tournament_html(root)
        data = tournament_page_data(root)
    elif (root / "tournament.yaml").exists():
        raise ConfigError(f"{root} is a tournament directory, but the tournament has no report.json yet (it is "
                          f"running or was interrupted: rerun it with the same --name to finish it)")
    elif is_run_dir(root):
        builder = ReportBuilder(root)
        try:
            data = builder.build()
        finally:
            builder.close()
        failed = builder.failed_sections
        data["generated"] = "colosseum report"
    else:
        raise ConfigError(f"{root} is neither a run directory (metrics.jsonl, config.resolved.yaml, matches.db or "
                          f"tournaments/) nor a tournament directory (tournament.yaml and report.json)")
    target = Path(out) if out is not None else root / REPORT_FILE
    path = write_text_atomic(target, render_html(data))
    if failed:
        raise ReportSectionsFailed(path, failed)
    return path
