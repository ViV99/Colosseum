/* Colosseum run report (SP4 spec block 5): renders the embedded JSON data; with config.live, refreshes
   every section from api/<section> (colosseum dashboard). No external resources: SVG charts drawn here. */
"use strict";
(function () {
  const CFG = JSON.parse(document.getElementById("colosseum-config").textContent);
  const DATA = JSON.parse(document.getElementById("colosseum-data").textContent);
  const ORDER = ["summary", "ratings", "matrix", "anchors", "opponents", "training", "system", "matches", "tournaments"];
  const TITLES = { summary: "Summary", ratings: "Ratings", matrix: "Results matrix", anchors: "Anchors",
    opponents: "Opponents", training: "Training", system: "System", matches: "Matches", tournaments: "Tournaments" };
  const CATEGORIES = ["latest", "snapshots", "rivals", "anchors", "fallback"];
  const STATE = {};          // per-section UI state (selected layout, agent, metrics, filters, sort)
  const COLOR_OF = {};       // entity / lineage / series name -> slot (color follows the entity, never its rank)
  let uid = 0;

  // ---------- helpers ----------
  function esc(v) {
    return String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function num(v, d) {
    if (v === null || v === undefined || Number.isNaN(v)) return "-";
    if (typeof v !== "number") return String(v);
    if (d === undefined) d = Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 1 ? 2 : 3;
    return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  /* a 95% interval; null (or a null bound: fewer than two games, an untestable cell) draws "-" */
  function ci(c, d) {
    return Array.isArray(c) && c.length === 2 && typeof c[0] === "number" && typeof c[1] === "number"
      ? "[" + num(c[0], d) + ", " + num(c[1], d) + "]" : "-";
  }
  /* "REPEATED" next to an interval that rests on repeated deterministic games (tournament marks) */
  function mark(where) {
    const list = (where || []).map((x) => (Array.isArray(x) ? x.join("/") : String(x)));
    return h("span", { class: "mark", title: "games repeat (not enough randomness)" + (list.length ? ": " + list.join(", ") : "") },
      "REPEATED" + (list.length ? " " + list.join(", ") : ""));
  }
  function withMark(text, where, on) { return on ? h("span", {}, text, mark(where)) : text; }
  /* the G2 non-transitivity test in one line; a null p-value means untestable (no pair with enough games) */
  function transitivityLine(t) {
    if (!t) return null;
    if (t.p_value === null || t.p_value === undefined)
      return h("p", { class: "note" }, "Transitivity (G²): untestable (no team pair with enough games).");
    if (t.flag) return null;
    return h("p", { class: "note" }, "Transitivity (G²): " + (t.g2 !== undefined ? "G² = " + num(t.g2, 1) + (t.dof !== undefined ? " on " + t.dof + " dof, " : ", ") : "") + "p = " + num(t.p_value, 4) + ": no sign of cycles.");
  }
  function slot(name) {
    if (!(name in COLOR_OF)) COLOR_OF[name] = Object.keys(COLOR_OF).length % 8;
    return "var(--s" + (COLOR_OF[name] + 1) + ")";
  }
  function catColor(name) { return "var(--s" + (CATEGORIES.indexOf(name) % 8 + 1) + ")"; }   // fixed per category
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
      else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid !== null && kid !== undefined) e.append(kid instanceof Node ? kid : String(kid));
    return e;
  }
  function notes(list) { return (list || []).map((n) => h("p", { class: "note" }, n)); }
  function st(name, key, dflt) {
    STATE[name] = STATE[name] || {};
    if (!(key in STATE[name])) STATE[name][key] = dflt;
    return STATE[name][key];
  }
  function select(name, key, label, options, onchange) {
    const cur = st(name, key, options.length ? options[0] : "");
    const s = h("select", { "aria-label": label, on: { change: (e) => { STATE[name][key] = e.target.value; onchange(); } } },
      options.map((o) => h("option", { value: o, selected: o === cur }, o === "" ? "(all)" : o)));
    return h("label", {}, label, s);
  }
  function tile(k, v, cls) { return h("div", { class: "tile" + (cls ? " " + cls : "") }, h("div", { class: "k" }, k), h("div", { class: "v" }, v)); }
  function statusBadge(status) {
    const cls = { running: "good", finished: "good", failed: "critical", stopped: "warning" }[status] || "";
    return h("span", { class: "status" }, h("i", { class: "dot " + cls, "aria-hidden": "true" }), status || "unknown");
  }
  const TIP = document.getElementById("tip");
  function showTip(ev, rows) {
    TIP.replaceChildren(...rows.map((r) => h("div", {}, r[0] !== null ? h("b", {}, r[0]) : null, r[1] ? " " + r[1] : "")));
    TIP.hidden = false;
    const x = Math.min(ev.clientX + 12, window.innerWidth - TIP.offsetWidth - 8);
    const y = Math.min(ev.clientY + 12, window.innerHeight - TIP.offsetHeight - 8);
    TIP.style.left = Math.max(4, x) + "px";
    TIP.style.top = Math.max(4, y) + "px";
  }
  function hideTip() { TIP.hidden = true; }

  // ---------- table: sortable, filterable, scrolls inside its own box ----------
  function table(name, cols, rows, opts) {
    opts = opts || {};
    const key = opts.key || "table";
    const sort = st(name, key + ":sort", opts.sort || null);
    const box = h("div", {});
    const filt = st(name, key + ":filter", "");
    const input = opts.filter === false ? null : h("input", { type: "search", placeholder: "filter", value: filt, "aria-label": "filter rows",
      on: { input: (e) => { STATE[name][key + ":filter"] = e.target.value; draw(); } } });
    const wrap = h("div", { class: "scroll tbl" });
    function draw() {
      const f = (STATE[name][key + ":filter"] || "").toLowerCase();
      let rs = rows.filter((r) => !f || cols.some((c) => String(c.text ? c.text(r) : r[c.key] ?? "").toLowerCase().includes(f)));
      const s = STATE[name][key + ":sort"];
      if (s) {
        const c = cols.find((x) => x.key === s.key);
        const val = (r) => (c && c.sortv ? c.sortv(r) : r[s.key]);
        rs = rs.slice().sort((a, b) => {
          const x = val(a), y = val(b);
          if (x === y) return 0;
          if (x === null || x === undefined) return 1;
          if (y === null || y === undefined) return -1;
          return (x < y ? -1 : 1) * (s.dir === "desc" ? -1 : 1);
        });
      }
      const head = h("tr", {}, cols.map((c) => h("th", {
        class: c.num ? "num" : null, scope: "col",
        "aria-sort": s && s.key === c.key ? (s.dir === "desc" ? "descending" : "ascending") : null, title: c.title || null,
        on: { click: () => { const cur = STATE[name][key + ":sort"]; STATE[name][key + ":sort"] = { key: c.key, dir: cur && cur.key === c.key && cur.dir === "asc" ? "desc" : "asc" }; draw(); } },
      }, c.label)));
      const body = rs.map((r) => h("tr", {}, cols.map((c) => {
        const v = c.cell ? c.cell(r) : c.text ? c.text(r) : c.fmt ? c.fmt(r[c.key]) : r[c.key] ?? "-";
        return h("td", { class: c.num ? "num" : null }, v);
      })));
      wrap.replaceChildren(h("table", {}, h("thead", {}, head), h("tbody", {}, body)));
      count.textContent = rs.length === rows.length ? rows.length + " rows" : rs.length + " of " + rows.length + " rows";
    }
    const count = h("span", { class: "muted" });
    box.append(h("div", { class: "controls" }, input, count), wrap);
    draw();
    return box;
  }

  // ---------- scales and SVG charts (drawn as markup inserted into an HTML element) ----------
  function extent(vals) {
    let lo = Infinity, hi = -Infinity;
    for (const v of vals) if (typeof v === "number" && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (lo === Infinity) return [0, 1];
    if (lo === hi) return [lo - 1, hi + 1];
    return [lo, hi];
  }
  function ticks(lo, hi, n) {
    const step0 = (hi - lo) / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  function tickLabel(v) { return Math.abs(v) >= 1e6 ? (v / 1e6).toPrecision(3) + "M" : Math.abs(v) >= 1e4 ? (v / 1e3).toPrecision(3) + "k" : String(+v.toPrecision(4)); }

  /* series: [{name, pts: [[x, y, lo, hi], ...]}]; hlines: [{name, y}]; y0/y1 fix the y range (e.g. 0..1). */
  function chartWidth() { return Math.max(300, Math.min(640, window.innerWidth - 56)); }
  function lineChart(opts) {
    const W = Math.min(opts.width || 640, chartWidth()), H = opts.height || 240, L = 48, R = 12, T = 22, B = 34;
    const series = opts.series.filter((s) => s.pts.length);
    const box = h("div", { class: "chart" });
    if (!series.length) { box.append(h("p", { class: "note" }, opts.empty || "no data yet")); return box; }
    const xs = series.flatMap((s) => s.pts.map((p) => p[0]));
    const ys = series.flatMap((s) => s.pts.flatMap((p) => [p[1], p[2], p[3]])).concat((opts.hlines || []).map((l) => l.y));
    const [x0, x1] = extent(xs);
    let [y0, y1] = extent(ys);
    if (opts.y0 !== undefined) y0 = Math.min(y0, opts.y0);
    if (opts.y1 !== undefined) y1 = Math.max(y1, opts.y1);
    const sx = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
    const sy = (y) => T + (1 - (y - y0) / (y1 - y0 || 1)) * (H - T - B);
    let svg = "";
    for (const t of ticks(y0, y1, 4)) svg += `<line x1="${L}" x2="${W - R}" y1="${sy(t)}" y2="${sy(t)}" stroke="var(--grid)"/><text x="${L - 6}" y="${sy(t) + 4}" text-anchor="end">${esc(tickLabel(t))}</text>`;
    for (const t of ticks(x0, x1, 5)) svg += `<text x="${sx(t)}" y="${H - B + 16}" text-anchor="middle">${esc(tickLabel(t))}</text>`;
    svg += `<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="var(--axis)"/>`;
    svg += `<text class="ttl" x="${W - R}" y="${H - 4}" text-anchor="end">${esc(opts.xlabel || "")}</text>`;
    if (opts.ylabel) svg += `<text class="ttl" x="${L}" y="12">${esc(opts.ylabel)}</text>`;
    for (const l of opts.hlines || []) {
      svg += `<line x1="${L}" x2="${W - R}" y1="${sy(l.y)}" y2="${sy(l.y)}" stroke="var(--ink-2)" stroke-dasharray="4 4"/>`;
      svg += `<text x="${W - R}" y="${sy(l.y) - 4}" text-anchor="end">${esc(l.name)}</text>`;
    }
    for (const s of series) {
      const c = slot(s.name);
      const band = s.pts.filter((p) => typeof p[2] === "number" && typeof p[3] === "number");
      if (band.length > 1) {
        const up = band.map((p) => `${sx(p[0])},${sy(p[3])}`), dn = band.slice().reverse().map((p) => `${sx(p[0])},${sy(p[2])}`);
        svg += `<polygon points="${up.concat(dn).join(" ")}" fill="${c}" fill-opacity="0.16" stroke="none"/>`;
      }
      const pts = s.pts.filter((p) => typeof p[1] === "number");
      svg += `<polyline points="${pts.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round"/>`;
      if (pts.length <= 40) for (const p of pts) svg += `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="3.5" fill="${c}" stroke="var(--surface)" stroke-width="2"/>`;
    }
    svg += `<line class="xh" x1="0" x2="0" y1="${T}" y2="${H - B}" stroke="var(--ink-2)" visibility="hidden"/>`;
    svg += `<rect class="hit" x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/>`;
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.label || "chart")}">${svg}</svg>`;
    const svgEl = box.firstChild, xh = svgEl.querySelector(".xh");
    const allX = Array.from(new Set(xs)).sort((a, b) => a - b);
    function move(ev) {
      const r = svgEl.getBoundingClientRect();
      const x = x0 + ((ev.clientX - r.left) * (W / r.width) - L) / (W - L - R) * (x1 - x0);
      let best = allX[0];
      for (const v of allX) if (Math.abs(v - x) < Math.abs(best - x)) best = v;
      xh.setAttribute("x1", sx(best)); xh.setAttribute("x2", sx(best)); xh.setAttribute("visibility", "visible");
      const rows = [[null, (opts.xlabel || "x") + " " + num(best, 0)]];
      for (const s of series) {
        const p = s.pts.find((q) => q[0] === best);
        if (p) rows.push([num(p[1], opts.digits), s.name + (typeof p[2] === "number" ? " " + ci([p[2], p[3]], opts.digits) : "") + (p[4] !== undefined ? ", n " + p[4] : "")]);
      }
      showTip(ev, rows);
    }
    svgEl.querySelector(".hit").addEventListener("pointermove", move);
    svgEl.querySelector(".hit").addEventListener("pointerleave", () => { xh.setAttribute("visibility", "hidden"); hideTip(); });
    if (series.length > 1 || (opts.hlines || []).length) {
      box.append(h("div", { class: "legend" }, series.map((s) => h("span", {}, h("i", { style: "background:" + slot(s.name) }), s.name)),
        (opts.hlines || []).map((l) => h("span", {}, h("i", { style: "background:var(--ink-2)" }), l.name + " (dashed)"))));
    }
    return box;
  }

  /* x: [...]; parts: [{name, values: [...]}] shares stacked from 0 (fixed category order). */
  function stackedArea(x, parts, opts) {
    const W = chartWidth(), H = 200, L = 40, R = 12, T = 8, B = 30;
    const box = h("div", { class: "chart" });
    if (!x.length) { box.append(h("p", { class: "note" }, "no data yet")); return box; }
    const [x0, x1] = extent(x);
    const sx = (v) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R);
    const sy = (v) => T + (1 - v) * (H - T - B);
    let svg = "", base = x.map(() => 0);
    for (const t of [0, 0.5, 1]) svg += `<line x1="${L}" x2="${W - R}" y1="${sy(t)}" y2="${sy(t)}" stroke="var(--grid)"/><text x="${L - 6}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`;
    for (const t of ticks(x0, x1, 5)) svg += `<text x="${sx(t)}" y="${H - B + 16}" text-anchor="middle">${esc(tickLabel(t))}</text>`;
    for (const p of parts) {
      const top = base.map((b, i) => b + (p.values[i] || 0));
      const up = x.map((v, i) => `${sx(v)},${sy(top[i])}`), dn = x.slice().reverse().map((v, j) => `${sx(v)},${sy(base[x.length - 1 - j])}`);
      svg += `<polygon points="${up.concat(dn).join(" ")}" fill="${catColor(p.name)}" stroke="var(--surface)" stroke-width="1"/>`;
      base = top;
    }
    svg += `<text class="ttl" x="${W - R}" y="${H - 4}" text-anchor="end">${esc(opts.xlabel || "")}</text>`;
    svg += `<rect class="hit" x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/>`;
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.label || "shares")}">${svg}</svg>`;
    const svgEl = box.firstChild;
    svgEl.querySelector(".hit").addEventListener("pointermove", (ev) => {
      const r = svgEl.getBoundingClientRect();
      const xv = x0 + ((ev.clientX - r.left) * (W / r.width) - L) / (W - L - R) * (x1 - x0);
      let i = 0;
      for (let j = 0; j < x.length; j++) if (Math.abs(x[j] - xv) < Math.abs(x[i] - xv)) i = j;
      showTip(ev, [[null, (opts.xlabel || "x") + " " + num(x[i], 0)]].concat(parts.map((p) => [num(p.values[i], 2), p.name])));
    });
    svgEl.querySelector(".hit").addEventListener("pointerleave", hideTip);
    box.append(h("div", { class: "legend" }, parts.map((p) => h("span", {}, h("i", { class: "box", style: "background:" + catColor(p.name) }), p.name))));
    return box;
  }

  function mix(a, b, t) {
    const pa = a.match(/\w\w/g).map((v) => parseInt(v, 16)), pb = b.match(/\w\w/g).map((v) => parseInt(v, 16));
    return "rgb(" + pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",") + ")";
  }
  function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

  /* axis: labels; cells[i][j] = {score, ci, n} | null (row i vs column j); cells with n < minGames are hatched. */
  function heatmap(axis, cells, minGames, info, label) {
    const n = axis.length, cs = Math.max(14, Math.min(34, Math.floor(560 / Math.max(1, n)))), LW = 120, TH = 96;
    const W = LW + n * cs + 4, H = TH + n * cs + 4, id = "hatch" + ++uid;
    const pos = cssVar("--div-pos") || "#2a78d6", neg = cssVar("--div-neg") || "#e34948", mid = cssVar("--div-mid") || "#f0efec";
    let svg = `<defs><pattern id="${id}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="var(--few)"/><line x1="0" y1="0" x2="0" y2="6" stroke="var(--axis)" stroke-width="2"/></pattern></defs>`;
    axis.forEach((a, i) => {
      const short = a.length > 18 ? a.slice(0, 17) + "…" : a;
      svg += `<text x="${LW - 4}" y="${TH + i * cs + cs / 2 + 4}" text-anchor="end">${esc(short)}</text>`;
      svg += `<text transform="translate(${LW + i * cs + cs / 2 + 4},${TH - 4}) rotate(-60)">${esc(short)}</text>`;
    });
    cells.forEach((row, i) => row.forEach((c, j) => {
      if (!c) return;
      const few = c.n < minGames;
      const t = Math.min(1, Math.abs(c.score - 0.5) * 2);
      const fill = few ? `url(#${id})` : mix(mid, c.score >= 0.5 ? pos : neg, t);
      svg += `<rect data-i="${i}" data-j="${j}" x="${LW + j * cs + 1}" y="${TH + i * cs + 1}" width="${cs - 2}" height="${cs - 2}" rx="2" fill="${fill}"/>`;
    }));
    const box = h("div", { class: "chart" });
    box.innerHTML = `<div class="scroll"><svg viewBox="0 0 ${W} ${H}" style="width:${W}px;max-width:none" role="img" aria-label="${esc(label || "results matrix")}">${svg}</svg></div>`;
    const line = h("div", { class: "cellinfo" }, "Row vs column: score (blue = row wins more, red = fewer); hatched: fewer than " + minGames + " games. Tap a cell.");
    function describe(ev) {
      const r = ev.target.closest("rect[data-i]");
      if (!r) return null;
      const i = +r.dataset.i, j = +r.dataset.j, c = cells[i][j];
      return [axis[i] + " vs " + axis[j], "score " + num(c.score, 3) + " " + ci(c.ci, 3) + ", " + c.n + " games" + (c.n < minGames ? " (few games)" : "")];
    }
    const svgEl = box.querySelector("svg");
    svgEl.addEventListener("pointermove", (ev) => { const d = describe(ev); if (d) showTip(ev, [[d[1], ""], [null, d[0]]]); else hideTip(); });
    svgEl.addEventListener("pointerleave", hideTip);
    svgEl.addEventListener("click", (ev) => { const d = describe(ev); if (d) line.textContent = d[0] + ": " + d[1]; });
    box.append(line);
    if (info) box.append(h("p", { class: "note" }, info));
    return box;
  }

  // ---------- sections ----------
  const R = {};

  R.summary = function (s) {
    const out = [];
    const budget = s.budget_frac !== null && s.budget_frac !== undefined ? num(100 * s.budget_frac, 1) + " %" : "-";
    out.push(h("div", { class: "tiles" },
      tile("State", statusBadge(s.status)),
      tile("Env steps", num(s.env_steps, 0) + (s.total_timesteps ? " / " + num(s.total_timesteps, 0) : "")),
      tile("Budget", budget),
      tile("Env steps/s", num(s.env_steps_per_sec, 0)),
      tile("Agents", String((s.agents || []).length))));
    if (s.status_detail) out.push(h("p", { class: "note" }, s.status_detail));
    if (s.retention) out.push(h("p", { class: "note" }, "Retention: " + s.retention));
    out.push(table("summary", [
      { key: "agent", label: "agent" }, { key: "kind", label: "kind" },
      { key: "bt", label: "BT ± CI (newest)", sortv: (r) => (r.bt ? r.bt.rating : null), text: (r) => (r.bt ? num(r.bt.rating, 0) + " " + ci(r.bt.ci, 0) + " (" + r.bt.entity + ", " + r.bt.layout + ")" : "-") },
      { key: "elo", label: "ELO", num: true, fmt: (v) => num(v, 0) },
      { key: "wr_vs_past", label: "wr_vs_past", num: true, fmt: (v) => num(v, 3) },
    ], s.agents || [], { filter: false }));
    if ((s.parents || []).length) out.push(h("p", { class: "note" }, "Resumed from: " + s.parents.join(" ← ")));
    return out.concat(notes(s.notes));
  };

  function layoutsOf(s) { return Object.keys(s.layouts || {}); }

  R.ratings = function (s) {
    const layouts = layoutsOf(s), out = [];
    if (!layouts.length) return [h("p", { class: "note" }, "No ratings yet.")].concat(notes(s.notes));
    const L = st("ratings", "layout", layouts[0]);
    const lay = s.layouts[L] || s.layouts[layouts[0]];
    const lineages = Array.from(new Set((lay.groups || []).flatMap((g) => g.entities.map((e) => e.lineage)).concat((lay.entities || []).map((e) => e.lineage)))).sort();
    out.push(h("div", { class: "controls" }, select("ratings", "layout", "layout", layouts, rerender("ratings")),
      select("ratings", "agent", "agent", [""].concat(lineages), rerender("ratings")),
      h("span", { class: "muted" }, "source: " + s.source + (s.uncertainty ? ", intervals: " + s.uncertainty : ""))));
    const agent = STATE.ratings.agent || "";
    if (lay.kind === "score") {
      out.push(h("h3", {}, L + " · mean score (one team)"));
      out.push(table("ratings", [
        { key: "entity", label: "entity" }, { key: "kind", label: "kind" }, { key: "mean", label: "mean score", num: true, fmt: (v) => num(v, 3) },
        { key: "ci", label: "95% CI", text: (r) => ci(r.ci, 3), sortv: (r) => (r.ci ? r.ci[0] : null) }, { key: "n", label: "matches", num: true },
      ], (lay.entities || []).filter((e) => !agent || e.lineage === agent), { key: "score", sort: { key: "mean", dir: "desc" } }));
      if (lay.truncated) out.push(h("p", { class: "note" }, lay.truncated + " more entities not shown (use the dashboard filters)."));
      out.push(h("h3", {}, L + " · team compositions (cross-play)"));
      out.push(table("ratings", [
        { key: "composition", label: "composition" }, { key: "mean", label: "mean score", num: true, fmt: (v) => num(v, 3) },
        { key: "ci", label: "95% CI", text: (r) => ci(r.ci, 3), sortv: (r) => (r.ci ? r.ci[0] : null) }, { key: "n", label: "matches", num: true },
        { key: "homogeneous", label: "homogeneous", text: (r) => (r.homogeneous ? "yes" : "no") },
      ], lay.compositions || [], { key: "comp", sort: { key: "mean", dir: "desc" } }));
      if (lay.compositions_truncated) out.push(h("p", { class: "note" }, lay.compositions_truncated + " more compositions not shown."));
      return out.concat(notes(s.notes));
    }
    out.push(transitivityLine(lay.transitivity));
    if (lay.transitivity && lay.transitivity.flag) {
      out.push(h("div", { class: "flag" }, "Possible cycles: G² = " + num(lay.transitivity.g2, 1) + " on " + lay.transitivity.dof + " dof, p = " + num(lay.transitivity.p_value, 4) + ". Pairs with the largest residuals:"));
      out.push(table("ratings", [{ key: "a", label: "team A" }, { key: "b", label: "team B" }, { key: "residual", label: "residual", num: true, fmt: (v) => num(v, 2) }, { key: "games", label: "games", num: true }], lay.transitivity.worst_pairs || [], { key: "g2", filter: false }));
    }
    if (lay.bias && Object.keys(lay.bias).length) out.push(h("p", { class: "note" }, "Team bias (Elo): " + Object.entries(lay.bias).map(([t, b]) => "team " + t + " " + num(b, 0)).join(", ")));
    (lay.groups || []).forEach((g, gi) => {
      out.push(h("h3", {}, L + " · group " + g.label + (g.component ? " · component " + g.component : "") + " · anchor " + g.anchor));
      const series = (g.series || []).filter((x) => !agent || x.lineage === agent).map((x) => ({ name: x.lineage, pts: x.points }));
      out.push(lineChart({ series, hlines: (g.anchors || []).map((a) => ({ name: a.entity, y: a.rating })), xlabel: "version", ylabel: "BT (Elo units)", digits: 0, label: "rating by version", empty: "no trainable entities in this group" }));
      out.push(table("ratings", [
        { key: "entity", label: "entity" }, { key: "kind", label: "kind" },
        { key: "rating", label: "BT", num: true, fmt: (v) => num(v, 0) },
        { key: "ci", label: "95% CI", text: (r) => ci(r.ci, 0), sortv: (r) => (r.ci ? r.ci[0] : null) },
        { key: "games", label: "games", num: true },
        { key: "p_best_lineage", label: "P(best of agent)", num: true, fmt: (v) => num(v, 2) },
        { key: "p_best_group", label: "P(best)", num: true, fmt: (v) => num(v, 2) },
        { key: "lcb", label: "LCB", num: true, fmt: (v) => num(v, 0), title: "lower bound of the deviation from the lineage mean; used by top-k" },
      ], g.entities.filter((e) => !agent || e.lineage === agent), { key: "g" + gi, sort: { key: "rating", dir: "desc" } }));
      if (g.truncated) out.push(h("p", { class: "note" }, g.truncated + " more entities not shown (use the dashboard filters)."));
    });
    return out.concat(notes(s.notes));
  };

  /* Live mode: the agent select asks the dashboard for that agent's snapshot map (and, for score layouts, the
     cross-play compositions with that agent: api/matrix?agent=); the layout is chosen here, from the full answer. */
  R.matrix = function (s) {
    const layouts = layoutsOf(s), out = [];
    const agentPick = CFG.live && (s.agents || []).length
      ? select("matrix", "agent", "agent", [""].concat(s.agents), () => refresh("matrix")) : null;
    if (!layouts.length) return [agentPick ? h("div", { class: "controls" }, agentPick) : null, h("p", { class: "note" }, "No matches yet.")].concat(notes(s.notes));
    const L = st("matrix", "layout", layouts[0]);
    const lay = s.layouts[L] || s.layouts[layouts[0]];
    out.push(h("div", { class: "controls" }, select("matrix", "layout", "layout", layouts, rerender("matrix")), agentPick));
    if (lay.kind === "score") {
      out.push(h("h3", {}, "Cross-play: team composition → mean score"));
      out.push(table("matrix", [{ key: "composition", label: "composition" }, { key: "mean", label: "mean score", num: true, fmt: (v) => num(v, 3) },
        { key: "ci", label: "95% CI", text: (r) => ci(r.ci, 3), sortv: (r) => (r.ci ? r.ci[0] : null) }, { key: "n", label: "matches", num: true },
        { key: "homogeneous", label: "homogeneous", text: (r) => (r.homogeneous ? "yes" : "no") }], lay.cross_play || [], { key: "cp", sort: { key: "mean", dir: "desc" } }));
      if (lay.truncated) out.push(h("p", { class: "note" }, lay.truncated + " more compositions not shown."));
      return out.concat(notes(s.notes));
    }
    if (lay.lineages && lay.lineages.axis.length) {
      out.push(h("h3", {}, "By agent (lineage), ordered by rating"));
      out.push(heatmap(lay.lineages.axis, lay.lineages.cells, s.min_games, null, "results by lineage"));
    }
    if (lay.snapshots && lay.snapshots.axis.length) {
      out.push(h("h3", {}, "Snapshots of " + lay.snapshots.agent));
      out.push(heatmap(lay.snapshots.axis, lay.snapshots.cells, s.min_games,
        lay.snapshots.thinned_from > lay.snapshots.axis.length ? "Showing " + lay.snapshots.axis.length + " of " + lay.snapshots.thinned_from + " entities (evenly spaced versions)." : null, "results by snapshot"));
    }
    return out.concat(notes(s.notes));
  };

  R.anchors = function (s) {
    const out = [];
    if (s.biased) out.push(h("div", { class: "flag" }, "From training games: biased by PFSP (opponents are chosen by the matchmaker, latest weights change). Enable evaluation (pulse) for an unbiased curve."));
    else if (s.source === "pulse") out.push(h("p", { class: "note" }, "From pulse evaluation: every snapshot against the same field on the same seeds."));
    const drawn = new Set();
    for (const [L, agents] of Object.entries(s.layouts || {})) for (const [agent, a] of Object.entries(agents)) {
      drawn.add(agent);
      const score = a.metric === "score";      // one-team layout: mean team score with the anchor as teammate
      out.push(h("h3", {}, L + " · " + agent + (score ? " · mean team score (teammate; (self): alone)" : "")));
      out.push(lineChart(Object.assign({ series: Object.entries(a.anchors || {}).map(([name, pts]) => ({ name, pts })), xlabel: "version", digits: 3 },
        score ? { ylabel: "mean team score", label: "mean score with anchors" } : { ylabel: "score vs anchor", y0: 0, y1: 1, label: "score against anchors" })));
      if ((a.unevaluated || []).length) out.push(h("p", { class: "note" }, "Not evaluated: v" + a.unevaluated.join(", v")));
    }
    for (const [agent, versions] of Object.entries(s.unevaluated || {})) {
      if (!drawn.has(agent) && (versions || []).length) out.push(h("p", { class: "note" }, "Not evaluated (" + agent + "): v" + versions.join(", v")));
    }
    if (!Object.keys(s.layouts || {}).length) out.push(h("p", { class: "note" }, "No games against anchors yet."));
    return out.concat(notes(s.notes));
  };

  R.opponents = function (s) {
    const out = [];
    for (const [agent, layouts] of Object.entries(s.agents || {})) for (const [L, d] of Object.entries(layouts)) {
      out.push(h("h3", {}, agent + " · " + L + " · played opponent categories"));
      out.push(stackedArea(d.env_steps, CATEGORIES.filter((c) => c in d.categories).map((c) => ({ name: c, values: d.categories[c] })), { xlabel: "env steps", label: "opponent shares" }));
    }
    for (const [L, owners] of Object.entries(s.pfsp || {})) for (const [owner, rows] of Object.entries(owners)) {
      out.push(h("h3", {}, "PFSP · " + L + " · " + owner + " (top " + rows.length + " by games)"));
      out.push(table("opponents", [{ key: "player", label: "player" }, { key: "score", label: "EMA score", num: true, fmt: (v) => num(v, 3) }, { key: "games", label: "games", num: true, fmt: (v) => num(v, 0) }], rows, { key: L + owner, filter: false }));
    }
    if (!out.length) out.push(h("p", { class: "note" }, "No opponent data yet."));
    return out.concat(notes(s.notes));
  };

  function seriesPicker(name, all, dflt, after) {
    const chosen = st(name, "metrics", dflt.filter((m) => all.includes(m)));
    return h("details", {}, h("summary", {}, "series (" + chosen.length + " of " + all.length + ")"),
      h("div", { class: "chips" }, all.map((m) => h("label", {}, h("input", { type: "checkbox", checked: chosen.includes(m), on: { change: (e) => {
        const set = new Set(STATE[name].metrics);
        if (e.target.checked) set.add(m); else set.delete(m);
        STATE[name].metrics = all.filter((x) => set.has(x)); after();
      } } }), m))));
  }

  R.training = function (s) {
    const agents = Object.keys(s.agents || {}), out = [];
    if (!agents.length) return [h("p", { class: "note" }, "No train records yet.")].concat(notes(s.notes));
    const A = st("training", "agent", agents[0]);
    const d = s.agents[A] || s.agents[agents[0]];
    const all = Object.keys(d.metrics).sort();
    out.push(h("div", { class: "controls" }, select("training", "agent", "agent", agents, rerender("training")), seriesPicker("training", all, s.default || [], rerender("training"))));
    out.push(h("div", { class: "grid2" }, STATE.training.metrics.map((m) => h("div", {}, h("h3", {}, m),
      lineChart({ series: [{ name: m, pts: d.train_step.map((x, i) => [x, d.metrics[m][i]]).filter((p) => p[1] !== null) }], xlabel: "train step", height: 180, width: 360, label: m })))));
    return out.concat(notes(s.notes));
  };

  R.system = function (s) {
    const out = [], latest = s.latest || {};
    const pick = (k, d) => (k in latest ? num(latest[k], d) : "-");
    out.push(h("div", { class: "tiles" },
      tile("Env steps/s", pick("env_steps_per_sec", 0)),
      tile("Dropped match results", pick("dropped_match_results", 0)),
      tile("Dropped reward episodes", pick("dropped_reward_episodes", 0)),
      tile("Journal", s.journal ? num(s.journal.file_mb, 1) + " MB" + (s.journal.counts_only ? " · counts only" : "") : "-"),
      tile("Free disk", s.free_disk_mb !== null && s.free_disk_mb !== undefined ? num(s.free_disk_mb / 1024, 1) + " GB" : "-")));
    if (s.journal && s.journal.counts_only) out.push(h("div", { class: "flag critical" }, "Journal in counts-only mode: free disk below match_log.min_free_disk_mb."));
    const all = Object.keys(s.series || {}).sort();
    if (all.length) {
      out.push(h("div", { class: "controls" }, seriesPicker("system", all, s.default || [], rerender("system"))));
      out.push(h("div", { class: "grid2" }, STATE.system.metrics.map((m) => h("div", {}, h("h3", {}, m),
        lineChart({ series: [{ name: m, pts: s.env_steps.map((x, i) => [x, s.series[m][i]]).filter((p) => p[1] !== null) }], xlabel: "env steps", height: 180, width: 360, label: m })))));
    }
    if (s.journal) out.push(h("p", { class: "note" }, "Journal: " + Object.entries(s.journal).map(([k, v]) => k + " " + num(v)).join(", ")));
    return out.concat(notes(s.notes));
  };

  function composition(m) {
    const teams = {};
    for (const x of m.seats || []) (teams[x.team] = teams[x.team] || []).push(x.agent_id + (x.rating_version !== null && x.rating_version !== undefined ? "@v" + x.rating_version : "") + (x.network_id && x.network_id !== "latest" && x.network_id !== "fixed" ? " (" + x.network_id + ")" : ""));
    const ranks = {};
    for (const t of m.teams || []) ranks[t.team] = t;
    return Object.keys(teams).map((t) => "t" + t + " [" + (ranks[t] ? (m.outcome_kind === "score" ? "score " + num(ranks[t].score, 2) : "rank " + ranks[t].rank) : "?") + "]: " + teams[t].join(", ")).join(" | ");
  }
  function outcomeFor(m, agent) {
    if (!agent) return "";
    const mine = (m.seats || []).filter((x) => x.agent_id === agent).map((x) => x.team);
    if (!mine.length || m.outcome_kind === "score") return "";
    const rank = {};
    for (const t of m.teams || []) rank[t.team] = t.rank;
    const own = Math.min(...mine.map((t) => rank[t])), others = Object.keys(rank).filter((t) => !mine.includes(+t)).map((t) => rank[t]);
    if (!others.length) return "";
    const best = Math.min(...others);
    return own < best ? "win" : own === best ? "draw" : "loss";
  }
  R.matches = function (s) {
    const rows = s.rows || [], out = [];
    const opts = (f) => [""].concat(Array.from(new Set(rows.flatMap(f))).sort());
    const flt = s.filters || {};
    for (const [k, v] of Object.entries(flt)) if (v !== null && v !== undefined && !(STATE.matches && k in STATE.matches)) st("matches", k, String(v));
    out.push(h("div", { class: "controls" },
      select("matches", "agent", "agent", opts((m) => (m.seats || []).map((x) => x.agent_id)), refilter),
      select("matches", "layout", "layout", opts((m) => [m.layout]), refilter),
      select("matches", "source", "source", opts((m) => [m.source]), refilter),
      select("matches", "outcome", "outcome", ["", "win", "draw", "loss"], refilter)));
    const f = STATE.matches;
    const shown = rows.filter((m) => (!f.agent || (m.seats || []).some((x) => x.agent_id === f.agent)) && (!f.layout || m.layout === f.layout)
      && (!f.source || m.source === f.source) && (!f.outcome || outcomeFor(m, f.agent) === f.outcome));
    out.push(table("matches", [
      { key: "env_steps", label: "env steps", num: true, fmt: (v) => num(v, 0) },
      { key: "source", label: "source" }, { key: "layout", label: "layout" },
      { key: "composition", label: "teams (versions)", text: composition },
      { key: "seed", label: "seed", num: true, fmt: (v) => (v === null || v === undefined ? "-" : String(v)) },
      { key: "episode_length", label: "length", num: true },
      { key: "replay", label: "replay", cell: replayCell },
      { key: "uid", label: "uid" },
    ], shown, { key: "m" }));
    out.push(h("p", { class: "note" }, "The latest " + rows.length + " matches" + (CFG.live ? " matching the filters." : " (filters apply to these).")));
    return out.concat(notes(s.notes));
  };
  /* A replay link from the env: a link only for http(s) and file URLs (an allow-list: "java\tscript:" and
     friends are normalized by the URL parser, so a deny-list can be bypassed); anything else is plain text. */
  const REPLAY_PROTOCOLS = ["http:", "https:", "file:"];
  function replayCell(r) {
    if (r.replay === null || r.replay === undefined || r.replay === "") return "-";
    let ok = false;
    try { ok = REPLAY_PROTOCOLS.includes(new URL(String(r.replay), location.href).protocol); } catch (e) { ok = false; }
    return ok ? h("a", { href: String(r.replay) }, "replay") : String(r.replay);
  }
  function refilter() { if (CFG.live) refresh("matches"); else rerender("matches")(); }

  /* One role group of a tournament (modes.report_view): candidates, the score matrix, the layout combination,
     the selection, rounds and flags. REPEATED marks sit next to every interval that rests on repeated
     deterministic games; a selection that is not confirmed is never drawn as a confirmed one. */
  function tournamentGroup(name, g, gi) {
    const out = [];
    const cands = g.candidates || [];
    out.push(table(name, [
      { key: "display", label: "candidate", text: (r) => r.display || r.name },
      { key: "score", label: "score vs field", num: true, fmt: (v) => num(v, 3) },
      { key: "score_ci", label: "95% CI", text: (r) => ci(r.score_ci, 3) + ((r.repeated || []).length ? " REPEATED" : ""),
        cell: (r) => withMark(ci(r.score_ci, 3), r.repeated, (r.repeated || []).length > 0), sortv: (r) => (r.score_ci ? r.score_ci[0] : null) },
      { key: "worst", label: "worst opponent", text: (r) => (r.worst ? r.worst.opponent + " " + num(r.worst.score, 3) : "-") },
      { key: "bt", label: "BT", num: true, fmt: (v) => num(v, 0) },
      { key: "bt_ci", label: "BT CI", text: (r) => ci(r.bt_ci, 0) },
      { key: "p_best", label: "P(best)", num: true, fmt: (v) => num(v, 2) },
      { key: "games", label: "games", num: true },
      { key: "eliminated_round", label: "out in round", num: true, fmt: (v) => (v === null || v === undefined ? "-" : String(v)) },
    ], cands, { key: "c" + gi, sort: { key: "score", dir: "desc" } }));
    if (g.matrix && Object.keys(g.matrix).length) {
      const field = Array.from(new Set(Object.values(g.matrix).flatMap((row) => Object.keys(row))));
      const txt = (c) => num(c.score, 3) + " " + ci(c.ci, 2) + " n " + (c.games === null || c.games === undefined ? "-" : c.games);
      out.push(table(name, [{ key: "cand", label: "candidate \\ field" }].concat(field.map((f) => ({
        key: f, label: f, text: (r) => (r[f] ? txt(r[f]) + (r[f].repeated ? " REPEATED" : "") : "-"),
        cell: (r) => (r[f] ? withMark(txt(r[f]), [], r[f].repeated) : "-"), sortv: (r) => (r[f] ? r[f].score : null) }))),
        Object.entries(g.matrix).map(([cand, row]) => Object.assign({ cand }, row)), { key: "mx" + gi, filter: false }));
    }
    const weights = Object.entries(g.candidate_weights || {});
    if (weights.length) out.push(h("p", { class: "note" }, "Candidates that are also field members play the field without themselves; their scores use the column weights " +
      weights.map(([c, w]) => c + ": [" + (w || []).map((x) => num(x, 2)).join(", ") + "]").join("; ") + "."));
    if (Array.isArray(g.combined) && g.combined.length) {
      out.push(h("p", { class: "note" }, "Combined over the layouts (layout weights):"));
      out.push(table(name, [
        { key: "name", label: "candidate" },
        { key: "score", label: "combined score", num: true, fmt: (v) => num(v, 3) },
        { key: "score_ci", label: "95% CI", text: (r) => ci(r.score_ci, 3) + ((r.repeated || []).length ? " REPEATED" : ""),
          cell: (r) => withMark(ci(r.score_ci, 3), r.repeated, (r.repeated || []).length > 0), sortv: (r) => (r.score_ci ? r.score_ci[0] : null) },
      ], g.combined, { key: "cb" + gi, filter: false, sort: { key: "score", dir: "desc" } }));
    }
    if (g.select) {
      const sel = g.select, rep = (sel.diff_repeated || []).length > 0;
      const diff = sel.runner_up ? h("span", {}, ", difference to the runner-up " + sel.runner_up + ": " + num(sel.diff, 3) + " ",
        withMark(ci(sel.diff_ci, 3), sel.diff_repeated, rep)) : (rep ? mark(sel.diff_repeated) : null);
      if (sel.confirmed) {
        out.push(h("div", { class: "flag good" }, h("span", { class: "status" }, h("i", { class: "dot good", "aria-hidden": "true" }), "confirmed"),
          " winner: " + sel.winner, diff));
      } else {
        out.push(h("div", { class: "flag" }, h("span", { class: "status" }, h("i", { class: "dot warning", "aria-hidden": "true" }), "not confirmed"),
          ": best stage B score " + sel.winner + " (no confirmed selection)", diff,
          (sel.indistinguishable || []).length ? "; indistinguishable: " + sel.indistinguishable.join(", ") : ""));
      }
      if (sel.note) out.push(h("p", { class: "note" }, "Note: " + sel.note + "."));
    }
    for (const r of g.rounds || []) out.push(h("p", { class: "note" }, "Round " + r.round + " (" + r.blocks + " blocks): eliminated " + ((r.eliminated || []).join(", ") || "none")));
    if (g.transitivity && g.transitivity.flag) out.push(h("div", { class: "flag" }, "Possible cycles among the tournament's players (p = " + num(g.transitivity.p_value, 4) + ")."));
    else out.push(transitivityLine(g.transitivity));
    for (const p of g.repeated_pairs || []) out.push(h("div", { class: "flag" }, "Games repeat (not enough randomness): " + (Array.isArray(p) ? p.join(" vs ") : String(p))));
    return out;
  }
  function tournamentWinner(r) {
    if (r.winner === undefined || r.winner === null) return null;
    if (r.confirmed === true) return h("div", { class: "flag good" }, h("span", { class: "status" }, h("i", { class: "dot good", "aria-hidden": "true" }), "Selected: " + r.winner + " (confirmed)"));
    return h("div", { class: "flag" }, h("span", { class: "status" }, h("i", { class: "dot warning", "aria-hidden": "true" }),
      "Not confirmed: " + r.winner + " has the best stage B score, but the selection could not be confirmed"));
  }
  R.tournaments = function (s) {
    const out = [];
    for (const item of s.items || []) {
      const r = item.report || {};
      out.push(h("h3", {}, (item.name || r.name || "tournament") + " · " + (r.mode || "") + (r.seed !== undefined && r.seed !== null ? " · seed " + r.seed : "")));
      out.push(tournamentWinner(r));
      for (const [L, lay] of Object.entries(r.layouts || {})) {
        const groups = lay.groups || { default: lay };
        if ((lay.self_excluded || []).length) out.push(h("p", { class: "note" }, "layout " + L + ": " + lay.self_excluded.join(", ") + " also in the field: played the field without themselves."));
        if ((lay.dropped || []).length) {
          out.push(h("div", { class: "flag" }, "layout " + L + ": compositions the schedule could not seat"));
          out.push(table("tournaments", [{ key: "candidate", label: "candidate" }, { key: "composition", label: "composition", text: (d) => (Array.isArray(d.composition) ? d.composition.join(", ") : String(d.composition ?? "-")) },
            { key: "reason", label: "reason" }], lay.dropped, { key: "dr" + (item.name || "") + L, filter: false }));
        }
        Object.entries(groups).forEach(([gname, g], gi) => {
          out.push(h("p", { class: "note" }, "layout " + L + (gname !== "default" ? " · group " + gname : "") + (lay.outcome_kind ? " · " + lay.outcome_kind : "")));
          out.push(...tournamentGroup("tournaments", g, (item.name || "") + L + gi));
        });
      }
      out.push(h("details", {}, h("summary", {}, "raw report"), h("pre", {}, JSON.stringify(r, null, 1))));
    }
    if (!out.length) out.push(h("p", { class: "note" }, "No tournaments in this run."));
    return out.concat(notes(s.notes));
  };

  // ---------- mount, navigation, live refresh ----------
  function mount(name) {
    const sec = document.getElementById("sec-" + name);
    if (!sec) return;
    const s = DATA.sections[name];
    if (!s) { sec.hidden = true; return; }
    sec.hidden = false;
    const body = sec.querySelector(".body");
    try {
      body.replaceChildren(...R[name](s).filter((x) => x !== null && x !== undefined));
    } catch (err) {
      body.replaceChildren(h("p", { class: "flag critical" }, "This section could not be drawn: " + err), h("details", {}, h("summary", {}, "data"), h("pre", {}, JSON.stringify(s, null, 1))));
    }
    body.dataset.rendered = "1";
  }
  function rerender(name) { return () => mount(name); }
  /* The filters a live refresh sends. Ratings and the matrix layout are filtered here, on the full answer (it is
     bounded per layout), so their selects keep every layout and agent; the matrix sends its agent (the snapshot
     map is built for it), the matches their filters (the latest rows that match). */
  function query(name) {
    const f = STATE[name] || {}, keys = { matrix: ["agent"], matches: ["agent", "layout", "source", "outcome"] }[name] || [];
    const q = keys.filter((k) => f[k]).map((k) => encodeURIComponent(k) + "=" + encodeURIComponent(f[k])).join("&");
    return q ? "?" + q : "";
  }
  let inflight = 0;
  const SEQ = {};            // per section: only the answer to the latest request is drawn
  function refresh(name) {
    inflight++;
    const seq = SEQ[name] = (SEQ[name] || 0) + 1;
    return fetch("api/" + name + query(name), { cache: "no-store" }).then((r) => (r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status))))
      .then((s) => { if (seq === SEQ[name] && JSON.stringify(s) !== JSON.stringify(DATA.sections[name])) { DATA.sections[name] = s; mount(name); } })
      .catch((e) => { document.getElementById("live-status").textContent = "live · update failed: " + e.message; })
      .finally(() => { inflight--; });
  }
  const EVERY = Math.max(2, CFG.refresh_sec);
  function tick() {
    if (inflight) return;
    Promise.all(ORDER.map(refresh)).then(() => {
      document.getElementById("live-status").textContent = "live · refreshed every " + EVERY + " s · last " + new Date().toLocaleTimeString();
    });
  }
  document.getElementById("nav").replaceChildren(...ORDER.filter((n) => DATA.sections[n]).map((n) => h("a", { href: "#sec-" + n }, TITLES[n])));
  for (const n of ORDER) mount(n);
  document.body.dataset.ready = "1";
  if (CFG.live) {
    document.getElementById("live-status").textContent = "live · refreshed every " + EVERY + " s";
    setInterval(tick, EVERY * 1000);
  } else if (DATA.generated) {
    document.getElementById("live-status").textContent = "static report · " + DATA.generated;
  }
})();
